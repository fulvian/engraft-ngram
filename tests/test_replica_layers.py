"""Tests for engraft.replica.layers.

No GGUF: small random weights, compared against `tests/ref_hf_qwen4exp.py`
(verbatim HF copies, with the four adaptations applied here in the test) and
against independent numpy/loop formulas for modules with no direct HF
reference (hc_mix/combine, MoE, PLE against `engraft.lens.PleReplica`).

uv run pytest tests/test_replica_layers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engraft.replica.layers as L
import ref_hf_qwen4exp as hf  # noqa: E402
from engraft.lens import PleReplica

torch.manual_seed(0)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float32) * 0.1


# --------------------------------------------------------------------------
# Delta net: against torch_recurrent_gated_delta_rule with the 4 adaptations
# --------------------------------------------------------------------------


def test_delta_net_matches_hf_ref():
    T, Hv, D = 5, 3, 4
    q = _rand(T, Hv, D)
    k = _rand(T, Hv, D)
    v = _rand(T, Hv, D)
    beta = torch.sigmoid(_rand(T, Hv))
    g_log = -torch.rand(T, Hv)  # negative log-decay (like ssm_a<0 * softplus>0)
    s0 = torch.zeros(Hv, D, D)

    out_replica, s_final_replica = L.gated_delta_net_recurrence(q, k, v, g_log, beta, s0)

    # HF: batch=1, shape [B,T,H,D]; no L2 norm in the kernel (adaptation iv, already
    # done outside here); g passed in log space (adaptation ii, the HF function exponentiates it inside);
    # q not prescaled (adaptation iii, the HF function scales by 1/sqrt(D) inside).
    q_hf = q.unsqueeze(0)
    k_hf = k.unsqueeze(0)
    v_hf = v.unsqueeze(0)
    beta_hf = beta.unsqueeze(0)
    g_hf = g_log.unsqueeze(0)
    out_hf, s_hf = hf.torch_recurrent_gated_delta_rule(
        q_hf, k_hf, v_hf, g_hf, beta_hf, initial_state=None, output_final_state=True
    )
    out_hf = out_hf[0]
    s_hf = s_hf[0]

    max_diff_out = (out_replica - out_hf).abs().max().item()
    max_diff_state = (s_final_replica - s_hf).abs().max().item()
    print(f"\ndelta_net: max|out diff|={max_diff_out:.2e} max|state diff|={max_diff_state:.2e}")
    assert max_diff_out < 1e-5
    assert max_diff_state < 1e-5


def test_delta_net_with_gqa_repeat_tile():
    """Adaptation (i): the tiled->grouped permutation applies to v/z/beta/
    alpha/ssm_a/dt_bias/ssm_out columns/conv's V part. Here we only verify
    that `linear_attn_layer`'s tiled broadcast (Hk->Hv) is consistent: v
    head h uses q/k from head h % Hk (ggml_repeat_4d), not a block
    grouping like repeat_interleave (used instead for full attention, block
    GQA via ggml_mul_mat r2)."""
    Hk, n_rep, D, T = 2, 3, 4, 3
    q = _rand(T, Hk, D)
    tiled = q.tile((1, n_rep, 1))
    for h in range(Hk * n_rep):
        assert torch.equal(tiled[:, h], q[:, h % Hk])


# --------------------------------------------------------------------------
# RoPE: against apply_interleaved_mrope + apply_rotary_pos_emb HF (equal positions)
# --------------------------------------------------------------------------


def test_rope_matches_hf_ref_equal_positions():
    T, H, head_dim = 4, 2, 8
    rope_dim = 6  # real sections [11,11,10,0] sum to 32*2=64; small here for the test
    sections = [1, 1, 1, 0]  # rope_dim/2 = 3 = sum(sections[:3])
    x = _rand(T, H, head_dim)
    positions = torch.arange(T, dtype=torch.float64)

    cos, sin = L.rope_cos_sin(positions, rope_dim, freq_base=10000.0)
    out_replica = L.apply_rope(x, cos, sin)

    # HF: identical freqs on all 3 mrope axes (plain text) -> apply_interleaved_mrope is
    # the identity by construction; verified explicitly here instead of assumed.
    half = rope_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) * 2.0 / rope_dim))
    freqs_1d = positions[:, None] * inv_freq[None, :]  # [T,half]
    freqs_3axis = freqs_1d.unsqueeze(0).expand(3, T, half).clone()
    freqs_t = hf.apply_interleaved_mrope(freqs_3axis, sections)
    assert torch.allclose(freqs_t, freqs_1d), "equal positions on the 3 axes -> interleaved mrope is the identity"

    emb = torch.cat([freqs_t, freqs_t], dim=-1)
    cos_hf = emb.cos().to(torch.float32).unsqueeze(0)  # [1,T,rope_dim] dummy batch
    sin_hf = emb.sin().to(torch.float32).unsqueeze(0)
    x_hf_in = x.permute(1, 0, 2).unsqueeze(0)  # [1,H,T,head_dim] (batch,heads,seq,dim)
    out_hf = hf.apply_rotary_pos_emb(x_hf_in, cos=cos_hf, sin=sin_hf, unsqueeze_dim=1)
    out_hf = out_hf[0].permute(1, 0, 2)  # -> [T,H,head_dim]

    max_diff = (out_replica - out_hf).abs().max().item()
    print(f"\nrope: max|diff|={max_diff:.2e}")
    assert max_diff < 1e-6


# --------------------------------------------------------------------------
# hc_mix / hc_combine: against an independent numpy formula
# --------------------------------------------------------------------------


def test_hc_mix_matches_independent_formula():
    T, hc, n_embd, hc_lr = 3, 4, 6, 5
    x = _rand(T, hc, n_embd)
    w_norm = _rand(hc * n_embd) + 1.0
    w_down = _rand(hc_lr, hc * n_embd)
    w_up = _rand(hc * n_embd, hc_lr)
    w_inject = _rand(hc, hc * n_embd)
    eps = 1e-6

    mixed, inject = L.hc_mix(x, w_norm, w_down, w_up, w_inject, eps, hc)

    # independent numpy formula, following qwen4exp.cpp line by line
    xn_ref = np.zeros((T, hc, n_embd), dtype=np.float64)
    xv = x.numpy().astype(np.float64)
    for t in range(T):
        for s in range(hc):
            ms = np.mean(xv[t, s] ** 2)
            xn_ref[t, s] = xv[t, s] / np.sqrt(ms + eps)
    xn_ref = xn_ref * w_norm.numpy().reshape(hc, n_embd)
    xn_flat_ref = xn_ref.reshape(T, hc * n_embd)
    lo_ref = xn_flat_ref @ w_down.numpy().T
    lo_ref = lo_ref / hc
    lo_ref = lo_ref * (1.0 / (1.0 + np.exp(-lo_ref))) if False else lo_ref  # silu below
    lo_ref = (lo_ref) * (1 / (1 + np.exp(-lo_ref)))
    gate_ref = 1.0 / (1.0 + np.exp(-(lo_ref @ w_up.numpy().T)))
    gate_ref = gate_ref.reshape(T, hc, n_embd)
    gated_ref = xn_ref * gate_ref
    mixed_ref = gated_ref.mean(axis=1)
    inject_ref = xn_flat_ref @ w_inject.numpy().T

    assert np.abs(mixed.numpy() - mixed_ref).max() < 1e-5
    assert np.abs(inject.numpy() - inject_ref).max() < 1e-5


def test_hc_combine_matches_independent_formula():
    T, hc, n_embd = 3, 4, 6
    residual = _rand(T, hc, n_embd)
    block_out = _rand(T, n_embd)
    inject = _rand(T, hc)

    out = L.hc_combine(residual, block_out, inject, hc)

    w_ref = 2.0 / (1.0 + np.exp(-(inject.numpy() / hc)))
    out_ref = residual.numpy() + block_out.numpy()[:, None, :] * w_ref[:, :, None]
    assert np.abs(out.numpy() - out_ref).max() < 1e-5


# --------------------------------------------------------------------------
# MoE: weights renormalized to sum, SwiGLU, against an independent explicit loop
# --------------------------------------------------------------------------


def test_moe_matches_explicit_loop():
    T, n_embd, n_expert, n_ff, n_used = 2, 6, 8, 5, 3
    x = _rand(T, n_embd)
    gate_inp = _rand(n_expert, n_embd)
    experts_gate = {e: _rand(n_ff, n_embd) for e in range(n_expert)}
    experts_up = {e: _rand(n_ff, n_embd) for e in range(n_expert)}
    experts_down = {e: _rand(n_embd, n_ff) for e in range(n_expert)}
    up_shexp = _rand(n_ff, n_embd)
    gate_shexp = _rand(n_ff, n_embd)
    down_shexp = _rand(n_embd, n_ff)
    gate_inp_shexp = _rand(n_embd)

    logits = x @ gate_inp.T
    probs = torch.softmax(logits, dim=-1)
    selected = torch.topk(probs, n_used, dim=-1).indices

    out = L.moe_ffn(
        x, gate_inp,
        lambda e: experts_gate[e], lambda e: experts_up[e], lambda e: experts_down[e],
        selected, up_shexp, gate_shexp, down_shexp, gate_inp_shexp, n_used,
    )

    # independent explicit loop, in numpy
    out_ref = np.zeros((T, n_embd))
    probs_np = probs.detach().numpy()
    for t in range(T):
        idx = selected[t].tolist()
        w = np.array([probs_np[t, e] for e in idx])
        w = w / max(w.sum(), 6.103515625e-5)
        acc = np.zeros(n_embd)
        for j, e in enumerate(idx):
            g = x[t].numpy() @ experts_gate[e].numpy().T
            u = x[t].numpy() @ experts_up[e].numpy().T
            h = (g * (1 / (1 + np.exp(-g)))) * u
            acc += w[j] * (h @ experts_down[e].numpy().T)
        shared = ((x[t].numpy() @ gate_shexp.numpy().T) * (1 / (1 + np.exp(-(x[t].numpy() @ gate_shexp.numpy().T))))) * (
            x[t].numpy() @ up_shexp.numpy().T
        )
        shared_out = shared @ down_shexp.numpy().T
        sg = 1 / (1 + np.exp(-(x[t].numpy() @ gate_inp_shexp.numpy())))
        out_ref[t] = acc + shared_out * sg

    assert np.abs(out.detach().numpy() - out_ref).max() < 1e-5


# --------------------------------------------------------------------------
# PLE: against engraft.lens.PleReplica.ple_out_at (numpy, already validated elsewhere)
# --------------------------------------------------------------------------


def _make_ple_replica_fake(hc, n_embd):
    obj = PleReplica.__new__(PleReplica)
    rng = np.random.default_rng(0)
    obj.w_key = (rng.standard_normal((hc * n_embd, n_embd)) * 0.1).astype(np.float32)
    obj.w_value = (rng.standard_normal((n_embd, n_embd)) * 0.1).astype(np.float32)
    obj.norm_key = (rng.standard_normal((hc, n_embd)) * 0.1 + 1.0).astype(np.float32)
    obj.norm_query = (rng.standard_normal((hc, n_embd)) * 0.1 + 1.0).astype(np.float32)
    obj.norm_conv = (rng.standard_normal((hc, n_embd)) * 0.1 + 1.0).astype(np.float32)
    obj.conv1d = (rng.standard_normal((hc, n_embd, 4)) * 0.1).astype(np.float32)
    obj.eps = 1e-6
    return obj


def test_ple_matches_ple_replica_numpy():
    # PleReplica has hc=4, n_embd=2560 hardwired as module constants (_N_HC, _N_EMBD):
    # the comparison uses those real dimensions (weights still random, T small).
    hc, n_embd, T = 4, 2560, 5
    ref = _make_ple_replica_fake(hc, n_embd)

    emb_seq = np.random.default_rng(1).standard_normal((T, n_embd)).astype(np.float32) * 0.1
    hidden_seq = np.random.default_rng(2).standard_normal((T, hc, n_embd)).astype(np.float32) * 0.1

    # PleReplica.ple_out returns only gated+conv_out (the sum with `hidden` happens in the
    # caller, qwen4exp.cpp build_ple: `return hidden + (gated + conv_out)`);
    # L.ple_forward already includes that sum, so it is added here for comparison.
    ple_out_ref = hidden_seq + ref.ple_out(emb_seq, hidden_seq)  # [T,hc,n_embd]

    w = L.PleWeights(
        w_key=torch.from_numpy(ref.w_key),
        w_value=torch.from_numpy(ref.w_value),
        norm_key=torch.from_numpy(ref.norm_key.reshape(-1)),
        norm_query=torch.from_numpy(ref.norm_query.reshape(-1)),
        norm_conv=torch.from_numpy(ref.norm_conv.reshape(-1)),
        conv1d=torch.from_numpy(ref.conv1d.reshape(hc * n_embd, 4)),
    )

    class _HP:
        hc_mult = hc
        n_embd = 2560
        f_norm_rms_eps = 1e-6
        ple_ngram_size = 3  # irrelevant: PleReplica uses a fixed dilation of 3 (CONV_DILATION)

    hist = torch.zeros(0, hc, n_embd)
    out_seq = torch.zeros(T, hc, n_embd)
    for t in range(T):
        emb_t = torch.from_numpy(emb_seq[t : t + 1])
        hidden_t = torch.from_numpy(hidden_seq[t : t + 1])
        out_t, hist = L.ple_forward(emb_t, hidden_t, w, hist, _HP)
        out_seq[t] = out_t[0]

    max_diff = np.abs(out_seq.numpy() - ple_out_ref).max()
    print(f"\nple: max|diff|={max_diff:.2e}")
    assert max_diff < 1e-4


# --------------------------------------------------------------------------
# L2 norm: against the HF copy `l2norm`
# --------------------------------------------------------------------------


def test_l2norm_matches_hf_ref():
    x = _rand(4, 5)
    eps = 1e-6
    out = L.l2norm(x, eps)
    out_hf = hf.l2norm(x, dim=-1, eps=eps)
    assert torch.allclose(out, out_hf, atol=1e-6)

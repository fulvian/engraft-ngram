"""Pure torch (f32) functions mirroring the fork's computation graph.

Axis convention: **torch-natural**, time first. A ggml tensor `[ne0,
ne1, ne2]` (ne0 fastest) becomes `[..., ne2, ne1, ne0]` here, with the "time"
axis (when present) always first. Weights come from `weights.GgufWeights`,
which already dequantizes them in `[out, in]` convention (numpy reverses the
ggml axes, and a ggml linear tensor is stored `{in, out}` in ne-order, checked
on `blk.1.ple_key.weight`): hence `y = x @ w.T` everywhere, as in `nn.Linear`.

Main references (llama.cpp fork, `fork-ple` branch):
  - `src/models/qwen4exp.cpp` build_hc_mix/build_hc_combine (lines ~370-422): HC mixer.
  - `src/models/qwen4exp.cpp` build_layer_attn (~947-1030): full attention, RoPE,
    sigmoid gate on the output, [q|gate] interleaved in `wq`.
  - `src/models/qwen4exp.cpp` build_layer_attn_linear (~1034-1160): gated delta net.
  - `src/models/delta-net-base.cpp` build_delta_net_autoregressive (~262-320): the
    exact one-token recurrence (used here for *every* position, prefix included:
    the fork's "chunking" form is only an optimization, mathematically equivalent
    up to floating-point summation order, tolerable at the 0.02 nat threshold of
    the replica-vs-engine validation).
  - `ggml/src/ggml-cpu/ops.cpp` ggml_compute_forward_ssm_conv_f32 (~9606-9657): tap 0 =
    oldest position, tap K-1 = current (same convention as the PLE conv in
    `engraft/lens.py`, dilation 1 instead of 3).
  - `ggml/src/ggml-cpu/ggml-cpu.c:1385` (`r2 = ne12/ne02; i02 = i12/r2`): the GQA
    broadcast of `ggml_mul_mat` is **by consecutive blocks** (head h -> kv head h//r2),
    the same convention as `repeat_kv` in HF; used for full attention.
  - `ggml/src/ggml-cpu/ops.cpp` ggml_compute_forward_repeat_f32 (~1716-1756)
    (`dst[i1*ne01+k1] = src[k1]`): the broadcast of `ggml_repeat_4d` is instead
    **tiled** (head h -> head h % n_k_heads); used to repeat the delta net's q/k
    from `ssm_n_group` to `ssm_dt_rank` heads.
  - `src/llama-graph.cpp` build_moe_ffn (~1941-2100): softmax, top-k, weights
    renormalized by their sum (clamp 6.1e-5), no scale (`expert_weights_scale=0`
    in the metadata), SwiGLU, no clamp (`swiglu_clamp_exp` absent = 0).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Norme
# --------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm on the last axis. `weight` is already "1+w" (or the direct weight for
    ssm_norm, which the GGUF conversion treats without an offset because the source
    HF module has none for RMSNormGated): direct multiplication, never +1 here."""
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(ms + eps)
    return normed * weight


def rmsnorm_grouped(x: torch.Tensor, weight_flat: torch.Tensor, eps: float, hc: int) -> torch.Tensor:
    """x: [..., hc, n_embd]. RMSNorm per stream (last axis), then multiplies by
    `weight_flat` [hc*n_embd] reshaped [hc, n_embd] (build_hc_mix: `ggml_rms_norm`
    reduces ne0=n_embd, then the weight is the flatten [hc_dim] with hc slower than
    n_embd -- same order as `weight_flat.reshape(hc, n_embd)`)."""
    n_embd = x.shape[-1]
    w = weight_flat.reshape(hc, n_embd)
    return rmsnorm(x, w, eps)


def l2norm(x: torch.Tensor, eps: float, dim: int = -1) -> torch.Tensor:
    """True L2 norm (not RMS): x / sqrt(sum(x^2)+eps). `use_qk_l2norm_in_kernel=False`
    in the fork, applied outside with the model's eps."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# --------------------------------------------------------------------------
# Hyper-connections
# --------------------------------------------------------------------------


def hc_mix(
    x: torch.Tensor,  # [T, hc, n_embd]
    w_norm: torch.Tensor,  # [hc*n_embd]
    w_down: torch.Tensor,  # [hc_lr, hc*n_embd] (out,in)
    w_up: torch.Tensor,  # [hc*n_embd, hc_lr] (out,in)
    w_inject: torch.Tensor | None,  # [hc, hc*n_embd] (out,in)
    eps: float,
    hc: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """qwen4exp.cpp build_hc_mix. Returns (mixed [T,n_embd], inject [T,hc] or None)."""
    t_len, _, n_embd = x.shape
    hc_dim = hc * n_embd

    xn = rmsnorm_grouped(x, w_norm, eps, hc)  # [T,hc,n_embd]
    xn_flat = xn.reshape(t_len, hc_dim)

    lo = xn_flat @ w_down.T  # [T, hc_lr]
    lo = F.silu(lo / hc)

    gate = torch.sigmoid(lo @ w_up.T)  # [T, hc_dim]
    gate = gate.reshape(t_len, hc, n_embd)

    gated = xn * gate  # [T,hc,n_embd]
    mixed = gated.mean(dim=1)  # [T,n_embd] (media sui flussi = qwen4exp.cpp collapse)

    inject = None
    if w_inject is not None:
        inject = xn_flat @ w_inject.T  # [T, hc]

    return mixed, inject


def hc_combine(
    residual: torch.Tensor,  # [T, hc, n_embd]
    block_out: torch.Tensor,  # [T, n_embd]
    inject: torch.Tensor,  # [T, hc]
    hc: int,
) -> torch.Tensor:
    """qwen4exp.cpp build_hc_combine: 2*sigmoid centers the scatter weights on 1."""
    w = torch.sigmoid(inject / hc) * 2.0  # [T,hc]
    return residual + block_out.unsqueeze(1) * w.unsqueeze(-1)


# --------------------------------------------------------------------------
# RoPE (mrope interleaved, simplified for plain text: same positions on all 3 axes)
# --------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions: torch.Tensor, rope_dim: int, freq_base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for NeoX-style RoPE, dimension `rope_dim` (half rotating, half
    concatenated).

    For plain text (no image: the three mrope components [T,H,W] share the same
    positions), `apply_interleaved_mrope` (HF `Qwen4ExpTextRotaryEmbedding`)
    reduces to the identity: overwriting `freqs[0]` with `freqs[1]`/`freqs[2]`
    changes nothing when they are equal. Verified by `test_rope_matches_hf_ref`
    against the HF functions copied into `tests/ref_hf_qwen4exp.py`, sections
    [11,11,10,0]."""
    half = rope_dim // 2
    inv_freq = 1.0 / (freq_base ** (torch.arange(0, half, dtype=torch.float64) * 2.0 / rope_dim))
    freqs = positions.to(torch.float64)[:, None] * inv_freq[None, :]  # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, rope_dim]
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, D]. cos/sin: [T, rope_dim]. Rotates only the first rope_dim channels."""
    rope_dim = cos.shape[-1]
    x_rope, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    c = cos.unsqueeze(1)  # [T,1,rope_dim]
    s = sin.unsqueeze(1)
    x_rope = x_rope * c + rotate_half(x_rope) * s
    return torch.cat([x_rope, x_pass], dim=-1)


# --------------------------------------------------------------------------
# Attenzione piena (GQA, gate sigmoide, indexer denso => saltato)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class AttnWeights:
    wq: torch.Tensor  # [2*n_embd_head*n_head, n_embd] -- [q|gate] interleaved per head
    wk: torch.Tensor  # [n_embd_head*n_head_kv, n_embd]
    wv: torch.Tensor  # [n_embd_head*n_head_kv, n_embd]
    wo: torch.Tensor  # [n_embd, n_embd_head*n_head]
    q_norm: torch.Tensor  # [n_embd_head]
    k_norm: torch.Tensor  # [n_embd_head]


def split_q_gate(qcur_full: torch.Tensor, n_head: int, n_embd_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """qcur_full: [T, 2*n_embd_head*n_head]. `wq` interleaves [q|gate] per head
    (qwen4exp.cpp:964-978: the view takes one element every 2*n_embd_head, then
    the first n_embd_head columns are q, the next n_embd_head are the gate)."""
    t_len = qcur_full.shape[0]
    x = qcur_full.reshape(t_len, n_head, 2 * n_embd_head)
    q = x[:, :, :n_embd_head]
    gate = x[:, :, n_embd_head:]
    return q, gate.reshape(t_len, n_head * n_embd_head)


def attention_full(
    x: torch.Tensor,  # [T, n_embd] -- only the NEW positions (query)
    w: AttnWeights,
    positions: torch.Tensor,  # [T] absolute positions of the queries
    hparams,
    k_cache: torch.Tensor | None,  # [T_prev, n_head_kv, n_embd_head] already rotated, or None
    v_cache: torch.Tensor | None,  # [T_prev, n_head_kv, n_embd_head]
    cache_positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """qwen4exp.cpp build_layer_attn. Dense causal attention (the QSA indexer is
    dense for T <= top_k, per qwen4exp.cpp:854): no selection, full causal mask
    over new queries + cache. Returns (block_out [T,n_embd], k_new, v_new) (K/V
    of the new slice, already rotated, to append to the caller's cache)."""
    t_len = x.shape[0]
    n_head, n_head_kv, d = hparams.n_head, hparams.n_head_kv, hparams.n_embd_head

    qcur_full = x @ w.wq.T  # [T, 2*d*n_head]
    q, gate = split_q_gate(qcur_full, n_head, d)  # q:[T,n_head,d], gate:[T,n_head*d]
    q = rmsnorm(q, w.q_norm, hparams.f_norm_rms_eps)

    k = (x @ w.wk.T).reshape(t_len, n_head_kv, d)
    k = rmsnorm(k, w.k_norm, hparams.f_norm_rms_eps)
    v = (x @ w.wv.T).reshape(t_len, n_head_kv, d)

    cos, sin = rope_cos_sin(positions.to(torch.float64), hparams.rope_dim, hparams.rope_freq_base)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if k_cache is not None:
        k_all = torch.cat([k_cache, k], dim=0)
        v_all = torch.cat([v_cache, v], dim=0)
        pos_all = torch.cat([cache_positions, positions], dim=0)
    else:
        k_all, v_all, pos_all = k, v, positions

    n_rep = n_head // n_head_kv
    k_rep = k_all.repeat_interleave(n_rep, dim=1)  # [Tkv, n_head, d] blocco consecutivo (ggml mul_mat r2)
    v_rep = v_all.repeat_interleave(n_rep, dim=1)

    scale = 1.0 / (d ** 0.5)
    # [n_head, T, Tkv]
    scores = torch.einsum("thd,shd->hts", q, k_rep) * scale
    causal = positions[:, None] >= pos_all[None, :]
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores.to(torch.float32), dim=-1)
    out = torch.einsum("hts,shd->thd", probs, v_rep)  # [T,n_head,d]

    out = out.reshape(t_len, n_head * d)
    out = out * torch.sigmoid(gate)
    block_out = out @ w.wo.T
    return block_out, k, v


# --------------------------------------------------------------------------
# Causal depthwise conv (used by both the delta net and PLE)
# --------------------------------------------------------------------------


def causal_depthwise_conv(
    x_new: torch.Tensor,  # [T, C]
    history: torch.Tensor,  # [K-1, C] (zeros if start of sequence)
    weight: torch.Tensor,  # [C, K] (tap 0 = oldest, tap K-1 = current)
    dilation: int = 1,
) -> torch.Tensor:
    """out[t,c] = sum_k weight[c,k] * x[t-(K-1-k)*dilation, c] (ggml_ssm_conv,
    dilation=1; PLE reuses this same function with dilation=ngram_size)."""
    kern = weight.shape[1]
    t_len, c = x_new.shape
    full = torch.cat([history, x_new], dim=0)  # [(K-1)+T, C] (positions -(K-1)..T-1)
    hist_len = history.shape[0]
    acc = torch.zeros(t_len, c, dtype=x_new.dtype)
    for k in range(kern):
        back = (kern - 1 - k) * dilation
        w_k = weight[:, k]
        for t in range(t_len):
            src_idx = hist_len + t - back
            if src_idx >= 0:
                acc[t] += full[src_idx] * w_k
    return acc


# --------------------------------------------------------------------------
# Gated delta net
# --------------------------------------------------------------------------


@dataclasses.dataclass
class DeltaNetWeights:
    wqkv: torch.Tensor  # [key_dim*2+value_dim, n_embd]
    wqkv_gate: torch.Tensor  # [value_dim, n_embd]
    ssm_conv1d: torch.Tensor  # [conv_dim, K]
    ssm_dt_bias: torch.Tensor  # [num_v_heads]
    ssm_a: torch.Tensor  # [num_v_heads] = -exp(A_log), already like this in the GGUF
    ssm_beta: torch.Tensor  # [num_v_heads, n_embd] (out,in)
    ssm_alpha: torch.Tensor  # [num_v_heads, n_embd]
    ssm_norm: torch.Tensor  # [head_v_dim]
    ssm_out: torch.Tensor  # [n_embd, value_dim]


@dataclasses.dataclass
class DeltaNetState:
    conv_hist: torch.Tensor  # [K-1, conv_dim] -- last pre-conv inputs (qkv_mixed)
    s: torch.Tensor  # [num_v_heads, head_dim, head_dim] -- recurrence state


def delta_net_init_state(hparams) -> DeltaNetState:
    k_minus_1 = hparams.ssm_d_conv - 1
    return DeltaNetState(
        conv_hist=torch.zeros(k_minus_1, hparams.conv_dim),
        s=torch.zeros(hparams.ssm_dt_rank, hparams.ssm_d_state, hparams.ssm_d_state),
    )


def gated_delta_net_recurrence(
    q: torch.Tensor,  # [T, Hv, D]  (already repeated from Hk to Hv heads, L2-normalized, not prescaled)
    k: torch.Tensor,  # [T, Hv, D]
    v: torch.Tensor,  # [T, Hv, D]
    g_log: torch.Tensor,  # [T, Hv] -- ssm_a * softplus(alpha+dt_bias), in spazio log
    beta: torch.Tensor,  # [T, Hv]
    s0: torch.Tensor,  # [Hv, D, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    """delta-net-base.cpp build_delta_net_autoregressive, repeated position by
    position (mathematically equivalent to the fork's chunked form for T>1: see
    the module note). Scales q by 1/sqrt(D) in here. Returns (out [T,Hv,D],
    final state [Hv,D,D])."""
    t_len, hv, d = q.shape
    scale = 1.0 / (d ** 0.5)
    q_scaled = q * scale
    s = s0
    outs = []
    for t in range(t_len):
        g_t = torch.exp(g_log[t]).reshape(hv, 1, 1)
        s = s * g_t
        kv_mem = torch.einsum("hij,hi->hj", s, k[t])  # sum_i s[i,j]*k[i]
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)
        s = s + torch.einsum("hi,hj->hij", k[t], delta)
        o_t = torch.einsum("hij,hi->hj", s, q_scaled[t])
        outs.append(o_t)
    out = torch.stack(outs, dim=0)  # [T,Hv,D]
    return out, s


def linear_attn_layer(
    x: torch.Tensor,  # [T, n_embd] -- only the new positions
    w: DeltaNetWeights,
    state: DeltaNetState,
    hparams,
) -> tuple[torch.Tensor, DeltaNetState]:
    """qwen4exp.cpp build_layer_attn_linear in full, for a block of T new
    positions given the prefix's tails (conv_hist, s). Returns (block_out
    [T,n_embd], new state)."""
    t_len = x.shape[0]
    d = hparams.ssm_d_state
    hk, hv = hparams.ssm_n_group, hparams.ssm_dt_rank
    key_dim = d * hk
    value_dim = d * hv

    qkv_mixed = x @ w.wqkv.T  # [T, 2*key_dim+value_dim]  (conv_dim)
    z = x @ w.wqkv_gate.T  # [T, value_dim]

    beta = torch.sigmoid(x @ w.ssm_beta.T)  # [T, hv]
    alpha = x @ w.ssm_alpha.T  # [T, hv]
    alpha_softplus = F.softplus(alpha + w.ssm_dt_bias)
    g_log = alpha_softplus * w.ssm_a  # [T, hv], spazio log (adattamento (ii))

    conv_out = causal_depthwise_conv(qkv_mixed, state.conv_hist, w.ssm_conv1d, dilation=1)
    conv_out = F.silu(conv_out)  # [T, conv_dim]

    q_conv = conv_out[:, :key_dim].reshape(t_len, hk, d)
    k_conv = conv_out[:, key_dim : 2 * key_dim].reshape(t_len, hk, d)
    v_conv = conv_out[:, 2 * key_dim :].reshape(t_len, hv, d)

    eps = hparams.f_norm_rms_eps
    q_conv = l2norm(q_conv, eps)
    k_conv = l2norm(k_conv, eps)

    if hk != hv:
        n_rep = hv // hk
        q_conv = q_conv.tile((1, n_rep, 1))  # piastrelle: testa h -> h % hk (ggml_repeat)
        k_conv = k_conv.tile((1, n_rep, 1))

    out, s_new = gated_delta_net_recurrence(q_conv, k_conv, v_conv, g_log, beta, state.s)

    z_heads = z.reshape(t_len, hv, d)
    normed = rmsnorm(out, w.ssm_norm, eps)
    gated = normed * torch.sigmoid(z_heads)
    final_output = gated.reshape(t_len, value_dim)

    block_out = final_output @ w.ssm_out.T  # [T, n_embd]

    k_minus_1 = hparams.ssm_d_conv - 1
    hist_full = torch.cat([state.conv_hist, qkv_mixed], dim=0)
    new_hist = hist_full[-k_minus_1:] if k_minus_1 > 0 else hist_full[:0]
    return block_out, DeltaNetState(conv_hist=new_hist, s=s_new)


# --------------------------------------------------------------------------
# MoE (softmax, top-k, weights renormalized to sum, sigmoid shared expert)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class MoeWeights:
    gate_inp: torch.Tensor  # [n_expert, n_embd]
    experts_gate: dict  # {e: [n_ff, n_embd]}  or supplied as a function (lazy, weights.py)
    experts_up: dict
    experts_down: dict
    gate_inp_shexp: torch.Tensor  # [n_embd] (actually [1,n_embd] in ggml, a vector)
    up_shexp: torch.Tensor  # [n_ff_shexp, n_embd]
    gate_shexp: torch.Tensor  # [n_ff_shexp, n_embd]
    down_shexp: torch.Tensor  # [n_embd, n_ff_shexp]


def moe_ffn(
    x: torch.Tensor,  # [T, n_embd]
    gate_inp: torch.Tensor,  # [n_expert, n_embd]
    expert_gate_fn,  # (e:int) -> [n_ff, n_embd] tensor (dequant lazy, via GgufWeights.expert)
    expert_up_fn,
    expert_down_fn,
    selected_experts: torch.Tensor,  # [T, n_expert_used] imposed indices (frozen routing)
    up_shexp: torch.Tensor,
    gate_shexp: torch.Tensor,
    down_shexp: torch.Tensor,
    gate_inp_shexp: torch.Tensor,
    n_expert_used: int,
) -> torch.Tensor:
    """build_layer_ffn + build_moe_ffn: routing with `selected_experts` imposed
    (the replica neither reorders nor chooses them, it receives them from the
    caller -- frozen at the base point or free, depending on the caller)."""
    t_len, n_embd = x.shape
    logits = x @ gate_inp.T  # [T, n_expert]
    probs = torch.softmax(logits, dim=-1)  # [T, n_expert]

    weights = torch.gather(probs, 1, selected_experts)  # [T, n_expert_used]
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(6.103515625e-5)
    weights = weights / weights_sum  # norm_w=True, no scale (w_scale=0)

    out = torch.zeros(t_len, n_embd, dtype=x.dtype)
    # one token at a time: each position has (in general) a different routing
    for t in range(t_len):
        acc = torch.zeros(n_embd, dtype=x.dtype)
        for j in range(n_expert_used):
            e = int(selected_experts[t, j].item())
            w_gate = expert_gate_fn(e)  # [n_ff, n_embd]
            w_up = expert_up_fn(e)
            w_down = expert_down_fn(e)  # [n_embd, n_ff]
            gate_act = F.silu(x[t] @ w_gate.T)
            up_act = x[t] @ w_up.T
            h = gate_act * up_act
            acc = acc + weights[t, j] * (h @ w_down.T)
        out[t] = acc

    shared = F.silu(x @ gate_shexp.T) * (x @ up_shexp.T)
    shared_out = shared @ down_shexp.T
    shared_gate = torch.sigmoid(x @ gate_inp_shexp)  # [T]
    out = out + shared_out * shared_gate.unsqueeze(-1)
    return out


# --------------------------------------------------------------------------
# PLE (blk.1): torch port of PleReplica (engraft/lens.py), differentiable
# with respect to `emb` (the 16 concatenated rows, 2560 = n_embd).
# --------------------------------------------------------------------------


@dataclasses.dataclass
class PleWeights:
    w_key: torch.Tensor  # [hc_dim, n_embd]
    w_value: torch.Tensor  # [n_embd, n_embd]
    norm_key: torch.Tensor  # [hc_dim]
    norm_query: torch.Tensor  # [hc_dim]
    norm_conv: torch.Tensor  # [hc_dim]
    conv1d: torch.Tensor  # [hc_dim, K]


def ple_forward(
    emb: torch.Tensor,  # [T, n_embd] -- the rows already concatenated (16*160=2560)
    hidden: torch.Tensor,  # [T, hc, n_embd] -- res_hc at the input of the PLE block
    w: PleWeights,
    hist: torch.Tensor,  # [(K-1)*ngram_size, hc, n_embd] or empty if start of sequence
    hparams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """qwen4exp.cpp build_ple. Returns (hidden + gated + conv_out, new history
    tail for the dilated conv) -- the final sum is what replaces res_hc."""
    hc = hparams.hc_mult
    n_embd = hparams.n_embd
    eps = hparams.f_norm_rms_eps
    t_len = emb.shape[0]

    key = (emb @ w.w_key.T).reshape(t_len, hc, n_embd)
    key = rmsnorm_grouped(key, w.norm_key, eps, hc)
    query = rmsnorm_grouped(hidden, w.norm_query, eps, hc)

    s = (key * query).sum(dim=-1) / (n_embd ** 0.5)  # [T,hc]
    mag = torch.sqrt(torch.clamp(s.abs(), min=1e-6))
    gate = torch.sigmoid(torch.sign(s) * mag)  # [T,hc]

    value = emb @ w.w_value.T  # [T, n_embd]
    gated = value.unsqueeze(1) * gate.unsqueeze(-1)  # [T,hc,n_embd]

    normed = rmsnorm_grouped(gated, w.norm_conv, eps, hc)
    normed_flat = normed.reshape(t_len, hc * n_embd)
    hist_flat = hist.reshape(hist.shape[0], hc * n_embd) if hist.numel() else hist.reshape(0, hc * n_embd)
    conv_out = causal_depthwise_conv(normed_flat, hist_flat, w.conv1d, dilation=hparams.ple_ngram_size)
    conv_out = F.silu(conv_out).reshape(t_len, hc, n_embd)

    out = hidden + gated + conv_out

    kern = w.conv1d.shape[1]
    hist_len = (kern - 1) * hparams.ple_ngram_size
    hist_full = torch.cat([hist, normed], dim=0) if hist.numel() or hist.shape[0] == 0 else normed
    new_hist = hist_full[-hist_len:] if hist_len > 0 else hist_full[:0]
    return out, new_hist

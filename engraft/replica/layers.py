"""Pure torch (f32) functions for the reference engine's forward graph.

Axis convention: torch-natural, time first. A ggml tensor `[ne0, ne1, ne2]`
(ne0 fastest) becomes here `[..., ne2, ne1, ne0]`, with the "time" axis (when
present) always first. Weights come from `weights.GgufWeights`, which
dequantizes them already in `[out, in]` convention (numpy reverses the ggml
axes, and a ggml linear tensor is stored `{in, out}` in ne-order): so
`y = x @ w.T` everywhere, like `nn.Linear`.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch.profiler import record_function

# --------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last axis. `weight` is already "1+w" (or the direct
    weight for ssm_norm, which has no offset in the source HF conversion):
    direct multiplication, never +1 here.

    Fixed F32 island: the reduction (`pow(2).mean`) is accumulated in F32,
    the output returns to `x`'s dtype (the active phase's dtype). With `x`
    already F32 (default) the casts are no-ops, bit for bit."""
    xf = x.to(torch.float32)
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(ms + eps)
    return (normed * weight.to(torch.float32)).to(x.dtype)


def rmsnorm_grouped(x: torch.Tensor, weight_flat: torch.Tensor, eps: float, hc: int) -> torch.Tensor:
    """x: [..., hc, n_embd]. RMSNorm per stream (last axis), then multiplies by
    `weight_flat` [hc*n_embd] reshaped [hc, n_embd]."""
    n_embd = x.shape[-1]
    w = weight_flat.reshape(hc, n_embd)
    return rmsnorm(x, w, eps)


def l2norm(x: torch.Tensor, eps: float, dim: int = -1) -> torch.Tensor:
    """True L2 norm (not RMS): x / sqrt(sum(x^2)+eps).

    Fixed F32 island: reduction in F32, output in `x`'s dtype. With `x`
    already F32 (default) the casts are no-ops, bit for bit."""
    x32 = x.to(torch.float32)
    return (x32 * torch.rsqrt((x32 * x32).sum(dim=dim, keepdim=True) + eps)).to(x.dtype)


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
    """Returns (mixed [T,n_embd], inject [T,hc] or None)."""
    t_len, _, n_embd = x.shape
    hc_dim = hc * n_embd

    xn = rmsnorm_grouped(x, w_norm, eps, hc)  # [T,hc,n_embd]
    xn_flat = xn.reshape(t_len, hc_dim)

    lo = xn_flat @ w_down.T  # [T, hc_lr]
    lo = F.silu(lo / hc)

    gate = torch.sigmoid(lo @ w_up.T)  # [T, hc_dim]
    gate = gate.reshape(t_len, hc, n_embd)

    gated = xn * gate  # [T,hc,n_embd]
    mixed = gated.mean(dim=1)  # [T,n_embd] (mean over streams)

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
    """2*sigmoid centers the scatter weights around 1."""
    w = torch.sigmoid(inject / hc) * 2.0  # [T,hc]
    return residual + block_out.unsqueeze(1) * w.unsqueeze(-1)


# --------------------------------------------------------------------------
# RoPE (interleaved mrope, simplified for text-only: same positions on all 3 axes)
# --------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions: torch.Tensor, rope_dim: int, freq_base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for NeoX-style RoPE, `rope_dim` (half rotating, half passed
    through unchanged).

    Fixed F32 island: computed in float64/float32 on `positions.device`
    regardless of the active phase's dtype -- the caller casts `cos`/`sin`
    back to the phase dtype before the next matmul."""
    half = rope_dim // 2
    inv_freq = 1.0 / (freq_base ** (torch.arange(0, half, dtype=torch.float64, device=positions.device) * 2.0 / rope_dim))
    freqs = positions.to(torch.float64)[:, None] * inv_freq[None, :]  # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, rope_dim]
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, D]. cos/sin: [T, rope_dim] (F32, fixed island). Rotates only
    the first rope_dim channels. `cos`/`sin` are cast to `x`'s dtype here,
    before the multiplication: with an F32 backend (default) this is a
    bit-for-bit no-op."""
    rope_dim = cos.shape[-1]
    x_rope, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    c = cos.to(x.dtype).unsqueeze(1)  # [T,1,rope_dim]
    s = sin.to(x.dtype).unsqueeze(1)
    x_rope = x_rope * c + rotate_half(x_rope) * s
    return torch.cat([x_rope, x_pass], dim=-1)


# --------------------------------------------------------------------------
# Full attention (GQA, sigmoid gate)
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
    """qcur_full: [T, 2*n_embd_head*n_head]. `wq` interleaves [q|gate] per head:
    the first n_embd_head columns per head are q, the next n_embd_head are the
    gate."""
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
    segment_ids: torch.Tensor | None = None,
    *,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense causal attention (the QSA indexer is dense for T <= top_k): no
    selection, full causal mask over new queries + cache. Returns
    (block_out [T,n_embd], k_new, v_new) (K/V of the new slice, already
    rotated, to append to the caller's cache).

    `segment_ids` (additive, default `None`, identical behavior without it):
    `[T]`, same length as `positions` -- if given, the mask becomes
    `causal ∧ (segment[t]==segment[s])` (a packed sequence: positions from
    different fragments never see each other even if causally ordered). Not
    composable with `k_cache`/`v_cache` (packing does not compose with an
    incremental prefix): raises if both are given.

    `attn_mask` (additive, keyword-only, default `None`): `[T, T]` bool (true
    = visible) -- if given, REPLACES the `causal`/`same_segment` computation
    (same mask, just already built: a packed-batch caller can compute it once
    per training step instead of recomputing it at every layer). Same
    precondition as `segment_ids`: not composable with `k_cache`/`v_cache`."""
    if attn_mask is not None and (
        k_cache is not None or v_cache is not None or cache_positions is not None
    ):
        raise ValueError(
            "attention_full: attn_mask is not composable with k_cache/v_cache "
            "(same precondition as segment_ids)"
        )
    if segment_ids is not None and (k_cache is not None or cache_positions is not None):
        raise ValueError(
            "attention_full: segment_ids is not composable with k_cache/v_cache "
            "(precondition of pack.packed_forward)"
        )
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
    k_rep = k_all.repeat_interleave(n_rep, dim=1)  # [Tkv, n_head, d] consecutive block (ggml mul_mat r2)
    v_rep = v_all.repeat_interleave(n_rep, dim=1)

    scale = 1.0 / (d ** 0.5)
    # [n_head, T, Tkv]
    scores = torch.einsum("thd,shd->hts", q, k_rep) * scale
    if attn_mask is not None:
        mask = attn_mask
    else:
        mask = positions[:, None] >= pos_all[None, :]
        if segment_ids is not None:
            same_segment = segment_ids[:, None] == segment_ids[None, :]
            mask = mask & same_segment
    scores.masked_fill_(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.einsum("hts,shd->thd", probs.to(v_rep.dtype), v_rep)  # [T,n_head,d]

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
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """out[t,c] = sum_k weight[c,k] * x[t-(K-1-k)*dilation, c] (PLE reuses this
    same function with dilation=ngram_size).

    `segment_ids` (additive, default `None`, identical behavior without it):
    `[T]`, one id per position of `x_new` (never for `history`: packing does
    not compose with a pre-existing history, a `packed_forward` precondition
    -- `history` must be empty when `segment_ids` is given, checked by the
    caller). Every tap whose source index falls in `history` (pre-existing
    history, never the packed batch's own) or at a position of `x_new` with a
    `segment_ids` different from the current position's is masked (zero
    contribution) -- same convention as the existing "before the start of the
    buffer" mask for `src_idx<0`."""
    kern = weight.shape[1]
    t_len, c = x_new.shape
    full = torch.cat([history, x_new], dim=0)  # [(K-1)+T, C] (positions -(K-1)..T-1)
    hist_len = history.shape[0]
    acc = torch.zeros(t_len, c, dtype=x_new.dtype, device=x_new.device)
    for k in range(kern):
        back = (kern - 1 - k) * dilation
        w_k = weight[:, k]
        for t in range(t_len):
            src_idx = hist_len + t - back
            if src_idx < 0:
                continue
            if segment_ids is not None:
                src_rel = src_idx - hist_len  # index relative to x_new (>=0 only if not history)
                if src_rel < 0 or bool(segment_ids[src_rel] != segment_ids[t]):
                    continue
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
    ssm_a: torch.Tensor  # [num_v_heads] = -exp(A_log), already so in the GGUF
    ssm_beta: torch.Tensor  # [num_v_heads, n_embd] (out,in)
    ssm_alpha: torch.Tensor  # [num_v_heads, n_embd]
    ssm_norm: torch.Tensor  # [head_v_dim]
    ssm_out: torch.Tensor  # [n_embd, value_dim]


@dataclasses.dataclass
class DeltaNetState:
    conv_hist: torch.Tensor  # [K-1, conv_dim] -- last pre-conv inputs (qkv_mixed)
    s: torch.Tensor  # [num_v_heads, head_dim, head_dim] -- recurrence state


def delta_net_init_state(
    hparams, device: "torch.device | str | None" = None, dtype: torch.dtype = torch.float32,
) -> DeltaNetState:
    """`device`/`dtype` (default `None`/`float32` -> CPU F32, identical
    behavior to a plain construction). `s` (the delta-net's recurrent state)
    stays **always F32** (a fixed island, accumulated in F32 along the whole
    prefix), regardless of the phase `dtype` passed for `conv_hist`."""
    device = torch.device("cpu") if device is None else device
    k_minus_1 = hparams.ssm_d_conv - 1
    return DeltaNetState(
        conv_hist=torch.zeros(k_minus_1, hparams.conv_dim, device=device, dtype=dtype),
        s=torch.zeros(hparams.ssm_dt_rank, hparams.ssm_d_state, hparams.ssm_d_state, device=device, dtype=torch.float32),
    )


def gated_delta_net_recurrence(
    q: torch.Tensor,  # [T, Hv, D]  (already repeated from Hk to Hv heads, L2-normed, not prescaled)
    k: torch.Tensor,  # [T, Hv, D]
    v: torch.Tensor,  # [T, Hv, D]
    g_log: torch.Tensor,  # [T, Hv] -- ssm_a * softplus(alpha+dt_bias), in log space
    beta: torch.Tensor,  # [T, Hv]
    s0: torch.Tensor,  # [Hv, D, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact recurrence, repeated position by position (mathematically
    equivalent to the chunked form for T>1, up to floating-point summation
    order). Scales q by 1/sqrt(D) here. Returns (out [T,Hv,D], final state
    [Hv,D,D]).

    Fixed F32 island: the recurrent state `s` stays accumulated in F32 along
    the whole prefix, regardless of `q`/`k`/`v`'s phase dtype; the output
    `out` is cast back to `q`'s dtype before returning. With an F32 backend
    (default) every cast is a bit-for-bit no-op."""
    t_len, hv, d = q.shape
    scale = 1.0 / (d ** 0.5)
    compute_dtype = torch.float32
    q_scaled = (q * scale).to(compute_dtype)
    k32 = k.to(compute_dtype)
    v32 = v.to(compute_dtype)
    g_log32 = g_log.to(compute_dtype)
    beta32 = beta.to(compute_dtype)
    s = s0.to(compute_dtype)
    outs = []
    for t in range(t_len):
        g_t = torch.exp(g_log32[t]).reshape(hv, 1, 1)
        s = s * g_t
        kv_mem = torch.einsum("hij,hi->hj", s, k32[t])  # sum_i s[i,j]*k[i]
        delta = (v32[t] - kv_mem) * beta32[t].unsqueeze(-1)
        s = s + torch.einsum("hi,hj->hij", k32[t], delta)
        o_t = torch.einsum("hij,hi->hj", s, q_scaled[t])
        outs.append(o_t)
    out = torch.stack(outs, dim=0).to(q.dtype)  # [T,Hv,D]
    return out, s


def linear_attn_layer(
    x: torch.Tensor,  # [T, n_embd] -- only the new positions
    w: DeltaNetWeights,
    state: DeltaNetState,
    hparams,
) -> tuple[torch.Tensor, DeltaNetState]:
    """The whole gated delta-net layer, for a block of T new positions given
    the prefix's tails (conv_hist, s). Returns (block_out [T,n_embd], new
    state)."""
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
    g_log = alpha_softplus * w.ssm_a  # [T, hv], log space

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
        q_conv = q_conv.tile((1, n_rep, 1))  # tiled: head h -> h % hk
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
    gate_inp_shexp: torch.Tensor  # [n_embd] (a [1,n_embd] vector in ggml)
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
    grouped: bool = False,
    capture: dict | None = None,
) -> torch.Tensor:
    """Routing with `selected_experts` imposed (the replica neither reorders
    nor chooses them, it receives them from the caller -- frozen at the base
    point or free, depending on the caller).

    `grouped=False` (default): legacy path, one token at a time -- invariant
    byte for byte. `grouped=True`: the same result within 1e-6, grouped by
    expert -- a single device->host sync per layer (`selected_experts
    .tolist()`) instead of one per (token, slot); each expert evaluated once
    over every token that uses it (a matrix product instead of vector x
    matrix).

    `capture` (dict or `None`, additive): if given, populated with
    `{"experts": out BEFORE summing with the shared/dense branch, "dense":
    shared_out*shared_gate (the always-active dense/shared branch, never
    routed), "weighted_sum": the final result (combination of both
    branches)}`, all `[T, n_embd]`, no `.detach()`/`.clone()` here (the
    caller decides). Default `None` -> nothing written, identical behavior."""
    t_len, n_embd = x.shape
    # Fixed F32 island: router logits, softmax and renormalized weights in
    # F32 regardless of `x`'s phase dtype; the weights return to `x`'s dtype
    # only for the expert combination. Under F32 this is a bit-for-bit no-op.
    logits = x.to(torch.float32) @ gate_inp.to(torch.float32).T  # [T, n_expert]
    probs = torch.softmax(logits, dim=-1)  # [T, n_expert]

    weights = torch.gather(probs, 1, selected_experts)  # [T, n_expert_used]
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(6.103515625e-5)
    weights = (weights / weights_sum).to(x.dtype)  # norm_w=True, no scale (w_scale=0)

    with record_function("moe/experts"):
        if grouped:
            # A single sync per layer: the routing is frozen for the whole
            # descent, so this list does not change from one step to the
            # next -- the cost is paid once.
            sel = selected_experts.detach().to("cpu").tolist()  # [T][n_expert_used]
            slots: dict[int, list[tuple[int, int]]] = {}
            for t in range(t_len):
                for j in range(n_expert_used):
                    slots.setdefault(sel[t][j], []).append((t, j))

            contrib = torch.zeros(t_len * n_expert_used, n_embd, dtype=x.dtype, device=x.device)
            for e in sorted(slots):
                pairs = slots[e]
                t_idx = torch.tensor([p[0] for p in pairs], dtype=torch.int64, device=x.device)
                j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device=x.device)
                xe = x[t_idx]  # [T_e, n_embd]
                w_gate = expert_gate_fn(e)  # [n_ff, n_embd]
                w_up = expert_up_fn(e)
                w_down = expert_down_fn(e)  # [n_embd, n_ff]
                h = F.silu(xe @ w_gate.T) * (xe @ w_up.T)  # [T_e, n_ff]
                ye = (h @ w_down.T) * weights[t_idx, j_idx].unsqueeze(-1)  # [T_e, n_embd]
                flat_idx = t_idx * n_expert_used + j_idx
                contrib = contrib.index_add(0, flat_idx, ye)  # out of place, differentiable

            contrib = contrib.view(t_len, n_expert_used, n_embd)
            out = contrib[:, 0]
            for j in range(1, n_expert_used):  # same summation order as the legacy loop
                out = out + contrib[:, j]
        else:
            out = torch.zeros(t_len, n_embd, dtype=x.dtype, device=x.device)
            # one token at a time: every position (in general) has a different routing
            for t in range(t_len):
                acc = torch.zeros(n_embd, dtype=x.dtype, device=x.device)
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

    if capture is not None:
        capture["experts"] = out

    with record_function("moe/shared"):
        shared = F.silu(x @ gate_shexp.T) * (x @ up_shexp.T)
        shared_out = shared @ down_shexp.T
        shared_gate = torch.sigmoid(x @ gate_inp_shexp)  # [T]
    dense_out = shared_out * shared_gate.unsqueeze(-1)
    if capture is not None:
        capture["dense"] = dense_out
    out = out + dense_out
    if capture is not None:
        capture["weighted_sum"] = out
    return out


# --------------------------------------------------------------------------
# PLE (n-gram table read layer): differentiable in `emb` (the concatenated
# rows, 2560 = n_embd).
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
    diag: dict | None = None,
    detach_value: bool = False,
    detach_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (hidden + gated + conv_out, new history tail for the dilated
    conv) -- the final sum is what replaces res_hc.

    Additive and backward-compatible (default `diag=None`,
    `detach_value=False`, `detach_gate=False` -> identical behavior): `diag`,
    if given, receives `s`, `gate`, `value_norm`, `gated_norm`, `hidden_norm`
    ([T,hc] except `value_norm` [T]) for the 4 hyper-connection streams into
    the PLE block. `detach_value`/`detach_gate` detach the corresponding
    factor from the graph BEFORE the product `gated = value*gate` (the
    numeric values do not change, only the gradient): by the product rule
    d(uv) = du*v + u*dv, the sum of the gradients computed with either factor
    detached is EXACTLY the total gradient (no approximation), because the
    rest of the downstream network sees the same numeric value of `gated`
    either way (detach does not alter the forward pass, only the backward)."""
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
    gate_for_gated = gate.detach() if detach_gate else gate
    value_for_gated = value.detach() if detach_value else value
    gated = value_for_gated.unsqueeze(1) * gate_for_gated.unsqueeze(-1)  # [T,hc,n_embd]

    if diag is not None:
        diag["s"] = s.detach().clone()
        diag["gate"] = gate.detach().clone()
        diag["value_norm"] = value.detach().norm(dim=-1).clone()
        diag["gated_norm"] = gated.detach().norm(dim=-1).clone()
        diag["hidden_norm"] = hidden.detach().norm(dim=-1).clone()

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

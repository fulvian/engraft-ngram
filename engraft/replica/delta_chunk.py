"""Chunked form of gated delta-net recurrence and causal depthwise convolution,
implemented in pure PyTorch without external dependencies.

Mathematical derivation (verified against the autoregressive recurrence
implementation): for a block of C local positions t=0..C-1 with incoming state
`S_in` [D,D] (axis i = key space, axis j = value space), let
`gc[t] = cumsum(g_log)[t]` (cumulative local sum, inclusive) and
`a[t] = exp(gc[t])` (cumulative decay from block start through t inclusive).
The exact recurrence (`s = s*g; kv = s^T k; delta = beta*(v-kv);
s = s + k(x)delta; o = s^T q`, where `o` uses the ALREADY-updated state)
rewrites in closed form at the intermediate state:

    s_t = a_t * S_in + sum_{s<=t} (a_t/a_s) * k_s (x) delta_s

Substituting into `kv_t = s_t_decayed^T k_t` (uses state BEFORE adding the
t-step contribution, i.e., the sum for s<t) yields a lower-triangular unitary
linear system in `delta_t` (WY / "UT transform" representation, Yang et al.
2024 "Parallelizing Linear Transformers with the Delta Rule over Sequence
Length"; same structure as FLA's `chunk_gated_delta_rule`):

    (I + M) @ Delta = RHS
    M[t,s]   = beta[t] * ratio[t,s] * (k_t . k_s)      for s < t, 0 elsewhere
    RHS[t]   = beta[t] * (v[t] - a[t] * (S_in^T k[t]))
    ratio[t,s] = exp(gc[t] - gc[s])   (= a_t/a_s, for t>=s)

solved by triangular substitution (`torch.linalg.solve_triangular`, unitary
diagonal by construction: `M` is strictly lower). Output and new state are
then derived in closed form from the known `Delta` values:

    o[t]     = a[t] * (S_in^T q[t]) + sum_{s<=t} ratio[t,s] * (k_s.q[t]) * delta[s]
    S_out    = a[C-1] * S_in + sum_s ratio[C-1,s] * k_s (x) delta[s]

Numerical guard: `g_log <= 0` always in the actual model (`ssm_a = -exp(A_log)`
already negative, `softplus(...) >= 0`), so `gc` is non-increasing and `a[t]`
never explodes. The term `ratio[t,s]` for `s>t` is never used (masked), but
computing its exponential before masking could overflow (`gc[t]-gc[s]` can be
large and positive for s>t): the exponent is therefore replaced with a very
negative value for masked pairs BEFORE `exp`, never after.
"""
from __future__ import annotations

import torch


def gated_delta_net_chunked(
    q: torch.Tensor,  # [T, Hv, D]
    k: torch.Tensor,  # [T, Hv, D]
    v: torch.Tensor,  # [T, Hv, D]
    g_log: torch.Tensor,  # [T, Hv]
    beta: torch.Tensor,  # [T, Hv]
    s0: torch.Tensor,  # [Hv, D, D]
    chunk: int = 64,
    segment_ids: torch.Tensor | None = None,
    batched_segments: bool = True,
    need_state: bool = True,
    s_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same signature and semantics as the autoregressive recurrence
    implementation (q already L2-normalized and not prescaled; scale 1/sqrt(D)
    applied here; state in F32; returns (out [T,Hv,D], final state [Hv,D,D])),
    chunked form: a Python loop over blocks (not tokens), recurrence between
    blocks only through state `S_in` [Hv,D,D]; inside each block no loop over T.

    `segment_ids` (additive, stage 0 T3, spec §5): `[T]`, piecewise-constant
    integers -- positions of the same head always belong to CONTIGUOUS segments
    by construction (`pack.pack_fragments`). If given, recurrence is applied
    **independently per segment**: `T` is divided into contiguous blocks with
    constant segment ID (segment boundaries found via tensor operations, no
    loop per position) and this same function is called ONCE per segment with
    `segment_ids=None` (incoming state `s0` only for the FIRST segment --
    `packed_forward` never composes with `base_state`/`cache`, so `s0` is
    always zero in practice anyway -- state ZEROED, not `s0`, for each
    subsequent segment), then outputs are concatenated. Equivalent by
    construction to "padding each segment to a multiple of `chunk` and zeroing
    `s` at boundaries" (recurrence is causal and purely sequential: a state
    restarting at zero at a segment boundary is indistinguishable, for
    positions in that segment, from an independent batch) -- implemented via
    composition (recursive call per segment) rather than fusing padding into a
    single chunk loop: each segment already truncates to its own `t_len` (line
    below, `out = torch.cat(outs, dim=0)[:t_len]`) before concatenation here,
    no padded row to explicitly discard. The final state returned is that of
    the LAST segment (the others are never reused: `packed_forward` never takes
    this state to another call). Default `None` -> behavior identical to
    current.

    `batched_segments` (phase B1 batch 1, additive, default `True`): with
    `segment_ids` given, selects which of two equivalent implementations
    processes segments -- `True` (new, `gated_delta_net_chunked_batched`,
    independent segments treated as batch dimension: ~86 to ~25-50 launches per
    layer with typical ~86 fragments/batch in B0) or `False`
    (`_gated_delta_net_chunked_by_segment`, a Python loop of calls, the
    pre-batch code). Equality between the two within F32 rounding (different
    batch of `bmm`/`solve_triangular`, same operations in same order per
    segment -- never bit-identical by construction: the gate is the dedicated
    equality test, not assumption). Ignored when `segment_ids` is `None` (no
    segment batch to choose).

    `need_state` (phase B1 batch 7, L7c, additive, default `True` = current
    behavior): with `False`, the final state update (`kᵀ·(r·δ)`,
    `decay·s + ...`) for the LAST chunk of the segment/sequence is NOT
    computed -- the returned state is `s0` UNCHANGED (same tensor as argument,
    never `None`, never recomputed): use only when the caller never consumes
    the final state (`packed_forward`, descent without `base_state`/`cache`).
    State updates for chunks BEFORE the last remain unchanged (needed for the
    next chunk).

    `s_zero` (phase B1 batch 7, L7c, additive, default `False`): declared by
    the CALLER, never verified at runtime on values -- if `True`, the
    incoming state to chunk 0 is zero by construction (`torch.bmm(k_c, s)`,
    `torch.bmm(q_c, s)` and their dependent element-wise ops become exact
    no-ops: `0·x=0`, `v - 0=v`, `0 + y=y`, guaranteed without NaN/Inf because
    `decay` stays finite with `g_log<=0`) and are skipped.
    """
    if segment_ids is not None:
        if batched_segments:
            return gated_delta_net_chunked_batched(q, k, v, g_log, beta, s0, chunk, segment_ids,
                                                     need_state=need_state)
        return _gated_delta_net_chunked_by_segment(q, k, v, g_log, beta, s0, chunk, segment_ids,
                                                    s_zero=s_zero, need_state=need_state)
    t_len, hv, d = q.shape
    device = q.device
    compute_dtype = torch.float32

    scale = 1.0 / (d ** 0.5)
    q32 = (q * scale).to(compute_dtype)
    k32 = k.to(compute_dtype)
    v32 = v.to(compute_dtype)
    g_log32 = g_log.to(compute_dtype)
    beta32 = beta.to(compute_dtype)
    s = s0.to(compute_dtype)

    pad = (chunk - t_len % chunk) % chunk
    t_pad = t_len + pad
    if pad:
        zeros_qkv = q32.new_zeros(pad, hv, d)
        zeros_scalar = q32.new_zeros(pad, hv)
        q32 = torch.cat([q32, zeros_qkv], dim=0)
        k32 = torch.cat([k32, zeros_qkv], dim=0)
        v32 = torch.cat([v32, zeros_qkv], dim=0)
        # g_log=0 -> decay 1 (no real state advancement during padding);
        # beta=0 -> delta=0 (no contribution from dummy k/v): padding steps are
        # exact no-ops on state.
        g_log32 = torch.cat([g_log32, zeros_scalar], dim=0)
        beta32 = torch.cat([beta32, zeros_scalar], dim=0)

    n_chunks = t_pad // chunk
    tril_incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=device))
    tril_strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=device), diagonal=-1)
    eye_c = torch.eye(chunk, dtype=compute_dtype, device=device)

    outs = []
    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        q_c = q32[sl].permute(1, 0, 2)  # [Hv,C,D]
        k_c = k32[sl].permute(1, 0, 2)
        v_c = v32[sl].permute(1, 0, 2)
        beta_c = beta32[sl].permute(1, 0)  # [Hv,C]
        g_c = g_log32[sl].permute(1, 0)  # [Hv,C]

        gc = torch.cumsum(g_c, dim=-1)  # [Hv,C]
        decay_start = torch.exp(gc)  # [Hv,C] = a_t

        diff = gc.unsqueeze(-1) - gc.unsqueeze(-2)  # diff[h,t,s] = gc[t]-gc[s]
        diff_masked = diff.masked_fill(~tril_incl, -1e30)
        ratio = torch.exp(diff_masked)  # [Hv,C,C], ratio[h,t,s] valid for t>=s

        kk = torch.bmm(k_c, k_c.transpose(-1, -2))  # [Hv,C,C], k_t . k_s
        m = beta_c.unsqueeze(-1) * ratio * kk
        m = m.masked_fill(~tril_strict, 0.0)
        i_plus_m = eye_c.unsqueeze(0) + m  # [Hv,C,C]

        # L7c: incoming state is zero ONLY at chunk 0, and only if caller
        # declares it (`s_zero`) -- bmm(k_c,s)==0 and dependent element-wise
        # are exact no-ops, never a runtime value test.
        chunk_s_zero = s_zero and c == 0
        if chunk_s_zero:
            rhs = beta_c.unsqueeze(-1) * v_c
        else:
            kv0 = torch.bmm(k_c, s)  # [Hv,C,D] = S_in^T k[t] (per row)
            rhs = beta_c.unsqueeze(-1) * (v_c - decay_start.unsqueeze(-1) * kv0)

        delta = torch.linalg.solve_triangular(i_plus_m, rhs, upper=False, unitriangular=True)  # [Hv,C,D]

        qk = torch.bmm(q_c, k_c.transpose(-1, -2))  # [Hv,C,C], q_t . k_s
        w = (ratio * qk).masked_fill(~tril_incl, 0.0)
        if chunk_s_zero:
            out_c = torch.bmm(w, delta)  # [Hv,C,D]; bmm(q_c,s)==0, decay*q_s_in is no-op
        else:
            q_s_in = torch.bmm(q_c, s)  # [Hv,C,D] = S_in^T q[t]
            out_c = decay_start.unsqueeze(-1) * q_s_in + torch.bmm(w, delta)  # [Hv,C,D]

        outs.append(out_c.permute(1, 0, 2))  # [C,Hv,D]

        # L7c: final state update (last chunk) is skipped if `need_state=False`
        # -- no one consumes it in descent (`packed_forward`); state updates for
        # earlier chunks remain (needed for the next one).
        if c == n_chunks - 1 and not need_state:
            continue
        decay_last = decay_start[:, -1]  # [Hv] = a_{C-1}
        r_last = ratio[:, -1, :]  # [Hv,C] = ratio[C-1,s], always valid (t=C-1 max)
        weighted = r_last.unsqueeze(-1) * delta  # [Hv,C,D]
        ks_delta = torch.bmm(k_c.transpose(-1, -2), weighted)  # [Hv,D,D]
        s = decay_last.unsqueeze(-1).unsqueeze(-1) * s + ks_delta

    out = torch.cat(outs, dim=0)[:t_len].to(q.dtype)
    final_state = s0 if not need_state else s  # L7c: "s0 unchanged", never recomputed
    return out, final_state


def _segment_boundaries(segment_ids: torch.Tensor) -> tuple[list[int], list[int]]:
    """Segment boundaries without Python loop over `.item()` (phase B1 batch 1,
    T1): a single `.tolist()` on a tensor of `n_seg` elements (not `T`) --
    segments are always contiguous and increasing (`pack.pack_fragments`), so a
    value change between `segment_ids[t-1]` and `segment_ids[t]` is the only
    possible boundary. Returns `(starts, lengths)`, parallel Python lists, one
    element per segment in order of appearance.
    """
    t_len = segment_ids.shape[0]
    device = segment_ids.device
    change = segment_ids[1:] != segment_ids[:-1]
    boundary_starts = torch.nonzero(change, as_tuple=False).flatten() + 1
    starts_t = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), boundary_starts])
    ends_t = torch.cat([starts_t[1:], torch.tensor([t_len], dtype=torch.int64, device=device)])
    both = torch.stack([starts_t, ends_t], dim=1).tolist()  # ONE device->host sync
    starts = [s for s, _ in both]
    lengths = [e - s for s, e in both]
    return starts, lengths


def _gated_delta_net_chunked_by_segment(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_log: torch.Tensor,
    beta: torch.Tensor, s0: torch.Tensor, chunk: int, segment_ids: torch.Tensor,
    s_zero: bool = False, need_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide `T` into contiguous blocks with constant `segment_ids` (segments
    are always contiguous by construction, `pack.pack_fragments`) and call
    `gated_delta_net_chunked` once per block, `s0` only for the first, zero
    elsewhere -- see `gated_delta_net_chunked` docstring for why this is
    equivalent to fused padding.

    `s_zero` (phase B1 batch 7, L7c, R2): declares that the argument `s0` IS
    zero (verified by caller, e.g., `packed_forward.patched_recurrence`) --
    applies to the FIRST segment (`i == 0`); subsequent segments (`i > 0`) are
    ALWAYS zero-state by construction of this function itself
    (`torch.zeros_like(s0)`), so they pass `s_zero=True` unconditionally,
    independent of the flag received as input.

    `need_state` (L7c, R1): forwarded to EVERY segment call -- final states of
    segments other than the last are already dead today (`s_last` is reassigned
    each iteration, never reused as `s_in` of `i+1`, which is always
    `torch.zeros_like(s0)`); with `False` no segment computes its final update
    and the function returns `s0` unchanged (uniform with
    `gated_delta_net_chunked`/`gated_delta_net_chunked_batched`, never a ghost
    `s_last` from the previous iteration).
    """
    starts, lengths = _segment_boundaries(segment_ids)

    outs = []
    s_last = s0
    for i, (a, ln) in enumerate(zip(starts, lengths)):
        b = a + ln
        s_in = s0 if i == 0 else torch.zeros_like(s0)
        seg_s_zero = s_zero if i == 0 else True
        out_seg, s_last = gated_delta_net_chunked(
            q[a:b], k[a:b], v[a:b], g_log[a:b], beta[a:b], s_in, chunk=chunk,
            s_zero=seg_s_zero, need_state=need_state,
        )
        outs.append(out_seg)
    return torch.cat(outs, dim=0), (s0 if not need_state else s_last)


def _group_segments_by_chunks(starts: list[int], lengths: list[int], chunk: int) -> list[dict]:
    """Group segments by `ceil(L/chunk)` (phase B1 batch 1, review disposition
    B5): a group batches only segments with the SAME number of chunks, so
    padding for each member stays exactly `ceil(L_i/chunk)*chunk` -- NEVER
    `n_seg*ceil(L_max/chunk)*chunk` (which would pad a short fragment to the
    longest in the batch, wasting memory and launches without bound as segment
    count grows). Order of first appearance of the chunk count, not numerical:
    no downstream requirement on that order, only determinism.
    """
    groups: dict[int, dict] = {}
    order: list[int] = []
    for seg_idx, (s, ln) in enumerate(zip(starts, lengths)):
        n_chunks = -(-ln // chunk) if ln > 0 else 1  # ceil division, empty segment never expected
        if n_chunks not in groups:
            groups[n_chunks] = {"n_chunks": n_chunks, "seg_indices": [], "starts": [], "lengths": []}
            order.append(n_chunks)
        g = groups[n_chunks]
        g["seg_indices"].append(seg_idx)
        g["starts"].append(s)
        g["lengths"].append(ln)
    return [groups[n] for n in order]


def compute_segment_groups(segment_ids: torch.Tensor, chunk: int) -> dict:
    """Boundaries + grouping (T1/B5) in one place, computable ONCE per `with`
    from `pack.packed_forward` (B1: otherwise ~108 syncs per step, one per
    layer) and passed to `gated_delta_net_chunked_batched` via `groups=`.
    Includes the padding factor for spec evidence: `n_rows_padded` (==
    Σ ceil(L_i/chunk)*chunk by grouping construction, so always 1.0 against
    itself) compared to `n_rows_naive` (`n_seg * ceil(L_max/chunk) * chunk`,
    the padding you'd get WITHOUT grouping by bucket -- what B5 avoids).
    """
    starts, lengths = _segment_boundaries(segment_ids)
    groups = _group_segments_by_chunks(starts, lengths, chunk)
    n_rows_padded = sum(len(g["seg_indices"]) * g["n_chunks"] * chunk for g in groups)
    l_max = max(lengths) if lengths else 0
    n_chunks_naive = (-(-l_max // chunk)) if l_max > 0 else 0
    n_rows_naive = len(lengths) * n_chunks_naive * chunk
    return {
        "groups": groups, "starts": starts, "lengths": lengths,
        "n_rows_padded": n_rows_padded, "n_rows_naive": n_rows_naive,
        "padding_ratio_vs_own_bucket": 1.0,
        "padding_ratio_vs_naive": (n_rows_padded / n_rows_naive) if n_rows_naive else 1.0,
    }


def gated_delta_net_chunked_batched(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_log: torch.Tensor,
    beta: torch.Tensor, s0: torch.Tensor, chunk: int, segment_ids: torch.Tensor,
    groups: list[dict] | None = None, need_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent segments treated as BATCH dimension (phase B1 batch 1, L1):
    the same 25 operations per chunk as `gated_delta_net_chunked` (F32, same
    order), but on `bmm`/`solve_triangular` with batch `n_seg_grupo*Hv` instead
    of running in a Python loop of `n_seg` separate calls -- the source of
    reduced launches (spec §1.1).

    `groups` (from `compute_segment_groups`, optional: if `None` computed here
    for standalone use/in tests): grouped by `ceil(L/chunk)` (B5) -- within a
    group each segment is padded to EXACTLY `n_chunks*chunk` (never to the
    batch max), zero beyond its true length (same exact no-op on state as
    `gated_delta_net_chunked`, g_log=0/beta=0).

    Packing and unpacking with a single `index_put`/gather per group via
    pre-computed indices (standard "ragged->padded" trick: each token's local
    position within its segment is obtained by subtracting the cumulative
    exclusive offset of lengths of earlier segments in the group from a flat
    `arange`) -- NEVER a `cat`/slice per segment, which would reintroduce the
    `n_seg` launches this batch eliminates.

    `s0` must be zero (raises `ValueError` otherwise): the packed path
    (`pack.packed_forward`) always zeros state at segment boundaries and never
    composes with a real prefix `s0` (see `gated_delta_net_chunked`/
    `packed_forward` docstring) -- nonzero `s0` here is always a caller error,
    never a legitimate case to silently handle (unlike
    `_gated_delta_net_chunked_by_segment`, which accepts it for the first
    segment).

    Output accumulation in F32 (like state) with cast to `q.dtype` only at the
    end: accumulating in bf16 would introduce an extra rounding per group,
    absent in the per-segment path (which casts once at the end of EVERY
    single `gated_delta_net_chunked` call -- here the output from different
    groups never sums, so the only real difference is the final gather
    rounding: doing it in F32 makes it negligible vs. the 1e-5/1e-4 gate).

    `need_state` (phase B1 batch 7, L7c, R1, default `True`): with `False` the
    final state update of the LAST chunk of each group is not computed and
    every segment's state is `s0` unchanged (identical by construction: the
    guard above enforces `s0` zero, and the zero-state branch of chunk 0
    below is already unconditional -- `s` starts from `torch.zeros(...)`, never
    from `s0` itself -- so with `need_state=False` no segment ever computes
    anything other than zero, coherent with `s0`).
    """
    if s0 is not None and s0.numel() and not torch.all(s0 == 0):
        raise ValueError(
            "gated_delta_net_chunked_batched: nonzero s0 with segment_ids is not "
            "supported -- the packed path always zeros state at segment boundaries "
            "(see packed_forward); for real s0 on a first segment use "
            "gated_delta_net_chunked(..., batched_segments=False)"
        )
    t_len, hv, d = q.shape
    device = q.device
    compute_dtype = torch.float32
    scale = 1.0 / (d ** 0.5)

    if groups is None:
        groups = compute_segment_groups(segment_ids, chunk)["groups"]

    n_seg_total = sum(len(g["seg_indices"]) for g in groups)
    out_f32 = torch.zeros(t_len, hv, d, dtype=compute_dtype, device=device)
    states: list[torch.Tensor | None] = [None] * n_seg_total

    q32 = (q * scale).to(compute_dtype)
    k32 = k.to(compute_dtype)
    v32 = v.to(compute_dtype)
    g32 = g_log.to(compute_dtype)
    b32 = beta.to(compute_dtype)

    for grp in groups:
        n_chunks_g = grp["n_chunks"]
        seg_idx_g = grp["seg_indices"]
        starts_g = grp["starts"]
        lengths_g = grp["lengths"]
        n_seg_g = len(seg_idx_g)
        c_total = n_chunks_g * chunk

        lengths_t = torch.tensor(lengths_g, dtype=torch.int64, device=device)
        starts_t = torch.tensor(starts_g, dtype=torch.int64, device=device)
        total_valid = sum(lengths_g)  # already known on Python side, no sync

        seg_of_flat = torch.repeat_interleave(torch.arange(n_seg_g, device=device), lengths_t)
        cum_excl = torch.cat([torch.zeros(1, dtype=torch.int64, device=device),
                               torch.cumsum(lengths_t, dim=0)[:-1]])
        offset_in_flat = torch.repeat_interleave(cum_excl, lengths_t)
        col_idx = torch.arange(total_valid, device=device) - offset_in_flat  # local position 0..L_i-1
        src_pos = torch.repeat_interleave(starts_t, lengths_t) + col_idx  # global position in q/k/v/...

        q_pad = q32.new_zeros(n_seg_g, c_total, hv, d)
        k_pad = k32.new_zeros(n_seg_g, c_total, hv, d)
        v_pad = v32.new_zeros(n_seg_g, c_total, hv, d)
        g_pad = g32.new_zeros(n_seg_g, c_total, hv)
        b_pad = b32.new_zeros(n_seg_g, c_total, hv)

        # Single index_put per tensor (scatter with pre-computed indices): no
        # cat/slice per segment. Beyond true length remains zero (exact no-op on
        # state, see module docstring).
        q_pad[seg_of_flat, col_idx] = q32[src_pos]
        k_pad[seg_of_flat, col_idx] = k32[src_pos]
        v_pad[seg_of_flat, col_idx] = v32[src_pos]
        g_pad[seg_of_flat, col_idx] = g32[src_pos]
        b_pad[seg_of_flat, col_idx] = b32[src_pos]

        nb = n_seg_g * hv  # combined batch: segment and head are both independent
        tril_incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=device))
        tril_strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=device), diagonal=-1)
        eye_c = torch.eye(chunk, dtype=compute_dtype, device=device)

        s = torch.zeros(nb, d, d, dtype=compute_dtype, device=device)
        outs_g = []
        for c_i in range(n_chunks_g):
            sl = slice(c_i * chunk, (c_i + 1) * chunk)
            # [n_seg_g, chunk, Hv, D] -> [n_seg_g*Hv, chunk, D] (same batch
            # merge as single-segment version, with an extra batch axis).
            q_c = q_pad[:, sl].permute(0, 2, 1, 3).reshape(nb, chunk, d)
            k_c = k_pad[:, sl].permute(0, 2, 1, 3).reshape(nb, chunk, d)
            v_c = v_pad[:, sl].permute(0, 2, 1, 3).reshape(nb, chunk, d)
            beta_c = b_pad[:, sl].permute(0, 2, 1).reshape(nb, chunk)
            g_c = g_pad[:, sl].permute(0, 2, 1).reshape(nb, chunk)

            gc = torch.cumsum(g_c, dim=-1)
            decay_start = torch.exp(gc)

            diff = gc.unsqueeze(-1) - gc.unsqueeze(-2)
            diff_masked = diff.masked_fill(~tril_incl, -1e30)
            ratio = torch.exp(diff_masked)

            kk = torch.bmm(k_c, k_c.transpose(-1, -2))
            m = beta_c.unsqueeze(-1) * ratio * kk
            m = m.masked_fill(~tril_strict, 0.0)
            i_plus_m = eye_c.unsqueeze(0) + m

            # L7c: `s` is literally `torch.zeros(...)` (line above) at chunk 0
            # of EVERY group -- the ValueError guard at function start already
            # enforces `s0` zero, so the zero-state branch here is unconditional
            # (no caller flag needed, unlike `gated_delta_net_chunked`, where the
            # first segment of `_gated_delta_net_chunked_by_segment` can have
            # real `s0`).
            if c_i == 0:
                rhs = beta_c.unsqueeze(-1) * v_c
            else:
                kv0 = torch.bmm(k_c, s)
                rhs = beta_c.unsqueeze(-1) * (v_c - decay_start.unsqueeze(-1) * kv0)
            delta = torch.linalg.solve_triangular(i_plus_m, rhs, upper=False, unitriangular=True)

            qk = torch.bmm(q_c, k_c.transpose(-1, -2))
            w = (ratio * qk).masked_fill(~tril_incl, 0.0)
            if c_i == 0:
                out_c = torch.bmm(w, delta)
            else:
                q_s_in = torch.bmm(q_c, s)
                out_c = decay_start.unsqueeze(-1) * q_s_in + torch.bmm(w, delta)

            outs_g.append(out_c.reshape(n_seg_g, hv, chunk, d).permute(0, 2, 1, 3))  # [n_seg_g,chunk,Hv,D]

            # L7c: final state update skipped at last chunk if `need_state=False`
            # -- see block below for returned value.
            if c_i == n_chunks_g - 1 and not need_state:
                continue
            decay_last = decay_start[:, -1]
            r_last = ratio[:, -1, :]
            weighted = r_last.unsqueeze(-1) * delta
            ks_delta = torch.bmm(k_c.transpose(-1, -2), weighted)
            s = decay_last.unsqueeze(-1).unsqueeze(-1) * s + ks_delta

        out_pad_g = torch.cat(outs_g, dim=1)  # [n_seg_g, c_total, Hv, D]
        out_f32[src_pos] = out_pad_g[seg_of_flat, col_idx]  # single unpack gather

        if need_state:
            s_by_seg = s.reshape(n_seg_g, hv, d, d)
            for local_i, seg_i in enumerate(seg_idx_g):
                states[seg_i] = s_by_seg[local_i]
        else:
            for seg_i in seg_idx_g:  # L7c (R3): "s0 unchanged", never recomputed
                states[seg_i] = s0

    s_last = states[-1]  # last segment by position (contiguous and increasing, see docstring)
    return out_f32.to(q.dtype), s_last


def causal_depthwise_conv_vec(
    x_new: torch.Tensor,  # [T, C]
    history: torch.Tensor,  # [K-1, C] (zeros if sequence start)
    weight: torch.Tensor,  # [C, K] (tap 0 = oldest, tap K-1 = current)
    dilation: int = 1,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same semantics as the autoregressive depthwise convolution
    (`out[t,c] = sum_k weight[c,k] * x[t-(K-1-k)*dilation, c]`, missing
    history treated as zero), without loop over T: sum of K shifted views
    (contiguous) of a buffer with zero-padding at front along
    `(K-1)*dilation` -- the desired index `hist_len + t - back` (where
    `back=(K-1-k)*dilation`), when negative (insufficient history at
    sequence start), falls exactly into the zero-padding zone: no explicit
    mask needed, the result is bit-identical to the original (no different
    rounding: same sum in same order over k, only indexing is vectorized).

    `segment_ids` (additive, stage 0 T3, spec §5): `[T]`, one id per position
    of `x_new` (never for `history`, always treated as "another segment" --
    padding/history zone with dummy id -1, never equal to a real id because
    `pack.pack_fragments` produces only id >=0): each shifted view is
    multiplied by the mask "same segment as current position"
    (`seg_full[s:s+t_len] == segment_ids`), on top of the zero-padding already
    present at buffer start. Equal (not bit-identical: same algorithm, different
    sum order than the autoregressive version, already true without
    `segment_ids`) to the loop version with the same `segment_ids` -- proved
    by the equality test between the two implementations. Default `None` ->
    behavior bit-identical to current.
    """
    kern = weight.shape[1]
    t_len, c = x_new.shape
    hist_len = history.shape[0]
    max_back = (kern - 1) * dilation

    pad_front = x_new.new_zeros(max_back, c)
    full = torch.cat([pad_front, history, x_new], dim=0)  # [max_back+hist_len+T, C]
    start0 = max_back + hist_len  # index of x_new[0] in `full`

    seg_full = None
    if segment_ids is not None:
        seg_pad = torch.full(
            (max_back + hist_len,), -1, dtype=segment_ids.dtype, device=segment_ids.device,
        )
        seg_full = torch.cat([seg_pad, segment_ids], dim=0)  # same indexing as `full`

    acc = torch.zeros(t_len, c, dtype=x_new.dtype, device=x_new.device)
    for k_idx in range(kern):
        back = (kern - 1 - k_idx) * dilation
        s = start0 - back
        contrib = full[s : s + t_len] * weight[:, k_idx]
        if seg_full is not None:
            same = (seg_full[s : s + t_len] == segment_ids).to(contrib.dtype).unsqueeze(-1)
            contrib = contrib * same
        acc = acc + contrib
    return acc

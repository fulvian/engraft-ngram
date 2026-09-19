"""Packing multiple fragments in canonical `[EOS]+frag` form into a single
sequence, with `segment_ids`/`positions` restarting from 0 at each fragment
-- attention, causal conv (PLE and delta-net), the chunked delta-net form,
AND the PLE n-gram addressing are all masked/recomputed at segment
boundaries (the fourth channel, `seq.OverlayEmb`, is not a `with`-scoped
patch: `segment_ids` is passed to its constructor -- see its docstring for
the hidden constraint that makes this necessary: the current position's own
EOS token does not cut its own context in the n-gram hash).

`pack_fragments` builds only data (no patching). `packed_forward` is the
context manager that makes the three patched channels (attention, PLE/
delta-net conv, chunked delta-net) boundary-aware, in the same style as
`seq.expert_op_patched`: temporary substitution, for the duration of the
`with` block, of module-level references read as free names by the calling
code (never a line of `model.py`/`layers.py` changed for THIS purpose --
the only lines touched in those files are the new, additive `segment_ids=None`
parameters).

`model.run_layer` calls `attention_full` as a name imported directly into
`model.py` (`from layers import (..., attention_full, ...)`): the same
situation `seq.expert_op_patched` already exploits for `moe_ffn` -- here
`model_mod.attention_full` is patched the same way. `linear_attn_layer`/
`ple_forward` instead call `gated_delta_net_recurrence`/
`causal_depthwise_conv` as free names resolved from `layers.py`'s own
namespace -- here `layers_mod.gated_delta_net_recurrence`/
`layers_mod.causal_depthwise_conv` are patched the same way.

Declared implementation choice: `packed_forward` is SELF-CONTAINED -- it
applies the chunked/vectorized delta-net form itself (calls
`gated_delta_net_chunked`/`causal_depthwise_conv_vec` directly) instead of
composing with a separately-activated patch: simpler, no fragile
introspection to check "is it already active", and a caller cannot forget to
activate it separately. The precondition "no base_state/cache" (zeroing at
segment boundaries would erase a real prefix state) is checked AT RUNTIME at
the three patch points (non-empty history/cache -> exception), not by
introspecting `Replica.prefix()`'s parameters (the context manager does not
see them)."""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import torch

from engraft.replica.delta_chunk import (
    causal_depthwise_conv_vec,
    compute_segment_groups,
    gated_delta_net_chunked,
    gated_delta_net_chunked_batched,
)
import engraft.replica.layers as layers_mod
import engraft.replica.model as model_mod


# --------------------------------------------------------------------------
# Packing: data only, no patching
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Packed:
    """A packed sequence (one greedily-filled bin).

    `tokens`: concatenation of the fragments in canonical `[eos]+frag` form
    (the last token is a label only, never processed -- same convention as
    `len(tokens)-1 = n_prefix` used everywhere else in the reference engine).
    `segment_ids`/`positions`: `[n_prefix]` (`n_prefix = len(tokens)-1`), one
    value for EVERY prefix position of the packed sequence (boundaries
    included: the position at a non-final fragment's last token EXISTS in the
    array, but its prediction is the boundary's, so it is never in
    `frag_slices`). `frag_slices[i] = (start, end)`, exclusive, index in
    prefix-position space (= `state.logits`) of fragment i's "significant"
    positions only (the same `len(canonical_frag_i)-1` rows an isolated
    `seq_forward` on that fragment alone would produce -- NEVER the boundary
    row)."""

    tokens: list[int]
    segment_ids: torch.Tensor  # [n_prefix] int64
    positions: torch.Tensor  # [n_prefix] float64
    frag_slices: list[tuple[int, int]]

    def to(self, device) -> "Packed":
        """Additive copy with `segment_ids`/`positions` moved to `device`
        (`tokens`/`frag_slices` are plain Python data, never tensors --
        nothing to move). `pack_fragments` always builds these two tensors on
        CPU; on a real device the activations live elsewhere -- without
        moving them, the three masked channels (`packed_forward`) fail with
        "Expected all tensors to be on the same device". A dedicated method
        instead of ad hoc patching at each call site: a caller does
        `packed.to(dev)` once and uses the result everywhere."""
        return Packed(
            tokens=self.tokens,
            segment_ids=self.segment_ids.to(device),
            positions=self.positions.to(device),
            frag_slices=self.frag_slices,
        )


def pack_fragments(frags: list[list[int]], eos: int, max_len: int) -> list[Packed]:
    """Packs `frags` (raw fragments, no leading EOS) into a list of `Packed`:
    each fragment becomes `[eos]+frag` (canonical form), GREEDY filling in
    the given order (a fragment enters the current bin if it fits, otherwise
    a new bin opens) -- no reordering, no best-fit. Raises `ValueError` if a
    single canonical fragment alone exceeds `max_len` (never packable into
    any bin, never silently truncated)."""
    canon = [[eos] + list(f) for f in frags]
    lengths = [len(c) for c in canon]
    for i, ln in enumerate(lengths):
        if ln > max_len:
            raise ValueError(
                f"pack_fragments: fragment {i} (canonical length {ln}, with EOS) "
                f"exceeds max_len={max_len} on its own"
            )

    bins: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    for i, ln in enumerate(lengths):
        if cur and cur_len + ln > max_len:
            bins.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len += ln
    if cur:
        bins.append(cur)

    out: list[Packed] = []
    for bin_idxs in bins:
        tok_offsets: list[int] = []
        tokens: list[int] = []
        for fi in bin_idxs:
            tok_offsets.append(len(tokens))
            tokens.extend(canon[fi])
        t_total = len(tokens)
        n_prefix = t_total - 1

        seg = torch.zeros(n_prefix, dtype=torch.int64)
        pos = torch.zeros(n_prefix, dtype=torch.float64)
        frag_slices: list[tuple[int, int]] = []
        for local_seg, fi in enumerate(bin_idxs):
            off = tok_offsets[local_seg]
            ln = lengths[fi]
            # This fragment's prefix positions in the global array:
            # off..off+ln-1 (ln local positions 0..ln-1), truncated to
            # n_prefix -- happens only for the LAST fragment of the bin (its
            # final token is the whole packed sequence's only label, exactly
            # as in the isolated case: `off+ln = t_total`, so `end_all =
            # n_prefix` gives `ln-1` positions, not `ln`).
            end_all = min(off + ln, n_prefix)
            if end_all > off:
                seg[off:end_all] = local_seg
                pos[off:end_all] = torch.arange(end_all - off, dtype=torch.float64)
            # frag_slices ALWAYS excludes the boundary position (the
            # fragment's last one, `off+ln-1`): for non-final fragments it
            # exists in the array (above) but predicts the NEXT fragment's
            # token, never significant; for the bin's last fragment it does
            # not exist at all (>= n_prefix). Either way the width is
            # `ln-1`, identical to what the isolated case would produce.
            frag_slices.append((off, off + ln - 1))
        out.append(Packed(tokens=tokens, segment_ids=seg, positions=pos, frag_slices=frag_slices))
    return out


# --------------------------------------------------------------------------
# packed_forward: temporary patches for the three channels
# --------------------------------------------------------------------------


@contextmanager
def packed_forward(replica, segment_ids: torch.Tensor | None, chunk: int = 64,
                    batched_segments: bool = True, info: dict | None = None,
                    need_state: bool = False):
    """Replaces, only for the duration of the `with` block:
      - `model.attention_full` (read by `Replica.run_layer` as a free name
        imported directly into `model.py`) with a version that passes
        `segment_ids` to `layers.attention_full` -- raises if the caller
        still passes non-`None` `k_cache`/`v_cache` (precondition: no
        `base_state`/`cache`).
      - `layers.gated_delta_net_recurrence` (a free name resolved from
        `layers.py`'s own namespace, called by `linear_attn_layer`) with
        `gated_delta_net_chunked(..., segment_ids=..., chunk=chunk)` --
        raises if `s0` is not all zero (a real prefix's recurrent state:
        `delta_net_init_state` always zero-initializes it for a fresh
        `LayerState()` -- a non-null `s0` can only come from a
        `base_state`/`cache`).
      - `layers.causal_depthwise_conv` (a free name, called by both
        `linear_attn_layer` and `ple_forward`) with
        `causal_depthwise_conv_vec(..., segment_ids=...)` -- raises if
        `history` is not all zero (the delta-net's history is ALWAYS
        `K-1` long, even for a fresh `LayerState()` -- `delta_net_init_state`
        fills it with zeros -- while the PLE's is 0 long only when fresh;
        length alone does not distinguish "no real history" from "real
        history", the CONTENT does: an accumulated real prefix is never
        exactly zero).

    With `segment_ids=None`, patches nothing (the context manager is a
    no-op): a caller that passes `segment_ids=None` downstream (`seq.
    seq_forward`) pays no cost and sees no behavior change.

    `batched_segments` (default `True`): boundaries and the grouping by
    `ceil(L/chunk)` are computed ONCE here, before entering the `with` --
    not at every call of `patched_recurrence` (one per layer, tens of times
    per forward, multiplied by the checkpoint recompute and the backward:
    without this, dozens of device->host syncs per step instead of one).
    With `batched_segments=True` the recurrence uses
    `gated_delta_net_chunked_batched(..., groups=groups)` (segments as a
    batch dimension); with `False`, a plain Python loop per segment (kept
    for A/B comparison, never the default measurement path).

    `need_state` (default `False`): forwarded to `gated_delta_net_chunked`/
    `gated_delta_net_chunked_batched` -- this descent never has
    `base_state`/`cache` (class precondition, guarded above), so nothing
    consumes the delta-net's final state returned by `patched_recurrence`:
    the default here is `False` (new behavior relative to the library
    functions' own default of `True`), which skips the final state update of
    the last chunk/segment. `True` restores the always-computed-state
    behavior (never consumed anyway in this descent).

    Always restores in `finally`, even on exception."""
    if segment_ids is None:
        yield
        return

    seg_groups = None
    if batched_segments:
        sg = compute_segment_groups(segment_ids, chunk)
        seg_groups = sg["groups"]
        if info is not None:
            info.update(n_groups=len(sg["groups"]), n_rows_padded=sg["n_rows_padded"],
                        n_rows_naive=sg["n_rows_naive"], padding_ratio_vs_naive=sg["padding_ratio_vs_naive"])

    orig_attention_full = model_mod.attention_full
    orig_recurrence = layers_mod.gated_delta_net_recurrence
    orig_conv = layers_mod.causal_depthwise_conv

    # `positions` is the SAME object for every layer inside this `with` --
    # the mask (causal ∧ same_segment) depends only on `positions`/
    # `segment_ids`, both fixed inside the `with`, so it is computed once and
    # passed as `attn_mask=` to the following calls instead of being
    # recomputed at every layer by `layers.attention_full`. The comparison is
    # by REFERENCE (`is`), never `id()`: `id()` of a collected object can be
    # reassigned to a new one, giving a silent false positive -- the
    # reference itself is kept in `last_positions`, so the object cannot be
    # collected while it is the memo's key.
    last_positions = None
    last_mask = None
    if info is not None:
        info["attn_mask_reuses"] = 0

    def patched_attention_full(x, w, positions, hparams, k_cache, v_cache, cache_positions):
        nonlocal last_positions, last_mask
        if k_cache is not None or v_cache is not None or cache_positions is not None:
            raise RuntimeError(
                "packed_forward: k_cache/v_cache/cache_positions are not None -- "
                "precondition violated (packing does not compose with base_state/cache)"
            )
        if positions is last_positions:
            mask = last_mask
            if info is not None:
                info["attn_mask_reuses"] += 1
        else:
            mask = (positions[:, None] >= positions[None, :]) & (segment_ids[:, None] == segment_ids[None, :])
            last_positions = positions
            last_mask = mask
        return orig_attention_full(
            x, w, positions, hparams, k_cache, v_cache, cache_positions, attn_mask=mask,
        )

    def patched_recurrence(q, k, v, g_log, beta, s0):
        if s0.numel() and not torch.all(s0 == 0):
            raise RuntimeError(
                "packed_forward: s0 (delta-net state) is not zero -- "
                "precondition violated (packing does not compose with base_state/cache)"
            )
        if batched_segments:
            return gated_delta_net_chunked_batched(q, k, v, g_log, beta, s0, chunk, segment_ids,
                                                     groups=seg_groups, need_state=need_state)
        # `s_zero=True`: the guard above already checked that `s0` is all
        # zero -- applies to the first segment inside the per-segment loop
        # (later segments are always zero-state by construction, regardless
        # of this flag).
        return gated_delta_net_chunked(q, k, v, g_log, beta, s0, chunk=chunk, segment_ids=segment_ids,
                                        batched_segments=False, s_zero=True, need_state=need_state)

    def patched_conv(x_new, history, weight, dilation=1):
        if history.numel() and not torch.all(history == 0):
            raise RuntimeError(
                "packed_forward: history is not zero -- precondition violated "
                "(packing does not compose with base_state/cache)"
            )
        return causal_depthwise_conv_vec(x_new, history, weight, dilation=dilation, segment_ids=segment_ids)

    model_mod.attention_full = patched_attention_full
    layers_mod.gated_delta_net_recurrence = patched_recurrence
    layers_mod.causal_depthwise_conv = patched_conv
    try:
        yield
    finally:
        model_mod.attention_full = orig_attention_full
        layers_mod.gated_delta_net_recurrence = orig_recurrence
        layers_mod.causal_depthwise_conv = orig_conv

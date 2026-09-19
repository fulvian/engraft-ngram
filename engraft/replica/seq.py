"""Sequence forward+backward with a differentiable PLE overlay, built entirely
by temporary monkey-patching of `engraft.replica.model`/`engraft.replica.layers`
(those two files are not touched): `Replica.prefix()` does not accept an
external differentiable PLE embedding or MoE implementation, so this module
shadows `replica.ple_true_emb`, `replica._expert_fns` and the module-level
`model.moe_ffn` for the duration of a single call, then restores them.

`Replica.prefix()` calls `self.ple_true_emb(tokens, t, overlay=...)`, always a
numpy gather (CPU, no gradient) concatenated with `torch.cat`. There is no
parameter to inject a differentiable tensor in its place -- but the call goes
through `self.ple_true_emb`, an attribute looked up first on the instance,
then on the class: a plain function placed in `replica.__dict__["ple_true_emb"]`
shadows it WITHOUT going through the descriptor protocol (it does not become a
bound method, it is called exactly as given, with no implicit `self`). The
same holds for `replica._expert_fns`. All patches are temporary, applied and
removed inside `expert_op_patched`/`seq_forward` (and their Triton
counterparts): no line of `model.py`/`layers.py` ever changes.

Verifying that a patch actually took effect (not just that it did not raise):
`seq_forward` counts calls to the patched gather and exposes them in
`PatchStats.n_ple_calls`, which the caller compares against the expected
number of positions -- a patch that "does not take" would produce a
numerically correct forward (identical to the real gather, if `rows_var`
starts from the real values) but a missing or zero gradient; the expected
call count is the proof the patched path was actually taken.

This file ships the reference (CPU-capable) forward/backward path
(`moe_ffn_seq`, `expert_op_patched`, `seq_forward`) used by the `--fake`
descent mode and by the maestro/teacher capture, plus the one Triton kernel
this repository publishes (`engraft.replica.triton_experts.TritonExpertMatmul`,
a plain per-expert matmul): `moe_ffn_seq_triton`, `expert_op_patched_triton`,
`seq_forward_triton`, `seq_step_triton`. The grouped/per-layer Triton kernel,
`torch.compile` fusion, and the associated memory/latency tuning knobs
(kernel tile sizes, launch configs, gate/up memoization, checkpoint memory
modes beyond the simplest one) are not part of this reference: they were a
private speed layer, not part of the technique, and are not published here.
"""
from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.profiler import record_function

from engraft.lens import RowSet
from engraft.replica.iq4_op import ExpertMatmul
import engraft.replica.layers as layers_mod
import engraft.replica.model as model_mod
import engraft.replica.pack as PACK

try:
    # Triton is an optional dependency: importable only when installed with a
    # matching GPU stack. The four symbols below stay module ATTRIBUTES (never
    # local to a function) so that tests can monkeypatch them (e.g.
    # `monkeypatch.setattr(seq, "TritonExpertMatmul", ...)`) even when Triton
    # is not installed -- a module `__getattr__` would not be enough, since
    # bare references inside this file are `LOAD_GLOBAL`, never routed
    # through `__getattr__`.
    from engraft.replica.triton_experts import TritonExpertMatmul
except ImportError:  # pragma: no cover -- exercised by the no-Triton import test
    TritonExpertMatmul = None  # type: ignore[assignment]


def _require_triton(what: str) -> None:
    """Runtime guard for the one use of the Triton kernel (`moe_ffn_seq_triton`,
    always, regardless of any other setting). Raises a clear `RuntimeError`
    naming the caller instead of the opaque `AttributeError` that would
    otherwise result from `None.apply(...)`."""
    if TritonExpertMatmul is None:
        raise RuntimeError(
            f"{what}: requires Triton (package not installed, or the import failed) -- "
            "the CPU reference path is seq_forward/expert_op_patched "
            "(ExpertMatmul, engraft.replica.iq4_op), or --fake mode.")


# --------------------------------------------------------------------------
# CPU reference: sequence MoE forward with a non-retaining IQ4 operator
# --------------------------------------------------------------------------


def moe_ffn_seq(
    x: torch.Tensor,  # [T, n_embd]
    gate_inp: torch.Tensor,
    expert_gate_fn,
    expert_up_fn,
    expert_down_fn,
    selected_experts: torch.Tensor,  # [T, n_expert_used]
    up_shexp: torch.Tensor,
    gate_shexp: torch.Tensor,
    down_shexp: torch.Tensor,
    gate_inp_shexp: torch.Tensor,
    n_expert_used: int,
    grouped: bool = True,
    capture: dict | None = None,
) -> torch.Tensor:
    """Same structure as `layers.moe_ffn(grouped=True)`, but the three
    per-expert products go through `ExpertMatmul.apply` instead of
    `xe @ w.T`: the dequantized expert matrix is never retained by the graph
    between forward and backward. `grouped` is accepted only for a signature
    identical to `moe_ffn` (the caller in `run_layer` always passes it as a
    keyword) -- this path is always grouped.

    One deliberate difference from `moe_ffn(grouped=True)`: a single
    `index_add` over all concatenated experts instead of one `index_add` per
    expert inside the loop -- same result (the `(t,j)` indices are disjoint
    by construction, one slot per pair), but without one `contrib` allocation
    per active expert at every layer.

    `capture` (dict or `None`, same semantics as `layers.moe_ffn`): populated
    with `{"experts":.., "dense":.., "weighted_sum":..}` if given -- needed
    because THIS function, not `layers.moe_ffn`, is the one actually active
    during `seq_forward` (`model_mod.moe_ffn` is patched to it by
    `expert_op_patched`, below)."""
    t_len, n_embd = x.shape
    logits = x.to(torch.float32) @ gate_inp.to(torch.float32).T
    probs = torch.softmax(logits, dim=-1)

    weights = torch.gather(probs, 1, selected_experts)
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(6.103515625e-5)
    weights = (weights / weights_sum).to(x.dtype)

    sel = selected_experts.detach().to("cpu").tolist()
    slots: dict[int, list[tuple[int, int]]] = {}
    for t in range(t_len):
        for j in range(n_expert_used):
            slots.setdefault(sel[t][j], []).append((t, j))

    ye_parts: list[torch.Tensor] = []
    idx_parts: list[torch.Tensor] = []
    for e in sorted(slots):
        pairs = slots[e]
        t_idx = torch.tensor([p[0] for p in pairs], dtype=torch.int64, device=x.device)
        j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device=x.device)
        xe = x[t_idx]  # [T_e, n_embd]

        gate_act = F.silu(ExpertMatmul.apply(xe, lambda e=e: expert_gate_fn(e)))
        up_act = ExpertMatmul.apply(xe, lambda e=e: expert_up_fn(e))
        h = gate_act * up_act  # [T_e, n_ff]
        ye = ExpertMatmul.apply(h, lambda e=e: expert_down_fn(e))  # [T_e, n_embd]
        ye = ye * weights[t_idx, j_idx].unsqueeze(-1)

        flat_idx = t_idx * n_expert_used + j_idx
        ye_parts.append(ye)
        idx_parts.append(flat_idx)

    contrib = torch.zeros(t_len * n_expert_used, n_embd, dtype=x.dtype, device=x.device)
    if ye_parts:
        all_ye = torch.cat(ye_parts, dim=0)
        all_idx = torch.cat(idx_parts, dim=0)
        contrib = contrib.index_add(0, all_idx, all_ye)

    contrib = contrib.view(t_len, n_expert_used, n_embd)
    out = contrib[:, 0]
    for j in range(1, n_expert_used):
        out = out + contrib[:, j]

    if capture is not None:
        capture["experts"] = out

    shared = F.silu(x @ gate_shexp.T) * (x @ up_shexp.T)
    shared_out = shared @ down_shexp.T
    shared_gate = torch.sigmoid(x @ gate_inp_shexp)
    dense_out = shared_out * shared_gate.unsqueeze(-1)
    if capture is not None:
        capture["dense"] = dense_out
    out = out + dense_out
    if capture is not None:
        capture["weighted_sum"] = out
    return out


# --------------------------------------------------------------------------
# Differentiable PLE overlay over a whole document
# --------------------------------------------------------------------------


_OVERLAY_CACHE_MAX_ENTRIES = 4096  # FIFO cap, ~100 MB expected over a corpus


def _gather_fragment_base(table, frag_tokens: list[int], row_map: dict, key_len: int | None,
                           n_heads: int, row_len: int) -> dict:
    """Raw gather (no overlay applied) for a whole fragment isolated from the
    rest of the packed batch (`frag_tokens` = only THIS fragment's tokens,
    never a neighbor's -- the n-gram hash only looks backward, so truncating
    the token list at the fragment boundary is a no-op relative to passing it
    the rest of the packed batch, see `OverlayEmb.build`). Same per-position
    formula as `OverlayEmb.build` (invariant: a long-key window, if present,
    is guaranteed entirely inside the fragment by the `t_local >= key_len-1`
    guard, never straddling a segment boundary -- otherwise the per-fragment
    cache would give a different result than the packed one). Reusable both
    for a direct gather and to populate `OverlayEmb._overlay_cache` -- one
    entry per DISTINCT combination of fragment tokens, valid across every
    packed batch and every pass (the content does not depend on neighbors in
    the bin). Also depends on `row_map`, built once for the whole descent run:
    a cache entry stays valid for the whole run, never across different runs
    (`row_map` may change)."""
    n_local = len(frag_tokens)
    base = torch.empty(n_local, n_heads, row_len, dtype=torch.float32)
    hits: list[tuple[int, int, int, str]] = []  # (t_local, h, variable_row, "key"|"id")
    for t_local in range(n_local):
        rs = RowSet.from_position(table, frag_tokens, t_local)
        base[t_local] = torch.from_numpy(rs.data.copy())
        window = None
        if key_len is not None and t_local >= key_len - 1:
            window = tuple(frag_tokens[t_local - key_len + 1 : t_local + 1])
        for h in range(n_heads):
            key = (window, h) if window is not None else None
            if key is not None and key in row_map:
                hits.append((t_local, h, row_map[key], "key"))
                continue
            g = int(rs.rows_global[h])
            if g in row_map:
                hits.append((t_local, h, row_map[g], "id"))
    return {"base": base, "hits": hits}


def overlay_key_len(row_map: dict) -> int | None:
    """Length `K` of the long-key window present in `row_map`, or `None` if it
    contains none (only `int` keys, global row ids -- unchanged behavior).
    Long keys are `(token_window, head)` tuples; all must share the same
    window length, otherwise the overlay would not know how far back to look
    at each position."""
    lengths = {len(key[0]) for key in row_map if isinstance(key, tuple)}
    if not lengths:
        return None
    assert len(lengths) == 1, (
        f"overlay_key_len: long keys of different lengths in row_map: {lengths}"
    )
    return next(iter(lengths))


@dataclasses.dataclass
class OverlayEmb:
    """Builds, for positions `start..n_positions-1` of `tokens`, the PLE
    embedding `[n_positions-start, n_embd]` with `row_map`'s rows replaced by
    `rows_var` -- differentiable in `rows_var`, identical to the real gather
    everywhere else.

    `table`: `PleTable` (or a fake compatible object) -- always read on CPU
    (the table never has its own gradient). `rows_var`: leaf tensor
    `[R, 160]` (`R` = distinct rows in `row_map`), on the device/dtype of the
    active phase. `row_map`: dict with mixed keys -- `int` = global row
    (always-available behavior) or `tuple[tuple[int, ...], int]` =
    `(K-token window, head)`, a long key that wins over the global row when
    both are present for the same position/head. Without tuple keys the
    long-key branch is never taken: bit-identical to the row-id-only case.

    `start` (default 0): builds only the new positions, as needed by a caller
    extending an already-computed prefix (`base_state`). Cache of the REAL
    embedding (`_base_cache`, key `(tuple(tokens), start, n_positions)`,
    never `id(tokens)`: an id can be recycled by a freed object): the gather
    from the table (`RowSet.from_position`, the expensive part, independent
    of `rows_var`) happens once per key; every subsequent call on the SAME
    `OverlayEmb` instance (even with `self.rows_var` reassigned to a
    different tensor) only does `index_put`. The cache lives on the
    INSTANCE, never a shared mutable default.

    `segment_ids` (default `None`, identical behavior without it): the
    n-gram hash (`table.ngram_addresses`) only looks backward in context, but
    the current position's own EOS token does not cut its own context.
    Without `segment_ids`, at the first position of a non-initial fragment of
    a packed batch, the hash reads the tail of the PREVIOUS fragment instead
    of only the EOS tokens of the isolated `[EOS]+frag`. With `segment_ids`
    given, every position `t` reads only `tokens[s(t):]` (`s(t)` = start of
    `t`'s segment, segments are contiguous and increasing as built by
    `pack.pack_fragments`), exactly reproducing the isolated hash.

    `overlay_cache` (default `None`, identical behavior without it): an
    EXTERNAL dict that lives for the whole run (passed to every call of
    `step_fn`/`forward_fn` in the descent loop) and caches the raw
    per-FRAGMENT gather (key: the TUPLE of the fragment's tokens, never a
    hash -- a hash collision would silently produce a wrong embedding, a
    Python dict still compares by full-tuple equality even at equal hash): a
    fragment re-drawn into a different bin on a later pass entirely skips
    `RowSet.from_position` (the expensive, `rows_var`-independent part) and
    only reapplies `index_put` with the current `rows_var`.

    Note on the cache key: it is not just the fragment's content, but content
    + SHAPE of its placement in the bin -- the last fragment of a bin loses
    its own boundary position (`pack.pack_fragments` truncates it), so its
    key is a tuple of `ln-1` tokens instead of `ln`. The same fragment can
    therefore get two different keys depending on whether it is last or not
    in the current bin: harmless (each key remains a pure, correct function
    of its own token range, never an aliasing between different fragments),
    it costs at most one extra cache entry per fragment."""

    table: object
    tokens: list[int]
    rows_var: torch.Tensor
    row_map: dict
    segment_ids: torch.Tensor | None = None
    overlay_cache: dict | None = None

    def __post_init__(self) -> None:
        self._base_cache: dict[tuple, dict] = {}
        self._seg_starts: list[int] | None = None
        self._seg_ranges: list[tuple[int, int]] | None = None
        if self.segment_ids is not None:
            # The gather is on CPU regardless (RowSet.from_position reads the
            # GGUF via numpy): nothing gained by keeping segment_ids on
            # rows_var's device.
            seg_cpu = self.segment_ids.detach().to("cpu").tolist()
            starts: dict[int, int] = {}
            seg_starts: list[int] = [0] * len(seg_cpu)
            ranges: list[tuple[int, int]] = []
            prev_v = None
            for i, v in enumerate(seg_cpu):
                if v not in starts:
                    starts[v] = i
                    if prev_v is not None:
                        ranges.append((starts[prev_v], i))
                seg_starts[i] = starts[v]
                prev_v = v
            if seg_cpu:
                ranges.append((starts[prev_v], len(seg_cpu)))
            self._seg_starts = seg_starts
            self._seg_ranges = ranges

    def _gather_base_via_fragment_cache(self, n_positions: int, key_len: int | None) -> dict:
        """Assembles the packed batch's `base`/`idx_*` by CONCATENATING the
        per-fragment pieces: used only when `overlay_cache` is given AND
        `segment_ids` is given -- `packed_forward` already forbids
        `base_state`/`cache` together with `segment_ids` (`start` is always
        0, `n_positions` is always `n_prefix`), so every `_seg_ranges`
        interval is ALWAYS requested whole -- no truncated-fragment case in
        practice, but the bounds below stay defensive regardless."""
        n_heads = self.table.n_heads
        row_len = self.rows_var.shape[1]
        base_parts: list[torch.Tensor] = []
        idx_t: list[int] = []
        idx_h: list[int] = []
        idx_r: list[int] = []
        n_key_hits = 0
        n_id_hits = 0
        for s_i, e_i in self._seg_ranges or []:
            if s_i >= n_positions:
                break  # segments are contiguous and increasing: none after this can fit
            frag_tokens = tuple(self.tokens[s_i:e_i])
            entry = self.overlay_cache.get(frag_tokens) if self.overlay_cache is not None else None
            if entry is None:
                entry = _gather_fragment_base(self.table, list(frag_tokens), self.row_map, key_len,
                                               n_heads, row_len)
                if self.overlay_cache is not None:
                    if len(self.overlay_cache) >= _OVERLAY_CACHE_MAX_ENTRIES:
                        self.overlay_cache.pop(next(iter(self.overlay_cache)))  # FIFO: oldest first
                    self.overlay_cache[frag_tokens] = entry
            e_local = min(e_i, n_positions)
            base_parts.append(entry["base"][: e_local - s_i].clone())  # clone: never write into the cached entry
            for t_local, h, r, kind in entry["hits"]:
                t_global = s_i + t_local
                if t_global >= n_positions:
                    continue
                idx_t.append(t_global)
                idx_h.append(h)
                idx_r.append(r)
                if kind == "key":
                    n_key_hits += 1
                else:
                    n_id_hits += 1
        base = (
            torch.cat(base_parts, dim=0) if base_parts
            else torch.empty(0, n_heads, row_len, dtype=torch.float32)
        )
        return {"base": base, "idx_t": idx_t, "idx_h": idx_h, "idx_r": idx_r,
                "n_key_hits": n_key_hits, "n_id_hits": n_id_hits}

    def build(self, n_positions: int, start: int = 0) -> torch.Tensor:
        n_heads = self.table.n_heads
        row_len = self.rows_var.shape[1]
        device = self.rows_var.device
        key_len = overlay_key_len(self.row_map)

        if self.overlay_cache is not None and self.segment_ids is not None and start == 0:
            cached = self._gather_base_via_fragment_cache(n_positions, key_len)
        else:
            seg_starts_key = tuple(self._seg_starts) if self._seg_starts is not None else None
            cache_key = (tuple(self.tokens), start, n_positions, seg_starts_key)
            cached = self._base_cache.get(cache_key)
            if cached is None:
                n_new = n_positions - start
                base = torch.empty(n_new, n_heads, row_len, dtype=torch.float32)
                idx_t: list[int] = []
                idx_h: list[int] = []
                idx_r: list[int] = []
                n_key_hits = 0
                n_id_hits = 0
                for t in range(start, n_positions):
                    if self._seg_starts is not None:
                        s_t = self._seg_starts[t]
                        t_local = t - s_t
                        rs = RowSet.from_position(self.table, self.tokens[s_t:], t_local)
                    else:
                        t_local = t
                        rs = RowSet.from_position(self.table, self.tokens, t)
                    base[t - start] = torch.from_numpy(rs.data.copy())
                    window = None
                    if key_len is not None and t_local >= key_len - 1:
                        window = tuple(self.tokens[t - key_len + 1 : t + 1])
                    for h in range(n_heads):
                        key = (window, h) if window is not None else None
                        if key is not None and key in self.row_map:
                            idx_t.append(t - start)
                            idx_h.append(h)
                            idx_r.append(self.row_map[key])
                            n_key_hits += 1
                            continue
                        g = int(rs.rows_global[h])
                        if g in self.row_map:
                            idx_t.append(t - start)
                            idx_h.append(h)
                            idx_r.append(self.row_map[g])
                            n_id_hits += 1
                cached = {
                    "base": base,
                    "idx_t": idx_t,
                    "idx_h": idx_h,
                    "idx_r": idx_r,
                    "n_key_hits": n_key_hits,
                    "n_id_hits": n_id_hits,
                }
                self._base_cache[cache_key] = cached

        self.n_key_hits = cached["n_key_hits"]
        self.n_id_hits = cached["n_id_hits"]

        base = cached["base"].to(device=device, dtype=self.rows_var.dtype)
        idx_t, idx_h, idx_r = cached["idx_t"], cached["idx_h"], cached["idx_r"]
        if idx_t:
            picked = self.rows_var[torch.tensor(idx_r, dtype=torch.int64, device=device)]
            emb3 = base.index_put(
                (
                    torch.tensor(idx_t, dtype=torch.int64, device=device),
                    torch.tensor(idx_h, dtype=torch.int64, device=device),
                ),
                picked,
            )
        else:
            emb3 = base
        return emb3.reshape(n_positions - start, n_heads * row_len)


# --------------------------------------------------------------------------
# Temporary patching (moe_ffn on the model module, ple_true_emb and
# _expert_fns on the Replica instance) and sequence-forward orchestration
# --------------------------------------------------------------------------


@dataclasses.dataclass
class PatchStats:
    """Counters exposed by the context managers to verify the patches took
    effect (a patch that does not take gives a numerically correct forward
    and a wrong/missing gradient -- the same failure mode as success, unless
    counted).

    `n_moe_calls`: counts calls to `moe_ffn_seq_triton`/`moe_ffn_seq` --
    without per-layer checkpointing it must equal `n_layer` (one call per
    layer with an FFN); with checkpointing (`torch.utils.checkpoint`,
    `use_reentrant=False`) it must double to `~2*n_layer` (one in the
    forward, one in the backward recompute): proof that the recompute really
    goes through the still-active patch, not through the already-restored
    original `moe_ffn`."""

    n_ple_calls: int = 0
    n_moe_calls: int = 0
    # Echo of `OverlayEmb.n_key_hits`/`n_id_hits` from the SAME overlay that
    # produced this forward -- not a second `OverlayEmb` rebuilt separately,
    # which would only prove the caller's construction matches, not that the
    # measured forward actually went through it. Zero with an int-only or
    # empty `row_map`: no new behavior for callers that never use long keys.
    n_key_hits: int = 0
    n_id_hits: int = 0
    # {il: n_token} -- positions whose expert selection (`selected_experts`,
    # [T,n_expert_used]) was REPLACED (not just observed) by the
    # checkpoint-freeze patch because it differed from what was captured at
    # the layer's first call -- populated only on a second call per layer
    # (checkpoint recompute); zero calls = zero entries = no freezing
    # happened (no active checkpoint, or no second call for that layer).
    # Direct measurement of how much routing "churn" the recompute actually
    # produces.
    routing_frozen_flips: dict = dataclasses.field(default_factory=dict)
    # {il: n_token} -- mismatch between the router's live routing and the
    # imposed locked routing (RBR, `routing_init`), counted ONCE per layer
    # (at the first call of `moe_ffn` for that layer after entering the
    # context manager -- never at later calls, those are
    # `routing_frozen_flips` above: the two measures are deliberately
    # disjoint). Empty without `routing_init` -- no new behavior.
    routing_flips_vs_init: dict = dataclasses.field(default_factory=dict)
    # {il: n_token} -- mismatch on ONLY the first top-k slot between live
    # routing and the imposed locked routing, same (first-per-layer) call as
    # `routing_flips_vs_init` above -- NEVER a second pass. Reason:
    # `routing_flips_vs_init` (any slot) is dominated by near-ties among the
    # top-k slots and does not discriminate a real router decision change
    # from a tie flipping at the tail of the list. Empty without
    # `routing_init` -- no new behavior.
    routing_flips_top1: dict = dataclasses.field(default_factory=dict)


def _apply_routing_patch(a: list, il: int | None, captured_routing: dict, seen: set,
                          stats: "PatchStats", routing_init: dict | None) -> None:
    """Replacement of `selected_experts` (`a[5]`, fixed position in `moe_ffn`'s
    signature), shared by `expert_op_patched` (CPU) and
    `expert_op_patched_triton`: mutates `a` IN PLACE. Called only when the
    caller has already checked `freeze_routing or routing_init is not None`
    (CPU path: only `routing_init is not None`, that path never had
    `freeze_routing`).

    `seen` (local to the caller's `with` block): distinguishes the FIRST call
    per layer (where, with `routing_init` given, the locked routing is
    imposed IMMEDIATELY and the mismatch against the live router is recorded
    in `routing_flips_vs_init[il]`; without `routing_init`, unchanged
    behavior -- capture only, no replacement) from later calls (the usual
    checkpoint-recompute freezing, `routing_frozen_flips` -- never the same
    layer key in both dicts for the same call)."""
    if il is None:
        if routing_init is not None:
            raise RuntimeError(
                "expert_op_patched: routing_init given but no current layer "
                "(current_il is None) -- the _expert_fns patch did not record "
                "the layer before the call to moe_ffn"
            )
        return
    selected = a[5]
    first = il not in seen
    seen.add(il)
    if first:
        if routing_init is not None:
            if il not in routing_init:
                raise RuntimeError(
                    f"expert_op_patched: routing_init is missing layer {il} "
                    f"(available layers: {sorted(routing_init)}) -- the locked "
                    "routing must cover every layer that calls moe_ffn, no "
                    "exception for the last layer"
                )
            ref = routing_init[il]
            if ref.shape[0] != selected.shape[0]:
                raise RuntimeError(
                    f"expert_op_patched: routing_init[{il}] has T={ref.shape[0]}, "
                    f"expected {selected.shape[0]} (current forward's T at layer {il})"
                )
            # `.contiguous()`: `.to(device=, dtype=)` is a no-op on strides
            # when device/dtype already match -- `stitch_routing_init`
            # produces non-contiguous views; `moe_ffn_seq` (CPU) tolerates any
            # stride, but `moe_ffn_seq_triton` is not guaranteed to -- forced
            # here, one single point for every caller of
            # `_apply_routing_patch`.
            ref = ref.to(device=selected.device, dtype=selected.dtype).contiguous()
            n_flipped = int((selected != ref).any(dim=-1).sum().item())
            if n_flipped:
                stats.routing_flips_vs_init[il] = n_flipped
            n_flipped_top1 = int((selected[:, 0] != ref[:, 0]).sum().item())
            if n_flipped_top1:
                stats.routing_flips_top1[il] = n_flipped_top1
            captured_routing[il] = ref
            a[5] = ref
        else:
            captured_routing[il] = selected.detach().clone()
    else:
        frozen = captured_routing[il]
        n_flipped = int((selected != frozen).any(dim=-1).sum().item())
        if n_flipped:
            stats.routing_frozen_flips[il] = stats.routing_frozen_flips.get(il, 0) + n_flipped
        a[5] = frozen


@contextmanager
def expert_op_patched(replica, precomputed_emb: torch.Tensor, stats: PatchStats, layer_hook=None, start: int = 0,
                       routing_init: "dict[int, torch.Tensor] | None" = None):
    """Replaces, only for the duration of the `with` block:
      - `model.moe_ffn` (module-level reference, read by `run_layer` as a
        free name) with `moe_ffn_seq`;
      - `replica.ple_true_emb` (instance attribute, shadows the class method
        without binding) with a gather from the precomputed tensor
        `precomputed_emb` (differentiable in whatever produced it, typically
        `OverlayEmb`);
      - `replica._expert_fns` with a version that forces `persist=True`.

    `start` (default 0): `precomputed_emb` covers only the new positions
    `start..n_prefix-1` -- the patch indexes `precomputed_emb[t-start]`, not
    `precomputed_emb[t]`. With `start=0` (default) it is the same index as
    always, identical behavior.

    `routing_init`: if given, `current_il` (recorded by
    `patched_expert_fns`) and a wrapper around `moe_ffn_seq` impose the
    locked routing at every layer -- same replacement, same two counters
    (`stats.routing_flips_vs_init`/`routing_frozen_flips`) and the same
    preconditions as `expert_op_patched_triton` (below), shared via
    `_apply_routing_patch`. With `routing_init=None` (default), no new
    behavior.

    Always restores in `finally`, even on exception."""
    had_ple = "ple_true_emb" in replica.__dict__
    orig_ple = replica.__dict__.get("ple_true_emb")
    had_expert_fns = "_expert_fns" in replica.__dict__
    orig_expert_fns_attr = replica.__dict__.get("_expert_fns")
    orig_expert_fns_bound = replica._expert_fns  # class-bound method (or a previous patch)
    orig_moe_ffn = model_mod.moe_ffn
    current_il = {"il": None}
    captured_routing: dict[int, torch.Tensor] = {}
    seen_layers: set[int] = set()

    def patched_ple_true_emb(tokens_arg, t, overlay=None):
        stats.n_ple_calls += 1
        return precomputed_emb[t - start].unsqueeze(0)

    def patched_expert_fns(il, persist, device=None, dtype=None):
        current_il["il"] = il
        if layer_hook is not None:
            layer_hook(il)
        return orig_expert_fns_bound(il, True, device=device, dtype=dtype)

    def counted_moe_ffn(*a, **kw):
        stats.n_moe_calls += 1
        if routing_init is not None:
            a = list(a)
            _apply_routing_patch(a, current_il["il"], captured_routing, seen_layers, stats, routing_init)
        return moe_ffn_seq(*a, **kw)

    replica.ple_true_emb = patched_ple_true_emb
    replica._expert_fns = patched_expert_fns
    model_mod.moe_ffn = counted_moe_ffn
    try:
        yield
    finally:
        model_mod.moe_ffn = orig_moe_ffn
        if had_ple:
            replica.__dict__["ple_true_emb"] = orig_ple
        else:
            del replica.__dict__["ple_true_emb"]
        if had_expert_fns:
            replica.__dict__["_expert_fns"] = orig_expert_fns_attr
        else:
            del replica.__dict__["_expert_fns"]


@contextmanager
def capture_layers(replica, capture: dict):
    """Read-only shadow of `replica.run_layer` (same pattern as
    `expert_op_patched`): calls the original method without touching the
    computation and records, per layer, `capture[il] = {"x": tensor
    [T,hc,n_embd] released from the graph (detached clone), "routing":
    ndarray [T,n_expert_used] or None}` -- used to attribute a difference
    layer by layer between the CPU reference path and a faster path, at both
    free and locked routing. Composable with `expert_op_patched`/
    `expert_op_patched_triton` (they shadow different attributes, no
    conflict); NOT composable with `run_layer_checkpointed` (both shadow
    `run_layer`: the last `with` wins, the first would be silently
    overridden)."""
    had = "run_layer" in replica.__dict__
    orig_attr = replica.__dict__.get("run_layer")
    orig_bound = replica.run_layer

    def patched(il, x, positions, state, routing_source, ple_emb, need_ffn_output=True,
                persist_experts=False, diag=None):
        new_x, new_state, routing_used = orig_bound(
            il, x, positions, state, routing_source, ple_emb,
            need_ffn_output=need_ffn_output, persist_experts=persist_experts, diag=diag,
        )
        capture[il] = {
            "x": new_x.detach().clone(),
            "routing": routing_used.detach().to("cpu").numpy() if routing_used is not None else None,
        }
        return new_x, new_state, routing_used

    replica.run_layer = patched
    try:
        yield
    finally:
        if had:
            replica.__dict__["run_layer"] = orig_attr
        else:
            del replica.__dict__["run_layer"]


@contextmanager
def capture_hidden(replica, hidden: dict, diag: dict | None = None):
    """Layer-by-layer determinism diagnostic: read-only shadow of
    `replica.run_layer`, same pattern as `capture_layers` -- but, unlike that
    one, moves the residual `x` to CPU float32 INSIDE the hook itself
    (`.to("cpu", torch.float32)`, never a `.clone()` that stays on the
    device): with many layers and a long sequence, one layer's residual can
    be tens of MB; the whole chain of one forward, offloaded to CPU, is many
    times that -- never accumulated on the accelerator, which stays at the
    cost of one layer at a time during the hook.

    `hidden[il]` = CPU float32 `torch.Tensor` `[T,hc,n_embd]` (the residual
    leaving layer `il`, after attention/delta-net AND the FFN if present).

    `diag` (dict or `None`): if given, replaces (does not merely add to)
    whatever `diag` `Replica.prefix()` would pass to `run_layer` (here always
    `None`, neither `seq_forward` nor `seq_forward_triton` forward it) --
    populated by the router's internal diagnostics with per-layer top-(k+1)
    indices/values, on CPU/numpy, only for layers where routing is computed
    live (never when `routing_source` is given for that layer). Note: passing
    `diag` changes the router's internal top-k call from `k` to `k+1` to get
    the same indices -- not guaranteed bit-identical to a forward without
    `diag`."""
    had = "run_layer" in replica.__dict__
    orig_attr = replica.__dict__.get("run_layer")
    orig_bound = replica.run_layer
    capture_diag = diag

    def patched(il, x, positions, state, routing_source, ple_emb, need_ffn_output=True,
                persist_experts=False, diag=None):
        new_x, new_state, routing_used = orig_bound(
            il, x, positions, state, routing_source, ple_emb,
            need_ffn_output=need_ffn_output, persist_experts=persist_experts,
            diag=capture_diag,
        )
        hidden[il] = new_x.detach().to("cpu", torch.float32)
        return new_x, new_state, routing_used

    replica.run_layer = patched
    try:
        yield
    finally:
        if had:
            replica.__dict__["run_layer"] = orig_attr
        else:
            del replica.__dict__["run_layer"]


@contextmanager
def capture_ple_gate(replica, out: dict, layer: int | None = None):
    """PLE-gate measurement: same pattern as `capture_hidden` above -- a
    read-only shadow of `replica.run_layer` that calls the original method
    unchanged. Unlike `capture_hidden`, this does not capture the residual
    `x` but `ple_forward`'s internal diagnostics (`layers.py`), which no
    caller forwards today (`Replica.prefix` never passes `ple_diag`).

    `layer` (default `None` -> `replica.hp.ple_layer`): the PLE layer to
    capture. `ValueError` if `not replica.hp.is_ple(layer)` -- wiring a
    non-PLE layer would silently capture nothing (`ple_forward` never runs
    on that layer).

    For the single `layer`, the patch passes `ple_diag={}` to the original
    call (never to the caller: if the caller had already given a non-null
    `ple_diag` for that layer, this raises `RuntimeError` instead of
    silently overwriting it -- never happens on the measurement path,
    `prefix` never passes it, but the contract stays explicit) and, on
    return, appends on CPU float32 `s`, `gate`, `gated_norm`, `hidden_norm`
    (`[T,hc]`) and `value_norm` (`[T]`) to `out` (`torch.cat` along `T` if
    the context sees more than one call -- only happens if the caller
    invokes `prefix()` more than once inside the same context, never during
    evaluation: one capture wraps one forward). Other layers forward
    `ple_diag` unchanged (`None` on the real path, `prefix` never passes it
    to any layer).

    Restores `run_layer` in `finally` even on exception (same pattern as
    `capture_hidden`/`capture_layers`). On exit without exception, if `out`
    is still empty this raises `RuntimeError`: the PLE layer was never
    crossed by the wrapped forward (happens on the fake path when `prefix()`
    receives fewer than two tokens and returns before the layer loop)."""
    if layer is None:
        layer = replica.hp.ple_layer
    if not replica.hp.is_ple(layer):
        raise ValueError(
            f"capture_ple_gate: layer {layer} is not the PLE layer (hp.ple_layer="
            f"{replica.hp.ple_layer})"
        )
    had = "run_layer" in replica.__dict__
    orig_attr = replica.__dict__.get("run_layer")
    orig_bound = replica.run_layer
    keys = ("s", "gate", "gated_norm", "hidden_norm", "value_norm")

    def patched(il, x, positions, state, routing_source, ple_emb, need_ffn_output=True,
                persist_experts=False, diag=None, ple_diag=None, detach_value=False, detach_gate=False):
        if il != layer:
            return orig_bound(
                il, x, positions, state, routing_source, ple_emb,
                need_ffn_output=need_ffn_output, persist_experts=persist_experts,
                diag=diag, ple_diag=ple_diag, detach_value=detach_value, detach_gate=detach_gate,
            )
        if ple_diag is not None:
            raise RuntimeError(
                f"capture_ple_gate: the caller already gave ple_diag for layer "
                f"{layer}, refusing to overwrite it silently"
            )
        local_diag: dict = {}
        new_x, new_state, routing_used = orig_bound(
            il, x, positions, state, routing_source, ple_emb,
            need_ffn_output=need_ffn_output, persist_experts=persist_experts,
            diag=diag, ple_diag=local_diag, detach_value=detach_value, detach_gate=detach_gate,
        )
        for key in keys:
            val = local_diag[key].detach().to("cpu", torch.float32)
            out[key] = torch.cat([out[key], val], dim=0) if key in out else val
        return new_x, new_state, routing_used

    replica.run_layer = patched
    try:
        yield
        if not out:
            raise RuntimeError(
                f"capture_ple_gate: PLE layer {layer} was never crossed by the "
                "wrapped forward (out is empty)"
            )
    finally:
        if had:
            replica.__dict__["run_layer"] = orig_attr
        else:
            del replica.__dict__["run_layer"]


def seq_forward(replica, tokens: list[int], rows_var: torch.Tensor, row_map: dict[int, int],
                 return_logits: bool = True, layer_hook=None,
                 routing_source: dict[int, np.ndarray] | None = None,
                 capture_routing: dict[int, np.ndarray] | None = None,
                 layer_capture: dict | None = None, grad_proxy: bool = True,
                 base_state: "model_mod.PrefixState | None" = None,
                 cache: "model_mod.PrefixCache | None" = None,
                 positions: torch.Tensor | None = None,
                 segment_ids: torch.Tensor | None = None,
                 delta_chunk_size: int = 64,
                 delta_batched: bool = True,
                 overlay_cache: dict | None = None,
                 routing_init: "dict[int, torch.Tensor] | None" = None):
    """Sequence forward: `Replica.prefix(tokens, grad_proxy=True,
    return_logits=...)` with the PLE gather replaced by `OverlayEmb` and the
    MoE by `moe_ffn_seq`, both patched only for the duration of this call.

    No `PrefixCache`/`overlay` dict passed to `prefix()`: a cache hit would
    skip the recompute -- and with it the whole differentiable graph this
    function builds on purpose every time; `overlay=None` leaves the row
    substitution entirely to the `ple_true_emb` patch.

    `layer_hook(il)`, if given, is called once per layer (at the point where
    `run_layer` asks for the expert closures) -- diagnostic only, never used
    by the computation itself.

    `routing_source`/`capture_routing`: forwarded unchanged to
    `Replica.prefix()` -- used for layer-by-layer attribution, to capture the
    routing from the CPU reference path and, optionally, to impose it on a
    faster path. `layer_capture` (dict or `None`): if given, wraps the call
    in `capture_layers` (above).

    `routing_init` (default `None`): forwarded to `expert_op_patched` -- see
    its docstring. Note: with `routing_init` given, `capture_routing` (if
    also given) keeps reporting the router's LIVE routing, not the imposed
    locked routing -- the mismatch between the two is exactly
    `stats.routing_flips_vs_init`.

    `grad_proxy`: forwarded unchanged to `Replica.prefix()`. Default `True`,
    backward-compatible with every existing caller. A caller that never calls
    `.backward()` on this forward should pass `False` explicitly to avoid
    retaining the whole per-layer autograd graph for nothing.

    Returns `(state, stats)`: `state.logits` (if `return_logits`) has shape
    `[n_prefix-start, n_vocab]`, `n_prefix = len(tokens)-1`; `stats.n_ple_calls`
    must equal `n_prefix-start` (one call per NEW position) -- the caller
    checks this to make sure the patch actually took.

    `base_state`/`cache` (default `None`): extend an already-computed prefix
    instead of recomputing it from scratch (typically the maestro/teacher,
    which reuses a shared document state per fragment, with empty
    `rows_var`/`row_map` -- a no-op overlay). Caution: this function always
    passes `overlay=None` to `Replica.prefix()` (row substitution lives in
    the `ple_true_emb` patch, not in the `overlay` dict) -- `PrefixCache`'s
    key over the rows read is therefore computed with an EMPTY overlay
    regardless of `rows_var`: a `cache` reused across steps with DIFFERENT
    `rows_var` (non-empty `row_map`, a real differentiable overlay) would
    collide on the same key and return a stale state computed with a
    previous `rows_var`. Safe only when `row_map` is empty or when every
    `cache` entry is used exactly once (the maestro/teacher's case): never in
    the overlay descent loop, which never passes `cache` to this function.

    `positions` (default `None`): forwarded to `Replica.prefix()` -- explicit
    RoPE positions instead of `arange(start, n_prefix)`.

    `segment_ids`/`delta_chunk_size` (default `None`/64): if `segment_ids` is
    given, wraps the call in `pack.packed_forward(replica, segment_ids,
    chunk=delta_chunk_size)` -- attention/conv/delta-net masked at segment
    boundaries (a packed sequence, `pack.pack_fragments`). Precondition
    (checked at runtime inside `packed_forward`): no `base_state`/`cache` --
    combining them with a non-`None` `segment_ids` raises `RuntimeError` from
    the first attention/conv layer that touches a non-empty state. With
    `segment_ids=None` (default), no extra patching, identical behavior.

    `delta_batched` (default `True`): forwarded to `pack.packed_forward` --
    `True` uses the batched chunked delta-net form (segments as a batch
    dimension), `False` a plain per-segment Python loop (kept for A/B
    comparison). Ignored with `segment_ids=None`.

    `overlay_cache` (default `None`): forwarded to `OverlayEmb` -- per-
    fragment raw-gather cache, keyed by its token tuple. `None` (default): no
    new behavior, every call recomputes the gather as before."""
    n_prefix = len(tokens) - 1
    if n_prefix <= 0:
        raise ValueError("seq_forward: need at least 2 tokens")
    _check_no_segment_ids_with_incremental_prefix("seq_forward", segment_ids, base_state, cache)

    start = 0
    base: model_mod.PrefixState | None = None
    if cache is not None:
        hit = cache.lookup(tokens[:n_prefix], {})
        if hit is not None:
            base, _ = hit
            start = base.n_prefix
    if base is None and base_state is not None:
        base = base_state
        start = base_state.n_prefix

    overlay_emb = OverlayEmb(replica.table, tokens, rows_var, row_map, segment_ids=segment_ids,
                            overlay_cache=overlay_cache)
    precomputed = overlay_emb.build(n_prefix, start=start)  # [n_prefix-start, n_embd], differentiable in rows_var

    stats = PatchStats()
    stats.n_key_hits = overlay_emb.n_key_hits
    stats.n_id_hits = overlay_emb.n_id_hits
    cap_cm = capture_layers(replica, layer_capture) if layer_capture is not None else _null_context()
    pack_cm = PACK.packed_forward(replica, segment_ids, chunk=delta_chunk_size, batched_segments=delta_batched)
    with expert_op_patched(replica, precomputed, stats, layer_hook=layer_hook, start=start,
                            routing_init=routing_init), cap_cm, pack_cm:
        state = replica.prefix(tokens, grad_proxy=grad_proxy, return_logits=return_logits,
                                routing_source=routing_source, capture_routing=capture_routing,
                                base_state=base_state, cache=cache, positions=positions)
    return state, stats


def exclusive_rows_at_position(table, tokens: list[int], n_positions: int, p: int, heads: range | None = None) -> set[int]:
    """Global rows of position `p` (among heads `heads`, default all) that do
    NOT appear at any other position `0..n_positions-1`, `!= p` -- the set
    over which a golden-value check can compare `seq_forward`'s gradient
    against an isolated single-step computation (a row reused at an earlier
    position receives, in the sequence forward, a gradient contribution from
    its causal effect on that earlier position too, which an isolated
    single-step computation -- frozen upstream state, no gradient -- does not
    see)."""
    heads = heads if heads is not None else range(table.n_heads)
    rs_p = RowSet.from_position(table, tokens, p)
    candidates = {int(rs_p.rows_global[h]) for h in heads}
    for t in range(n_positions):
        if t == p:
            continue
        rs_t = RowSet.from_position(table, tokens, t)
        candidates -= set(int(g) for g in rs_t.rows_global)
    return candidates


# ==========================================================================
# Triton per-expert path: the one published Triton kernel
# (`engraft.replica.triton_experts.TritonExpertMatmul`) in place of
# `ExpertMatmul` (torch dequant + separate matmul) -- `W` is never
# materialized in memory, `x` stays bf16, `wdot="split"` by default.
# ==========================================================================


@dataclasses.dataclass
class ExpertRef:
    """A reference (not a tensor) to a quantized expert: `moe_ffn_seq_triton`
    resolves it to raw bytes only at the point of use (`expert_byte_view`),
    never a dequantization. `w`: a `DeviceIQ4` instance with every tensor's
    bytes already resident on the device (loaded once at construction)."""

    w: object
    name: str
    e: int
    n_in: int
    n_out: int
    qtype: str


def expert_byte_view(w, name: str, e: int) -> torch.Tensor:
    """A view (no copy: reshape + first-axis indexing on a contiguous
    tensor) of expert `e`'s raw bytes for tensor `name`, from the whole blob
    already resident at `w._dev_bytes[name]` (`DeviceIQ4`, populated once at
    construction). Same shape convention as `DeviceIQ4`'s uncached expert
    dequantization: requires the expert axis to be outermost in the gguf
    reader's byte-shaped blob (true for `*_exps` tensors)."""
    tens = w._dev_bytes[name]
    n_expert = tens.shape[0]
    per_expert = tens.reshape(n_expert, -1, tens.shape[-1])
    return per_expert[e]


def _triton_expert_fns(replica, w, il: int):
    """Replacement for `Replica._expert_fns` on the Triton path: the three
    closures return `ExpertRef`, not tensors -- `w` is captured here (not on
    `replica._expert_fns`, which on the reference path always reads
    `self.w`) so a caller could pass a different `DeviceIQ4` than
    `replica.w` if ever needed."""
    p = f"blk.{il}."
    n_embd = replica.hp.n_embd
    n_ff = replica.hp.n_ff_exp

    def _qtype(name: str) -> str:
        return w._index[name][1].tensor_type.name

    gate_name = p + "ffn_gate_exps.weight"
    up_name = p + "ffn_up_exps.weight"
    down_name = p + "ffn_down_exps.weight"
    gate_qtype = _qtype(gate_name)
    up_qtype = _qtype(up_name)
    down_qtype = _qtype(down_name)

    def gate_fn(e: int) -> ExpertRef:
        return ExpertRef(w, gate_name, e, n_embd, n_ff, gate_qtype)

    def up_fn(e: int) -> ExpertRef:
        return ExpertRef(w, up_name, e, n_embd, n_ff, up_qtype)

    def down_fn(e: int) -> ExpertRef:
        return ExpertRef(w, down_name, e, n_ff, n_embd, down_qtype)

    return gate_fn, up_fn, down_fn


def moe_ffn_seq_triton(
    x: torch.Tensor,  # [T, n_embd]
    gate_inp: torch.Tensor,
    expert_gate_fn,  # (e) -> ExpertRef
    expert_up_fn,
    expert_down_fn,
    selected_experts: torch.Tensor,
    up_shexp: torch.Tensor,
    gate_shexp: torch.Tensor,
    down_shexp: torch.Tensor,
    gate_inp_shexp: torch.Tensor,
    n_expert_used: int,
    grouped: bool = True,
    wdot: str = "split",
    capture: dict | None = None,
) -> torch.Tensor:
    """Like `moe_ffn_seq`, but the three per-expert products go through the
    fused Triton kernels (`TritonExpertMatmul.apply` on raw bytes) instead of
    `ExpertMatmul.apply` on a torch-dequantized tensor: `W` is never
    materialized, neither in forward nor in backward.

    `capture`: same additive semantics as `moe_ffn_seq`/`layers.moe_ffn` --
    replicated here because `run_layer` always passes `capture=...` (`None`
    or a dict) to `model_mod.moe_ffn`, whichever implementation is patched in
    at that moment."""
    _require_triton("moe_ffn_seq_triton")
    t_len, n_embd = x.shape
    logits = x.to(torch.float32) @ gate_inp.to(torch.float32).T
    probs = torch.softmax(logits, dim=-1)

    weights = torch.gather(probs, 1, selected_experts)
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(6.103515625e-5)
    weights = (weights / weights_sum).to(x.dtype)

    sel = selected_experts.detach().to("cpu").tolist()
    slots: dict[int, list[tuple[int, int]]] = {}
    for t in range(t_len):
        for j in range(n_expert_used):
            slots.setdefault(sel[t][j], []).append((t, j))

    ye_parts: list[torch.Tensor] = []
    idx_parts: list[torch.Tensor] = []
    for e in sorted(slots):
        pairs = slots[e]
        t_idx = torch.tensor([p[0] for p in pairs], dtype=torch.int64, device=x.device)
        j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device=x.device)
        xe = x[t_idx]  # [T_e, n_embd]

        gref = expert_gate_fn(e)
        uref = expert_up_fn(e)
        dref = expert_down_fn(e)
        gate_act = F.silu(TritonExpertMatmul.apply(
            xe, expert_byte_view(gref.w, gref.name, gref.e), gref.qtype, gref.n_in, gref.n_out, wdot))
        up_act = TritonExpertMatmul.apply(
            xe, expert_byte_view(uref.w, uref.name, uref.e), uref.qtype, uref.n_in, uref.n_out, wdot)
        h = gate_act * up_act  # [T_e, n_ff]
        ye = TritonExpertMatmul.apply(
            h, expert_byte_view(dref.w, dref.name, dref.e), dref.qtype, dref.n_in, dref.n_out, wdot)  # [T_e, n_embd]
        ye = ye * weights[t_idx, j_idx].unsqueeze(-1)

        flat_idx = t_idx * n_expert_used + j_idx
        ye_parts.append(ye)
        idx_parts.append(flat_idx)

    contrib = torch.zeros(t_len * n_expert_used, n_embd, dtype=torch.float32, device=x.device)  # F32 accumulation: the Triton kernels return F32, index_add needs matching dtypes
    if ye_parts:
        all_ye = torch.cat(ye_parts, dim=0).to(torch.float32)
        all_idx = torch.cat(idx_parts, dim=0)
        contrib = contrib.index_add(0, all_idx, all_ye)

    contrib = contrib.view(t_len, n_expert_used, n_embd)
    out = contrib[:, 0]
    for j in range(1, n_expert_used):
        out = out + contrib[:, j]
    out = out.to(x.dtype)  # cast once after summing over experts (F32 islands)

    if capture is not None:
        capture["experts"] = out

    shared = F.silu(x @ gate_shexp.T) * (x @ up_shexp.T)
    shared_out = shared @ down_shexp.T
    shared_gate = torch.sigmoid(x @ gate_inp_shexp)
    dense_out = shared_out * shared_gate.unsqueeze(-1)
    if capture is not None:
        capture["dense"] = dense_out
    out = out + dense_out
    if capture is not None:
        capture["weighted_sum"] = out
    return out


@contextmanager
def expert_op_patched_triton(replica, w, precomputed_emb: torch.Tensor, stats: PatchStats,
                              wdot: str = "split", layer_hook=None, freeze_routing: bool = False,
                              start: int = 0,
                              routing_init: "dict[int, torch.Tensor] | None" = None):
    """Like `expert_op_patched`, but for the Triton path: `moe_ffn` becomes
    `moe_ffn_seq_triton` and `_expert_fns` returns `ExpertRef` (raw bytes)
    instead of calling `w.expert()` -- no `persist` patch needed: the Triton
    path never goes through the dequant cache (it does not dequantize at
    all).

    `freeze_routing`: if `True`, the first call of `moe_ffn` for each layer
    (identified by `il`, stashed here by `patched_expert_fns`) captures
    `selected_experts` (`[T,n_expert_used]`, a FIXED shape independent of its
    values); a second call for the SAME layer (the recompute of
    `torch.utils.checkpoint`, only observable with checkpointing active)
    replaces the incoming argument with the captured one -- it does not
    compare, it REPLACES: `moe_ffn_seq_triton` will group tokens by expert
    exactly as at the first call, same shape by construction, so the step's
    gradient is taken at the routing frozen at that point (refreshed between
    steps, never within one). `stats.routing_frozen_flips` counts, per
    layer, the positions whose live selection at the second call would have
    differed from the frozen one (a direct measurement of the engine's
    non-determinism on this device, not just a guard).

    `start`: same meaning as `expert_op_patched` (above).

    `routing_init`: if given, the `selected_experts` substitution happens
    EVEN WITHOUT `freeze_routing`, already at the FIRST call per layer -- the
    locked routing in `routing_init[il]` is imposed and the live mismatch is
    recorded in `stats.routing_flips_vs_init[il]`. Preconditions (raised as
    `RuntimeError` with the offending values): every layer that calls
    `moe_ffn` must have an entry in `routing_init` (no exception for the last
    layer); `routing_init[il].shape[0]` must match the current forward's
    `T`; `current_il["il"]` must not be `None`. With `routing_init=None`
    (default), no new behavior."""
    had_ple = "ple_true_emb" in replica.__dict__
    orig_ple = replica.__dict__.get("ple_true_emb")
    had_expert_fns = "_expert_fns" in replica.__dict__
    orig_expert_fns_attr = replica.__dict__.get("_expert_fns")
    orig_moe_ffn = model_mod.moe_ffn
    current_il = {"il": None}
    captured_routing: dict[int, torch.Tensor] = {}
    seen_layers: set[int] = set()

    def patched_ple_true_emb(tokens_arg, t, overlay=None):
        stats.n_ple_calls += 1
        return precomputed_emb[t - start].unsqueeze(0)

    def patched_expert_fns(il, persist, device=None, dtype=None):
        current_il["il"] = il
        if layer_hook is not None:
            layer_hook(il)
        return _triton_expert_fns(replica, w, il)

    def counted_moe_ffn(*a, **kw):
        stats.n_moe_calls += 1
        if freeze_routing or routing_init is not None:
            a = list(a)
            _apply_routing_patch(a, current_il["il"], captured_routing, seen_layers, stats, routing_init)
        return moe_ffn_seq_triton(*a, wdot=wdot, **kw)

    replica.ple_true_emb = patched_ple_true_emb
    replica._expert_fns = patched_expert_fns
    model_mod.moe_ffn = counted_moe_ffn
    try:
        yield
    finally:
        model_mod.moe_ffn = orig_moe_ffn
        if had_ple:
            replica.__dict__["ple_true_emb"] = orig_ple
        else:
            del replica.__dict__["ple_true_emb"]
        if had_expert_fns:
            replica.__dict__["_expert_fns"] = orig_expert_fns_attr
        else:
            del replica.__dict__["_expert_fns"]


def seq_forward_triton(replica, w, tokens: list[int], rows_var: torch.Tensor, row_map: dict[int, int],
                        return_logits: bool = True, wdot: str = "split", layer_hook=None,
                        grad_proxy: bool = True,
                        routing_source: dict[int, np.ndarray] | None = None,
                        capture_routing: dict[int, np.ndarray] | None = None,
                        layer_capture: dict | None = None,
                        freeze_routing: bool = False,
                        base_state: "model_mod.PrefixState | None" = None,
                        cache: "model_mod.PrefixCache | None" = None,
                        positions: torch.Tensor | None = None,
                        segment_ids: torch.Tensor | None = None,
                        delta_chunk_size: int = 64,
                        delta_batched: bool = True,
                        overlay_cache: dict | None = None,
                        routing_init: "dict[int, torch.Tensor] | None" = None):
    """Like `seq_forward`, but on the Triton path: `expert_op_patched_triton`
    in place of `expert_op_patched`. `w`: the same `DeviceIQ4` that built
    `replica` -- passed explicitly (not re-read from `replica.w`), for the
    same reason as `_triton_expert_fns`.

    `grad_proxy`: a forward-only exactness check (`.backward()` never
    called) has no reason to go through `torch.enable_grad()`: with
    `grad_proxy=True` (the default) `prefix()` makes `emb_new` a
    differentiable leaf and retains the whole per-layer graph even if no one
    ever uses it -- costly, and unnecessary for a check that never
    backpropagates. A caller that does not call `.backward()` must pass
    `grad_proxy=False` explicitly.

    `routing_source`/`capture_routing`/`layer_capture`: same meaning as
    `seq_forward` (above) -- forwarded to `Replica.prefix()` unchanged;
    `layer_capture` wraps the call in `capture_layers`.

    `freeze_routing`: forwarded unchanged to `expert_op_patched_triton` --
    see its docstring (freezes `selected_experts` between the first and a
    SECOND call of `moe_ffn` for the same layer, never observable here
    without `torch.utils.checkpoint`: this function calls `Replica.prefix()`
    exactly once). Default `False`: no new behavior for callers that do not
    use it.

    `base_state`/`cache`/`positions`/`segment_ids`: same meaning as
    `seq_forward` (above, including the caution about `cache`'s restrictive
    key with an always-empty overlay) -- forwarded to
    `Replica.prefix()`/`pack.packed_forward`. Not runnable in a CPU-only
    session (no CUDA/ROCm device): the only proof available here is that the
    signature accepts and forwards the parameters, not numerical exactness on
    the Triton path (that is left to a real device).

    `delta_batched`/`overlay_cache`: same meaning as `seq_forward` (above) --
    forwarded to `pack.packed_forward`/`OverlayEmb`.

    `routing_init`: forwarded to `expert_op_patched_triton` -- see its
    docstring (same note about `capture_routing`/`routing_flips_vs_init` as
    `seq_forward`)."""
    n_prefix = len(tokens) - 1
    if n_prefix <= 0:
        raise ValueError("seq_forward_triton: need at least 2 tokens")
    _check_no_segment_ids_with_incremental_prefix("seq_forward_triton", segment_ids, base_state, cache)

    start = 0
    base: model_mod.PrefixState | None = None
    if cache is not None:
        hit = cache.lookup(tokens[:n_prefix], {})
        if hit is not None:
            base, _ = hit
            start = base.n_prefix
    if base is None and base_state is not None:
        base = base_state
        start = base_state.n_prefix

    overlay_emb = OverlayEmb(replica.table, tokens, rows_var, row_map, segment_ids=segment_ids,
                            overlay_cache=overlay_cache)
    precomputed = overlay_emb.build(n_prefix, start=start)

    stats = PatchStats()
    stats.n_key_hits = overlay_emb.n_key_hits
    stats.n_id_hits = overlay_emb.n_id_hits
    cap_cm = capture_layers(replica, layer_capture) if layer_capture is not None else _null_context()
    pack_cm = PACK.packed_forward(replica, segment_ids, chunk=delta_chunk_size, batched_segments=delta_batched)
    with expert_op_patched_triton(replica, w, precomputed, stats, wdot=wdot, layer_hook=layer_hook,
                                   freeze_routing=freeze_routing,
                                   start=start, routing_init=routing_init), cap_cm, pack_cm:
        state = replica.prefix(tokens, grad_proxy=grad_proxy, return_logits=return_logits,
                                routing_source=routing_source, capture_routing=capture_routing,
                                base_state=base_state, cache=cache, positions=positions)
    return state, stats


# ==========================================================================
# Per-layer checkpointing and the fused forward+backward training step
# ==========================================================================


@contextmanager
def run_layer_checkpointed(replica):
    """Replaces, only for the duration of the `with` block, `replica.run_layer`
    (same shadowing pattern as `_expert_fns`/`ple_true_emb`) with a version
    that wraps the whole call in `torch.utils.checkpoint.checkpoint(...,
    use_reentrant=False)`: the layer's intermediate activations (including
    the bf16 tensors produced by the Triton kernels for every expert the
    sequence activates) are not retained by the graph, they are fully
    recomputed in the backward.

    A finer-grained checkpoint (only the MoE region, or only the non-MoE
    region, per layer) is a memory-tuning variant of this same idea that
    requires extra parameters on `Replica.run_layer` beyond what this
    reference implementation's `model.py` exposes -- not part of this public
    reference; only the whole-layer form is provided here.

    Checkpoints only the tensor `x`: within one call of `Replica.prefix()`
    without `base_state`/`cache` (the only way `seq_step_triton` uses it),
    `layers[il]` always starts from an empty `LayerState()` by construction
    (never passed from one layer to another), so the state/routing returned
    by `run_layer` are never read back by the caller on this path
    (`capture_routing`/`diag` are always `None` here) -- they can be
    recomputed/discarded without loss. The asserts below make the constraint
    explicit instead of leaving it implicit: an incremental `state` or an
    external `routing_source` with this patch active would silently produce
    a wrong gradient."""
    had = "run_layer" in replica.__dict__
    orig_attr = replica.__dict__.get("run_layer")
    orig_bound = replica.run_layer  # class-bound method (or a previous patch)

    def patched(il, x, positions, state, routing_source, ple_emb, need_ffn_output=True,
                persist_experts=False, diag=None):
        if state.attn is not None or state.delta is not None or state.ple_hist is not None:
            raise RuntimeError(
                "run_layer_checkpointed: requires an empty per-layer state "
                f"(state.attn/delta/ple_hist all None) -- layer {il} has non-empty state: "
                "this patch discards the returned state, an incremental forward "
                "(base_state/cache) would silently lose it"
            )
        if routing_source is not None:
            raise RuntimeError(
                "run_layer_checkpointed: external routing_source not supported "
                "(freezing routing between forward and recompute happens at the "
                "moe_ffn boundary -- expert_op_patched_triton -- not here)"
            )

        def fn(x_inner):
            new_x, _new_state, _routing = orig_bound(
                il, x_inner, positions, state, routing_source, ple_emb,
                need_ffn_output=need_ffn_output, persist_experts=persist_experts, diag=diag,
            )
            return new_x

        new_x = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
        return new_x, state, None

    replica.run_layer = patched
    try:
        yield
    finally:
        if had:
            replica.__dict__["run_layer"] = orig_attr
        else:
            del replica.__dict__["run_layer"]


def seq_step_triton(replica, w, tokens: list[int], rows_var: torch.Tensor, row_map: dict[int, int],
                     loss_fn, wdot: str = "split", checkpoint: bool = False, layer_hook=None,
                     base_state: "model_mod.PrefixState | None" = None,
                     cache: "model_mod.PrefixCache | None" = None,
                     positions: torch.Tensor | None = None,
                     segment_ids: torch.Tensor | None = None,
                     delta_batched: bool = True,
                     overlay_cache: dict | None = None,
                     routing_init: "dict[int, torch.Tensor] | None" = None,
                     *,
                     delta_chunk_size: int = 64):
    """Sequence forward+backward, Triton path, in a SINGLE `with` block:
    unlike `seq_forward_triton` (which returns the state and leaves the
    caller to build the loss and call `.backward()` AFTER leaving the
    context manager), here the patch (`expert_op_patched_triton`, and, if
    checkpointing is active, `run_layer_checkpointed`) stays active during
    `.backward()` too.

    `checkpoint` (bool, default `False`): enables the whole-layer
    checkpointing of `run_layer_checkpointed`. `freeze_routing` (toward
    `expert_op_patched_triton`) is turned on whenever checkpointing is
    active (the recompute, when it happens, must redo the same top-k) and
    off otherwise (`moe_ffn` is called exactly once per layer: no recompute
    to freeze, numerically identical either way, but leaving it off avoids
    needless cloning and keeps the default path bit-identical).

    Necessary with checkpointing active: `torch.utils.checkpoint` (with
    `use_reentrant=False`) does not keep the layer's graph, it recomputes it
    inside `.backward()` -- if the patch had already been restored by then
    (as happens if `.backward()` is called after `seq_forward_triton`
    returns), the recompute would invoke the ORIGINAL `moe_ffn`/`_expert_fns`
    (expecting dequantized tensors instead of `ExpertRef`) instead of the
    Triton path -- an error, or worse, a silently wrong number. Without
    checkpointing (`checkpoint=False`) this function is equivalent to
    `seq_forward_triton` + `loss_fn` + `.backward()` outside the `with` (the
    graph is already built entirely from `TritonExpertMatmul` nodes, whose
    backward does not need the patch) -- used anyway for uniform timing
    (`t_fwd`/`t_bwd` always measured at the same point).

    Returns `(loss, timings, stats)`: `timings = {"fwd_s", "bwd_s", "pre_s",
    "delta_pad_ratio", "delta_n_groups"}` (`torch.cuda.synchronize()` before/
    after each phase if the device is CUDA; `pre_s` is the time from entering
    this function to `t0`, i.e. building the overlay and entering the
    patches' context managers, BEFORE the forward); `stats.n_moe_calls` is
    expected to equal the number of FFN layers without checkpointing (MoE
    never recomputed), roughly double that with checkpointing (proof the
    recompute goes through the patch, not through the original `moe_ffn`).

    `base_state`/`cache`/`positions`/`segment_ids`: same meaning as
    `seq_forward_triton` (above) -- forwarded to
    `Replica.prefix()`/`pack.packed_forward`. `checkpoint=True` together with
    `base_state`/`cache` is not supported (`run_layer_checkpointed` requires
    an EMPTY per-layer state at every layer -- raises at runtime, never a
    silently wrong gradient). Not runnable in a CPU-only session (no CUDA/
    ROCm device).

    `delta_batched`/`overlay_cache`: same meaning as `seq_forward`/
    `seq_forward_triton` -- forwarded to `pack.packed_forward`/`OverlayEmb`.
    `overlay_cache`, if given, must live for the WHOLE run (a single dict
    passed at every step by the descent loop), never recreated at each call
    -- otherwise it would never hit (same principle as `row_map`, built
    once).

    `routing_init`: forwarded to `expert_op_patched_triton` -- see its
    docstring. With `checkpoint=True` AND `routing_init` given, the first
    call per layer imposes the locked routing (and records
    `routing_flips_vs_init`); the checkpoint RECOMPUTE (second call) stays
    frozen against the imposed routing from the first call, not against the
    live one -- and counts in `routing_frozen_flips` as always (the two
    measures never accumulate on the same call)."""
    _t_entry = time.perf_counter()  # reference point for timings["pre_s"], below
    n_prefix = len(tokens) - 1
    if n_prefix <= 0:
        raise ValueError("seq_step_triton: need at least 2 tokens")
    _check_no_segment_ids_with_incremental_prefix("seq_step_triton", segment_ids, base_state, cache)

    device = torch.device(w.device)
    is_cuda = device.type == "cuda"

    start = 0
    base: model_mod.PrefixState | None = None
    if cache is not None:
        hit = cache.lookup(tokens[:n_prefix], {})
        if hit is not None:
            base, _ = hit
            start = base.n_prefix
    if base is None and base_state is not None:
        base = base_state
        start = base_state.n_prefix

    overlay_emb = OverlayEmb(replica.table, tokens, rows_var, row_map, segment_ids=segment_ids,
                            overlay_cache=overlay_cache)
    precomputed = overlay_emb.build(n_prefix, start=start)

    stats = PatchStats()
    stats.n_key_hits = overlay_emb.n_key_hits
    stats.n_id_hits = overlay_emb.n_id_hits
    freeze_routing = bool(checkpoint)
    ckpt_cm = run_layer_checkpointed(replica) if checkpoint else _null_context()
    pack_info: dict = {}
    pack_cm = PACK.packed_forward(replica, segment_ids, chunk=delta_chunk_size, batched_segments=delta_batched,
                                  info=pack_info)
    with expert_op_patched_triton(replica, w, precomputed, stats, wdot=wdot, layer_hook=layer_hook,
                                   freeze_routing=freeze_routing,
                                   start=start, routing_init=routing_init), ckpt_cm, pack_cm:
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pre_s = t0 - _t_entry
        state = replica.prefix(tokens, grad_proxy=True, return_logits=True,
                                base_state=base_state, cache=cache, positions=positions)
        if is_cuda:
            torch.cuda.synchronize(device)
        t_fwd = time.perf_counter() - t0

        loss = loss_fn(state)

        t1 = time.perf_counter()
        with record_function("replica/backward"):
            loss.backward()
        if is_cuda:
            torch.cuda.synchronize(device)
        t_bwd = time.perf_counter() - t1

    assert stats.n_ple_calls == n_prefix - start, (
        f"seq_step_triton: n_ple_calls={stats.n_ple_calls} != n_prefix-start="
        f"{n_prefix - start} (the ple_true_emb patch did not take -- see PatchStats)"
    )
    return loss, {"fwd_s": t_fwd, "bwd_s": t_bwd, "pre_s": pre_s,
                  "delta_pad_ratio": pack_info.get("padding_ratio_vs_naive"),
                  "delta_n_groups": pack_info.get("n_groups")}, stats


@contextmanager
def _null_context():
    yield


def _check_no_segment_ids_with_incremental_prefix(
    fn_name: str, segment_ids, base_state, cache,
) -> None:
    """Precondition of `pack.packed_forward` ("does not compose with
    base_state/cache") checked HERE, right on function entry -- before even
    building the overlay embedding -- so incorrect use raises a clear
    `ValueError` instead of a deep `RuntimeError` inside the first attention/
    conv layer that touches a non-empty state (which `pack.packed_forward`
    raises anyway, as an independent guard: the two checks deliberately
    overlap)."""
    if segment_ids is not None and (base_state is not None or cache is not None):
        raise ValueError(
            f"{fn_name}: segment_ids is not composable with base_state/cache "
            "(precondition of pack.packed_forward)"
        )

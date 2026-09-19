"""Stage 0 (spec docs/foundation/ design and plans): teacher targets (single-fact
function `make_targets`, and corpus function `teacher_targets`), KD loss with tail
bucket support, and graft variables for corpus (`build_rows_var`).

Teacher targets (stage 0 tasks) and the evaluation loop run on fake forward (CPU, no
GPU/Triton). The only part outside scope here is the correctness of the real Triton
path. The descent loop is in a separate module.

Fixed F32 isolation: every reduction here (`log_softmax`, `logsumexp`) is in F32 (`logits.
to(torch.float32)`), matching the rest of the replica -- teacher targets are already
F32 when loaded from `.npz` (`np.float32`, `TeacherTargets`).
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import torch

from engraft.lens import RowSet
from engraft.table import ROW_LEN


# --------------------------------------------------------------------------
# Teacher targets (format .npz) and KD loss with tail bucket
# --------------------------------------------------------------------------


@dataclasses.dataclass
class TeacherTargets:
    """Teacher targets for a corpus: for each position, the ids and log-probabilities
    of the top-k tokens (sorted by log-probability descending) and the log-probability
    of the tail mass (tokens OUTSIDE the top-k). `frag_id`/`pos` identify the position
    in the corpus (fragment index, position index within the fragment) -- read by the
    caller to align targets to the student's logits during descent.

    `ids`: `[N,k]` int64. `logp`: `[N,k]` float32. `tail_logp`: `[N]` float32.
    `frag_id`/`pos`: `[N]` (integers, dtype compatible with `np.asarray`).
    """

    ids: np.ndarray
    logp: np.ndarray
    tail_logp: np.ndarray
    frag_id: np.ndarray
    pos: np.ndarray

    def save(self, path) -> None:
        """Uncompressed `.npz` (`np.savez`, never `savez_compressed`): values come back
        bit-identical on read, no compression approximation to justify."""
        np.savez(
            path, ids=self.ids, logp=self.logp, tail_logp=self.tail_logp,
            frag_id=self.frag_id, pos=self.pos,
        )

    @classmethod
    def load(cls, path) -> "TeacherTargets":
        with np.load(path) as z:
            return cls(
                ids=z["ids"], logp=z["logp"], tail_logp=z["tail_logp"],
                frag_id=z["frag_id"], pos=z["pos"],
            )


def subset_targets(targets: "TeacherTargets", keep_idx: list[int]) -> "TeacherTargets":
    """Capacity curve, branch I: re-indexes teacher targets for a SUBSET of `fragments_all`
    -- `targets.frag_id` is an index in the FULL list used to generate `targets`; when the
    caller filters `fragments_all` to `[fragments_all[i] for i in keep_idx]` (for a single
    fact), the same targets must be filtered to only rows with `frag_id in keep_idx` AND
    their `frag_id` must be REMAPPED to the new index (`keep_idx.index(old)`, i.e., the
    position of `old` in the order of `keep_idx` -- the same order the caller uses to build
    the filtered list).

    `keep_idx` may have duplicates or be unordered in principle, but the caller (descent
    loop) always builds it as `[i for i, f in enumerate(fragments_all) if ...]`, so
    monotonic and without duplicates -- not enforced here (no extra assumptions on the
    caller; the remap works for any list of distinct indices)."""
    remap = {int(old): new for new, old in enumerate(keep_idx)}
    mask = np.array([int(fid) in remap for fid in targets.frag_id], dtype=bool)
    new_frag_id = np.array(
        [remap[int(fid)] for fid in targets.frag_id[mask]], dtype=targets.frag_id.dtype,
    )
    return TeacherTargets(
        ids=targets.ids[mask], logp=targets.logp[mask], tail_logp=targets.tail_logp[mask],
        frag_id=new_frag_id, pos=targets.pos[mask],
    )


# --------------------------------------------------------------------------
# Locked routing (LR) -- capture, format .npz, and resolution per fragment.
# Locked routing is a property of the corpus, the base model, and the compute
# configuration it was captured with (`dense_dtype`/`moe_kernel`/`wdot`): routing
# depends on dtype and kernel -- saved in the file, verified by the consumer.
# --------------------------------------------------------------------------


def save_routing_base(path, routing_by_frag: "dict[str, dict[int, np.ndarray]]",
                       dense_dtype: str, moe_kernel: str, wdot: str) -> dict:
    """Writes routing base (used to measure damage with frozen routing): flattens
    `routing_by_frag` (`{frag_key: {layer_id: ndarray [n_frag, n_expert_used]}}`,
    accumulated output from `teacher_targets(routing_out=...)`) into four aligned
    arrays per ROW -- `frag_key [N] str`, `pos [N] int64` (local position 0..n_frag-1),
    `layers [L] int64` (sorted union of all layers seen, typically all 48 MoE layers),
    `routing [N, L, k] int16` (`n_expert_used` small, `int16` sufficient and halves size
    vs `int64`). Missing layer for a position raises `ValueError`, never silent fill:
    the precondition "every layer that calls moe_ffn" must hold already here, not only
    on load.

    Returns `{"n_positions", "n_layers", "n_expert_used"}` (diagnostics for caller/log)."""
    if not routing_by_frag:
        raise ValueError("save_routing_base: routing_by_frag empty -- no positions to write")

    layers_set: set[int] = set()
    for per_layer in routing_by_frag.values():
        layers_set.update(int(il) for il in per_layer)
    layers = np.array(sorted(layers_set), dtype=np.int64)

    frag_key_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] = []
    routing_parts: list[np.ndarray] = []
    n_expert_used: int | None = None

    for frag_key, per_layer in routing_by_frag.items():
        missing = layers_set - set(int(il) for il in per_layer)
        if missing:
            raise ValueError(
                f"save_routing_base: fragment {frag_key!r} missing layers {sorted(missing)} "
                f"(expected layers: {sorted(layers_set)}) -- incomplete capture"
            )
        arrs = [np.asarray(per_layer[int(il)]) for il in layers.tolist()]
        n_frag = arrs[0].shape[0]
        for il, arr in zip(layers.tolist(), arrs):
            if arr.shape[0] != n_frag:
                raise ValueError(
                    f"save_routing_base: fragment {frag_key!r} layer {il} has {arr.shape[0]} "
                    f"rows, expected {n_frag} (mismatch with another layer)"
                )
            if n_expert_used is None:
                n_expert_used = arr.shape[1]
            elif arr.shape[1] != n_expert_used:
                raise ValueError(
                    f"save_routing_base: fragment {frag_key!r} layer {il} has n_expert_used="
                    f"{arr.shape[1]}, expected {n_expert_used}"
                )
        stacked = np.stack(arrs, axis=1)  # [n_frag, L, k]
        frag_key_parts.append(np.full(n_frag, str(frag_key), dtype=object))
        pos_parts.append(np.arange(n_frag, dtype=np.int64))
        routing_parts.append(stacked)

    frag_key_arr = np.concatenate(frag_key_parts, axis=0).astype(str)
    pos_arr = np.concatenate(pos_parts, axis=0)
    routing_arr = np.concatenate(routing_parts, axis=0).astype(np.int16)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, frag_key=frag_key_arr, pos=pos_arr, layers=layers, routing=routing_arr,
        dense_dtype=str(dense_dtype), moe_kernel=str(moe_kernel), wdot=str(wdot),
    )
    return {
        "n_positions": int(routing_arr.shape[0]), "n_layers": int(layers.shape[0]),
        "n_expert_used": int(n_expert_used) if n_expert_used is not None else 0,
    }


def load_routing_base(path, *, dense_dtype: str | None = None, moe_kernel: str | None = None,
                       wdot: str | None = None, mismatch_ok: bool = False,
                       ) -> "tuple[dict[str, np.ndarray], np.ndarray, dict]":
    """Reads `routing_base_<lang>.npz`: returns `(by_frag, layers, config)` -- `by_frag =
    {frag_key: ndarray [T, L, k]}` (int64, rows REORDERED by `pos` ascending -- the caller
    must not assume write order), `layers = ndarray [L] int64` (SAME order as L axis of
    each `by_frag[key]`), `config = {"dense_dtype", "moe_kernel", "wdot"}` (three strings
    saved by `save_routing_base`).

    **The key is the fragment's string `id`, never list index** -- fragments here are keyed
    by id, not index, since consumers may work with filtered lists.

    Configuration verification: if any of `dense_dtype`/`moe_kernel`/`wdot` is given AND
    differs from the saved value, raises `ValueError` with both values -- unless
    `mismatch_ok=True` (the caller CLI activates this with a flag and DECLARES the
    deviation in its report, never silently)."""
    with np.load(path, allow_pickle=False) as z:
        frag_key = z["frag_key"]
        pos = z["pos"]
        layers = z["layers"].astype(np.int64)
        routing = z["routing"]
        cfg = {
            "dense_dtype": str(z["dense_dtype"]), "moe_kernel": str(z["moe_kernel"]),
            "wdot": str(z["wdot"]),
        }

    requested = {"dense_dtype": dense_dtype, "moe_kernel": moe_kernel, "wdot": wdot}
    mismatches = {
        k: {"routing_base": cfg[k], "requested": v} for k, v in requested.items()
        if v is not None and str(v) != cfg[k]
    }
    if mismatches and not mismatch_ok:
        raise ValueError(
            f"load_routing_base: configuration of {path!r} differs from requested: "
            f"{mismatches} -- routing depends on dtype/kernel; pass mismatch_ok=True "
            "to proceed anyway, DECLARING the deviation in the report"
        )

    by_frag: dict[str, np.ndarray] = {}
    by_key_idx: dict[str, list[int]] = {}
    for i, fk in enumerate(frag_key.tolist()):
        by_key_idx.setdefault(str(fk), []).append(i)
    for fk, idxs in by_key_idx.items():
        idxs_sorted = sorted(idxs, key=lambda i: int(pos[i]))
        expected = list(range(len(idxs_sorted)))
        actual = [int(pos[i]) for i in idxs_sorted]
        if actual != expected:
            raise ValueError(
                f"load_routing_base: non-contiguous/unordered positions for frag_key={fk!r}: "
                f"{actual} (expected 0..{len(idxs_sorted) - 1})"
            )
        by_frag[fk] = routing[idxs_sorted].astype(np.int64)  # [T, L, k]

    return by_frag, layers, cfg


def routing_array_for_fragment(by_frag: "dict[str, np.ndarray]", frag_id: str, n_expected: int) -> np.ndarray:
    """ndarray `[n_expected, L, k]` for ONE fragment: verifies `frag_id` and length,
    never silent truncation/fill. Base of `routing_init_for_fragment` (below, per-layer
    dict) and other consumers."""
    arr = by_frag.get(str(frag_id))
    if arr is None:
        raise ValueError(
            f"routing_array_for_fragment: no LR for frag_id={frag_id!r} "
            f"(available fragments in file: {len(by_frag)})"
        )
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"routing_array_for_fragment: LR for {frag_id!r} has T={arr.shape[0]}, "
            f"expected {n_expected} (len(tokens)-1 of requested fragment)"
        )
    return arr


def routing_init_for_fragment(by_frag: "dict[str, np.ndarray]", layers: np.ndarray,
                               frag_id: str, n_expected: int) -> "dict[int, np.ndarray]":
    """`{layer_id: ndarray [n_expected, k]}` for ONE fragment: used by forward and
    evaluation. `ValueError` if `frag_id` absent from LR or length mismatch."""
    arr = routing_array_for_fragment(by_frag, frag_id, n_expected)
    return {int(layers[j]): arr[:, j, :] for j in range(layers.shape[0])}


def log_prob_tail(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Log-probability of TAIL mass (vocabulary minus `ids`), computed as
    `logsumexp(logits with ids set to -inf) - logsumexp(logits)` -- **never**
    `log(1 - Σp_topk)` (catastrophic cancellation in F32 precisely in the regime where
    the student concentrates on top-k, when `Σp_topk -> 1`). `logits`: `[...,V]`;
    `ids`: `[...,k]` (int64, same batch dimensions as logits, one id per column -- must
    not contain duplicates on the same row; scatter would write multiple times to the
    same position, harmless here since the value is always the same scalar `-inf`, but
    a duplicate would indicate a bug upstream). Returns `[...]` (V axis removed).

    `logits.scatter(-1, ids, -inf)` (scalar VALUE, not a tensor source) is the
    gradient-safe form: backward of this scatter writes exact ZERO (`masked_fill`)
    to replaced positions, never a product `grad*0` that with local NaN gradient
    (softmax of all-inf row, edge case `ids` = full vocabulary, k=V) would give
    `0*NaN=NaN` -- verified empirically (no NaN gradient rows when k=V, see tests)."""
    logz = torch.logsumexp(logits, dim=-1)
    masked = logits.scatter(-1, ids, float("-inf"))
    return torch.logsumexp(masked, dim=-1) - logz


def make_targets(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher targets for a block of positions: `logits` `[T,V]` (or `[...,V]`) ->
    `(ids [...,k] int64, logp [...,k] float32, tail_logp [...] float32)`. `logits` is
    detached and converted to F32 before every reduction -- targets are always constants,
    never differentiable."""
    logits32 = logits.detach().to(torch.float32)
    logz = torch.logsumexp(logits32, dim=-1, keepdim=True)
    top = torch.topk(logits32, k, dim=-1)
    ids = top.indices.to(torch.int64)
    logp = top.values - logz
    tail_logp = log_prob_tail(logits32, ids)
    return ids, logp, tail_logp


def kd_loss(
    logits: torch.Tensor, ids: torch.Tensor, logp: torch.Tensor, tail_logp: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated KL with tail bucket: `logits` `[T,V]` of STUDENT (differentiable),
    `ids`/`logp`/`tail_logp` TEACHER targets (constants, typically from `TeacherTargets`/
    `make_targets`), `loss_mask` `[T]` (1.0 included, 0.0 excluded -- `loss_mask_for_fragment`
    constructs the canonical one).

    `logq_k = gather(log_softmax(logits), ids)`, `log q_tail` as `log_prob_tail` above
    (same guard against cancellation, applied HERE to the student: precisely the regime
    described, "cancellation where the student concentrates on top-k").
    `KL_t = Σ_k p_k(logp_k-logq_k) + p_tail(tail_logp-logq_tail)`. Tail term is guarded
    (`torch.where`) when `p_tail=exp(tail_logp)=0` (edge case k=V, empty tail:
    `tail_logp=-inf` EXACT, not very negative -- `tail_logp-logq_tail` becomes
    `-inf-(-inf)=NaN` without the guard, even though `p_tail` is zero: `0*NaN=NaN`,
    never simplified by float arithmetic). With `p_tail>0` and `logq_tail=-inf` (student
    assigns EXACTLY zero probability to tail while teacher doesn't) the term stays `+inf`:
    true divergence, not a bug -- the guard covers only `p_tail=0`, not that case.

    `loss = Σ_t mask_t·KL_t / Σ_t mask_t` (`Σmask` with small floor to avoid
    divide-by-zero if `loss_mask` is all null -- never observable for `Σmask>=1`,
    the normal case of a 0/1 mask). Also returns `per_position` `[T]` **detached**
    (for mass profile and logging), NEVER multiplied by `loss_mask` (it's raw KL
    per position; the caller applies their own selection)."""
    logits32 = logits.to(torch.float32)
    logz = torch.logsumexp(logits32, dim=-1, keepdim=True)
    logq_k = torch.gather(logits32, -1, ids) - logz
    logq_tail = log_prob_tail(logits32, ids)

    p_k = logp.exp()
    p_tail = tail_logp.exp()
    kl_topk = (p_k * (logp - logq_k)).sum(dim=-1)
    kl_tail_raw = p_tail * (tail_logp - logq_tail)
    kl_tail = torch.where(torch.isfinite(tail_logp), kl_tail_raw, torch.zeros_like(kl_tail_raw))

    per_position = kl_topk + kl_tail

    mask = loss_mask.to(per_position.dtype)
    denom = mask.sum().clamp_min(1e-8)
    loss = (mask * per_position).sum() / denom
    return loss, per_position.detach()


def lm_plus_base_loss(
    logits: torch.Tensor, targets_base: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    answer_mask: torch.Tensor, loss_mask: torch.Tensor, y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branch "LM + divergence from base": `-log q(y_t)` (forced teaching) on positions
    where `answer_mask` is true, KD from BASE targets (`targets_base = (ids, logp,
    tail_logp)`, the student WITHOUT graft/document) elsewhere -- same `kd_loss`,
    reused for the KD term (its `per_position`, NEVER its scalar `loss`: the branch
    selection happens here, not inside `kd_loss`). `y`: `[T]` int64, the true token at
    each position (used only where `answer_mask` is true).

    `loss = Σ_t mask_t·per_position_t / Σ_t mask_t`, same normalization as `kd_loss`.
    Also returns `per_position` detached."""
    logits32 = logits.to(torch.float32)
    logz = torch.logsumexp(logits32, dim=-1, keepdim=True)
    logq_y = torch.gather(logits32, -1, y.unsqueeze(-1)).squeeze(-1) - logz.squeeze(-1)
    nll = -logq_y

    ids, logp, tail_logp = targets_base
    _, kd_per_position = kd_loss(logits32, ids, logp, tail_logp, loss_mask)

    answer = answer_mask.to(torch.bool)
    per_position = torch.where(answer, nll, kd_per_position)

    mask = loss_mask.to(per_position.dtype)
    denom = mask.sum().clamp_min(1e-8)
    loss = (mask * per_position).sum() / denom
    return loss, per_position.detach()


def loss_mask_for_fragment(n: int, n_excl: int = 9, extra_excl: tuple[int, ...] = ()) -> torch.Tensor:
    """Mask `[n]` (1.0 included, 0.0 excluded, float32): the first `n_excl` positions
    are excluded, COUNTED FROM the EOS-including position (default 9 = (K-1)*PLE conv
    dilation, INITIAL value to read against `per_position` profile, not a guarantee --
    the model's attention is recurrent and doesn't exhaust in 9 positions), plus positions
    indicated by `extra_excl` (absolute indices in the fragment, e.g., separators). Out-of-range
    indices (`n_excl>n` or `extra_excl` value outside `[0,n)`) are tolerated (truncated/ignored),
    never an error: a mask shorter than the requested exclusions is simply all zero."""
    mask = torch.ones(n, dtype=torch.float32)
    mask[: min(n_excl, n)] = 0.0
    for idx in extra_excl:
        if 0 <= idx < n:
            mask[idx] = 0.0
    return mask


# --------------------------------------------------------------------------
# Graft variables for corpus
# --------------------------------------------------------------------------


def _read_true_rows(table, rows_global: list[int]) -> np.ndarray:
    """Reads TRUE (dequantized) rows for a list of global row indices, in any order
    -- same addressing algorithm as delta_map.global_to_head_local/read_original_rows
    (reimplemented here, not imported: this module remains independent from offline
    analysis of delta mappings, which run only on already-written overlays)."""
    offsets = np.asarray(table.head_offsets, dtype=np.int64)
    voc = np.asarray(table.head_vocab_sizes, dtype=np.int64)
    ends = offsets + voc
    out = np.empty((len(rows_global), ROW_LEN), dtype=np.float32)
    for i, g in enumerate(rows_global):
        g = int(g)
        h = int(np.searchsorted(offsets, g, side="right") - 1)
        h = max(0, min(h, len(offsets) - 1))
        if not (offsets[h] <= g < ends[h]):
            raise ValueError(f"_read_true_rows: global row {g} outside all heads")
        local = g - int(offsets[h])
        out[i] = table.read_rows(h, local, 1)[0]
    return out


_POLICIES = ("all-read", "private", "entity", "t8-answer")


def build_rows_var(
    table, sequences: list[list[int]], policy: str, row_sets: dict[str, list[int]],
    exclude_positions: dict[int, list[int]] | None = None,
) -> tuple[dict[int, int], torch.Tensor]:
    """Builds `(row_map, rows_var)` for a locality policy:

    | policy | variable rows |
    |---|---|
    | `all-read` | `row_sets["all_read"]` |
    | `private` | `row_sets["all_read"] - row_sets["neutral_read"]` |
    | `entity` | `row_sets["entity"] ∪ row_sets["terme"]` |
    | `t8-answer` | `row_sets["t8_answer"]` |

    `row_sets` is the dict of global row lists from census (`census_<lang>.json`, section
    `row_sets`). Rows read at positions in `exclude_positions` (`{sequence_index: [positions]}`,
    typically separators) are ALWAYS excluded from the set, regardless of policy -- all 16
    heads at the position, not just T8 (a separator never carries a fact; freezing it entirely
    is correct by construction).

    Returns `row_map` (`{global_row: index_in_rows_var}`, deterministic order = ascending
    global row) and `rows_var` (`[R,160]` float32, TRUE values read from table -- optimization
    starting point, NEVER a placeholder). `R=0` (no rows selected) is valid: `rows_var`
    returns `[0,160]`."""
    if policy not in _POLICIES:
        raise ValueError(f"build_rows_var: unknown policy {policy!r} (expected: {_POLICIES})")

    def rows_of(key: str) -> set[int]:
        return {int(r) for r in row_sets[key]}

    if policy == "all-read":
        candidate = rows_of("all_read")
    elif policy == "private":
        candidate = rows_of("all_read") - rows_of("neutral_read")
    elif policy == "entity":
        candidate = rows_of("entity") | rows_of("terme")
    else:  # "t8-answer"
        candidate = rows_of("t8_answer")

    excluded: set[int] = set()
    for seq_idx, positions in (exclude_positions or {}).items():
        tokens = sequences[seq_idx]
        for p in positions:
            rs = RowSet.from_position(table, tokens, p)
            excluded.update(int(g) for g in rs.rows_global)

    selected = sorted(candidate - excluded)
    row_map = {g: i for i, g in enumerate(selected)}
    if not selected:
        return row_map, torch.zeros(0, ROW_LEN, dtype=torch.float32)

    data = _read_true_rows(table, selected)
    rows_var = torch.from_numpy(data.copy())
    return row_map, rows_var


# --------------------------------------------------------------------------
# Teacher: `doc_prefix_state`/`teacher_targets`.
# --------------------------------------------------------------------------


def fake_seq_forward(replica, w, tokens: list[int], rows_var: torch.Tensor, row_map: dict[int, int],
                      return_logits: bool = True, grad_proxy: bool = True,
                      base_state=None, cache=None, positions=None, segment_ids=None,
                      capture_routing: "dict[int, np.ndarray] | None" = None,
                      routing_init: "dict[int, torch.Tensor] | None" = None,
                      routing_source: "dict[int, np.ndarray] | None" = None,
                      delta_batched: bool = True, overlay_cache: dict | None = None,
                      delta_chunk_size: int = 64, **_ignored):
    """Fake adapter of `forward_fn` for `doc_prefix_state`/`teacher_targets` and the
    descent loop: same positional signature as `seq.seq_forward_triton` (real engine)
    but forwards to `seq.seq_forward` (F32 CPU path, no Triton) discarding `w` and
    every Triton-only argument (`wdot`, `freeze_routing`, ...) passed for call uniformity
    -- same pattern as the fake step adapter for the sequence forward.

    `capture_routing`/`routing_init`: explicit parameters, forwarded UNCHANGED to
    `seq.seq_forward` (already public there, never touched). `routing_source`: the
    parameter that truly forces locked routing in the expert operator. `delta_batched`/
    `overlay_cache`/`delta_chunk_size`: explicit, forwarded UNCHANGED to
    `seq.seq_forward` (never absorbed in `**_ignored`, so the on/on vs off/off
    comparison of `--fake` is the same path twice)."""
    from engraft.replica.seq import seq_forward
    return seq_forward(
        replica, tokens, rows_var, row_map, return_logits=return_logits, grad_proxy=grad_proxy,
        base_state=base_state, cache=cache, positions=positions, segment_ids=segment_ids,
        capture_routing=capture_routing, routing_init=routing_init, routing_source=routing_source,
        delta_batched=delta_batched, overlay_cache=overlay_cache, delta_chunk_size=delta_chunk_size,
    )


def compute_doc_hash(doc_tokens) -> str:
    """`sha256` hex of the document's tokens (int64, given order) -- identifies the
    document on which a PrefixState was built (spec: `prefix()` does not verify that
    `base_state` matches the tokens; verification lives here, in the caller). Encoding
    declared: `np.asarray(doc_tokens, dtype=np.int64).tobytes()` (platform byte order,
    stable within ONE process/run -- hash is not for cross-machine comparison with
    different endianness, never the case here)."""
    arr = np.asarray(list(doc_tokens), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def doc_prefix_state(replica, w, doc_tokens: list[int], eos: int, forward_fn) -> tuple[object, str]:
    """`(state, doc_hash)`: `forward_fn(replica, w, doc_tokens+[eos], rows_var=[0,160]
    empty, row_map={}, return_logits=False, grad_proxy=False)` -- once, in RAM. `forward_fn`
    is `seq.seq_forward_triton` (real engine) or `fake_seq_forward` (above, on fakes)
    -- same positional signature for both.

    Resulting state has `state.n_prefix == len(doc_tokens)` (the final `eos` is ONLY
    the label of the document's last position, never processed -- same convention as
    `pack.Packed.tokens`): a caller extending with `tokens = doc_tokens + frag_tokens`
    (`frag_tokens` already in canonical form `[EOS]+frag`) and this `state` as
    `base_state` gets `start = len(doc_tokens)`, whose first new position IS the fragment's
    EOS position (the document's final EOS and fragment's opening EOS are the SAME token/
    position; no EOS duplication).

    `doc_hash`: `compute_doc_hash(doc_tokens)` -- the caller reusing this `state` for
    multiple fragments (`teacher_targets`, below) verifies it at every reuse."""
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)
    tokens = list(doc_tokens) + [int(eos)]
    state, _stats = forward_fn(
        replica, w, tokens, empty_rows, {}, return_logits=False, grad_proxy=False,
    )
    return state, compute_doc_hash(doc_tokens)


def teacher_targets(
    replica, w, doc_tokens: list[int], fragments: list[dict], k: int, forward_fn, eos: int,
    sample_full: int = 0, doc_state=None, doc_hash: str | None = None, seed: int = 0,
    routing_out: "dict[str, dict[int, np.ndarray]] | None" = None,
) -> tuple[TeacherTargets, dict | None]:
    """Teacher targets for an entire corpus: for each fragment in `fragments` (dict with
    `tokens` field already in canonical form `[EOS]+frag`), one forward -- with non-empty
    `doc_tokens`, `forward_fn(..., base_state=doc_state)` extends the document prefix
    (computed once above); with `doc_tokens=[]` (default, empty corpus) each fragment is
    forwarded ISOLATED (`base_state=None`) -- **the SAME targets as BASE** (same functions
    with `doc_tokens=[]` produce base targets, used by the "LM + divergence from base" branch
    and the floor). `rows_var`/`row_map` are ALWAYS empty (the teacher reads TRUE rows from
    the table, never an overlay).

    Each fragment's `state.logits` has `len(frag_tokens)-1` rows (the fragment's last token
    is ALWAYS only the label of the previous position, never processed; row 0 = position of
    the EOS predicting `frag[0]`) -- verified here (`RuntimeError` if mismatch; no silent
    misalignment). `frag_id` in output is the INDEX of `fragments` (given order, stable: the
    descent loop caller must pass the SAME list, same order, used to build `row_map`/batches,
    so `frag_id` stays valid in both places); `pos` is local position 0..n_frag-1.

    `doc_hash` (runtime verification): with `doc_state` given by the caller (explicit reuse,
    e.g., a loop generating targets for multiple corpora from one document), `doc_hash` must
    match `compute_doc_hash(doc_tokens)` -- else `ValueError` (the `base_state` doesn't match
    the document passed). With `doc_state=None` (default, the common case "run once per corpus"),
    `doc_prefix_state` is called here internally and verification is always true by construction
    -- the parameter exists for testing and for a caller wanting to compute `doc_state` once
    and pass it to multiple `teacher_targets` calls.

    Truncation bias (with `sample_full > 0` AND `doc_tokens` non-empty; the comparison only makes
    sense for CONDITIONED teacher -- with `doc_tokens=[]` teacher and "base student" would be
    IDENTICAL, bias trivially negative by construction, never the case described: gate declared,
    not silent assumption): a sample of `sample_full` positions (seed `seed`, `np.random.default_rng`,
    without replacement) receives a second ISOLATED forward of their fragment (`base_state=None`)
    -- logits from that second forward, at the same local positions, ARE the base student (true
    rows, same meaning as `doc_tokens=[]` above). Per sampled position: `KL_full = Σ_v
    p_teacher(v)(logp_teacher(v) - logq_base(v))` (F32 direct, no truncation) and `KL_truncated
    = kd_loss(...)` with targets truncated to `k` at THAT position against the same `q_base` --
    `bias = KL_full - KL_truncated` (a lower bound of full KL, so `bias >= 0` always). `bias_report`:
    `{"n_sample", "k", "mean_bias", "max_bias", "min_bias"}`, or `None` if the gate above
    is not satisfied.

    `routing_out` (dict or `None`): with EMPTY `doc_tokens` (BASE run), each isolated fragment
    forward passes `capture_routing={}` -- populated by `Replica.prefix()` with live routing
    at each layer -- and the result `{layer_id: ndarray [n_frag, n_expert_used]}` is written
    to `routing_out[frag["id"]]`. With non-empty `doc_tokens` (CONDITIONED teacher), capture
    is not offered -- `ValueError` if `routing_out` is given together with non-empty `doc_tokens`
    (the conditioned teacher always has free routing)."""
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)
    have_doc = bool(doc_tokens)

    if routing_out is not None and have_doc:
        raise ValueError(
            "teacher_targets: routing_out given with non-empty doc_tokens -- LR capture "
            "is offered ONLY for the BASE run (doc_tokens=[]), conditioned teacher stays "
            "with free routing"
        )

    if have_doc:
        if doc_state is None:
            doc_state, doc_hash = doc_prefix_state(replica, w, doc_tokens, eos, forward_fn)
        expected_hash = compute_doc_hash(doc_tokens)
        if doc_hash != expected_hash:
            raise ValueError(
                f"teacher_targets: doc_hash={doc_hash!r} does not match recalculated hash "
                f"of doc_tokens ({expected_hash!r}) -- the given base_state does not match "
                "the document passed (spec: verify on every reuse)"
            )

    # `lengths[fi]` = number of LOGIT ROWS expected for fragment fi (= n_frag = len(frag_tokens)-1,
    # spec: the last token is only a label) -- used for bias sampling below, must match
    # `state.logits.shape[0]` verified below in the main loop.
    lengths = [len(frag["tokens"]) - 1 for frag in fragments]
    for fi, ln in enumerate(lengths):
        if ln < 1:
            raise ValueError(
                f"teacher_targets: fragment {fi} (id={fragments[fi].get('id')!r}) has "
                f"{ln + 1} tokens, needs at least 2 (canonical form [EOS]+fragment non-empty)"
            )
    total_positions = sum(lengths)

    sample_positions: dict[int, set[int]] = {}
    if sample_full > 0 and have_doc and total_positions > 0:
        n_sample = min(sample_full, total_positions)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(total_positions, size=n_sample, replace=False)
        boundaries: list[tuple[int, int]] = []
        offset = 0
        for ln in lengths:
            boundaries.append((offset, offset + ln))
            offset += ln
        for gi in (int(x) for x in chosen):
            for fi, (a, b) in enumerate(boundaries):
                if a <= gi < b:
                    sample_positions.setdefault(fi, set()).add(gi - a)
                    break

    ids_parts: list[np.ndarray] = []
    logp_parts: list[np.ndarray] = []
    tail_parts: list[np.ndarray] = []
    frag_id_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] = []
    biases: list[float] = []

    for fi, frag in enumerate(fragments):
        frag_tokens = list(frag["tokens"])
        if have_doc:
            full_tokens = list(doc_tokens) + frag_tokens
            state, _stats = forward_fn(
                replica, w, full_tokens, empty_rows, {}, return_logits=True, grad_proxy=False,
                base_state=doc_state,
            )
        else:
            routing_cap = {} if routing_out is not None else None
            state, _stats = forward_fn(
                replica, w, frag_tokens, empty_rows, {}, return_logits=True, grad_proxy=False,
                capture_routing=routing_cap,
            )
            if routing_out is not None:
                routing_out[frag["id"]] = routing_cap
        logits = state.logits
        n_frag = logits.shape[0]
        # n_frag = len(frag_tokens)-1 (spec: the last token is ALWAYS only the label of the
        # previous position, never processed -- same convention as pack.Packed.frag_slices,
        # "width = canonical fragment length - 1"): row 0 = position of the EOS (tokens[0])
        # predicting frag[0] (tokens[1]).
        if n_frag != len(frag_tokens) - 1:
            raise RuntimeError(
                f"teacher_targets: fragment {fi} -- state.logits has {n_frag} rows, "
                f"expected len(frag_tokens)-1={len(frag_tokens) - 1} (spec: "
                "'no slice'; the fragment's last token is only a label)"
            )
        ids_f, logp_f, tail_f = make_targets(logits, k)
        ids_parts.append(ids_f.detach().cpu().numpy())
        logp_parts.append(logp_f.detach().cpu().numpy())
        tail_parts.append(tail_f.detach().cpu().numpy())
        frag_id_parts.append(np.full(n_frag, fi, dtype=np.int64))
        pos_parts.append(np.arange(n_frag, dtype=np.int64))

        wanted = sample_positions.get(fi)
        if wanted:
            base_state_frag, _bstats = forward_fn(
                replica, w, frag_tokens, empty_rows, {}, return_logits=True, grad_proxy=False,
            )
            base_logits = base_state_frag.logits
            for local_p in sorted(wanted):
                teacher_row = logits[local_p].detach().to(torch.float32)
                q_row = base_logits[local_p].detach().to(torch.float32)
                logp_full = teacher_row - torch.logsumexp(teacher_row, dim=-1)
                logq_full = q_row - torch.logsumexp(q_row, dim=-1)
                kl_full = float((logp_full.exp() * (logp_full - logq_full)).sum().item())

                ids_row, logp_row, tail_row = make_targets(teacher_row.unsqueeze(0), k)
                _, per_pos = kd_loss(
                    q_row.unsqueeze(0), ids_row, logp_row, tail_row,
                    torch.ones(1, dtype=torch.float32, device=q_row.device),
                )
                kl_trunc = float(per_pos[0].item())
                biases.append(kl_full - kl_trunc)

    targets = TeacherTargets(
        ids=np.concatenate(ids_parts, axis=0) if ids_parts else np.zeros((0, k), dtype=np.int64),
        logp=np.concatenate(logp_parts, axis=0) if logp_parts else np.zeros((0, k), dtype=np.float32),
        tail_logp=np.concatenate(tail_parts, axis=0) if tail_parts else np.zeros((0,), dtype=np.float32),
        frag_id=np.concatenate(frag_id_parts, axis=0) if frag_id_parts else np.zeros((0,), dtype=np.int64),
        pos=np.concatenate(pos_parts, axis=0) if pos_parts else np.zeros((0,), dtype=np.int64),
    )

    bias_report = None
    if sample_full > 0 and have_doc:
        if biases:
            bias_report = {
                "n_sample": len(biases), "k": int(k),
                "mean_bias": float(np.mean(biases)),
                "max_bias": float(np.max(biases)),
                "min_bias": float(np.min(biases)),
            }
        else:
            bias_report = {"n_sample": 0, "k": int(k), "mean_bias": None, "max_bias": None, "min_bias": None}

    return targets, bias_report

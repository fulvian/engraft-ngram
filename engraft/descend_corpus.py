"""Corpus-level ENGRAFT descent: packed fragments (`engraft.replica.pack`),
three arms (`kd`, `lm-base`, `kd-base-only`), four locality policies
(`engraft.replica.distill.build_rows_var`), guards before and during the
loop.

Dry run (fake): `--fake` substitutes `seq_step_triton`/`seq_forward_triton`
with `fake_seq_step_packed`/`engraft.replica.distill.fake_seq_forward` (the
F32 CPU path, the same tiny replica as `engraft.testing.fake_full_weights`).

Real run: requires a real GGUF (weights + n-gram table) and a CUDA/ROCm
device with Triton installed for the one published kernel
(`engraft.replica.triton_experts.TritonExpertMatmul`) -- the module import
itself never requires Triton (see `engraft.replica.seq`), only actually
running against a real backend does.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from engraft.lens import read_pleo

from engraft.lens import write_pleo
from engraft.table import ROW_LEN
import engraft.replica.distill as D
from engraft.replica.pack import pack_fragments
from engraft.replica.seq import seq_forward, seq_step_triton, seq_forward_triton
from engraft.replica.corpus_metrics import add_counters, count_pack
from engraft.replica.plateau import (
    LOCAL_RATE_PERSISTENCE, LOCAL_RATE_PHI, LOCAL_RATE_WINDOW, LocalRateDetector,
    PlateauDetector,
)
from engraft.replica.overlay_sum import sum_overlays
from engraft.replica.regime import (
    EVAL_EVERY_FREE, INSTAB_DROP, INSTAB_GRAD_FACTOR, REENTRY_READINGS,
    STATE_FREE, STATE_LOCKED, STATE_REENTRY, STATE_STOP,
    SWITCH_NORM_GROWTH, SWITCH_PERSISTENCE, SWITCH_PHI, SWITCH_WINDOW, RegimeArbiter,
)

ARMS = ("kd", "lm-base", "kd-base-only")
POLICIES = ("all-read", "t8-answer", "answer-only", "b8-t8-answer")  # engraft.replica.distill.build_rows_var
PLATEAU_REL_IMPROVEMENT = 0.0  # a pass "improves" only with a strictly lower mean loss

# A regime value of "misto" (the private prototype's Italian name) is accepted
# as a deprecated alias for "mixed" -- never a silent third behavior.
ROUTING_REGIME_ALIASES = {"misto": "mixed", "bloccato": "locked"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_routing_regime(value: str) -> str:
    value = ROUTING_REGIME_ALIASES.get(value, value)
    if value not in ("locked", "mixed"):
        raise ValueError(f"routing_regime must be 'locked' or 'mixed' (or the alias 'misto'), got {value!r}")
    return value


# --------------------------------------------------------------------------
# Injectable fake step: CPU F32 path, same positional shape as seq_step_triton
# --------------------------------------------------------------------------


def fake_seq_step_packed(replica, w, tokens: list[int], rows_var: torch.Tensor, row_map: dict[int, int],
                          loss_fn, wdot: str = "split", checkpoint: bool = False,
                          positions: torch.Tensor | None = None, segment_ids: torch.Tensor | None = None,
                          delta_batched: bool = True, overlay_cache: dict | None = None,
                          routing_init: "dict[int, torch.Tensor] | None" = None,
                          *, delta_chunk_size: int = 64):
    """Fake equivalent of `seq_step_triton` for `--fake`/tests (F32 CPU path,
    `seq.seq_forward`, no Triton). `checkpoint=True` is not supported here
    (the F32 test path never needs it): raises `NotImplementedError` instead
    of silently ignoring it. `wdot` is accepted for a uniform call signature
    but unused (the CPU reference path has no Triton wdot precision knob)."""
    if checkpoint:
        raise NotImplementedError(
            "fake_seq_step_packed: checkpoint=True is not supported on the F32 test "
            "path (--fake) -- only checkpoint=False is runnable here"
        )
    t0 = time.perf_counter()
    state, stats = seq_forward(
        replica, tokens, rows_var, row_map, return_logits=True, grad_proxy=True,
        positions=positions, segment_ids=segment_ids, delta_batched=delta_batched,
        delta_chunk_size=delta_chunk_size, routing_init=routing_init, overlay_cache=overlay_cache,
    )
    t_fwd = time.perf_counter() - t0
    loss = loss_fn(state)
    t1 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t1
    return loss, {"fwd_s": t_fwd, "bwd_s": t_bwd}, stats


# --------------------------------------------------------------------------
# Packing with bin membership: pack.pack_fragments does not expose it --
# duplicated here ONLY for the binning (identical logic), never for building
# Packed itself (that stays in pack_fragments).
# --------------------------------------------------------------------------


def _greedy_bin_membership(frags: list[list[int]], eos: int, max_len: int) -> list[list[int]]:
    lengths = [len(f) + 1 for f in frags]  # canonical = [eos]+f, same formula as pack_fragments
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
    return bins


def pack_with_membership(fragments: list[dict], eos: int, max_len: int):
    """`(packed_list, membership)`: `membership[i]` = list of the original
    `fragments` dicts, in the SAME order as the segments of bin
    `packed_list[i]` (needed to align targets/masks per fragment --
    `pack.Packed` does not carry this reference). `fragments[j]["tokens"]` is
    already in canonical `[EOS]+frag` form: passed to `pack_fragments`
    without the leading EOS (`tokens[1:]`), which re-adds it."""
    frags_raw = [list(f["tokens"])[1:] for f in fragments]
    bins = _greedy_bin_membership(frags_raw, eos, max_len)
    packed_list = pack_fragments(frags_raw, eos, max_len)
    assert len(bins) == len(packed_list), (
        "pack_with_membership: duplicated binning diverges from pack_fragments -- a bug, "
        "never a silent approximation"
    )
    membership = [[fragments[i] for i in bin_idxs] for bin_idxs in bins]
    return packed_list, membership


# --------------------------------------------------------------------------
# Targets/mask/answer for a packed sequence
# --------------------------------------------------------------------------


def answer_rows(frag: dict) -> list[int]:
    """`frag["answer_spans"]` are TOKEN indices (0-based in `frag["tokens"]`),
    not logit rows: the row predicting the token at index `s` is `s-1`
    (`Replica.prefix`: row `r` predicts `tokens[r+1]`). `s=0` (the opening
    EOS) never has an associated row, excluded here."""
    return [int(s) - 1 for s in frag.get("answer_spans", []) if int(s) >= 1]


def index_targets(targets: "D.TeacherTargets") -> dict[int, dict[int, int]]:
    """`frag_id -> {pos: row index in targets.ids/logp/tail_logp}` (O(1)
    lookup, built once per run)."""
    idx: dict[int, dict[int, int]] = {}
    frag_id = targets.frag_id
    pos = targets.pos
    for i in range(frag_id.shape[0]):
        idx.setdefault(int(frag_id[i]), {})[int(pos[i])] = i
    return idx


def build_packed_targets(packed, frag_dicts: list[dict], frag_id_by_str: dict[str, int],
                          targets_idx: dict[int, dict[int, int]], targets: "D.TeacherTargets", k: int):
    """`(ids, logp, tail_logp)` `[n_prefix, k]`/`[n_prefix, k]`/`[n_prefix]`
    aligned to the packed sequence's prefix positions: for each fragment in
    the bin, its `frag_slices` rows are read from the `.npz` by
    `(frag_id, local pos)`. BOUNDARY positions (never in any `frag_slices`)
    get a placeholder -- the FIRST real row already filled in the bin (finite
    by construction, never NaN/Inf): `loss_mask` always excludes them, the
    placeholder only keeps `kd_loss`/`lm_plus_base_loss` finite over the
    whole `[T,V]`."""
    n_prefix = len(packed.tokens) - 1
    ids_full = np.zeros((n_prefix, k), dtype=np.int64)
    logp_full = np.zeros((n_prefix, k), dtype=np.float32)
    tail_full = np.zeros((n_prefix,), dtype=np.float32)
    filled = np.zeros(n_prefix, dtype=bool)

    for local_seg, frag in enumerate(frag_dicts):
        a, b = packed.frag_slices[local_seg]
        fid = frag_id_by_str[frag["id"]]
        pos_idx = targets_idx.get(fid)
        if pos_idx is None:
            raise KeyError(
                f"build_packed_targets: no target in .npz for frag_id={fid} "
                f"(id={frag['id']!r}) -- targets/fragments are misaligned"
            )
        for local_p in range(b - a):
            row = pos_idx.get(local_p)
            if row is None:
                raise KeyError(
                    f"build_packed_targets: no target for (frag_id={fid}, pos={local_p}) "
                    f"(id={frag['id']!r}) -- targets/fragments are misaligned"
                )
            ids_full[a + local_p] = targets.ids[row]
            logp_full[a + local_p] = targets.logp[row]
            tail_full[a + local_p] = targets.tail_logp[row]
            filled[a + local_p] = True

    if filled.any() and not filled.all():
        first = int(np.argmax(filled))
        ids_full[~filled] = ids_full[first]
        logp_full[~filled] = logp_full[first]
        tail_full[~filled] = tail_full[first]

    return torch.from_numpy(ids_full), torch.from_numpy(logp_full), torch.from_numpy(tail_full)


def stitch_routing_init(packed, frag_dicts: list[dict], rbr: "dict[str, np.ndarray]",
                         layers: np.ndarray, device) -> "dict[int, torch.Tensor]":
    """`{il: Tensor [T_pack, k]}` for packed batch `packed`: for each
    fragment in the bin, its `frag_slices` rows are read from the locked
    routing resolved for `frag["id"]`
    (`engraft.replica.distill.routing_array_for_fragment`). Every non-last
    segment's BOUNDARY row (exists in the array, never read by the loss)
    takes a copy of the PRECEDING row: it can never land at index 0
    (`pack.pack_fragments` always opens the first segment at `off=0`,
    already covered by its own `frag_slices`)."""
    n_prefix = len(packed.tokens) - 1
    n_layers = layers.shape[0]
    stitched: np.ndarray | None = None
    filled = np.zeros(n_prefix, dtype=bool)

    for local_seg, frag in enumerate(frag_dicts):
        a, b = packed.frag_slices[local_seg]
        arr = D.routing_array_for_fragment(rbr, frag["id"], b - a)  # [b-a, L, k]
        if stitched is None:
            stitched = np.zeros((n_prefix, n_layers, arr.shape[2]), dtype=np.int64)
        stitched[a:b] = arr
        filled[a:b] = True

    if stitched is None:
        raise ValueError("stitch_routing_init: packed batch with no fragments (frag_dicts empty)")

    for t in range(n_prefix):
        if filled[t]:
            continue
        if t == 0:
            raise RuntimeError(
                "stitch_routing_init: boundary row at position 0 -- no preceding row to copy "
                "(pack_fragments should guarantee that the first segment starts at index 0, "
                "always covered by frag_slices)"
            )
        stitched[t] = stitched[t - 1]

    return {
        int(layers[j]): torch.from_numpy(stitched[:, j, :]).to(device=device, dtype=torch.int64)
        for j in range(n_layers)
    }


def build_packed_loss_mask(packed, frag_dicts: list[dict], n_excl: int,
                            weight_fact_positions: float = 1.0,
                            fact_weights: "dict[str, float] | None" = None) -> torch.Tensor:
    """`[n_prefix]`: per fragment, `distill.loss_mask_for_fragment(n, n_excl)`
    -- already excludes the first `n_excl` positions (EOS included) and the
    boundary positions (never covered by any `frag_slices`, stay 0 by
    construction). `weight_fact_positions`: scales the weight of answer
    positions. `fact_weights` (mass balance, default `None` = current bit-
    for-bit behavior): with a dict, every segment is scaled by
    `fact_weights[frag["fact_ids"][0]]` (multiplicative, commutes with
    `weight_fact_positions`)."""
    n_prefix = len(packed.tokens) - 1
    mask = torch.zeros(n_prefix, dtype=torch.float32)
    for local_seg, frag in enumerate(frag_dicts):
        a, b = packed.frag_slices[local_seg]
        frag_mask = D.loss_mask_for_fragment(b - a, n_excl=n_excl)
        if weight_fact_positions != 1.0:
            for row in answer_rows(frag):
                if 0 <= row < (b - a):
                    frag_mask[row] = frag_mask[row] * weight_fact_positions
        if fact_weights is not None:
            frag_mask = frag_mask * fact_weights[frag["fact_ids"][0]]
        mask[a:b] = frag_mask
    return mask


def compute_fact_weights(train_frags: list[dict], n_excl: int) -> "tuple[dict[str, float], dict[str, float], dict]":
    """Per-fact weight from the mass admitted by the loss mask (fact
    balance): `M_f = sum` of the positions admitted by
    `distill.loss_mask_for_fragment(len(tokens), n_excl)` over every
    `train_frags` fragment with `fact_ids == [f]`.

    A fragment with missing, empty, or non-singleton `fact_ids` raises
    `ValueError` naming the fragment id. A fact at zero mass (every one of
    its train fragments has <= `n_excl` tokens) raises `ValueError` naming
    the fact id, BEFORE any downstream use (never a division by zero).

    `w_f = median(M) / M_f`, median over the facts present in `train_frags`.
    Returns `(weights, mass, stats)` with `stats = {n_facts, mass_min,
    mass_median, mass_max, w_min, w_max, n_frags_zero_mass}`."""
    mass: dict[str, float] = {}
    n_frags_zero_mass = 0
    for frag in train_frags:
        fact_ids = frag.get("fact_ids")
        if not fact_ids or len(fact_ids) != 1:
            raise ValueError(
                f"compute_fact_weights: fragment {frag.get('id')!r} has invalid fact_ids: {fact_ids!r}"
            )
        fid = fact_ids[0]
        frag_mass = float(D.loss_mask_for_fragment(len(frag["tokens"]), n_excl=n_excl).sum().item())
        if frag_mass == 0.0:
            n_frags_zero_mass += 1
        mass[fid] = mass.get(fid, 0.0) + frag_mass

    for fid, m in mass.items():
        if m == 0.0:
            raise ValueError(
                f"compute_fact_weights: fact {fid!r} at zero mass "
                f"(every train fragment has <= {n_excl} tokens)"
            )

    values = sorted(mass.values())
    n = len(values)
    mass_median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2.0
    weights = {fid: mass_median / m for fid, m in mass.items()}
    w_values = list(weights.values())
    stats = {
        "n_facts": n, "mass_min": min(values), "mass_median": mass_median, "mass_max": max(values),
        "w_min": min(w_values), "w_max": max(w_values), "n_frags_zero_mass": n_frags_zero_mass,
    }
    return weights, mass, stats


def build_packed_y_and_answer_mask(packed, frag_dicts: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """`(y, answer_mask)` `[n_prefix]` for the `lm-base` arm: `y[t]` = the
    true token predicted from position `t`; `answer_mask` = 1.0 at every
    fragment's `answer_rows(frag)`."""
    n_prefix = len(packed.tokens) - 1
    tokens_t = packed.tokens
    y = torch.tensor(tokens_t[1:n_prefix + 1], dtype=torch.int64)
    answer_mask = torch.zeros(n_prefix, dtype=torch.float32)
    for local_seg, frag in enumerate(frag_dicts):
        a, b = packed.frag_slices[local_seg]
        for row in answer_rows(frag):
            if 0 <= row < (b - a):
                answer_mask[a + row] = 1.0
    return y, answer_mask


def make_loss_fn(arm: str, ids_t=None, logp_t=None, tail_t=None, loss_mask=None, *,
                  y=None, answer_mask=None, base_ids=None, base_logp=None, base_tail=None):
    """Builds `loss_fn(state) -> loss` for the given arm (same `kd_loss`/
    `lm_plus_base_loss`, different targets):

    - `kd`: `ids_t/logp_t/tail_t` = the teacher's targets.
    - `kd-base-only` (negative control): the SAME call as `kd`, but
      `ids_t/logp_t/tail_t` are the base's targets -- no dedicated loss
      function, the difference is ONLY in the data the caller loads.
    - `lm-base`: `-log q(y_t)` at `answer_mask` positions, KD from the base's
      targets (`base_ids/base_logp/base_tail`) elsewhere."""
    if arm in ("kd", "kd-base-only"):
        def loss_fn(state):
            loss, per_position = D.kd_loss(state.logits, ids_t, logp_t, tail_t, loss_mask)
            loss_fn.per_position = per_position
            return loss
        return loss_fn
    if arm == "lm-base":
        def loss_fn(state):
            loss, per_position = D.lm_plus_base_loss(
                state.logits, (base_ids, base_logp, base_tail), answer_mask, loss_mask, y,
            )
            loss_fn.per_position = per_position
            return loss
        return loss_fn
    raise ValueError(f"make_loss_fn: unknown arm {arm!r} (expected: {ARMS})")


# --------------------------------------------------------------------------
# Heldout evaluation
# --------------------------------------------------------------------------


def eval_kd_on_packed(replica, w, heldout_packed: list, heldout_membership: list[list[dict]],
                       frag_id_by_str: dict[str, int], targets_idx: dict[int, dict[int, int]],
                       targets: "D.TeacherTargets", n_excl: int, k: int, forward_fn,
                       rows_var: torch.Tensor, row_map: dict[int, int],
                       routing_inits: "list[dict[int, torch.Tensor]] | None" = None,
                       delta_batched: bool = True, overlay_cache: dict | None = None) -> float | None:
    """Mean KD (weighted by admitted position count) on the fixed heldout
    packed batches, with the CURRENT `rows_var` (`.detach()`: no retained
    graph, never an Adam step here). `None` if there is no heldout fragment."""
    if not heldout_packed:
        return None
    dev = getattr(getattr(replica, "backend", None), "device", "cpu")
    total_loss = 0.0
    total_mask = 0.0
    for bi, (packed, frag_dicts) in enumerate(zip(heldout_packed, heldout_membership)):
        packed_dev = packed.to(dev)
        loss_mask = build_packed_loss_mask(packed, frag_dicts, n_excl).to(dev)
        ids_t, logp_t, tail_t = build_packed_targets(packed, frag_dicts, frag_id_by_str, targets_idx, targets, k)
        ids_t, logp_t, tail_t = ids_t.to(dev), logp_t.to(dev), tail_t.to(dev)
        routing_init = routing_inits[bi] if routing_inits is not None else None
        state, _stats = forward_fn(
            replica, w, packed_dev.tokens, rows_var.detach(), row_map, return_logits=True, grad_proxy=False,
            positions=packed_dev.positions, segment_ids=packed_dev.segment_ids, routing_init=routing_init,
            delta_batched=delta_batched, overlay_cache=overlay_cache,
        )
        loss, _per_pos = D.kd_loss(state.logits, ids_t, logp_t, tail_t, loss_mask)
        m = float(loss_mask.sum().item())
        total_loss += float(loss.item()) * m
        total_mask += m
    return total_loss / total_mask if total_mask > 0 else None


def eval_heldout_indicators(replica, w, heldout_packed: list, heldout_membership: list[list[dict]],
                             frag_id_by_str: dict[str, int], targets_idx: dict[int, dict[int, int]],
                             targets: "D.TeacherTargets", n_excl: int, k: int, forward_fn,
                             rows_var: torch.Tensor, row_map: dict[int, int],
                             routing_inits: "list[dict[int, torch.Tensor]] | None" = None,
                             delta_batched: bool = True, overlay_cache: dict | None = None,
                             want_routing_stats: bool = False) -> dict | None:
    """Same loop as `eval_kd_on_packed` (same `forward_fn`, same heldout bins
    -- ONE forward per bin, never an extra one), but from the same logits it
    also computes cheap token-level indicators at `answer_mask` positions
    (`y` is always the corpus's TRUE token, regardless of which arm is
    training).

    Returns `None` if there is no heldout bin. Otherwise a dict with
    `kd_heldout` (IDENTICAL to `eval_kd_on_packed`), `acc_heldout` (fraction
    of answer positions with `argmax(logits) == y`, weighted by position
    count), `margin_heldout` (mean of `logit[y] - max_{v!=y} logit[v]`),
    `margin_heldout_median` (median over the GLOBAL set of margins across
    every bin), and `n_answer_pos_heldout`.

    `want_routing_stats` (mixed-routing descent, default `False` -- return
    dict otherwise unchanged): with `True` and `routing_inits` given, the
    return dict gains `routing_flips_top1_total`, `routing_flips_top1_frac`
    and `routing_flips_vs_init_total`, read from the locked-routing
    forward's `_stats`."""
    if not heldout_packed:
        return None
    dev = getattr(getattr(replica, "backend", None), "device", "cpu")
    total_loss = 0.0
    total_mask = 0.0
    counters_validation = add_counters()
    n_correct = 0
    total_answer_pos = 0
    margin_sum = 0.0
    margin_values: list[float] = []
    routing_flips_top1_total = 0
    routing_flips_vs_init_total = 0
    routing_denominator_total = 0
    for bi, (packed, frag_dicts) in enumerate(zip(heldout_packed, heldout_membership)):
        packed_dev = packed.to(dev)
        loss_mask_cpu = build_packed_loss_mask(packed, frag_dicts, n_excl)
        loss_mask = loss_mask_cpu.to(dev)
        counters_validation = add_counters(
            counters_validation,
            count_pack(
                len(packed.tokens), len(frag_dicts), loss_mask_cpu.tolist(),
                [False] * len(loss_mask_cpu), answer_nll=False,
            ),
        )
        ids_t, logp_t, tail_t = build_packed_targets(packed, frag_dicts, frag_id_by_str, targets_idx, targets, k)
        ids_t, logp_t, tail_t = ids_t.to(dev), logp_t.to(dev), tail_t.to(dev)
        routing_init = routing_inits[bi] if routing_inits is not None else None
        state, _stats = forward_fn(
            replica, w, packed_dev.tokens, rows_var.detach(), row_map, return_logits=True, grad_proxy=False,
            positions=packed_dev.positions, segment_ids=packed_dev.segment_ids, routing_init=routing_init,
            delta_batched=delta_batched, overlay_cache=overlay_cache,
        )
        loss, _per_pos = D.kd_loss(state.logits, ids_t, logp_t, tail_t, loss_mask)
        m = float(loss_mask.sum().item())
        total_loss += float(loss.item()) * m
        total_mask += m

        if want_routing_stats and routing_init is not None:
            routing_flips_top1_total += sum(_stats.routing_flips_top1.values())
            routing_flips_vs_init_total += sum(_stats.routing_flips_vs_init.values())
            routing_denominator_total += len(packed_dev.tokens) * len(routing_init)

        y_true, answer_mask = build_packed_y_and_answer_mask(packed, frag_dicts)
        y_true, answer_mask = y_true.to(dev), answer_mask.to(dev)
        ans_idx = torch.nonzero(answer_mask > 0.5, as_tuple=True)[0]
        n_ans = int(ans_idx.numel())
        if n_ans == 0:
            continue
        logits_ans = state.logits[ans_idx].detach().to(torch.float32)  # [n_ans, V]
        y_ans = y_true[ans_idx]
        pred = logits_ans.argmax(dim=-1)
        n_correct += int((pred == y_ans).sum().item())
        logit_y = logits_ans.gather(1, y_ans.view(-1, 1)).squeeze(1)
        masked = logits_ans.clone()
        masked.scatter_(1, y_ans.view(-1, 1), float("-inf"))
        max_other = masked.max(dim=-1).values
        margin = (logit_y - max_other)
        margin_sum += float(margin.sum().item())
        margin_values.extend(float(v) for v in margin.tolist())
        total_answer_pos += n_ans

    kd_heldout = total_loss / total_mask if total_mask > 0 else None
    routing_stats_extra = {}
    if want_routing_stats and routing_inits is not None:
        routing_stats_extra = {
            "routing_flips_top1_total": routing_flips_top1_total,
            "routing_flips_vs_init_total": routing_flips_vs_init_total,
            "routing_flips_top1_frac": (
                routing_flips_top1_total / routing_denominator_total
                if routing_denominator_total > 0 else None
            ),
        }
    if total_answer_pos == 0:
        return {
            "kd_heldout": kd_heldout, "acc_heldout": None, "margin_heldout": None,
            "margin_heldout_median": None, "n_answer_pos_heldout": 0,
            "counters_validation": counters_validation,
            **routing_stats_extra,
        }
    return {
        "kd_heldout": kd_heldout,
        "acc_heldout": n_correct / total_answer_pos,
        "margin_heldout": margin_sum / total_answer_pos,
        "margin_heldout_median": float(np.median(margin_values)),
        "n_answer_pos_heldout": total_answer_pos,
        **routing_stats_extra,
        "counters_validation": counters_validation,
    }


def _cos_rows(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


class StepTimeGuard:
    """Per-step time guard: fires if `window` (default 20) CONSECUTIVE steps
    exceed `factor` (default 3) times the MEDIAN of the previous steps
    (median computed BEFORE adding the current step to the history --
    never self-referential). Never fires under 3 steps of history."""

    def __init__(self, window: int = 20, factor: float = 3.0):
        self.window = window
        self.factor = factor
        self._history: list[float] = []
        self._consecutive_over = 0

    def check(self, step_seconds: float) -> bool:
        triggered = False
        if len(self._history) >= 3:
            median = float(np.median(self._history))
            if median > 0.0 and step_seconds > self.factor * median:
                self._consecutive_over += 1
                if self._consecutive_over >= self.window:
                    triggered = True
            else:
                self._consecutive_over = 0
        self._history.append(step_seconds)
        return triggered


def check_device_memory_guard(device, memory_fraction: float, margin_bytes: int,
                               *, stat: str = "allocated") -> bool:
    """`True` if the guard FIRES (memory over the ceiling). No-op (`False`)
    without CUDA (a guard on `torch.cuda.memory_allocated` is meaningless on
    CPU -- the `--fake` path always runs on CPU).

    `stat` (default `"allocated"`): `"max_allocated"` uses
    `torch.cuda.max_memory_allocated` (the PEAK, which can exceed a step's
    snapshot because of transient forward/backward buffers already freed by
    the time this guard is checked) -- the caller checks it BEFORE the step
    with `"allocated"` (budget already exceeded before starting) and AFTER
    with `"max_allocated"` (the peak of the step just taken, reset by
    `torch.cuda.reset_peak_memory_stats` right before the step)."""
    if stat not in ("allocated", "max_allocated"):
        raise ValueError(f"check_device_memory_guard: stat must be 'allocated' or 'max_allocated', got {stat!r}")
    if not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(device).total_memory
    ceiling = memory_fraction * total - margin_bytes
    used = torch.cuda.max_memory_allocated(device) if stat == "max_allocated" else torch.cuda.memory_allocated(device)
    return used > ceiling


def set_oom_score_adj(value: int = 800, path: Path = Path("/proc/self/oom_score_adj")) -> bool:
    """Writes `value` to `path`: `True` on success, `False` (with a warning,
    NEVER an exception) if the file is not writable (permissions, sandbox,
    non-Linux)."""
    try:
        path.write_text(str(int(value)))
        return True
    except OSError as exc:
        log(f"set_oom_score_adj: {path} not writable ({exc}) -- a warning, not an error")
        return False


CONCURRENT_RUN_PATTERN = r"(^|[ /])engraft/(descend_corpus)\.py( |$)"


def check_no_concurrent_run() -> None:
    """`pgrep -f` for another instance of this script, EXCLUDING this
    process's own PID (the pattern is anchored to the script name itself, so
    it never matches the invocation of `pgrep` looking for it). Raises
    `RuntimeError` if another run is found -- never called from `--fake`."""
    r = subprocess.run(
        ["pgrep", "-f", CONCURRENT_RUN_PATTERN], stdout=subprocess.PIPE,
    )
    pids = [p for p in r.stdout.decode().split() if p and p != str(os.getpid())]
    if pids:
        raise RuntimeError(
            f"check_no_concurrent_run: another engraft.descend_corpus run is already "
            f"active (pid {pids}) -- never run two at once"
        )


# --------------------------------------------------------------------------
# Output: merged.pleo (variable rows + FROZEN all_read rows)
# --------------------------------------------------------------------------


def write_merged_pleo(table, row_map: dict[int, int], rows_var: torch.Tensor,
                       all_read_rows: list[int], out_path: Path) -> dict:
    """Writes `merged.pleo`: the VARIABLE rows (`row_map`/`rows_var`, current
    value) PLUS the `all_read_rows` non-variable rows, at their TRUE value
    (Delta=0 by construction) -- not only the variable ones (a downstream
    consumer that checks freezing needs every head represented and every
    frozen row present in the overlay).

    Returns `{"n_rows_variable", "n_rows_frozen", "frozen_rows"}`."""
    variable_set = set(row_map.keys())
    frozen_rows = sorted(set(int(r) for r in all_read_rows) - variable_set)

    rows_all = np.array(list(row_map.keys()) + frozen_rows, dtype=np.int32)
    data_var = rows_var.detach().to("cpu", torch.float32).numpy()
    if frozen_rows:
        data_frozen = D._read_true_rows(table, frozen_rows)
        data_all = np.concatenate([data_var, data_frozen], axis=0) if data_var.shape[0] else data_frozen
    else:
        data_all = data_var
    write_pleo(out_path, rows_all, data_all)
    return {"n_rows_variable": len(row_map), "n_rows_frozen": len(frozen_rows), "frozen_rows": frozen_rows}


def _dtype_manifest_fields(replica) -> dict:
    from engraft.replica.model import head_dtype_to_name
    return {
        "dense_dtype": replica.backend.dtypes.prefix,
        "head_dtype": head_dtype_to_name(replica.head_dtype),
        "head_out_dtype": head_dtype_to_name(replica.head_out_dtype),
    }


@contextmanager
def _null_context():
    yield


# --------------------------------------------------------------------------
# The descent loop
# --------------------------------------------------------------------------


def descend_corpus(
    replica, w, table, step_fn, forward_fn,
    fragments_all: list[dict], row_sets: dict[str, list[int]],
    policy: str, arm: str, targets: "D.TeacherTargets", out_dir: Path,
    *, k: int = 256, pack_len: int = 2048, n_excl: int = 9, patience: int = 10,
    max_steps: int | None = None, gpu_hours_cap: float | None = None,
    mu: float = 0.1, checkpoint: bool = False, seed: int = 0, eval_every: int = 20,
    stop_criterion: str = "kd_train", stop_window: int = 8, stop_confidence: float = 0.95,
    stop_local_window: int = LOCAL_RATE_WINDOW, stop_phi: float = LOCAL_RATE_PHI,
    stop_persistence: int = LOCAL_RATE_PERSISTENCE,
    wdot: str = "split", delta_chunk_size: int = 64,
    delta_batched: bool = True, overlay_cache: dict | None = None,
    rows_start_pleo: Path | None = None, step_offset: int = 0, snapshot_every: int = 0,
    weight_fact_positions: float = 1.0, fact_weight: str = "none",
    lr_scale: float = 1.0, time_guard: "StepTimeGuard | None" = None,
    device_guard_kwargs: dict | None = None,
    record_peak_memory: bool = False, memory_device: str | None = None,
    routing_base: "tuple[dict[str, np.ndarray], np.ndarray] | None" = None,
    max_passes: int | None = None, eval_every_passes: int | None = None,
    snapshot_every_passes: int | None = None,
    routing_regime: str = "locked",
    switch_norm_growth: float = SWITCH_NORM_GROWTH, switch_phi: float = SWITCH_PHI,
    switch_window: int = SWITCH_WINDOW, switch_persistence: int = SWITCH_PERSISTENCE,
    instab_drop: float = INSTAB_DROP, instab_grad_factor: float = INSTAB_GRAD_FACTOR,
    reentry_readings: int = REENTRY_READINGS, eval_every_free: int = EVAL_EVERY_FREE,
) -> dict:
    """ENGRAFT descent over a packed corpus: `fragments_all` must be the SAME
    list, in the SAME order, used to generate `targets`
    (`distill.teacher_targets`) -- `targets.frag_id` is an index into this
    list. `max_steps` is an ABSOLUTE ceiling on the step counter (useful with
    `step_offset`: resuming at `step_offset=N` with `max_steps=N+K` runs K
    new steps, never `N+K` from scratch). `stop_reason` is one of
    `{"plateau","max_steps","budget","guard_step_time","guard_device_memory",
    "plateau_acc_heldout","plateau_acc_heldout_rate", ...regime.STOP_REASON_*}`.

    `routing_base` (optional -- `(by_frag, layers)` from
    `distill.load_routing_base`): if given, ONE `routing_init` per `Packed`
    (`stitch_routing_init`, never the same `routing_init` reused across
    different packed batches) -- built once for the `heldout` bins (fixed for
    the whole run) and REBUILT every pass for the train bins (reshuffled and
    repacked every pass, below). The per-step record gains
    `routing_flips_vs_init` (total and per layer) and `n_moe_calls` -- ONLY
    with `routing_base` given.

    `routing_regime` (default `"locked"`, identical behavior to a plain
    routing-locked descent): `"mixed"` (alias `"misto"` resolved by the CLI)
    turns on the `RegimeArbiter` state machine (`engraft.replica.regime`) --
    requires `routing_base` given (the LOCKED/REENTRY phases need it) and
    `stop_criterion == "acc_heldout_rate"` (the free phase's exit criterion;
    the old detector is internal to the arbiter, `acc_plateau_detector`
    stays `None` in this regime). `switch_norm_growth`/`switch_phi`/
    `switch_window`/`switch_persistence`/`instab_drop`/`instab_grad_factor`/
    `reentry_readings`/`eval_every_free` are the arbiter's parameters, passed
    through as given; `stop_local_window`/`stop_phi`/`stop_persistence`
    parametrize the arbiter's INTERNAL exit detector (the SAME
    `LocalRateDetector` class, never a second instance outside the arbiter).

    `max_passes`/`eval_every_passes`/`snapshot_every_passes` (capacity-curve
    cadence in PASSES instead of steps, additive, default `None` = identical
    behavior): mutually exclusive with the homologous `max_steps`/
    `eval_every`/`snapshot_every` (checked by `main()`).

    Returns the manifest dict, and also writes it to
    `out_dir/merged_manifest.json`/`out_dir/summary.json`."""
    if policy not in POLICIES:
        raise ValueError(f"descend_corpus: unknown policy {policy!r} (expected: {POLICIES})")
    if arm not in ARMS:
        raise ValueError(f"descend_corpus: unknown arm {arm!r} (expected: {ARMS})")
    if fact_weight not in ("none", "mass"):
        raise ValueError(f"descend_corpus: unknown fact_weight {fact_weight!r} (expected: none, mass)")
    routing_regime = resolve_routing_regime(routing_regime)
    if routing_regime == "mixed":
        if routing_base is None:
            raise ValueError("descend_corpus: routing_regime='mixed' requires routing_base "
                              "(the LOCKED/REENTRY phases need it) -- CLI: --routing-base")
        if stop_criterion != "acc_heldout_rate":
            raise ValueError(
                f"descend_corpus: routing_regime='mixed' requires stop_criterion='acc_heldout_rate' "
                f"(the free phase's exit criterion), got {stop_criterion!r}"
            )

    train_frags = [f for f in fragments_all if f.get("split") == "train"]
    heldout_frags = [f for f in fragments_all if f.get("split") != "train"]
    if not train_frags:
        raise ValueError("descend_corpus: no fragment with split=='train'")

    # Fact balance: BEFORE building row_map/rows_var (below, via
    # D.build_rows_var(table, ...)) so a zero-mass fact or an invalid
    # fact_ids stops the run immediately, not after minutes of preparation.
    # Heldout stays UNWEIGHTED (eval_kd_on_packed/eval_heldout_indicators
    # never receive fact_weights_map).
    fact_weights_map: "dict[str, float] | None" = None
    fact_mass_map: "dict[str, float] | None" = None
    fact_weight_stats: dict | None = None
    if fact_weight == "mass":
        fact_weights_map, fact_mass_map, fact_weight_stats = compute_fact_weights(train_frags, n_excl)
        log(f"fact_weight=mass: {fact_weight_stats}")

    eos_ids = {frag["tokens"][0] for frag in fragments_all}
    if len(eos_ids) != 1:
        raise ValueError(f"descend_corpus: fragments with different opening EOS tokens: {eos_ids}")
    eos = next(iter(eos_ids))

    frag_id_by_str = {frag["id"]: i for i, frag in enumerate(fragments_all)}
    targets_idx = index_targets(targets)

    exclude_positions = {i: [0] for i in range(len(train_frags))}  # the opening EOS
    row_map, rows_var_init = D.build_rows_var(
        table, [f["tokens"] for f in train_frags], policy, row_sets, exclude_positions=exclude_positions,
    )
    if rows_start_pleo is not None:
        rows_var_init = _load_warm_start_rows(rows_start_pleo, row_map, rows_var_init)
    rows0 = rows_var_init.clone().detach()
    rows_var = rows_var_init.clone().detach().requires_grad_(True)
    rows_pre = rows_var.detach().clone()

    mean_row_norm = float(rows0.norm(dim=1).mean().item()) if rows0.shape[0] else 0.0
    overlay_norm0 = float(rows0.norm().item()) if rows0.shape[0] else 0.0
    # `lr_scale` multiplies the base step 0.01/sqrt(ROW_LEN) relative to the
    # mean row norm; at 1.0 it reproduces the base formula.
    lr = lr_scale * 0.01 * mean_row_norm / (ROW_LEN ** 0.5)
    opt = torch.optim.Adam([rows_var], lr=lr)  # always reset on resume

    heldout_packed, heldout_membership = (
        pack_with_membership(heldout_frags, eos, pack_len) if heldout_frags else ([], [])
    )
    dev = getattr(getattr(replica, "backend", None), "device", "cpu")
    heldout_routing_inits: "list[dict[int, torch.Tensor]] | None" = None
    if routing_base is not None:
        rbr_by_frag, rbr_layers = routing_base
        heldout_routing_inits = [
            stitch_routing_init(packed, frag_dicts, rbr_by_frag, rbr_layers, dev)
            for packed, frag_dicts in zip(heldout_packed, heldout_membership)
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"descend_{arm}_{policy}.jsonl"
    if step_offset == 0 and log_path.exists():
        log_path.unlink()
    log_f = open(log_path, "a")

    step = step_offset
    optimizer_updates_executed = 0
    counters_executed = add_counters()
    counters_before_update = dict(counters_executed)
    counters_validation = add_counters()
    pass_idx = 0
    best_pass_mean = float("inf")
    passes_without_improve = 0
    plateau_k_stop_step: dict[int, int | None] = {k: None for k in range(3, 11)}
    regime_arbiter: "RegimeArbiter | None" = None
    if routing_regime == "mixed":
        acc_plateau_detector = None
        regime_arbiter = RegimeArbiter(
            switch_norm_growth=switch_norm_growth, switch_window=switch_window,
            switch_phi=switch_phi, switch_persistence=switch_persistence,
            instab_drop=instab_drop, instab_grad_factor=instab_grad_factor,
            reentry_readings=reentry_readings, eval_every_free=eval_every_free,
            exit_window=stop_local_window, exit_phi=stop_phi, exit_persistence=stop_persistence,
        )
    elif stop_criterion == "acc_heldout_rate":
        acc_plateau_detector = LocalRateDetector(stop_local_window, stop_phi, stop_persistence)
    elif stop_criterion == "acc_heldout":
        acc_plateau_detector = PlateauDetector(stop_window, stop_confidence)
    else:
        acc_plateau_detector = None
    grad_kd_norm_window: "deque[float]" = deque(maxlen=10)  # INSTAB_GRAD_STEPS, engraft.replica.regime
    acc_readings: "list[list]" = []
    acc_stop_step: "int | None" = None
    acc_t_stat_at_stop: "float | None" = None
    acc_rate_ratio_at_stop: "float | None" = None
    stop_series_from_step = step
    stop_reason: str | None = None
    routing_flips_top1_frac_readings: "list[list]" = []
    acc_readings_rbr: "list[list]" = []
    acc_readings_free: "list[list]" = []
    steps_per_pass: int | None = None
    t_start = time.time()
    gpu_seconds_cap = gpu_hours_cap * 3600.0 if gpu_hours_cap is not None else None
    time_guard = time_guard if time_guard is not None else StepTimeGuard()
    device_guard_kwargs = device_guard_kwargs or {}

    while True:
        if max_passes is not None and pass_idx >= max_passes:
            stop_reason = "max_steps"
            break
        order = list(range(len(train_frags)))
        random.Random(seed + pass_idx).shuffle(order)  # a declared per-pass seed
        shuffled = [train_frags[i] for i in order]
        packed_list, membership = pack_with_membership(shuffled, eos, pack_len)
        if steps_per_pass is None:
            steps_per_pass = len(packed_list)
        train_routing_inits: "list[dict[int, torch.Tensor]] | None" = None
        if routing_base is not None:
            rbr_by_frag, rbr_layers = routing_base
            train_routing_inits = [
                stitch_routing_init(packed, frag_dicts, rbr_by_frag, rbr_layers, dev)
                for packed, frag_dicts in zip(packed_list, membership)
            ]

        pass_losses: list[float] = []
        for pack_idx, (packed, frag_dicts) in enumerate(zip(packed_list, membership)):
            if max_steps is not None and step >= max_steps:
                stop_reason = "max_steps"
                break
            if device_guard_kwargs and check_device_memory_guard(
                device_guard_kwargs.get("device"), device_guard_kwargs.get("memory_fraction", 0.8),
                device_guard_kwargs.get("margin_bytes", 6 * (1 << 30)),
            ):
                stop_reason = "guard_device_memory"
                break
            if device_guard_kwargs and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device_guard_kwargs.get("device"))
            t_step0 = time.time()

            packed_dev = packed.to(dev)
            loss_mask_cpu = build_packed_loss_mask(
                packed, frag_dicts, n_excl, weight_fact_positions, fact_weights_map,
            )
            loss_mask = loss_mask_cpu.to(dev)
            if arm in ("kd", "kd-base-only"):
                ids_t, logp_t, tail_t = build_packed_targets(
                    packed, frag_dicts, frag_id_by_str, targets_idx, targets, k,
                )
                ids_t, logp_t, tail_t = ids_t.to(dev), logp_t.to(dev), tail_t.to(dev)
                loss_fn = make_loss_fn(arm, ids_t, logp_t, tail_t, loss_mask)
                answer_mask_cpu = torch.zeros_like(loss_mask_cpu)
            else:  # lm-base: `targets` are the BASE's targets
                base_ids, base_logp, base_tail = build_packed_targets(
                    packed, frag_dicts, frag_id_by_str, targets_idx, targets, k,
                )
                base_ids, base_logp, base_tail = base_ids.to(dev), base_logp.to(dev), base_tail.to(dev)
                y, answer_mask_cpu = build_packed_y_and_answer_mask(packed, frag_dicts)
                y, answer_mask = y.to(dev), answer_mask_cpu.to(dev)
                loss_fn = make_loss_fn(
                    arm, loss_mask=loss_mask, y=y, answer_mask=answer_mask,
                    base_ids=base_ids, base_logp=base_logp, base_tail=base_tail,
                )
            counters_pack = count_pack(
                len(packed.tokens), len(frag_dicts),
                loss_mask_cpu.tolist(), answer_mask_cpu.tolist(), answer_nll=arm == "lm-base",
            )

            if regime_arbiter is not None:
                step_routing_init = (
                    train_routing_inits[pack_idx]
                    if train_routing_inits is not None and regime_arbiter.state in (STATE_LOCKED, STATE_REENTRY)
                    else None
                )
            else:
                step_routing_init = train_routing_inits[pack_idx] if train_routing_inits is not None else None

            opt.zero_grad()
            loss, timings, step_stats = step_fn(
                replica, w, packed_dev.tokens, rows_var, row_map, loss_fn, wdot=wdot,
                checkpoint=checkpoint, positions=packed_dev.positions,
                segment_ids=packed_dev.segment_ids, delta_batched=delta_batched,
                overlay_cache=overlay_cache, routing_init=step_routing_init,
            )
            grad_kd = (
                rows_var.grad.detach().clone() if rows_var.grad is not None
                else torch.zeros_like(rows_var)
            )
            grad_kd_norm = float(grad_kd.norm().item())
            grad_kd_norm_window.append(grad_kd_norm)

            if rows0.shape[0]:
                reg = mu * ((rows_var - rows0).pow(2).sum() / rows0.pow(2).sum().clamp_min(1e-12))
                reg.backward()
            else:
                reg = torch.zeros((), dtype=torch.float32)
            grad_total = (
                rows_var.grad.detach().clone() if rows_var.grad is not None
                else torch.zeros_like(rows_var)
            )
            grad_reg_norm = float((grad_total - grad_kd).norm().item())

            rows_pre = rows_var.detach().clone()  # BEFORE the last opt.step()
            opt.step()

            if rows0.shape[0]:
                norm_ratio_max = float(
                    (rows_var.detach().norm(dim=1) / rows0.norm(dim=1).clamp_min(1e-12)).max().item()
                )
                cos_min = float(min(
                    _cos_rows(rows_var.detach()[i].numpy(), rows0[i].numpy()) for i in range(rows0.shape[0])
                ))
            else:
                norm_ratio_max, cos_min = 0.0, float("nan")

            if rows0.shape[0]:
                overlay_step_norm = float((rows_var.detach() - rows_pre).norm().item())
                overlay_dist0 = float((rows_var.detach() - rows0).norm().item())
            else:
                overlay_step_norm, overlay_dist0 = 0.0, 0.0

            kd_train = float(loss.item())
            pass_losses.append(kd_train)
            counters_before_update = dict(counters_executed)
            counters_executed = add_counters(counters_executed, counters_pack)
            optimizer_updates_executed += 1

            _snap_is_last_pack_of_pass = pack_idx == len(packed_list) - 1
            _do_snapshot = (
                (_snap_is_last_pack_of_pass and (pass_idx + 1) % snapshot_every_passes == 0)
                if snapshot_every_passes is not None
                else snapshot_every > 0 and (step + 1 - step_offset) % snapshot_every == 0
            )
            if _do_snapshot:
                _snap_label = step + 1 - step_offset
                snap_dir = out_dir / "snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_manifest = write_merged_pleo(
                    table, row_map, rows_pre, row_sets.get("all_read", []),
                    snap_dir / f"merged_step{_snap_label:04d}.pleo",
                )
                (snap_dir / f"merged_step{_snap_label:04d}_manifest.json").write_text(json.dumps(
                    {
                        **snap_manifest, "step": _snap_label, "pass_idx": pass_idx,
                        "stop_reason": "snapshot",
                        "optimizer_updates_executed": optimizer_updates_executed,
                        "overlay_updates": max(0, optimizer_updates_executed - 1),
                        "counters_executed": counters_executed,
                        "counters_overlay": counters_before_update,
                        "counters_validation": counters_validation,
                    },
                    indent=2, default=str,
                ))

            _do_eval = (
                (step - regime_arbiter.switch_step) % eval_every_free == 0
                if regime_arbiter is not None and regime_arbiter.switch_step is not None
                else (pack_idx == 0 and pass_idx % eval_every_passes == 0)
                if eval_every_passes is not None
                else (step - step_offset) % max(eval_every, 1) == 0
            )
            kd_heldout = None
            heldout_indicators = None
            heldout_indicators_rbr = None
            heldout_indicators_free = None
            counters_validation_reading = None
            if heldout_packed and _do_eval:
                if regime_arbiter is not None:
                    # Two forwards per reading, in EVERY phase -- with locked
                    # routing (the series charted today) and without (the
                    # baseline / the free phase's exit criterion). Validation
                    # counters come ONLY from the locked-routing forward.
                    heldout_indicators_rbr = eval_heldout_indicators(
                        replica, w, heldout_packed, heldout_membership, frag_id_by_str,
                        targets_idx, targets, n_excl, k, forward_fn, rows_var, row_map,
                        routing_inits=heldout_routing_inits,
                        delta_batched=delta_batched, overlay_cache=overlay_cache,
                        want_routing_stats=True,
                    )
                    heldout_indicators_free = eval_heldout_indicators(
                        replica, w, heldout_packed, heldout_membership, frag_id_by_str,
                        targets_idx, targets, n_excl, k, forward_fn, rows_var, row_map,
                        routing_inits=None,
                        delta_batched=delta_batched, overlay_cache=overlay_cache,
                    )
                    if heldout_indicators_rbr is not None:
                        counters_validation_reading = heldout_indicators_rbr.get("counters_validation")
                        if counters_validation_reading is not None:
                            counters_validation = add_counters(counters_validation, counters_validation_reading)
                        routing_flips_top1_frac_readings.append(
                            [step, heldout_indicators_rbr.get("routing_flips_top1_frac")]
                        )
                    heldout_indicators = (
                        heldout_indicators_rbr if regime_arbiter.state in (STATE_LOCKED, STATE_REENTRY)
                        else heldout_indicators_free
                    )
                    kd_heldout = heldout_indicators["kd_heldout"] if heldout_indicators else None
                else:
                    heldout_indicators = eval_heldout_indicators(
                        replica, w, heldout_packed, heldout_membership, frag_id_by_str,
                        targets_idx, targets, n_excl, k, forward_fn, rows_var, row_map,
                        routing_inits=heldout_routing_inits,
                        delta_batched=delta_batched, overlay_cache=overlay_cache,
                    )
                    kd_heldout = heldout_indicators["kd_heldout"] if heldout_indicators else None
                    if heldout_indicators is not None:
                        counters_validation_reading = heldout_indicators.get("counters_validation")
                        if counters_validation_reading is not None:
                            counters_validation = add_counters(counters_validation, counters_validation_reading)

            record = {
                "step": step, "pass": pass_idx, "arm": arm, "policy": policy,
                "kd_train": kd_train, "reg": float(reg.item()), "norm_ratio_max": norm_ratio_max,
                "cos_min": cos_min, "grad_kd_norm": grad_kd_norm, "grad_reg_norm": grad_reg_norm,
                "grad_kd_over_reg": (grad_kd_norm / grad_reg_norm) if grad_reg_norm > 1e-12 else None,
                "t_fwd_s": timings["fwd_s"], "t_bwd_s": timings["bwd_s"],
                "t_s": time.time() - t_start,
                "overlay_step_norm": overlay_step_norm, "overlay_dist0": overlay_dist0,
                "optimizer_updates_executed": optimizer_updates_executed,
                "counters_pack": counters_pack,
                "counters_cumulative": counters_executed,
                "counters_before_update": counters_before_update,
                "counters_validation_reading": counters_validation_reading,
                "counters_validation_cumulative": counters_validation,
            }
            for _k_src, _k_dst in (("pre_s", "t_pre_s"), ("delta_pad_ratio", "delta_pad_ratio"),
                                   ("delta_n_groups", "delta_n_groups")):
                if timings.get(_k_src) is not None:
                    record[_k_dst] = timings[_k_src]
            if kd_heldout is not None:
                record["kd_heldout"] = kd_heldout
            if heldout_indicators is not None:
                record["acc_heldout"] = heldout_indicators["acc_heldout"]
                record["margin_heldout"] = heldout_indicators["margin_heldout"]
                record["margin_heldout_median"] = heldout_indicators["margin_heldout_median"]
                record["n_answer_pos_heldout"] = heldout_indicators["n_answer_pos_heldout"]
            if regime_arbiter is not None:
                if heldout_indicators_rbr is not None:
                    record["acc_heldout_rbr"] = heldout_indicators_rbr["acc_heldout"]
                if heldout_indicators_free is not None:
                    record["acc_heldout_free"] = heldout_indicators_free["acc_heldout"]
            if routing_base is not None:
                record["n_moe_calls"] = step_stats.n_moe_calls
                record["routing_flips_vs_init_total"] = sum(step_stats.routing_flips_vs_init.values())
                record["routing_flips_vs_init_by_layer"] = {
                    str(il): n for il, n in step_stats.routing_flips_vs_init.items()
                }
                record["routing_frozen_flips_total"] = sum(step_stats.routing_frozen_flips.values())
                record["routing_frozen_flips_by_layer"] = {
                    str(il): n for il, n in step_stats.routing_frozen_flips.items()
                }
            if regime_arbiter is not None:
                if step_routing_init is not None:
                    record["routing_flips_top1_total"] = sum(step_stats.routing_flips_top1.values())
                    _denom = len(packed_dev.tokens) * len(step_routing_init)
                    record["routing_flips_top1_frac"] = (
                        record["routing_flips_top1_total"] / _denom if _denom > 0 else None
                    )
                else:
                    record["routing_flips_top1_total"] = None
                    record["routing_flips_top1_frac"] = None
                    record["routing_flips_vs_init_total"] = None
                    record["routing_flips_vs_init_by_layer"] = None
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

            if acc_plateau_detector is not None and heldout_indicators is not None:
                acc_reading = heldout_indicators["acc_heldout"]
                acc_readings.append([step, acc_reading])
                if acc_plateau_detector.push(step, acc_reading):
                    if stop_criterion == "acc_heldout_rate":
                        stop_reason = "plateau_acc_heldout_rate"
                        acc_rate_ratio_at_stop = acc_plateau_detector.rate_ratio
                    else:
                        stop_reason = "plateau_acc_heldout"
                        acc_t_stat_at_stop = acc_plateau_detector.t_stat
                    acc_stop_step = step
                    step += 1
                    break
            elif regime_arbiter is not None and _do_eval and heldout_packed:
                acc_rbr_reading = heldout_indicators_rbr["acc_heldout"] if heldout_indicators_rbr else None
                acc_free_reading = heldout_indicators_free["acc_heldout"] if heldout_indicators_free else None
                grad_kd_norm_recent = (
                    sum(grad_kd_norm_window) / len(grad_kd_norm_window) if grad_kd_norm_window else None
                )
                decision = regime_arbiter.push(
                    step, acc_rbr_reading, acc_free_reading, norm_ratio_max, grad_kd_norm_recent,
                )
                acc_readings.append([step, acc_rbr_reading])
                acc_readings_rbr.append([step, acc_rbr_reading])
                acc_readings_free.append([step, acc_free_reading])
                if decision.event == "switch":
                    switch_pleo_manifest = write_merged_pleo(
                        table, row_map, rows_pre, row_sets.get("all_read", []),
                        out_dir / "merged_switch.pleo",
                    )
                    (out_dir / "merged_switch_manifest.json").write_text(json.dumps(
                        {**switch_pleo_manifest, "step": step, "switch_step": regime_arbiter.switch_step},
                        indent=2, default=str,
                    ))
                if decision.next_state == STATE_STOP:
                    stop_reason = decision.reason
                    acc_stop_step = step
                    acc_rate_ratio_at_stop = regime_arbiter.exit_rate_ratio
                    step += 1
                    break

            step_dt = time.time() - t_step0

            if time_guard.check(step_dt):
                stop_reason = "guard_step_time"
                step += 1
                break
            if device_guard_kwargs and check_device_memory_guard(
                device_guard_kwargs.get("device"), device_guard_kwargs.get("memory_fraction", 0.8),
                device_guard_kwargs.get("margin_bytes", 6 * (1 << 30)),
                stat="max_allocated",
            ):
                stop_reason = "guard_device_memory"
                step += 1
                break
            step += 1

            if gpu_seconds_cap is not None and (time.time() - t_start) >= gpu_seconds_cap:
                stop_reason = "budget"
                break

        if stop_reason is not None:
            break

        pass_mean = float(np.mean(pass_losses)) if pass_losses else float("nan")
        if pass_mean < best_pass_mean * (1.0 - PLATEAU_REL_IMPROVEMENT):
            best_pass_mean = pass_mean
            passes_without_improve = 0
        else:
            passes_without_improve += 1

        for k_stop, stopped_at in plateau_k_stop_step.items():
            if stopped_at is None and passes_without_improve >= k_stop:
                plateau_k_stop_step[k_stop] = step
                log(f"plateau_k_stop_step[{k_stop}]={step} (passes_without_improve={passes_without_improve})")
                if k_stop in (5, 10):
                    snap_pleo_manifest = write_merged_pleo(
                        table, row_map, rows_pre, row_sets.get("all_read", []),
                        out_dir / f"merged_plateau{k_stop}.pleo",
                    )
                    (out_dir / f"merged_plateau{k_stop}_manifest.json").write_text(json.dumps(
                        {
                            **snap_pleo_manifest, "step": step, "pass_idx": pass_idx,
                            "stop_reason": f"plateau{k_stop}_snapshot",
                            "optimizer_updates_executed": optimizer_updates_executed,
                            "overlay_updates": max(0, optimizer_updates_executed - 1),
                            "counters_executed": counters_executed,
                            "counters_overlay": counters_before_update,
                            "counters_validation": counters_validation,
                        },
                        indent=2, default=str,
                    ))

        if stop_criterion == "kd_train" and passes_without_improve >= patience:
            stop_reason = "plateau"
            break
        pass_idx += 1

    log_f.close()

    merged_path = out_dir / "merged.pleo"
    pleo_manifest = write_merged_pleo(table, row_map, rows_pre, row_sets.get("all_read", []), merged_path)

    _eval_every_steps_effective = (
        eval_every_free if regime_arbiter is not None
        else (steps_per_pass or 0) * eval_every_passes if eval_every_passes is not None
        else eval_every
    )
    manifest = {
        "policy": policy, "arm": arm, "n_rows_variable": len(row_map),
        "frozen_rows": pleo_manifest["frozen_rows"], "n_rows_frozen": pleo_manifest["n_rows_frozen"],
        "seed": seed, "k": k, "pack_len": pack_len, "n_excl": n_excl, "mu": mu,
        "checkpoint": checkpoint, "step_offset": step_offset,
        "fact_weight": fact_weight,
        **({"fact_mass": fact_mass_map, "fact_weights": fact_weights_map,
            "fact_weight_stats": fact_weight_stats} if fact_weight == "mass" else {}),
        "delta_batched": delta_batched, "overlay_cache_enabled": overlay_cache is not None,
        "stop_reason": stop_reason, "n_steps": step, "n_passes": pass_idx + (1 if stop_reason == "plateau" else 0),
        "plateau_k_stop_step": plateau_k_stop_step,
        "steps_per_pass": steps_per_pass, "eval_every_passes": eval_every_passes,
        "eval_every_steps": _eval_every_steps_effective,
        "eval_every_steps_locked": eval_every if regime_arbiter is not None else None,
        "max_passes": max_passes, "snapshot_every_passes": snapshot_every_passes,
        "stop_criterion": stop_criterion, "stop_window": stop_window, "stop_confidence": stop_confidence,
        "stop_window_steps": stop_window * _eval_every_steps_effective,
        "acc_stop_step": acc_stop_step, "acc_t_stat_at_stop": acc_t_stat_at_stop,
        "stop_local_window": stop_local_window, "stop_phi": stop_phi,
        "stop_persistence": stop_persistence,
        "stop_local_window_steps": stop_local_window * _eval_every_steps_effective,
        "acc_rate_ratio_at_stop": acc_rate_ratio_at_stop,
        "acc_stop_a0": (
            regime_arbiter.a0_free if regime_arbiter is not None
            else getattr(acc_plateau_detector, "a0", None)
        ),
        "acc_readings": acc_readings,
        "acc_readings_skipped": (
            regime_arbiter.n_skipped if regime_arbiter is not None
            else (acc_plateau_detector.n_skipped if acc_plateau_detector is not None else 0)
        ),
        "stop_series_from_step": stop_series_from_step,
        "kd_train_plateau10_step": plateau_k_stop_step[10],
        "routing_regime": routing_regime,
        "phases": regime_arbiter.phases_as_dicts() if regime_arbiter is not None else [],
        "switch_step": regime_arbiter.switch_step if regime_arbiter is not None else None,
        "switch_norm_growth_at": regime_arbiter.switch_norm_growth_at if regime_arbiter is not None else None,
        "switch_rate_ratio_at": regime_arbiter.switch_rate_ratio_at if regime_arbiter is not None else None,
        "switch_acc_rbr": regime_arbiter.switch_acc_rbr if regime_arbiter is not None else None,
        "switch_acc_free": regime_arbiter.switch_acc_free if regime_arbiter is not None else None,
        "switch_grad_kd_ref": regime_arbiter.switch_grad_kd_ref if regime_arbiter is not None else None,
        "n_reentries": regime_arbiter.n_reentries if regime_arbiter is not None else 0,
        "n_instability_triggers": regime_arbiter.n_instability_triggers if regime_arbiter is not None else 0,
        "acc_readings_rbr": acc_readings_rbr,
        "acc_readings_free": acc_readings_free,
        "routing_flips_top1_frac_readings": routing_flips_top1_frac_readings,
        "optimizer_updates_executed": optimizer_updates_executed,
        "overlay_updates": max(0, optimizer_updates_executed - 1),
        "counters_executed": counters_executed,
        "counters_overlay": counters_before_update,
        "counters_validation": counters_validation,
        "lr": lr, "lr_scale": lr_scale, "mean_row_norm": mean_row_norm, "overlay_norm0": overlay_norm0,
        "total_time_s": time.time() - t_start,
        "log_path": str(log_path), "merged_pleo_path": str(merged_path),
        **_dtype_manifest_fields(replica),
    }
    if record_peak_memory and torch.cuda.is_available():
        manifest["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(memory_device))
        manifest["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(memory_device))
    (out_dir / "merged_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    summary = dict(manifest)
    summary["row_map"] = {str(g): i for g, i in row_map.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return manifest


# --------------------------------------------------------------------------
# Warm start, preflight, and backend loading (real: MAI eseguito qui -- see
# `load_real_backend`'s docstring; fake: a real but tiny CPU Replica)
# --------------------------------------------------------------------------


def _load_warm_start_rows(rows_start_pleo: Path, row_map: dict[int, int], rows_var_init: torch.Tensor) -> torch.Tensor:
    """Warm start (continuing a run that hit its step ceiling): reads a
    previous run's final `.pleo` overlay and substitutes its values, for the
    global ids present in `row_map`, in place of `rows_var_init`'s true rows.
    Fails fast if a current fact's variable row is missing from the starting
    overlay -- would happen with a `.pleo` from a different fact or a run
    with a different `row_map`."""
    warm_rows_g, warm_data = read_pleo(rows_start_pleo)
    warm_map = {int(g): d for g, d in zip(warm_rows_g.tolist(), warm_data)}
    missing = [g for g in row_map if g not in warm_map]
    if missing:
        raise RuntimeError(
            f"--rows-start-pleo {rows_start_pleo}: variable rows missing from the "
            f"starting overlay (global ids): {missing} -- never a silent assumption"
        )
    rows_var_start = rows_var_init.clone()
    for g, idx in row_map.items():
        rows_var_start[idx] = torch.from_numpy(warm_map[g].copy()).to(rows_var_init.device)
    return rows_var_start


def _preflight_real(min_avail_gb: float = 15.0, meminfo_path: Path = Path("/proc/meminfo")) -> None:
    """Checked before EVERY real load: available memory (`>= min_avail_gb`)
    and no other production inference server for this model already running.
    Exits with a clear message (never a silent OOM or a silently interrupted
    production run) -- NEVER called from the dry run (`--fake`)."""
    meminfo = meminfo_path.read_text()
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    if m is None:
        raise RuntimeError("_preflight_real: MemAvailable not found in /proc/meminfo")
    avail_gb = int(m.group(1)) / (1024 * 1024)
    if avail_gb < min_avail_gb:
        raise RuntimeError(
            f"_preflight_real: MemAvailable={avail_gb:.1f} GB < {min_avail_gb:.1f} GB required "
            "-- real run NOT started"
        )
    log(f"_preflight_real: OK (MemAvailable={avail_gb:.1f} GB)")


def _fake_replica_and_table(seed: int = 1, head_dtype: str = "f32"):
    """A tiny but real `Replica`, `engraft.replica.layers`, fake weights
    (`engraft.testing.fake_full_weights.FakeFullWeights`) and a fake table
    (`engraft.testing.fake_table.FakeTable`): the `--fake`/test path,
    entirely CPU, no Triton."""
    from engraft.testing.fake_full_weights import FakeFullWeights, tiny_hparams
    from engraft.testing.fake_table import FakeTable
    import engraft.replica.backend as Bk
    from engraft.replica.model import Replica, head_dtype_from_name

    hp = tiny_hparams()
    w = FakeFullWeights(hp)
    table = FakeTable(seed=seed)
    # `table.eos_token_id` is overridden here, on the instance, to an
    # embeddable value (`n_vocab-1`): `FakeTable`'s real sentinel
    # (`EOS_TOKEN_ID=999_999_999`) is not a valid embedding index in this
    # tiny `Replica` (`n_vocab=20`).
    table.eos_token_id = hp.n_vocab - 1
    replica = Replica(hp, w, table, backend=Bk.Backend.cpu_f32(), head_dtype=head_dtype_from_name(head_dtype))
    return replica, None, table, fake_seq_step_packed, D.fake_seq_forward


def load_real_backend(args):  # pragma: no cover -- requires a real GGUF + CUDA/ROCm device
    """Loads the real backend: a `DeviceIQ4` weight store resident on
    `args.device`, the real PLE table, and the Triton per-expert step/forward
    functions (`engraft.replica.seq.seq_step_triton`/`seq_forward_triton`).
    Not runnable in a CPU-only session -- scaffolding for completeness, not
    exercised here.

    `--table-path`/`--shard-paths` name the GGUF files explicitly (no
    hardcoded machine paths): the caller supplies them, typically from a
    local model download."""
    from engraft.replica.backend import Backend, PhaseDtypes, ResidentPolicy
    from engraft.replica.model import Replica, head_dtype_from_name
    from engraft.replica.weights import DeviceIQ4
    from engraft.replica.hparams import Hparams
    from engraft.table import PleTable

    _preflight_real(args.min_avail_gb)
    check_no_concurrent_run()

    GiB = 1 << 30
    working_set_bytes = int(args.working_set_gb * GiB)
    dequant_cache_bytes = int(args.dequant_cache_gb * GiB)
    backend = Backend(
        device=args.device, weight_store="device_iq4",
        resident_policy=ResidentPolicy("working_set", working_set_bytes, dequant_cache_bytes=dequant_cache_bytes),
        dtypes=PhaseDtypes(prefix=args.dense_dtype, grad=args.dense_dtype), moe_grouped=True,
    )
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=args.device)

    shard_paths = list(args.shard_paths)
    hp = Hparams.from_gguf_paths(*shard_paths[:2])
    w = DeviceIQ4(shard_paths, device=args.device, dtype=torch.float32)
    table = PleTable(args.table_path)
    head_dtype = head_dtype_from_name(getattr(args, "head_dtype", "f32"))
    replica = Replica(hp, w, table, backend=backend, head_dtype=head_dtype)
    set_oom_score_adj(800)

    from engraft.replica.seq import seq_step_triton as _step_fn, seq_forward_triton as _forward_fn
    return replica, w, table, _step_fn, _forward_fn


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_fragments_all(usage_corpus_resolved_path: Path) -> list[dict]:
    data = json.loads(Path(usage_corpus_resolved_path).read_text())
    return data["fragments"] if "fragments" in data else data


def _read_descend_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rows_read_by_prefix(table, tokens: list[int]) -> set[int]:
    """Every global row read by `tokens`'s prefix (positions 0..len-2)."""
    from engraft.lens import RowSet

    out: set[int] = set()
    n = len(tokens)
    for t in range(max(n - 1, 0)):
        rs = RowSet.from_position(table, tokens, t)
        out.update(int(g) for g in rs.rows_global)
    return out


def build_all_read_row_set(table, train_frags_tokens: list[list[int]]) -> dict[str, list[int]]:
    """`{"all_read": [...]}`: the union of every global row read by every
    train fragment's prefix, used by `write_merged_pleo` to also freeze the
    non-variable rows the model actually reads. A minimal, self-contained
    reference builder -- the private pipeline's row-set calibration tooling
    (budget/weighting across many candidate rows) is out of scope here."""
    all_read: set[int] = set()
    for toks in train_frags_tokens:
        all_read |= _rows_read_by_prefix(table, toks)
    return {"all_read": sorted(all_read)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--usage-corpus", required=True, help="usage_corpus_resolved.json")
    parser.add_argument("--census", default=None,
                         help="census.json (a {'row_sets': {'all_read': [...], ...}} document) -- "
                              "when given, row_sets comes from here instead of being recomputed by "
                              "build_all_read_row_set(); see docs/formats.md")
    parser.add_argument("--targets", required=True, help=".npz from engraft.teacher (TeacherTargets.save)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--pack-len", type=int, default=2048)
    parser.add_argument("--n-excl", type=int, default=9)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-passes", type=int, default=None)
    parser.add_argument("--gpu-hours-cap", type=float, default=None)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--checkpoint", action="store_true",
                         help="whole-layer checkpointing (torch.utils.checkpoint) during the backward pass -- "
                              "does not change the math, may change peak memory; see docs/formats.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-every-passes", type=int, default=None)
    parser.add_argument("--stop-criterion", default="kd_train",
                         choices=["kd_train", "acc_heldout", "acc_heldout_rate"])
    parser.add_argument("--stop-window", type=int, default=8)
    parser.add_argument("--stop-confidence", type=float, default=0.95)
    parser.add_argument("--stop-local-window", type=int, default=LOCAL_RATE_WINDOW)
    parser.add_argument("--stop-phi", type=float, default=LOCAL_RATE_PHI)
    parser.add_argument("--stop-persistence", type=int, default=LOCAL_RATE_PERSISTENCE)
    parser.add_argument("--wdot", default="split", choices=["bf16", "split", "f32"])
    parser.add_argument("--delta-chunk-size", type=int, default=64)
    parser.add_argument("--delta-batched", dest="delta_batched", action="store_true", default=True)
    parser.add_argument("--no-delta-batched", dest="delta_batched", action="store_false")
    parser.add_argument("--overlay-cache", action="store_true",
                         help="enable the per-fragment raw-gather cache for the whole run")
    parser.add_argument("--rows-start-pleo", default=None)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--snapshot-every-passes", type=int, default=None)
    parser.add_argument("--weight-fact-positions", type=float, default=1.0)
    parser.add_argument("--fact-weight", default="none", choices=["none", "mass"])
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--routing-base", default=None,
                         help="routing_base_*.npz from engraft.teacher --routing-out (locked routing, RBR)")
    parser.add_argument("--routing-regime", default="locked",
                         choices=["locked", "mixed", "misto", "bloccato"],
                         help="'misto'/'bloccato' are deprecated aliases for 'mixed'/'locked'")
    parser.add_argument("--switch-norm-growth", type=float, default=SWITCH_NORM_GROWTH)
    parser.add_argument("--switch-phi", type=float, default=SWITCH_PHI)
    parser.add_argument("--switch-window", type=int, default=SWITCH_WINDOW)
    parser.add_argument("--switch-persistence", type=int, default=SWITCH_PERSISTENCE)
    parser.add_argument("--instab-drop", type=float, default=INSTAB_DROP)
    parser.add_argument("--instab-grad-factor", type=float, default=INSTAB_GRAD_FACTOR)
    parser.add_argument("--reentry-readings", type=int, default=REENTRY_READINGS)
    parser.add_argument("--eval-every-free", type=int, default=EVAL_EVERY_FREE)
    parser.add_argument("--record-peak-memory", action="store_true")
    parser.add_argument("--fake", action="store_true")
    # Backend (real run only)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--working-set-gb", type=float, default=2.0)
    parser.add_argument("--dequant-cache-gb", type=float, default=24.0)
    parser.add_argument("--dense-dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--head-dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--table-path", default=None, help="GGUF n-gram table shard path (real run only)")
    parser.add_argument("--shard-paths", nargs="+", default=None, help="GGUF weight shard paths (real run only)")
    parser.add_argument("--min-avail-gb", type=float, default=100.0)
    parser.add_argument("--device-memory-fraction", type=float, default=None,
                         help="if given, enables the per-step device memory guard")
    parser.add_argument("--device-memory-margin-gb", type=float, default=6.0)
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_passes is not None:
        raise SystemExit("--max-steps and --max-passes are mutually exclusive")
    if args.eval_every_passes is not None and args.routing_regime in ("mixed", "misto"):
        raise SystemExit("--eval-every-passes is not supported with --routing-regime mixed "
                          "(the free phase's cadence is --eval-every-free)")

    if args.fake:
        replica, w, table, step_fn, forward_fn = _fake_replica_and_table()
    else:  # pragma: no cover -- requires a real GGUF/CUDA device
        if not args.table_path or not args.shard_paths:
            raise SystemExit("a real run requires --table-path and --shard-paths")
        replica, w, table, step_fn, forward_fn = load_real_backend(args)

    fragments_all = _load_fragments_all(args.usage_corpus)
    targets = D.TeacherTargets.load(args.targets)

    if args.census:
        census = json.loads(Path(args.census).read_text())
        row_sets = census["row_sets"]
    else:
        train_frags_tokens = [f["tokens"] for f in fragments_all if f.get("split") == "train"]
        row_sets = build_all_read_row_set(table, train_frags_tokens)

    routing_base = None
    if args.routing_base:
        by_frag, layers, cfg = D.load_routing_base(args.routing_base, mismatch_ok=True)
        routing_base = (by_frag, layers)
        log(f"routing_base loaded from {args.routing_base}: config={cfg}")

    device_guard_kwargs = None
    if args.device_memory_fraction is not None:
        device_guard_kwargs = {
            "device": args.device, "memory_fraction": args.device_memory_fraction,
            "margin_bytes": int(args.device_memory_margin_gb * (1 << 30)),
        }

    out_dir = Path(args.out)
    manifest = descend_corpus(
        replica, w, table, step_fn, forward_fn, fragments_all, row_sets,
        args.policy, args.arm, targets, out_dir,
        k=args.k, pack_len=args.pack_len, n_excl=args.n_excl, patience=args.patience,
        max_steps=args.max_steps, gpu_hours_cap=args.gpu_hours_cap, mu=args.mu,
        checkpoint=args.checkpoint, seed=args.seed, eval_every=args.eval_every,
        stop_criterion=args.stop_criterion, stop_window=args.stop_window, stop_confidence=args.stop_confidence,
        stop_local_window=args.stop_local_window, stop_phi=args.stop_phi, stop_persistence=args.stop_persistence,
        wdot=args.wdot, delta_chunk_size=args.delta_chunk_size,
        delta_batched=args.delta_batched, overlay_cache=({} if args.overlay_cache else None),
        rows_start_pleo=Path(args.rows_start_pleo) if args.rows_start_pleo else None,
        step_offset=args.step_offset, snapshot_every=args.snapshot_every,
        weight_fact_positions=args.weight_fact_positions, fact_weight=args.fact_weight,
        lr_scale=args.lr_scale, device_guard_kwargs=device_guard_kwargs,
        record_peak_memory=args.record_peak_memory, memory_device=args.device if not args.fake else None,
        routing_base=routing_base,
        max_passes=args.max_passes, eval_every_passes=args.eval_every_passes,
        snapshot_every_passes=args.snapshot_every_passes,
        routing_regime=args.routing_regime,
        switch_norm_growth=args.switch_norm_growth, switch_phi=args.switch_phi, switch_window=args.switch_window,
        switch_persistence=args.switch_persistence, instab_drop=args.instab_drop,
        instab_grad_factor=args.instab_grad_factor, reentry_readings=args.reentry_readings,
        eval_every_free=args.eval_every_free,
    )
    log(f"descend_corpus: stop_reason={manifest['stop_reason']!r} n_steps={manifest['n_steps']} "
        f"merged_pleo={manifest['merged_pleo_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

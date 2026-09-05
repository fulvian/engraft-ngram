"""Chained fact grafting: graft a multi-token answer position by position.

`graft_fact` grafts positions 0..n-1 of a fact's answer, in series: position i
reads, in its own prefix, the already-grafted rows of positions < i (a chain
overlay) and builds `routing_trigger` from scratch (never a free-routing
prefix) before calling `descend` (row_mask T8, refresh_every=1). Resumable per
graft via `state.json`: an already-closed graft is skipped, its rows re-enter
the chain overlay by re-reading its `.pleo`.

`merge_fact_overlays` applies the key-conflict resolution decided by
`engraft.facts` (`keys.json`: the fact that comes first in `facts.json` wins,
the other is excluded in full) to the union of per-fact `.pleo` overlays: no
new conflict detection happens here, `keys.json` is the source of truth.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from engraft.lens import RowSet, read_pleo, write_pleo

from engraft.replica.model import check_precondition
import engraft.replica.descend as D

ROW_MASK_T8 = np.array([False] * 8 + [True] * 8)

DEFAULT_CFG = {
    "lam": 0.0,
    "row_mask": ROW_MASK_T8,
    "refresh_every": 1,
    "thresholds": [],
    "p_stop": 0.95,
    "plateau_steps": 150,
}


class PreconditionError(RuntimeError):
    """A graft (fid, position) reads one of its own T8 rows in its own prefix:
    fail-fast, never continued."""

    def __init__(self, fid: str, position: int, precondition: dict):
        self.fid = fid
        self.position = position
        self.precondition = precondition
        super().__init__(
            f"{fid} position {position}: precondition failed, hits={precondition.get('hits')}"
        )


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"done": {}}
    return json.loads(path.read_text())


def _overlay_map_from_parts(parts: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for rows_g, data_g in parts:
        for i in range(rows_g.shape[0]):
            out[int(rows_g[i])] = data_g[i]
    return out


def _build_routing_trigger(
    cap_prefix: dict[int, np.ndarray], cap_last: dict[int, np.ndarray], n_layer: int,
) -> dict[int, np.ndarray]:
    """Concatenates, layer by layer, the routing captured from the prefix
    (positions 0..T-2, absent for layer n_layer-1) with the routing captured by
    a gradient-free `last_step` on the true rows, on the same prefix state."""
    routing_trigger: dict[int, np.ndarray] = {}
    for il in range(n_layer):
        last_row = np.asarray(cap_last[il]).reshape(1, -1).astype(np.int32)
        if il == n_layer - 1:
            routing_trigger[il] = last_row
        else:
            prefix_rows = np.asarray(cap_prefix[il], dtype=np.int32).reshape(-1, last_row.shape[1])
            routing_trigger[il] = np.concatenate([prefix_rows, last_row], axis=0)
    return routing_trigger


def prepare_innesto(
    replica, table, fact: dict, i: int, overlay_map: dict[int, np.ndarray], out_dir: Path,
) -> dict:
    """Live prefix (with the chain overlay), `routing_trigger` over 48 (here
    `replica.hp.n_layer`) layers, and precondition check for graft `i` of
    `fact`. Shared by `graft_fact` and a perturbed-seed repeat run: a single
    place that builds `routing_trigger`, never a free-routing prefix. Writes
    `<fid>_<i>_precondition.json` and raises `PreconditionError` if the
    precondition fails (fail-fast, before any descent).

    Returns {"tokens", "prefix_state", "routing_trigger", "rowset", "rows_true",
    "prefix_time_s", "precondition"}."""
    fid = fact["id"]
    trigger_tokens = list(fact["trigger_tokens"])
    answer_tokens = list(fact["answer_tokens"])
    tokens_i = trigger_tokens + answer_tokens[:i]

    rows_i = np.asarray(
        fact["trigger_rows_global"] if i == 0 else fact["chain_rows_global"][str(i)]
    )
    precond_rows_t8 = rows_i[8:]

    t0 = time.time()
    cap_prefix: dict[int, np.ndarray] = {}
    prefix_state = replica.prefix(
        tokens_i, routing_source=None, capture_routing=cap_prefix, overlay=overlay_map,
    )
    rowset_i = RowSet.from_position(table, tokens_i, len(tokens_i) - 1)
    rows_true_i = torch.from_numpy(rowset_i.data.copy())

    cap_last: dict[int, np.ndarray] = {}
    with torch.no_grad():
        replica.last_step(
            tokens_i, prefix_state, rows_true_i, routing_source=None,
            persist_experts=False, capture_routing=cap_last,
        )
    prefix_time_s = time.time() - t0

    n_layer = replica.hp.n_layer
    routing_trigger = _build_routing_trigger(cap_prefix, cap_last, n_layer)

    precondition = check_precondition(table, {f"{fid}_{i}": tokens_i}, precond_rows_t8)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / f"{fid}_{i}_precondition.json", precondition)
    if not precondition["ok"]:
        raise PreconditionError(fid, i, precondition)

    return {
        "tokens": tokens_i, "prefix_state": prefix_state, "routing_trigger": routing_trigger,
        "rowset": rowset_i, "rows_true": rows_true_i, "prefix_time_s": prefix_time_s,
        "precondition": precondition,
    }


def graft_fact(
    replica, table, tok, fact: dict, cfg: dict, out_dir: Path, state_path: Path,
    on_step=None,
) -> dict:
    """Grafts, in chain, every position of `fact`'s answer.

    `fact`: a resolved entry with at least `id`, `trigger_tokens`,
    `answer_tokens`, `trigger_rows_global`, `chain_rows_global`. `tok` is not
    used directly here (the tokens are already resolved in `fact`): it stays
    in the signature for uniformity and for future diagnostic use (decoding
    on error). `on_step`, if given, is forwarded to `descend` (a caller-side
    step-time guardian)."""
    fid = fact["id"]
    trigger_tokens = list(fact["trigger_tokens"])
    answer_tokens = list(fact["answer_tokens"])
    n = len(answer_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {**DEFAULT_CFG, **cfg}

    state = _load_state(state_path)
    done: dict[str, dict] = state.setdefault("done", {})

    overlay_parts: list[tuple[np.ndarray, np.ndarray]] = []
    innesti: list[dict] = []
    for i_str in sorted(done.keys(), key=int):
        pleo_path = out_dir / f"ckpt_{fid}_{i_str}_final.pleo"
        overlay_parts.append(read_pleo(pleo_path))
        innesti.append(done[i_str])

    for i in range(n):
        if str(i) in done:
            continue
        overlay_map = _overlay_map_from_parts(overlay_parts)
        prep = prepare_innesto(replica, table, fact, i, overlay_map, out_dir)
        tokens_i = prep["tokens"]
        prefix_state = prep["prefix_state"]
        routing_trigger = prep["routing_trigger"]
        rowset_i = prep["rowset"]
        rows_true_i = prep["rows_true"]

        log_path = out_dir / f"descend_{fid}_{i}.jsonl"
        t0 = time.time()
        summary = D.descend(
            replica, tokens_i, prefix_state, routing_trigger, rows_true_i,
            answer_tokens[i], {}, rowset_i.rows_global, cfg["lam"], out_dir, log_path,
            row_mask=cfg["row_mask"], refresh_every=cfg["refresh_every"],
            thresholds=cfg["thresholds"], p_stop=cfg["p_stop"],
            plateau_steps=cfg["plateau_steps"], tag=f"{fid}_{i}", on_step=on_step,
        )
        descend_time_s = time.time() - t0

        rows_g, data_g = read_pleo(out_dir / f"ckpt_{fid}_{i}_final.pleo")
        overlay_parts.append((rows_g, data_g))

        innesto_summary = {
            "position": i,
            "n_steps": summary.get("n_steps"),
            "stop_reason": summary.get("stop_reason"),
            "final_p_free": summary.get("final_p_free"),
            "cos_min_final": summary.get("cos_min_final"),
            "norm_ratio_max_final": summary.get("norm_ratio_max_final"),
            "prefix_time_s": prep["prefix_time_s"],
            "descend_time_s": descend_time_s,
            "precondition_ok": prep["precondition"]["ok"],
        }
        innesti.append(innesto_summary)
        done[str(i)] = innesto_summary
        _atomic_write_json(state_path, state)

    rows_all, data_all = RowSet.build_overlay(overlay_parts)
    fact_pleo_path = out_dir / f"{fid}.pleo"
    write_pleo(fact_pleo_path, rows_all, data_all)

    p_free_values = [it.get("final_p_free") for it in innesti if it.get("final_p_free") is not None]
    p_free_product = float(np.prod(p_free_values)) if len(p_free_values) == n else None

    fact_summary = {
        "id": fid,
        "n_positions": n,
        "innesti": sorted(innesti, key=lambda it: it["position"]),
        "p_free_product": p_free_product,
        "overlay_path": str(fact_pleo_path),
        "n_rows_overlay": int(rows_all.shape[0]),
    }
    _atomic_write_json(out_dir / f"{fid}.json", fact_summary)
    return fact_summary


def merge_fact_overlays(
    order: list[str], excluded_facts: set[str] | list[str], fact_pleo_paths: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Union of the per-fact overlays in `order` (the `facts.json` order),
    honoring the exclusion already decided by `keys.json`: no new conflict
    detection happens here. If two non-excluded facts collide anyway (same
    row with different values, which should not happen by construction),
    `RowSet.build_overlay` raises -- an error to report, never silently
    absorbed."""
    excluded_set = set(excluded_facts)
    included: list[str] = []
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for fid in order:
        if fid in excluded_set:
            continue
        rows_g, data_g = read_pleo(fact_pleo_paths[fid])
        parts.append((rows_g, data_g))
        included.append(fid)
    rows_all, data_all = RowSet.build_overlay(parts)
    manifest = {
        "included": included,
        "excluded": sorted(excluded_set),
        "n_rows": int(rows_all.shape[0]),
    }
    return rows_all, data_all, manifest

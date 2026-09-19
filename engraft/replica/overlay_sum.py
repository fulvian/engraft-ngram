"""Sum overlays from multiple facts over the union of present rows.

Each `.pleo` file of a fact is written by `write_merged_pleo` with ALL rows
from `all_read` (sparse gradient variables plus frozen), not only those affected.
Thus "row affected by fact i" is not "row present in file" (that would always be
true), but |Delta_i| > 0 where `Delta_i = rows_in_file - true_rows`
(`distill._read_true_rows`, same addressing as used by the quantized table).
"""
from __future__ import annotations

import numpy as np

from engraft.lens import read_pleo
import engraft.replica.distill as D

# Quantization floor for IQ4_NL: only recorded in manifest, never a threshold
# that stops anything here -- the decision remains with the operator downstream.
QUANT_FLOOR = 0.055


def delta_for_pleo(table, pleo_path) -> tuple[np.ndarray, np.ndarray]:
    """`(rows_global, delta)` for a single `.pleo` file:
    `delta = rows_in_file - true_rows` (`distill._read_true_rows`, same addressing
    as used by `write_merged_pleo`), in the SAME order as `rows_global` written in
    the file (never reordered here -- the caller, `sum_overlays`, decides the union
    and order). Both terms are already float32 (`read_pleo`/`_read_true_rows`):
    the subtraction stays in float32, so `Delta` of an untouched row (Adam with
    null gradient) is exact zero, not a rounding residue from a float64 pass.
    """
    rows, data = read_pleo(pleo_path)
    true_rows = D._read_true_rows(table, rows.tolist())
    delta = data.astype(np.float32) - true_rows.astype(np.float32)
    return rows, delta


def sum_overlays(table, pleo_paths: list) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sum overlays from N facts: for each file `Delta_i = rows - true_rows`;
    union of rows present in at least one file (same for all in practice,
    `all_read` written by `write_merged_pleo`);
    `merged = true_rows_of_union + Sum_i Delta_i` (sum in float32, per row,
    over union).

    Returns `(rows_union, data_merged, manifest)`:
    - `rows_union`: `[R]` int32, ascending order (deterministic, independent
      of `pleo_paths` order);
    - `data_merged`: `[R,160]` float32 = true rows + sum of overlays;
    - `manifest`: `n_files`, `n_rows_union`, `n_rows_moved_any` (|Delta| > 0
      in >= 1 fact), `n_rows_shared` (|Delta| > 0 in >= 2 facts), histogram of
      facts-per-row on moved rows (`facts_per_row_histogram`, string keys =
      number of facts moving that row), and for shared rows: `sum_abs_norm` =
      |Sum Delta| (norm of sum -- can cancel if facts pull in opposite
      directions) and `abs_sum_norm` = Sum |Delta| (sum of norms -- never
      cancels) per row, aggregated as mean/max; `max_row_delta_norm` = maximum
      |Sum Delta| over ALL rows (not just shared) compared to quantization
      floor `QUANT_FLOOR` (only recorded).
    """
    if not pleo_paths:
        raise ValueError("sum_overlays: no .pleo files provided")

    per_file = [delta_for_pleo(table, p) for p in pleo_paths]
    row_len = per_file[0][1].shape[1]

    rows_union_set: set[int] = set()
    for rows, _delta in per_file:
        rows_union_set.update(int(r) for r in rows.tolist())
    rows_union = np.array(sorted(rows_union_set), dtype=np.int32)
    row_index = {int(g): i for i, g in enumerate(rows_union)}
    n = rows_union.shape[0]

    delta_sum = np.zeros((n, row_len), dtype=np.float32)
    moved_count = np.zeros(n, dtype=np.int64)  # number of facts with |Delta_i| > 0 per row
    abs_sum = np.zeros(n, dtype=np.float64)  # Sum_i |Delta_i| (L2 norm per fact, summed)

    for rows, delta in per_file:
        row_norms = np.linalg.norm(delta.astype(np.float64), axis=1)
        moved_mask = row_norms > 0.0
        idx = np.array([row_index[int(g)] for g in rows.tolist()], dtype=np.int64)
        np.add.at(delta_sum, idx, delta)
        np.add.at(moved_count, idx[moved_mask], 1)
        np.add.at(abs_sum, idx[moved_mask], row_norms[moved_mask])

    true_union = D._read_true_rows(table, rows_union.tolist())
    data_merged = (true_union.astype(np.float32) + delta_sum).astype(np.float32)

    sum_abs_norm_per_row = np.linalg.norm(delta_sum.astype(np.float64), axis=1)
    moved_any_mask = moved_count >= 1
    shared_mask = moved_count >= 2

    row_hist: dict[str, int] = {}
    for c in moved_count[moved_any_mask].tolist():
        key = str(int(c))
        row_hist[key] = row_hist.get(key, 0) + 1

    shared_sum_abs = sum_abs_norm_per_row[shared_mask]
    shared_abs_sum = abs_sum[shared_mask]
    max_row_delta_norm = float(sum_abs_norm_per_row.max()) if n else 0.0

    manifest = {
        "n_files": len(pleo_paths),
        "n_rows_union": int(n),
        "n_rows_moved_any": int(moved_any_mask.sum()),
        "n_rows_shared": int(shared_mask.sum()),
        "facts_per_row_histogram": row_hist,
        "shared_rows_sum_abs_norm_mean": float(shared_sum_abs.mean()) if shared_sum_abs.size else None,
        "shared_rows_sum_abs_norm_max": float(shared_sum_abs.max()) if shared_sum_abs.size else None,
        "shared_rows_abs_sum_norm_mean": float(shared_abs_sum.mean()) if shared_abs_sum.size else None,
        "shared_rows_abs_sum_norm_max": float(shared_abs_sum.max()) if shared_abs_sum.size else None,
        "max_row_delta_norm": max_row_delta_norm,
        "quant_floor": QUANT_FLOOR,
        "max_row_delta_norm_over_quant_floor": max_row_delta_norm > QUANT_FLOOR,
    }
    return rows_union, data_merged, manifest

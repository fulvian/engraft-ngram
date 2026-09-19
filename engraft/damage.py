"""Collateral-damage measurement of an overlay: two subcommands.

`plan` (CPU, table only, no weights/GPU): for every row of the overlay
`.pleo` computes the delta against the true row (`rows_delta.npz`); scans the
large neutral corpus in blocks and finds the positions that read at least one
overlay row (`hits.npz`); samples blocks (`chunks.json`) for the real forward
-- `n_random` blocks at uniform offset, `n_hit` blocks centered on a TARGET
row (see below).

`run` (free-routing forward, fake in tests / GPU for a real measurement): two
forwards per block (base = true rows, student = overlay), `dnll`/`kl`/`flips`
per position (`per_chunk/chunk_NNNN.npz`), aggregated summary
(`damage.json`/`damage.md`).

Hit-block sampling: two strata by TARGET ROW (never an unchanged row) --
`top` = the `n_hit//2` CHANGED rows with the highest `read_count*dnorm`;
`tail` = `n_hit - n_top` CHANGED rows chosen at random (seed) among the rest
with `read_count>=2`. One block per target row, centered on one of its
occurrences chosen at random (seed) in the corpus -- never a repeated row
across blocks. (A private predecessor of this sampling stratified by
`dnorm_max` quartile; half the overlay's rows were unchanged and 64% of the
hits read only unchanged rows, wasting half the blocks on a base==student
bit-identical comparison -- the by-row-mass stratification above replaced
it.)

Position <-> logit-row convention (load-bearing, not obvious): `run` builds
`tokens = [eos] + corpus[start:start+chunk_len]` (length `chunk_len+1`).
`state.logits[r]` (r=0..chunk_len-1) predicts `tokens[r+1] =
corpus[start+r]` -- and the position whose 16 rows determine that prediction
is `tokens[r]`, i.e. (for r>=1) `corpus[start+r-1]`. So a corpus position
`pos` (the convention below: the rows read at `pos` predict `corpus[pos+1]`)
corresponds to logit row `r = pos - start + 1`. For `r<2` the context is
truncated by the opening EOS (intentional zeroing) and does NOT match the
corpus's real context there -- irrelevant here because "hit" blocks center
their position at `chunk_len//4` from the start, always >=2 tokens inside
the block.

Diagnostic subcommands the private original also had (`--diag-determinism`,
`--diag-layers`, `--diag-sublayer`) are not part of this port: they isolated
a specific non-determinism incident on the private fast path and
`--diag-layers` needs an intermediate per-sub-block capture inside
`Replica.run_layer` this reference's public interface does not expose (same
constraint documented for `engraft.eval`'s dropped `--ple-contesa`). Neither
diagnostic produced a number this repository publishes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from engraft.lens import read_pleo
import engraft.replica.distill as D

BLOCK_TOKENS = 8_000_000


# --------------------------------------------------------------------------
# Neutral-corpus scanning (offline addressing, no forward)
# --------------------------------------------------------------------------


def ngram_rows(tokens: np.ndarray, table) -> tuple[np.ndarray, np.ndarray]:
    """Global bigram [n,8] and trigram [n,8] rows for every position of
    `tokens`. Reproduces `table.ngram_addresses`'s uint64 hash, vectorized
    (numpy wraps the product modulo 2**64 without raising, like the scalar
    version's masking)."""
    n = tokens.shape[0]
    eos = int(table.eos_token_id)
    tok64 = tokens.astype(np.int64)
    idx = np.arange(n)

    x0 = tok64
    valid1 = idx >= 1
    x1 = np.where(valid1, tok64[np.clip(idx - 1, 0, None)], eos)
    cut1 = (~valid1) | (x1 == eos)
    valid2 = idx >= 2
    x2_raw = np.where(valid2, tok64[np.clip(idx - 2, 0, None)], eos)
    x2 = np.where(cut1, eos, x2_raw)

    mult = [np.uint64(m) for m in table.layer_multipliers]
    u0 = x0.astype(np.uint64)
    u1 = x1.astype(np.uint64)
    u2 = x2.astype(np.uint64)

    mixed_bi = (u0 * mult[0]) ^ (u1 * mult[1])
    mixed_tri = mixed_bi ^ (u2 * mult[2])

    heads_per_ngram = table.heads_per_ngram
    rows_big = np.empty((n, heads_per_ngram), dtype=np.int64)
    rows_tri = np.empty((n, heads_per_ngram), dtype=np.int64)
    for g in range(heads_per_ngram):
        h_bi = g
        vocab_bi = np.uint64(table.head_vocab_sizes[h_bi])
        rows_big[:, g] = (mixed_bi % vocab_bi).astype(np.int64) + table.head_offsets[h_bi]
        h_tri = heads_per_ngram + g
        vocab_tri = np.uint64(table.head_vocab_sizes[h_tri])
        rows_tri[:, g] = (mixed_tri % vocab_tri).astype(np.int64) + table.head_offsets[h_tri]
    return rows_big, rows_tri


def iter_blocks(tokens: np.ndarray, block: int = BLOCK_TOKENS) -> Iterator[tuple[int, np.ndarray]]:
    """(offset, tokens[offset-2:offset+block]) -- 2 tokens of overlap (0 on
    the first block)."""
    n = len(tokens)
    offset = 0
    first = True
    while offset < n:
        this_block = min(block, n - offset)
        start = offset if first else offset - 2
        yield offset, tokens[start:offset + this_block]
        offset += this_block
        first = False


# --------------------------------------------------------------------------
# Task 1.1 -- delta per overlay row
# --------------------------------------------------------------------------


def compute_rows_delta(table, rows_global: np.ndarray, data: np.ndarray) -> dict:
    """(dnorm, ratio, cos) per overlay row against the dequantized true row.

    `dnorm = ||ov-true||`; `ratio = ||ov||/||true||` (NaN if the true row is
    null, never observed on the real table but the formula stays defined);
    `cos` = cosine between ov and true (NaN if either has zero norm)."""
    true = D._read_true_rows(table, rows_global.tolist())
    diff = data.astype(np.float64) - true.astype(np.float64)
    dnorm = np.linalg.norm(diff, axis=1).astype(np.float32)
    norm_true = np.linalg.norm(true, axis=1).astype(np.float64)
    norm_ov = np.linalg.norm(data, axis=1).astype(np.float64)
    ratio = np.where(norm_true > 0, norm_ov / np.maximum(norm_true, 1e-30), np.nan).astype(np.float32)
    denom = norm_ov * norm_true
    cos = np.where(
        denom > 0,
        np.sum(data.astype(np.float64) * true.astype(np.float64), axis=1) / np.maximum(denom, 1e-30),
        np.nan,
    ).astype(np.float32)
    return {
        "rows_global": np.asarray(rows_global, dtype=np.int32),
        "dnorm": dnorm, "ratio": ratio, "cos": cos, "true": true,
    }


def rows_delta_summary(delta: dict) -> dict:
    dnorm = delta["dnorm"]
    ratio = delta["ratio"]
    cos = delta["cos"]
    q50, q90, q99 = np.percentile(dnorm, [50, 90, 99]).tolist()
    return {
        "n_rows": int(len(dnorm)),
        "dnorm_quantiles": {"p50": q50, "p90": q90, "p99": q99, "max": float(np.max(dnorm))},
        "n_ratio_gt_1_5": int(np.sum(np.nan_to_num(ratio, nan=0.0) > 1.5)),
        "n_cos_lt_0_5": int(np.sum(np.nan_to_num(cos, nan=1.0) < 0.5)),
    }


# --------------------------------------------------------------------------
# Task 1.2 -- neutral-corpus positions that read the overlay
# --------------------------------------------------------------------------


def scan_hits(tokens: np.ndarray, table, overlay_sorted: np.ndarray, dnorm_sorted: np.ndarray,
              block: int = BLOCK_TOKENS) -> dict:
    """Block scan: for every position its 16 rows (8 bigram + 8 trigram);
    `hit` = at least one in `overlay_sorted`. Returns, for every hit
    position, `pos`/`n_hit_rows`/`dnorm_max`/`dnorm_sum`, plus `read_count`
    (how many times each overlay row -- indexed like `overlay_sorted` -- was
    read, for the most-read-rows ranking).

    Excludes the corpus's last position (no following token)."""
    n = int(tokens.shape[0])
    n_ov = int(overlay_sorted.shape[0])
    read_count = np.zeros(n_ov, dtype=np.int64)
    pos_parts: list[np.ndarray] = []
    n_hit_rows_parts: list[np.ndarray] = []
    dnorm_max_parts: list[np.ndarray] = []
    dnorm_sum_parts: list[np.ndarray] = []

    t0 = time.time()
    for offset, blk in iter_blocks(tokens, block=block):
        rows_big, rows_tri = ngram_rows(blk, table)
        start_local = 0 if offset == 0 else 2
        rows16 = np.concatenate([rows_big[start_local:], rows_tri[start_local:]], axis=1)  # [m,16]
        if rows16.shape[0] == 0:
            continue
        idx = np.searchsorted(overlay_sorted, rows16)
        idx_c = np.clip(idx, 0, max(n_ov - 1, 0))
        matched = (n_ov > 0) & (idx < n_ov) & (overlay_sorted[idx_c] == rows16)
        if matched.any():
            flat_idx = idx_c[matched]
            np.add.at(read_count, flat_idx, 1)
            matched_dnorm = np.where(matched, dnorm_sorted[idx_c], 0.0)
            n_hit_rows = matched.sum(axis=1)
            hit_local = np.flatnonzero(n_hit_rows > 0)
            if hit_local.size:
                pos_here = (offset + hit_local).astype(np.int64)
                keep = pos_here < (n - 1)
                hit_local = hit_local[keep]
                pos_here = pos_here[keep]
                if hit_local.size:
                    pos_parts.append(pos_here)
                    n_hit_rows_parts.append(n_hit_rows[hit_local].astype(np.int32))
                    dnorm_max_parts.append(matched_dnorm[hit_local].max(axis=1).astype(np.float32))
                    dnorm_sum_parts.append(matched_dnorm[hit_local].sum(axis=1).astype(np.float32))
    dt = time.time() - t0

    def _cat(parts, dtype):
        return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

    return {
        "pos": _cat(pos_parts, np.int64),
        "n_hit_rows": _cat(n_hit_rows_parts, np.int32),
        "dnorm_max": _cat(dnorm_max_parts, np.float32),
        "dnorm_sum": _cat(dnorm_sum_parts, np.float32),
        "read_count": read_count,
        "scan_time_s": dt,
        "n_tokens": n,
    }


def hits_summary(hits: dict, overlay_sorted: np.ndarray, dnorm_sorted: np.ndarray,
                  ratio_sorted: np.ndarray, cos_sorted: np.ndarray, top_n: int = 20) -> dict:
    pos = hits["pos"]
    n_tokens = hits["n_tokens"]
    total_hits = int(pos.shape[0])
    hits_per_million = (total_hits / n_tokens * 1e6) if n_tokens else 0.0
    dnorm_max = hits["dnorm_max"]
    if dnorm_max.size:
        q = np.percentile(dnorm_max, [25, 50, 75, 100]).tolist()
        frac_hits_only_unchanged = float(np.mean(dnorm_max == 0.0))
    else:
        q = [0.0, 0.0, 0.0, 0.0]
        frac_hits_only_unchanged = 0.0
    read_count = hits["read_count"]
    changed_mask = dnorm_sorted > 0
    n_changed_rows = int(changed_mask.sum())
    n_unchanged_rows = int(overlay_sorted.shape[0] - n_changed_rows)
    n_changed_rows_read = int(np.sum(changed_mask & (read_count > 0)))
    order = np.argsort(-read_count)[:top_n]
    top_rows = [
        {
            "row_global": int(overlay_sorted[i]),
            "read_count": int(read_count[i]),
            "dnorm": float(dnorm_sorted[i]),
            "ratio": float(ratio_sorted[i]) if np.isfinite(ratio_sorted[i]) else None,
            "cos": float(cos_sorted[i]) if np.isfinite(cos_sorted[i]) else None,
        }
        for i in order.tolist()
    ]
    return {
        "total_hits": total_hits,
        "hits_per_million_tokens": hits_per_million,
        "dnorm_max_quartiles": {"p25": q[0], "p50": q[1], "p75": q[2], "p100": q[3]},
        "n_changed_rows": n_changed_rows, "n_unchanged_rows": n_unchanged_rows,
        "n_changed_rows_read": n_changed_rows_read,
        "frac_hits_only_unchanged": frac_hits_only_unchanged,
        "top_rows_by_read_count": top_rows,
        "scan_time_s": hits["scan_time_s"],
    }


# --------------------------------------------------------------------------
# Task 1.3 -- block sampling for the real forward
# --------------------------------------------------------------------------


def _hit_positions_within(pos_sorted: np.ndarray, dnorm_max_sorted: np.ndarray, lo: int, hi: int) -> list[dict]:
    """Hits in `[lo, hi)`: RELATIVE offset (`pos-lo`) + `dnorm_max` of each
    (`run` must be able to tell, inside a block, hits on UNCHANGED rows
    -- `dnorm_max==0` -- apart from changed ones, without reopening
    `hits.npz`)."""
    i0 = int(np.searchsorted(pos_sorted, lo, side="left"))
    i1 = int(np.searchsorted(pos_sorted, hi, side="left"))
    rel = (pos_sorted[i0:i1] - lo).astype(int).tolist()
    dn = dnorm_max_sorted[i0:i1].astype(float).tolist()
    return [{"pos_rel": r, "dnorm_max": d} for r, d in zip(rel, dn)]


def collect_occurrences_for_rows(tokens: np.ndarray, table, target_rows: np.ndarray,
                                  block: int = BLOCK_TOKENS) -> dict[int, list[int]]:
    """Block scan (same schema as `scan_hits`) for a SMALL, already-chosen
    set of target rows: every corpus position that reads them (full list, no
    aggregation) -- feeds `sample_chunks` to pick one occurrence per row at
    random."""
    target_sorted = np.unique(np.asarray(target_rows, dtype=np.int64))
    n_t = int(target_sorted.shape[0])
    occ: dict[int, list[int]] = {int(g): [] for g in target_sorted.tolist()}
    if n_t == 0:
        return occ
    n = int(tokens.shape[0])
    for offset, blk in iter_blocks(tokens, block=block):
        rows_big, rows_tri = ngram_rows(blk, table)
        start_local = 0 if offset == 0 else 2
        rows16 = np.concatenate([rows_big[start_local:], rows_tri[start_local:]], axis=1)
        if rows16.shape[0] == 0:
            continue
        idx = np.searchsorted(target_sorted, rows16)
        idx_c = np.clip(idx, 0, n_t - 1)
        matched = (idx < n_t) & (target_sorted[idx_c] == rows16)
        if not matched.any():
            continue
        rloc, cloc = np.nonzero(matched)
        positions = offset + rloc
        keep = positions < (n - 1)
        rloc, cloc, positions = rloc[keep], cloc[keep], positions[keep]
        rows_hit = target_sorted[idx_c[rloc, cloc]]
        for p, g in zip(positions.tolist(), rows_hit.tolist()):
            occ[int(g)].append(int(p))
    return occ


def sample_chunks(hits: dict, table, tokens: np.ndarray,
                   overlay_sorted: np.ndarray, dnorm_sorted: np.ndarray, cos_sorted: np.ndarray,
                   chunk_len: int, n_random: int, n_hit: int, seed: int) -> list[dict]:
    """`n_random` blocks at uniform offset + `n_hit` blocks, one per TARGET
    row:

    - Never an UNCHANGED row (`dnorm=0`) as a target -- by construction,
      targets are drawn only from `dnorm_sorted > 0`.
    - `top` stratum: the `n_hit//2` CHANGED rows with the highest
      `read_count*dnorm`.
    - `tail` stratum: `n_hit - n_top` CHANGED rows chosen at random (seed)
      among the rest with `read_count>=2` (excluding the `top` stratum's
      rows).
    - No repeated target row (top and tail are disjoint sets by
      construction; if the pools are smaller than requested, every
      available row is used and the shortfall is declared, never a silent
      error)."""
    rng = np.random.default_rng(seed)
    n_tokens = int(tokens.shape[0])
    max_start = n_tokens - chunk_len - 1  # -1: the corpus's last position is excluded from hits
    if max_start < 0:
        raise ValueError(f"sample_chunks: corpus too short ({n_tokens}) for chunk_len={chunk_len}")

    pos = hits["pos"]
    order_by_pos = np.argsort(pos)
    pos_sorted = pos[order_by_pos]
    dnorm_max_sorted_by_pos = hits["dnorm_max"][order_by_pos]

    chunks: list[dict] = []

    n_random_eff = min(n_random, max_start + 1)
    starts = sorted(rng.choice(max_start + 1, size=n_random_eff, replace=False).tolist())
    for s in starts:
        chunks.append({
            "start": int(s), "kind": "random",
            "hit_positions": _hit_positions_within(pos_sorted, dnorm_max_sorted_by_pos, s, s + chunk_len),
        })

    read_count = hits["read_count"]
    changed_idx = np.flatnonzero(dnorm_sorted > 0)
    if changed_idx.size == 0:
        return chunks  # no changed row read by the corpus: no hit block possible

    n_top = n_hit // 2
    n_tail = n_hit - n_top

    metric = read_count[changed_idx].astype(np.float64) * dnorm_sorted[changed_idx].astype(np.float64)
    top_order = np.argsort(-metric, kind="stable")
    top_sel = changed_idx[top_order[:n_top]]

    remaining = np.setdiff1d(changed_idx, top_sel, assume_unique=True)
    tail_pool = remaining[read_count[remaining] >= 2]
    n_tail_eff = min(n_tail, int(tail_pool.size))
    tail_sel = (rng.choice(tail_pool, size=n_tail_eff, replace=False)
                if n_tail_eff > 0 else np.zeros(0, dtype=np.int64))

    target_rows_all = np.concatenate([top_sel, tail_sel]).astype(np.int64)
    strata = ["top"] * len(top_sel) + ["tail"] * len(tail_sel)

    target_row_globals = overlay_sorted[target_rows_all]
    occ = collect_occurrences_for_rows(tokens, table, target_row_globals)

    center = chunk_len // 4
    for local_i, row_idx in enumerate(target_rows_all.tolist()):
        g = int(overlay_sorted[row_idx])
        positions = occ.get(g, [])
        if not positions:
            continue  # defensive: read_count>0 implies occurrences, should not happen
        p = int(rng.choice(positions))
        s = max(0, min(max_start, p - center))
        cos_v = cos_sorted[row_idx]
        chunks.append({
            "start": s, "kind": "hit", "stratum": strata[local_i],
            "target_row": g, "target_dnorm": float(dnorm_sorted[row_idx]),
            "target_cos": float(cos_v) if np.isfinite(cos_v) else None,
            "target_read_count": int(read_count[row_idx]),
            "center_pos": p,
            "hit_positions": _hit_positions_within(pos_sorted, dnorm_max_sorted_by_pos, s, s + chunk_len),
        })
    return chunks


# --------------------------------------------------------------------------
# Human-readable report
# --------------------------------------------------------------------------


def write_plan_report(out_dir: Path, delta_summary: dict, h_summary: dict, chunks: list[dict],
                       chunk_len: int, n_tokens: int) -> None:
    lines = [
        "# Collateral-damage plan", "",
        f"Neutral corpus: {n_tokens} tokens; chunk_len={chunk_len}.", "",
        "## Delta per row (overlay vs true rows)", "",
        f"- rows: {delta_summary['n_rows']}",
        f"- dnorm: p50={delta_summary['dnorm_quantiles']['p50']:.4f}, "
        f"p90={delta_summary['dnorm_quantiles']['p90']:.4f}, "
        f"p99={delta_summary['dnorm_quantiles']['p99']:.4f}, "
        f"max={delta_summary['dnorm_quantiles']['max']:.4f}",
        f"- rows with ratio>1.5: {delta_summary['n_ratio_gt_1_5']}",
        f"- rows with cos<0.5: {delta_summary['n_cos_lt_0_5']}", "",
        "## Positions reading the overlay (large neutral corpus)", "",
        f"- scan time: {h_summary['scan_time_s']:.1f} s",
        f"- total hits: {h_summary['total_hits']}",
        f"- hits per million tokens: {h_summary['hits_per_million_tokens']:.2f}",
        f"- CHANGED rows (dnorm>0): {h_summary['n_changed_rows']} "
        f"(read from corpus: {h_summary['n_changed_rows_read']}); UNCHANGED rows (dnorm=0): "
        f"{h_summary['n_unchanged_rows']}",
        f"- fraction of hits reading ONLY unchanged rows (base==student bit-identical, "
        f"no damage measurable there): {h_summary['frac_hits_only_unchanged']:.3f}",
        f"- dnorm_max quartiles (informative only -- no longer used for sampling, see module "
        f"docstring): p25={h_summary['dnorm_max_quartiles']['p25']:.4f}, "
        f"p50={h_summary['dnorm_max_quartiles']['p50']:.4f}, "
        f"p75={h_summary['dnorm_max_quartiles']['p75']:.4f}, "
        f"p100={h_summary['dnorm_max_quartiles']['p100']:.4f}", "",
        "### The 20 most-read rows from the neutral corpus", "",
        "| global row | reads | dnorm | ratio | cos |", "|---|---|---|---|---|",
    ]
    for r in h_summary["top_rows_by_read_count"]:
        ratio_s = f"{r['ratio']:.3f}" if r["ratio"] is not None else "n/a"
        cos_s = f"{r['cos']:.3f}" if r["cos"] is not None else "n/a"
        lines.append(f"| {r['row_global']} | {r['read_count']} | {r['dnorm']:.4f} | {ratio_s} | {cos_s} |")

    n_random = sum(1 for c in chunks if c["kind"] == "random")
    hit_chunks = [c for c in chunks if c["kind"] == "hit"]
    top_chunks = [c for c in hit_chunks if c["stratum"] == "top"]
    tail_chunks = [c for c in hit_chunks if c["stratum"] == "tail"]
    lines += [
        "", "## Sampled blocks", "",
        f"- random: {n_random}", f"- hit: {len(hit_chunks)} (top: {len(top_chunks)}, tail: {len(tail_chunks)})",
        f"- total: {len(chunks)}", "",
        "### Target rows, `top` stratum", "",
        "| global row | reads | dnorm | cos | center position |", "|---|---|---|---|---|",
    ]
    for c in sorted(top_chunks, key=lambda c: -c["target_read_count"] * c["target_dnorm"]):
        cos_s = f"{c['target_cos']:.3f}" if c["target_cos"] is not None else "n/a"
        lines.append(f"| {c['target_row']} | {c['target_read_count']} | {c['target_dnorm']:.4f} | "
                     f"{cos_s} | {c['center_pos']} |")
    lines += ["", "### Target rows, `tail` stratum", "",
              "| global row | reads | dnorm | cos | center position |", "|---|---|---|---|---|"]
    for c in sorted(tail_chunks, key=lambda c: c["target_row"]):
        cos_s = f"{c['target_cos']:.3f}" if c["target_cos"] is not None else "n/a"
        lines.append(f"| {c['target_row']} | {c['target_read_count']} | {c['target_dnorm']:.4f} | "
                     f"{cos_s} | {c['center_pos']} |")
    lines.append("")
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI -- plan
# --------------------------------------------------------------------------


def _load_table(fake: bool, table_path: str | None):
    if fake:
        from engraft.descend_corpus import _fake_replica_and_table
        _replica, _w, table, _step_fn, _forward_fn = _fake_replica_and_table()
        return table
    from engraft.table import PleTable
    if not table_path:
        raise SystemExit("a real run requires --table-path")
    return PleTable(table_path)


def cmd_plan(args: argparse.Namespace) -> int:
    table = _load_table(args.fake, args.table_path)
    rows_g, data = read_pleo(args.overlay)

    delta = compute_rows_delta(table, rows_g, data)
    d_summary = rows_delta_summary(delta)

    order = np.argsort(delta["rows_global"])
    overlay_sorted = delta["rows_global"][order]
    dnorm_sorted = delta["dnorm"][order]
    ratio_sorted = delta["ratio"][order]
    cos_sorted = delta["cos"][order]

    tokens = np.load(args.neutral_tokens, mmap_mode="r")
    hits = scan_hits(tokens, table, overlay_sorted, dnorm_sorted)
    h_summary = hits_summary(hits, overlay_sorted, dnorm_sorted, ratio_sorted, cos_sorted)

    chunks = sample_chunks(hits, table, tokens, overlay_sorted, dnorm_sorted, cos_sorted,
                            args.chunk_len, args.n_random, args.n_hit, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "rows_delta.npz", rows_global=delta["rows_global"], dnorm=delta["dnorm"],
              ratio=delta["ratio"], cos=delta["cos"])
    np.savez(out_dir / "hits.npz", pos=hits["pos"], n_hit_rows=hits["n_hit_rows"],
              dnorm_max=hits["dnorm_max"], dnorm_sum=hits["dnorm_sum"],
              overlay_sorted=overlay_sorted, read_count=hits["read_count"])
    (out_dir / "chunks.json").write_text(json.dumps({
        "meta": {
            "overlay": str(args.overlay), "neutral_tokens": str(args.neutral_tokens),
            "chunk_len": args.chunk_len, "n_random": args.n_random, "n_hit": args.n_hit,
            "seed": args.seed, "n_tokens": hits["n_tokens"], "fake": bool(args.fake),
        },
        "chunks": chunks,
    }, indent=2))
    write_plan_report(out_dir, d_summary, h_summary, chunks, args.chunk_len, hits["n_tokens"])

    print(f"rows_delta: n={d_summary['n_rows']} dnorm_p50={d_summary['dnorm_quantiles']['p50']:.4f} "
          f"dnorm_max={d_summary['dnorm_quantiles']['max']:.4f}")
    print(f"hits: total={h_summary['total_hits']} per_million={h_summary['hits_per_million_tokens']:.2f} "
          f"time={h_summary['scan_time_s']:.1f}s")
    print(f"changed rows={h_summary['n_changed_rows']} (read={h_summary['n_changed_rows_read']}) "
          f"unchanged={h_summary['n_unchanged_rows']} frac_hit_only_unchanged="
          f"{h_summary['frac_hits_only_unchanged']:.3f}")
    n_hit_chunks = sum(1 for c in chunks if c["kind"] == "hit")
    n_top = sum(1 for c in chunks if c["kind"] == "hit" and c["stratum"] == "top")
    n_tail = sum(1 for c in chunks if c["kind"] == "hit" and c["stratum"] == "tail")
    print(f"chunks: {len(chunks)} written to {out_dir / 'chunks.json'} (hit={n_hit_chunks}: top={n_top} tail={n_tail})")
    print(f"plan.md: {out_dir / 'plan.md'}")
    return 0


# --------------------------------------------------------------------------
# CLI -- run
# --------------------------------------------------------------------------


def _kl_and_nll(logits_base: torch.Tensor, logits_student: torch.Tensor, target: torch.Tensor,
                 chunk_rows: int = 128, k_top: int = 3) -> tuple:
    """`dnll`/`kl` PER ROW, in blocks of `chunk_rows` rows (a whole-tensor
    `[T,V]` `log_softmax`/`exp`/difference with V~250k and T=2048 creates
    several ~2 GB temporaries -- OOM with a resident model at high VRAM
    occupancy). Each block computes `log_softmax` of base/student ONLY on
    those rows, also extracts the top-`k_top` (id, p) of both (serves the
    "worst N with context" report without ever holding the full logp again),
    moves everything to CPU/numpy and frees the block's temporaries (`del`)
    before the next block.

    Returns `dnll`, `kl` (numpy `[T]`, float32) and `top3_base_ids`/
    `top3_base_p`/`top3_student_ids`/`top3_student_p` (numpy `[T,k_top]`)."""
    n = int(logits_base.shape[0])
    dnll = np.empty(n, dtype=np.float32)
    kl = np.empty(n, dtype=np.float32)
    top3_base_ids = np.empty((n, k_top), dtype=np.int64)
    top3_base_p = np.empty((n, k_top), dtype=np.float32)
    top3_student_ids = np.empty((n, k_top), dtype=np.int64)
    top3_student_p = np.empty((n, k_top), dtype=np.float32)

    for lo in range(0, n, chunk_rows):
        hi = min(n, lo + chunk_rows)
        logp_b = torch.log_softmax(logits_base[lo:hi], dim=-1)
        logp_a = torch.log_softmax(logits_student[lo:hi], dim=-1)
        tgt = target[lo:hi]
        nll_b = -logp_b.gather(1, tgt.unsqueeze(1)).squeeze(1)
        nll_a = -logp_a.gather(1, tgt.unsqueeze(1)).squeeze(1)
        dnll_chunk = nll_a - nll_b
        p_b = torch.exp(logp_b)
        kl_chunk = (p_b * (logp_b - logp_a)).sum(dim=-1)
        vb, ib = torch.topk(logp_b, k_top, dim=-1)
        va, ia = torch.topk(logp_a, k_top, dim=-1)

        dnll[lo:hi] = dnll_chunk.detach().cpu().numpy()
        kl[lo:hi] = kl_chunk.detach().cpu().numpy()
        top3_base_ids[lo:hi] = ib.detach().cpu().numpy()
        top3_base_p[lo:hi] = torch.exp(vb).detach().cpu().numpy()
        top3_student_ids[lo:hi] = ia.detach().cpu().numpy()
        top3_student_p[lo:hi] = torch.exp(va).detach().cpu().numpy()

        del logp_b, logp_a, tgt, nll_b, nll_a, dnll_chunk, p_b, kl_chunk, vb, ib, va, ia

    return dnll, kl, top3_base_ids, top3_base_p, top3_student_ids, top3_student_p


def _flips_per_position(base_cap: dict, student_cap: dict, n_prefix: int) -> np.ndarray:
    """`capture_routing` captures arrays PER LAYER: `torch.Tensor` on the F32
    (`seq_forward`) or Triton path, indistinctly -- normalized here to numpy
    before comparing."""
    flips = np.zeros(n_prefix, dtype=np.int32)
    for il in sorted(set(base_cap) & set(student_cap)):
        b, a = base_cap[il], student_cap[il]
        b_np = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
        a_np = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        diff = (b_np != a_np).any(axis=-1)
        flips += diff.astype(np.int32)
    return flips


def _top3_from_arrays(ids_row: np.ndarray, p_row: np.ndarray) -> list[dict]:
    return [{"token": int(t), "p": float(p)} for t, p in zip(ids_row.tolist(), p_row.tolist())]


def _decode_context(decode_fn, tok_list: list[int], r: int, n_before: int = 20) -> str | None:
    if decode_fn is None:
        return None
    lo = max(0, r - n_before + 1)
    return "".join(decode_fn(t) for t in tok_list[lo:r + 1])


def cmd_run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan) / "chunks.json"
    plan = json.loads(plan_path.read_text())
    meta = plan["meta"]
    chunk_len = meta["chunk_len"]
    chunks = plan["chunks"]

    rows_g, data = read_pleo(args.overlay)
    row_map = {int(g): i for i, g in enumerate(rows_g.tolist())}
    rows_var_overlay = torch.from_numpy(np.asarray(data, dtype=np.float32))

    if args.fake:
        from engraft.descend_corpus import _fake_replica_and_table
        replica, w, table, _step_fn, forward_fn = _fake_replica_and_table(head_dtype=args.head_dtype)
    else:  # pragma: no cover -- requires a real GGUF/CUDA device
        from engraft.descend_corpus import load_real_backend
        replica, w, table, _step_fn, forward_fn = load_real_backend(args)

    decode_fn = None
    if not args.fake:  # pragma: no cover -- requires a real GGUF/CUDA device
        from engraft.config import load as load_config
        from engraft.table import PleTokenizer
        engraft_cfg = load_config(args.config)
        tok = PleTokenizer(engraft_cfg.get_path("model.tokenizer"))
        decode_fn = tok.decode_token

    tokens = np.load(args.neutral_tokens, mmap_mode="r")
    eos = int(table.eos_token_id)
    # The base forward must go through the SAME `OverlayEmb`/`index_put` as
    # the student (never `row_map={}`), on the SAME rows -- with TRUE
    # values, so base/student differ ONLY in the overlay's content (same or
    # different rows), never in the embedding construction path.
    rows_var_true = torch.from_numpy(D._read_true_rows(table, rows_g.tolist()).astype(np.float32))
    dev = getattr(getattr(replica, "backend", None), "device", "cpu")
    rows_var_overlay = rows_var_overlay.to(dev)
    rows_var_true = rows_var_true.to(dev)

    out_dir = Path(args.out)
    per_chunk_dir = out_dir / "per_chunk"
    per_chunk_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = len(chunks) if args.n_chunks_max is None else min(len(chunks), args.n_chunks_max)
    cap_s = args.gpu_minutes_cap * 60.0
    t_start = time.time()
    done_meta: list[dict] = []
    forward_times: list[float] = []
    global_records: list[dict] = []  # {"dnll":..,"kl":.., "chunk_i":.., "r":..}
    hit_records: list[dict] = []  # one record per hit block (target row)
    control_records: list[dict] = []  # hits on UNCHANGED rows inside a hit block (control)

    is_cuda = torch.cuda.is_available() and str(dev).startswith("cuda")

    n_chunks_done = 0
    for i, ch in enumerate(chunks[:n_chunks]):
        if time.time() - t_start > cap_s:
            break
        start = int(ch["start"])
        sub = tokens[start:start + chunk_len]
        tok_list = [eos] + [int(x) for x in sub.tolist()]

        base_cap: dict = {}
        student_cap: dict = {}
        t0 = time.time()
        with torch.no_grad():
            state_base, _ = forward_fn(
                replica, w, tok_list, rows_var_true, row_map, return_logits=True, grad_proxy=False,
                capture_routing=base_cap,
            )
            # `--rbr`: `base_cap`, already populated by the base forward
            # above, goes DIRECTLY to `routing_source` -- the only parameter
            # that forces `_routing_for` to reuse a given routing instead of
            # computing it live. Without `--rbr` (default) `routing_source`
            # stays `None`, free routing, today's behavior.
            state_student, _ = forward_fn(
                replica, w, tok_list, rows_var_overlay, row_map, return_logits=True, grad_proxy=False,
                capture_routing=student_cap,
                routing_source=base_cap if getattr(args, "rbr", False) else None,
            )
            dt = time.time() - t0
            forward_times.append(dt)

            logits_base = state_base.logits.detach().to(torch.float32)
            logits_student = state_student.logits.detach().to(torch.float32)
            n_prefix = logits_base.shape[0]
            target = torch.tensor(tok_list[1:], dtype=torch.long, device=logits_base.device)
            dnll, kl, top3_base_ids, top3_base_p, top3_student_ids, top3_student_p = _kl_and_nll(
                logits_base, logits_student, target, chunk_rows=args.kl_rows_chunk,
            )
            flips = _flips_per_position(base_cap, student_cap, n_prefix)

        del state_base, state_student, logits_base, logits_student, target
        if is_cuda:
            torch.cuda.empty_cache()

        np.savez_compressed(
            per_chunk_dir / f"chunk_{i:04d}.npz",
            start=start, dnll=dnll, kl=kl, flips=flips, forward_s=dt,
        )
        rec_meta = {
            "i": i, "start": start, "kind": ch["kind"], "stratum": ch.get("stratum"),
            "target_row": ch.get("target_row"),
            "center_pos": ch.get("center_pos"), "n_hit_positions": len(ch.get("hit_positions", [])),
            "forward_s": dt,
        }
        done_meta.append(rec_meta)
        n_chunks_done += 1

        if ch["kind"] == "random":
            for r in range(n_prefix):
                global_records.append({"i": i, "start": start, "r": int(r), "dnll": float(dnll[r]),
                                        "kl": float(kl[r])})
        else:  # "hit": one block = one target row (never an unchanged row, see plan)
            target_r = int(ch["center_pos"]) - start + 1  # pos -> logit row, see module docstring
            if 0 <= target_r < n_prefix:
                lo1, hi1 = target_r + 1, min(n_prefix, target_r + 9)
                lo2, hi2 = target_r + 9, min(n_prefix, target_r + 65)
                hit_records.append({
                    "i": i, "start": start, "stratum": ch.get("stratum"), "target_row": ch.get("target_row"),
                    "target_dnorm": ch.get("target_dnorm"), "target_cos": ch.get("target_cos"),
                    "target_read_count": ch.get("target_read_count"),
                    "r": target_r, "dnll": float(dnll[target_r]), "kl": float(kl[target_r]),
                    "dnll_p1_8": float(dnll[lo1:hi1].mean()) if hi1 > lo1 else None,
                    "kl_p1_8": float(kl[lo1:hi1].mean()) if hi1 > lo1 else None,
                    "dnll_p9_64": float(dnll[lo2:hi2].mean()) if hi2 > lo2 else None,
                    "kl_p9_64": float(kl[lo2:hi2].mean()) if hi2 > lo2 else None,
                })
            # controls: hits on UNCHANGED rows (dnorm_max==0) in the same
            # block -- expected: kl==0 upstream of the target row (causal
            # locality); this is only an observational control, not an
            # assertion.
            for hp in ch.get("hit_positions", []):
                if float(hp.get("dnorm_max", 0.0)) == 0.0:
                    r = int(hp["pos_rel"]) + 1
                    if 0 <= r < n_prefix:
                        control_records.append({
                            "i": i, "start": start, "r": r, "kl": float(kl[r]),
                            "upstream_of_target": r < target_r,
                        })

        # "worst 20 with context" report: keeps the top-K by absolute dnll
        # (random blocks only) with decoded context.
        if ch["kind"] == "random":
            k_worst = min(20, n_prefix)
            worst_idx = np.argpartition(-np.abs(dnll), k_worst - 1)[:k_worst] if n_prefix else np.zeros(0, dtype=int)
            for r in worst_idx.tolist():
                global_records[-n_prefix + r]["context"] = _decode_context(decode_fn, tok_list, r)
                global_records[-n_prefix + r]["top3_base"] = _top3_from_arrays(top3_base_ids[r], top3_base_p[r])
                global_records[-n_prefix + r]["top3_student"] = _top3_from_arrays(
                    top3_student_ids[r], top3_student_p[r])
                global_records[-n_prefix + r]["target"] = int(tok_list[r + 1])

    total_s = time.time() - t_start

    damage = _build_damage_summary(
        global_records, hit_records, control_records, done_meta, forward_times, total_s,
        n_chunks_requested=len(chunks), n_chunks_done=n_chunks_done,
        gpu_minutes_cap=args.gpu_minutes_cap,
        routing_mode="locked" if getattr(args, "rbr", False) else "free",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "damage.json").write_text(json.dumps(damage, indent=2, default=str))
    _write_damage_md(out_dir, damage)
    print(f"run: n_chunks_done={n_chunks_done}/{len(chunks)} total_time={total_s:.1f}s")
    print(f"damage.json/damage.md written to {out_dir}")
    return 0


def _stratum_summary(recs: list[dict]) -> dict:
    at_hit_dnll = np.array([r["dnll"] for r in recs], dtype=np.float64)
    at_hit_kl = np.array([r["kl"] for r in recs], dtype=np.float64)

    def _mean_of(key: str) -> float | None:
        vals = [r[key] for r in recs if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n": len(recs),
        "dnll_at_hit_mean": float(at_hit_dnll.mean()), "kl_at_hit_mean": float(at_hit_kl.mean()),
        "dnll_p1_8_mean": _mean_of("dnll_p1_8"), "kl_p1_8_mean": _mean_of("kl_p1_8"),
        "dnll_p9_64_mean": _mean_of("dnll_p9_64"), "kl_p9_64_mean": _mean_of("kl_p9_64"),
    }


def _build_damage_summary(global_records, hit_records, control_records, done_meta, forward_times, total_s,
                           n_chunks_requested, n_chunks_done, gpu_minutes_cap, routing_mode: str = "free") -> dict:
    dnll_g = np.array([r["dnll"] for r in global_records], dtype=np.float64)
    kl_g = np.array([r["kl"] for r in global_records], dtype=np.float64)
    global_stats = {}
    if kl_g.size:
        global_stats = {
            "n_positions": int(kl_g.size),
            "dnll_mean": float(dnll_g.mean()), "dnll_median": float(np.median(dnll_g)),
            "kl_mean": float(kl_g.mean()), "kl_median": float(np.median(kl_g)),
            "frac_kl_gt_0_01": float(np.mean(kl_g > 0.01)),
            "frac_kl_gt_0_1": float(np.mean(kl_g > 0.1)),
            "frac_kl_gt_1": float(np.mean(kl_g > 1.0)),
        }
        worst = sorted(
            (r for r in global_records if "context" in r or "top3_base" in r),
            key=lambda r: -abs(r["dnll"]),
        )[:20]
        global_stats["worst_20"] = [dict(r) for r in worst]
    else:
        global_stats = {"n_positions": 0}

    hit_stats: dict = {}
    if hit_records:
        by_stratum = {}
        for stratum in ("top", "tail"):
            recs = [r for r in hit_records if r.get("stratum") == stratum]
            if recs:
                by_stratum[stratum] = _stratum_summary(recs)
        worst_hit = sorted(hit_records, key=lambda r: -abs(r["dnll"]))[:20]
        hit_stats = {
            "by_stratum": by_stratum,
            "per_target_row": hit_records,  # one record per block/target row, already compact
            "worst_20": worst_hit,
        }

    control_stats: dict = {}
    if control_records:
        upstream = [r for r in control_records if r["upstream_of_target"]]
        downstream = [r for r in control_records if not r["upstream_of_target"]]
        eps = 1e-6
        upstream_violations = [r for r in upstream if abs(r["kl"]) > eps]
        control_stats = {
            "n_controls": len(control_records),
            "n_upstream": len(upstream), "n_downstream": len(downstream),
            "n_upstream_violations": len(upstream_violations),
            "upstream_violations_examples": upstream_violations[:10],
            "downstream_kl_mean": float(np.mean([r["kl"] for r in downstream])) if downstream else None,
        }

    return {
        "n_chunks_requested": n_chunks_requested, "n_chunks_done": n_chunks_done,
        "gpu_minutes_cap": gpu_minutes_cap, "total_s": total_s,
        "forward_s_mean": float(np.mean(forward_times)) if forward_times else None,
        "chunks": done_meta,
        "global": global_stats, "hits": hit_stats, "controls": control_stats,
        # "free" (default, today's behavior) or "locked" (--rbr:
        # routing_source=base_cap, flips=0 by construction) -- the mode
        # actually used by the student forward.
        "routing_mode": routing_mode,
    }


def _write_damage_md(out_dir: Path, damage: dict) -> None:
    g = damage["global"]

    def fmt(x):
        return f"{x:.4f}" if x is not None else "n/a"

    lines = [
        "# Collateral damage -- summary", "",
        f"Blocks requested: {damage['n_chunks_requested']}, done: {damage['n_chunks_done']} "
        f"(cap {damage['gpu_minutes_cap']} min).",
        f"Total time: {damage['total_s']:.1f} s; mean forward: "
        f"{damage['forward_s_mean']:.2f} s" if damage["forward_s_mean"] else "", "",
        f"Student forward routing: **{damage.get('routing_mode', 'free')}** "
        + ("(`--rbr`: routing_source=base_cap, flips=0 by construction)."
           if damage.get("routing_mode") == "locked"
           else "(default, live routing -- same as earlier measurements)."),
        "",
        "## Global (random blocks)", "",
    ]
    if g.get("n_positions"):
        lines += [
            f"- positions: {g['n_positions']}",
            f"- dnll: mean={g['dnll_mean']:.4f}, median={g['dnll_median']:.4f}",
            f"- kl: mean={g['kl_mean']:.4f}, median={g['kl_median']:.4f}",
            f"- fraction kl>0.01/0.1/1: {g['frac_kl_gt_0_01']:.3f} / {g['frac_kl_gt_0_1']:.3f} / "
            f"{g['frac_kl_gt_1']:.3f}",
        ]
    else:
        lines.append("(no random block run)")

    lines += ["", "## At the hits, by stratum (target rows: `top`/`tail`)", ""]
    h = damage.get("hits") or {}
    if h.get("by_stratum"):
        lines.append("| stratum | n | dnll@hit | kl@hit | dnll +1..+8 | kl +1..+8 | dnll +9..+64 | kl +9..+64 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for stratum, s in sorted(h["by_stratum"].items()):
            lines.append(
                f"| {stratum} | {s['n']} | {fmt(s['dnll_at_hit_mean'])} | {fmt(s['kl_at_hit_mean'])} | "
                f"{fmt(s['dnll_p1_8_mean'])} | {fmt(s['kl_p1_8_mean'])} | "
                f"{fmt(s['dnll_p9_64_mean'])} | {fmt(s['kl_p9_64_mean'])} |"
            )
        lines += ["", "### Target rows, `top` stratum", "",
                   "| row | reads (/corpus) | dnorm | cos | dnll@hit | kl@hit | kl +1..+8 |",
                   "|---|---|---|---|---|---|---|"]
        top_rows = sorted(
            (r for r in h["per_target_row"] if r.get("stratum") == "top"),
            key=lambda r: -(r.get("target_read_count") or 0) * (r.get("target_dnorm") or 0.0),
        )
        for r in top_rows:
            cos_s = fmt(r.get("target_cos"))
            lines.append(
                f"| {r['target_row']} | {r.get('target_read_count')} | {fmt(r.get('target_dnorm'))} | "
                f"{cos_s} | {fmt(r['dnll'])} | {fmt(r['kl'])} | {fmt(r.get('kl_p1_8'))} |"
            )
    else:
        lines.append("(no hit block run)")

    lines += ["", "## Controls (hits on UNCHANGED rows inside a hit block)", ""]
    c = damage.get("controls") or {}
    if c.get("n_controls"):
        lines += [
            f"- total controls: {c['n_controls']} (upstream of the target row: {c['n_upstream']}, "
            f"downstream: {c['n_downstream']})",
            f"- upstream violations (kl!=0 where 0 was expected, causal locality): {c['n_upstream_violations']}",
            f"- mean kl of downstream controls (propagation expected, no constraint): "
            f"{fmt(c['downstream_kl_mean'])}",
        ]
    else:
        lines.append("(no control -- no hit on an unchanged row inside the executed blocks)")
    lines.append("")
    (out_dir / "damage.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--memory-fraction", type=float, default=0.8)
    p.add_argument("--working-set-gb", type=float, default=2.0)
    p.add_argument("--dequant-cache-gb", type=float, default=24.0)
    p.add_argument("--dense-dtype", default="f32", choices=["f32", "bf16"])
    p.add_argument("--head-dtype", default="f32", choices=["f32", "bf16"],
                   help="GEMM input dtype of the vocabulary head (explicit cast, output ALWAYS F32); "
                        "default f32 unchanged")
    p.add_argument("--wdot", default="split", choices=["bf16", "split", "f32"])
    p.add_argument("--delta-chunk-size", type=int, default=64)
    p.add_argument("--min-avail-gb", type=float, default=100.0)
    p.add_argument("--table-path", default=None, help="GGUF n-gram table shard path (real run only)")
    p.add_argument("--shard-paths", nargs="+", default=None, help="GGUF weight shard paths (real run only)")
    p.add_argument("--config", default=None, help="engraft.toml path (real run only, tokenizer resolution)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--overlay", required=True)
    p_plan.add_argument("--neutral-tokens", required=True)
    p_plan.add_argument("--chunk-len", type=int, default=2048)
    p_plan.add_argument("--n-random", type=int, default=16)
    p_plan.add_argument("--n-hit", type=int, default=32)
    p_plan.add_argument("--seed", type=int, default=0)
    p_plan.add_argument("--out", required=True)
    p_plan.add_argument("--fake", action="store_true")
    p_plan.add_argument("--table-path", default=None, help="GGUF n-gram table shard path (real run only)")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run")
    p_run.add_argument("--plan", required=True)
    p_run.add_argument("--overlay", required=True)
    p_run.add_argument("--neutral-tokens", required=True)
    p_run.add_argument("--n-chunks-max", type=int, default=None)
    p_run.add_argument("--gpu-minutes-cap", type=float, default=45.0)
    p_run.add_argument("--kl-rows-chunk", type=int, default=128,
                        help="rows per block in _kl_and_nll (log_softmax/exp on the whole [T,V] with "
                             "V~250k creates too many temporaries)")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--fake", action="store_true")
    p_run.add_argument("--rbr", action="store_true",
                        help="locked routing in the damage measurement: the student forward uses "
                             "EXACTLY the base forward's routing from the same block "
                             "(routing_source=base_cap, capture_routing already used for flips) instead "
                             "of free routing -- flips=0 by construction, isolates row-value damage from "
                             "router-flip damage. Default off (free routing, today's behavior); the mode "
                             "used is declared in damage.md.")
    _add_backend_args(p_run)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

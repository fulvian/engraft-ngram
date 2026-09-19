"""Corpus statistics for the language comparison (`data/quail/results/languages/`),
recomputed from public data only: the corpus files under `data/quail/corpus/<cell>/`
and the public `engraft.descend_corpus.compute_fact_weights` function.

For each cell (`s0b` Italian, `z0b` Chinese -- corpus directory `z0b-preliminary`):

- `n_train_fragments`: count of `usage_corpus_resolved.json` fragments with
  `split == "train"` and `ok == True` (the set `compute_fact_weights` and the
  descent itself train on).
- `rows_per_fragment`: `len(census.json["row_sets"]["all_read"])` (the count of
  DISTINCT n-gram table rows the corpus's train fragments read, deduplicated)
  divided by `n_train_fragments`. This is an average, not a per-fragment
  count -- a row read by several fragments is counted once in the numerator
  and once per fragment that reads it in the denominator's implicit sum, so
  the ratio reads as "how many of the distinct rows a fragment newly
  contributes coverage of, on average".
- `mass_median`: `compute_fact_weights(train_frags, n_excl)["mass_median"]`
  (the median, over the 100 facts, of the total loss-mask-admitted token
  mass across that fact's train fragments -- see
  `engraft.descend_corpus.compute_fact_weights` docstring). `n_excl=9` is
  the published value for `s0b` (`data/quail/config/s0b.json`); no public
  config exists for `z0b`, so the same `n_excl=9` is used there too (stated,
  not hidden) -- and it reproduces the z0b `mass_median` already published
  in `results/languages/fact_balance.json`'s `per_cell.z0b.mass_stats`
  exactly, which corroborates the choice.
- `contention`: `win_heavier`/`n_cmp` from
  `results/languages/fact_balance.json`'s `per_cell.<cell>`, when not null
  (s0b has both; z0b's are published as `null`, sourced from a private path
  -- reported as "not derivable from public data", not guessed at).

This script only reads files already in this repository; it writes
`data/quail/results/languages/corpus_stats.json`.

Usage: `python scripts/language_corpus_stats.py` from the repository root.
"""
from __future__ import annotations

import json
from pathlib import Path

from engraft.descend_corpus import compute_fact_weights

ROOT = Path(__file__).resolve().parent.parent

CELLS = {
    "s0b": {"corpus_dir": "data/quail/corpus/s0b", "n_excl": 9},
    "z0b": {"corpus_dir": "data/quail/corpus/z0b-preliminary", "n_excl": 9},
}


def _load(rel: str):
    return json.loads((ROOT / rel).read_text())


def stats_for_cell(cell: str, corpus_dir: str, n_excl: int) -> dict:
    usage = _load(f"{corpus_dir}/usage_corpus_resolved.json")
    census = _load(f"{corpus_dir}/census.json")
    train_frags = [f for f in usage["fragments"] if f["split"] == "train" and f["ok"]]
    n_train = len(train_frags)

    n_rows = len(census["row_sets"]["all_read"])
    rows_per_fragment = n_rows / n_train

    _weights, _mass, mass_stats = compute_fact_weights(train_frags, n_excl)

    balance_cell = _load("data/quail/results/languages/fact_balance.json")["per_cell"].get(cell, {})
    win_heavier = balance_cell.get("win_heavier")
    n_cmp = balance_cell.get("n_cmp")
    if win_heavier is None or n_cmp is None:
        contention = "not derivable from public data"
    else:
        contention = f"{win_heavier}/{n_cmp}"

    return {
        "n_train_fragments": n_train,
        "n_rows_all_read": n_rows,
        "rows_per_fragment": round(rows_per_fragment, 1),
        "mass_median": mass_stats["mass_median"],
        "n_excl_used": n_excl,
        "contention": contention,
    }


def main() -> int:
    out = {cell: stats_for_cell(cell, cfg["corpus_dir"], cfg["n_excl"]) for cell, cfg in CELLS.items()}
    out_path = ROOT / "data/quail/results/languages/corpus_stats.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwritten: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

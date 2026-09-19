"""Recomputes the numbers in `paper/engraft.tex`'s Table 1 (`tab:cells`) that
have a public file behind them, and compares the recomputed value against
the number actually written in the paper. Companion to
`scripts/check_readme_numbers.py` (same design, different source document);
see that module's docstring for the general approach.

Table 1 has seven rows: `base`, three non-public pairs (`seed N, captured`
and `seed N, released`, N in {0, 1}) and two public rows (`seed N,
mass-weighted`, which are cells `s0b`/`s1b`, released with this repository
-- see the table's own caption). Only `base` and the two mass-weighted rows
are checked here: the four `captured`/`released` rows have no public file
(no per-step log of the actual step count reached, and no public
`engine_results.json` for those specific runs) and are reported as "not in
public data", not silently skipped and not guessed at. Within the two
public rows, the `steps` column is *also* "not in public data": the public
`config/s0b.json`/`s1b.json` carry `max_steps` (a cap), not the actual step
count the run stopped at, so it is not the same number as the table's
`steps` column.

Usage: `python scripts/check_paper_numbers.py` from the repository root.
Exit code 0 iff every checkable entry reads "same" (rows/columns without a
public file never fail the run -- they are reported, not checked).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "engraft.tex"


class NotPublicData(Exception):
    """Raised by a compute() that has no public file to recompute from."""


def _load(rel_path: str):
    return json.loads((ROOT / rel_path).read_text())


def _engine_results(cell: str):
    return _load(f"data/quail/results/{cell}/engine_results.json")


def _macro_exact_match(rows: list, column: str) -> float:
    """Mean, over the 100 facts, of that fact's own exact-match rate
    (`column` is 'student' or 'base'). A fragment can name more than one
    fact_id; it counts toward every fact it names, matching the paper's
    "macro per fact" framing (one number per fact, then averaged)."""
    by_fact: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        hit = row["greedy"][column]["exact_match"]
        for fact_id in row["fact_ids"]:
            by_fact[fact_id].append(hit)
    rates = [sum(v) / len(v) for v in by_fact.values()]
    return sum(rates) / len(rates)


def c_base_row():
    d = _engine_results("s0b")  # base column is identical across cells sharing the test set
    n = len(d)
    rank1 = sum(1 for x in d if x["base"]["rank_first"] == 1) / n
    em = sum(1 for x in d if x["greedy"]["base"]["exact_match"]) / n
    macro = _macro_exact_match(d, "base")
    return (f"{rank1:.3f}", f"{em:.3f}", f"{macro:.3f}", "0")


def c_mass_weighted_row(cell: str):
    d = _engine_results(cell)
    n = len(d)
    rank1 = sum(1 for x in d if x["student"]["rank_first"] == 1) / n
    em = sum(1 for x in d if x["greedy"]["student"]["exact_match"]) / n
    macro = _macro_exact_match(d, "student")
    kl = _load(f"data/quail/results/{cell}/damage_it_text.json")["global"]["kl_mean"]
    return (f"{rank1:.3f}", f"{em:.3f}", f"{macro:.3f}", f"{kl:.4f}")


def c_not_public(*_a, **_k):
    raise NotPublicData


CHECKS = [
    dict(
        label="Table 1: base row (first token rank1, exact match, exact match macro, KL)",
        paper_pattern=r"base & -- & -- & ([\d.]+) & ([\d.]+) & ([\d.]+) & (\d) \\\\",
        compute=c_base_row,
        public=True,
    ),
    dict(
        label="Table 1: seed 0, captured -- steps and all four metrics",
        paper_pattern=r"seed 0, captured & captured throughout & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\",
        compute=c_not_public,
        public=False,
    ),
    dict(
        label="Table 1: seed 1, captured -- steps and all four metrics",
        paper_pattern=r"seed 1, captured & captured throughout & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\",
        compute=c_not_public,
        public=False,
    ),
    dict(
        label="Table 1: seed 0, released -- steps and all four metrics",
        paper_pattern=r"seed 0, released & captured to 140, then free & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\",
        compute=c_not_public,
        public=False,
    ),
    dict(
        label="Table 1: seed 1, released -- steps and all four metrics",
        paper_pattern=r"seed 1, released & captured to 140, then free & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\",
        compute=c_not_public,
        public=False,
    ),
    dict(
        label="Table 1: seed 0, mass-weighted -- steps (not in public data)",
        paper_pattern=r"seed 0, mass-weighted & captured to 140, then free & (\d+) & [\d.]+ & [\d.]+ & [\d.]+ & [\d.]+ \\\\",
        compute=c_not_public,
        public=False,
    ),
    dict(
        label="Table 1: seed 0, mass-weighted (s0b) -- first token rank1, exact match, exact match macro, KL",
        paper_pattern=r"seed 0, mass-weighted & captured to 140, then free & \d+ & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\",
        compute=lambda: c_mass_weighted_row("s0b"),
        public=True,
    ),
    dict(
        label="Table 1: seed 1, mass-weighted -- steps (not in public data)",
        paper_pattern=r"seed 1, mass-weighted & captured to 160, then free & (\d+) & [\d.]+ & [\d.]+ & [\d.]+ & [\d.]+ \\\\",
        compute=c_not_public,
        public=False,
    ),
    dict(
        label="Table 1: seed 1, mass-weighted (s1b) -- first token rank1, exact match, exact match macro, KL",
        paper_pattern=r"seed 1, mass-weighted & captured to 160, then free & \d+ & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\",
        compute=lambda: c_mass_weighted_row("s1b"),
        public=True,
    ),
]


def main() -> int:
    text = PAPER.read_text()
    n_fail = 0
    n_not_public = 0
    for check in CHECKS:
        matches = list(re.finditer(check["paper_pattern"], text))
        if len(matches) != 1:
            print(f"FAIL  {check['label']}: paper pattern matched {len(matches)} times (expected 1)")
            n_fail += 1
            continue
        groups = matches[0].groups()
        paper_value = groups[0] if len(groups) == 1 else groups
        try:
            computed = check["compute"]()
        except NotPublicData:
            n_not_public += 1
            print(f"not in public data  {check['label']}: paper={paper_value!r}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {check['label']}: compute() raised {exc!r}")
            n_fail += 1
            continue
        same = (computed == paper_value)
        status = "same" if same else "DIFFERENT"
        if not same:
            n_fail += 1
        print(f"{'same' if same else 'DIFFERENT':<10} {check['label']}: paper={paper_value!r} "
              f"computed={computed!r} ({status})")

    n_checked = len(CHECKS) - n_not_public
    print(f"\n{len(CHECKS)} entries, {n_checked} checkable, "
          f"{n_checked - n_fail} same, {n_fail} different, {n_not_public} not in public data")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

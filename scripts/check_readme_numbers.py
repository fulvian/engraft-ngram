"""Recomputes every headline number in the repository root `README.md` from
the public JSON file the README cites next to it, and compares the
recomputed value against the text actually written in the README.

This is NOT a parser of the README's markdown -- it is a fixed checklist,
one entry per number, each with:
  - `label`: a short description (for the printed report only).
  - `readme_pattern`: a regex, applied to the whole README.md text, whose
    capture group(s) are the number(s) as WRITTEN in the README.
  - `compute`: a function that opens the cited JSON file(s) (paths relative
    to the repository root) and returns the same number(s), independently
    recomputed, formatted the same way the README displays them (so the
    comparison is a plain string equality, not a fuzzy numeric tolerance --
    a rounding-convention mismatch is a real discrepancy, not noise).

Every entry must find exactly one match in the README (`readme_pattern`
matches once); zero or multiple matches is a script bug or a README edit
this checklist has not been updated for, and is reported as a FAIL, not
silently skipped.

Usage: `python scripts/check_readme_numbers.py` from the repository root.
Exit code 0 iff every check reads "same".
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _load(*rel_paths: str):
    if len(rel_paths) == 1:
        return json.loads((ROOT / rel_paths[0]).read_text())
    return [json.loads((ROOT / p).read_text()) for p in rel_paths]


def _frac(n: int, d: int, decimals: int = 3) -> str:
    return f"{n / d:.{decimals}f}"


# ---------------------------------------------------------------------
# Compute functions -- one per README number, returning the exact string
# the README should show.
# ---------------------------------------------------------------------


def c_s0b_exact_match():
    d = _load("data/quail/results/s0b/engine_results.json")
    return _frac(sum(1 for x in d if x["greedy"]["student"]["exact_match"]), len(d))


def c_s0b_base_exact_match():
    d = _load("data/quail/results/s0b/engine_results.json")
    return _frac(sum(1 for x in d if x["greedy"]["base"]["exact_match"]), len(d))


def c_s0b_rank1_engine():
    d = _load("data/quail/results/s0b/engine_results.json")
    return _frac(sum(1 for x in d if x["student"]["rank_first"] == 1), len(d))


def c_s1b_exact_match():
    d = _load("data/quail/results/s1b/engine_results.json")
    return _frac(sum(1 for x in d if x["greedy"]["student"]["exact_match"]), len(d))


def c_s0b_replica_free_pinned():
    d = _load("data/quail/results/s0b/replica_eval.json")["fragments"]
    free = _frac(sum(1 for x in d if x["rank_student_free"] == 1), len(d))
    pinned = _frac(sum(1 for x in d if x["correct"]["student"]), len(d))
    return f"{free} / {pinned}"


def c_s0b_base_prior_pinned():
    d = _load("data/quail/results/s0b/replica_eval.json")["fragments"]
    return _frac(sum(1 for x in d if x["correct"]["base"]), len(d))


def c_s0b_damage_kl_mean():
    d = _load("data/quail/results/s0b/damage_it_text.json")
    return f"{d['global']['kl_mean']:.4f}"


def c_s0b_probes():
    d = _load("data/quail/results/s0b/probe_results.json")
    both = sum(1 for x in d if x["both"])
    atleast = sum(1 for x in d if x["hit_a"] or x["hit_b"])
    return f"{both} / {len(d)}, {atleast} / {len(d)}"


def c_s0b_probe_truncation():
    """Composition probes: (number of probes, generation cap, probes carrying an
    empty think block). The cap is asserted, not read: if the generations ever
    stop at different lengths the README sentence is wrong and this must fail."""
    d = _load("data/quail/results/s0b/probe_results.json")
    lengths = {len(x["tokens"]) for x in d}
    if len(lengths) != 1:
        raise ValueError(f"probe generations are not all one length: {sorted(lengths)}")
    n_think = sum(1 for x in d if "<think>" in x["output"])
    return (str(len(d)), str(lengths.pop()), str(n_think))


def c_yardstick_floor():
    d = _load("results/2026-09-12/yardstick/damage_noise_quant_s0.json")
    return f"{d['global']['kl_mean']:.4f}"


def c_24fact_seed0():
    d = _load("results/2026-09-12/s0/damage.json")
    return f"{d['global']['kl_mean']:.4f}"


def c_24fact_seed1():
    d = _load("results/2026-09-12/s1/damage.json")
    return f"{d['global']['kl_mean']:.4f}"


def c_100fact_vs_yardstick_ratio():
    hundred = _load("data/quail/results/s0b/damage_it_text.json")["global"]["kl_mean"]
    floor = _load("results/2026-09-12/yardstick/damage_noise_quant_s0.json")["global"]["kl_mean"]
    ratio = hundred / floor
    return f"{ratio:.0f}"  # README says "about 4x" -- nearest integer


def c_capacity_ratio_vs_yardstick(damage_file: str):
    floor = _load("results/2026-09-12/yardstick/damage_noise_quant_s0.json")["global"]["kl_mean"]
    dmg = _load(f"data/quail/results/capacity-curve/{damage_file}")["global"]["kl_mean"]
    return f"{dmg / floor:.1f}"


def c_capacity_row(n_facts: int, damage_file: str):
    curve = _load("data/quail/results/capacity-curve/capacity_curve.json")
    row = next(r for r in curve["rows"] if r["n"] == n_facts)
    top1 = f"{row['rbr']['top1_frac']:.3f}"
    dmg = _load(f"data/quail/results/capacity-curve/{damage_file}")
    kl = f"{dmg['global']['kl_mean']:.4f}"
    return f"{top1} | {kl}"


def c_zh_exact_match_pair():
    it = _load("data/quail/results/s0b/engine_results.json")
    zh = _load("data/quail/results/z0b-preliminary/engine_results.json")
    it_em = _frac(sum(1 for x in it if x["greedy"]["student"]["exact_match"]), len(it))
    zh_em = _frac(sum(1 for x in zh if x["greedy"]["student"]["exact_match"]), len(zh))
    it_base = _frac(sum(1 for x in it if x["greedy"]["base"]["exact_match"]), len(it))
    zh_base = _frac(sum(1 for x in zh if x["greedy"]["base"]["exact_match"]), len(zh))
    return f"{zh_em} against Italian {it_em}; base models {zh_base} and {it_base}"


def c_zh_gain_pair():
    it = _load("data/quail/results/s0b/replica_eval.json")["fragments"]
    zh = _load("data/quail/results/z0b-preliminary/replica_eval.json")["fragments"]
    it_pinned = sum(1 for x in it if x["correct"]["student"]) / len(it)
    it_base = sum(1 for x in it if x["correct"]["base"]) / len(it)
    zh_pinned = sum(1 for x in zh if x["correct"]["student"]) / len(zh)
    zh_base = sum(1 for x in zh if x["correct"]["base"]) / len(zh)
    it_gain = f"{it_pinned - it_base:.2f}"
    zh_gain = f"{zh_pinned - zh_base:.2f}"
    return f"{zh_gain}\n{it_gain}"


def c_zh_language_coef_ci():
    d = _load("data/quail/results/languages/mass_ols.json")
    lo, hi = d["regression"]["bootstrap"]["ci95_coef_zh"]
    return f"[−{-lo:.3f}, +{hi:.3f}]"


def c_zh_damage_it_zh():
    it_text = _load("data/quail/results/z0b-preliminary/damage_it_text.json")
    zh_text = _load("data/quail/results/z0b-preliminary/damage_zh_text.json")
    return f"{it_text['global']['kl_mean']:.4f}\n{zh_text['global']['kl_mean']:.4f}"


def _lang_cells():
    return {
        "it": "s0b",
        "en": "e0b-preliminary",
        "zh": "z0b-preliminary",
    }


def _lang_engine(cell):
    return _load(f"data/quail/results/{cell}/engine_results.json")


def c_lang_row(kind: str):
    """One row of the three-language table, in README column order (it, en, zh)."""
    out = []
    for lang in ("it", "en", "zh"):
        cell = _lang_cells()[lang]
        if kind == "damage":
            f = {"it": "s0b/damage_it_text.json",
                 "en": "e0b-preliminary/damage_en_text.json",
                 "zh": "z0b-preliminary/damage_zh_text.json"}[lang]
            out.append(f"{_load(f'data/quail/results/{f}')['global']['kl_mean']:.4f}")
            continue
        d = _lang_engine(cell)
        if kind == "n":
            out.append(str(len(d)))
        elif kind == "em":
            out.append(_frac(sum(1 for x in d if x["greedy"]["student"]["exact_match"]), len(d)))
        elif kind == "em_base":
            out.append(_frac(sum(1 for x in d if x["greedy"]["base"]["exact_match"]), len(d)))
        elif kind == "rank1":
            out.append(_frac(sum(1 for x in d if x["student"]["rank_first"] == 1), len(d)))
    return tuple(out)


def c_en_facts_covered():
    d = _lang_engine("e0b-preliminary")
    fids = {f for x in d for f in x["fact_ids"]}
    return str(len(fids))


def _specular(name="engine_results.json"):
    return _load(f"data/quail/results/languages/specular-zh-on-it/{name}")


def c_specular_em():
    d = _specular()
    n = len(d)
    return (f"{sum(1 for x in d if x['greedy']['student']['exact_match']) / n:.4f}",
            f"{sum(1 for x in d if x['greedy']['base']['exact_match']) / n:.4f}")


def c_specular_rank1():
    d = _specular()
    n = len(d)
    return (f"{sum(1 for x in d if x['student']['rank_first'] == 1) / n:.4f}",
            f"{sum(1 for x in d if x['base']['rank_first'] == 1) / n:.4f}")


def c_specular_identical():
    d = _specular()
    return (str(sum(1 for x in d if x["student"]["p_first"] == x["base"]["p_first"])), str(len(d)))


def c_specular_overlay_hits():
    return str(sum(x["student"]["overlay_hits"] for x in _specular()))


def c_specular_probes():
    d = _specular("probe_results.json")
    return (str(sum(1 for x in d if x["both"])), str(len(d)),
            str(sum(1 for x in d if x["hit_a"] or x["hit_b"])), str(len(d)))


def c_specular_mixed():
    a = _specular("probe_results_mixed_it.json")
    b = _specular("probe_results_mixed_zh.json")
    return (str(sum(1 for x in a if x["hit_a"] or x["hit_b"])), str(len(a)),
            str(sum(1 for x in b if x["hit_a"] or x["hit_b"])), str(len(b)))


def _row_sharing():
    return _load("data/quail/results/s0b/row_sharing.json")


def c_row_sharing_windows():
    d = _row_sharing()
    return (f"{d['n_rows_overlay']:,}", str(d["n_rows_multi_window"]))


def c_row_sharing_collision_auc():
    return f"{_row_sharing()['contrasts']['frac_pure_hash_collision']['auc']:.3f}"


def c_row_sharing_content_share():
    return f"{_row_sharing()['groups']['success']['shared_by_content']:.3f}"


def c_row_sharing_siblings():
    d = _row_sharing()
    return (str(d["subject_or_template"]["n_compared"]), str(d["subject_or_template"]["same_subject"]))


def c_overlay_kb_per_fact():
    n_bytes = (ROOT / "data/quail/overlays/s0b/merged.pleo").stat().st_size
    return str(round(n_bytes / _n_facts_s0b() / 1000))


def c_train_tokens_per_fact():
    frags = _load("data/quail/corpus/s0b/usage_corpus_resolved.json")["fragments"]
    n_tok = sum(f["n_positions"] for f in frags if f["split"] == "train")
    return str(round(n_tok / _n_facts_s0b()))


def _n_facts_s0b() -> int:
    frags = _load("data/quail/corpus/s0b/usage_corpus_resolved.json")["fragments"]
    return len({i for f in frags for i in f["fact_ids"]})


def c_row_sharing_heavier_sibling():
    d = _row_sharing()["subject_or_template"]
    return (str(d["winner_has_more_training_mass"]), str(d["n_compared"]))


CHECKS = [
    dict(
        label="Quail table: exact answer, engine, free routing (s0b)",
        readme_pattern=r"free routing \(841 test sentences\) \| \*\*([\d.]+)\*\*",
        compute=c_s0b_exact_match,
    ),
    dict(
        label="Quail table: base model without overlay",
        readme_pattern=r"base model without overlay \| \*\*([\d.]+)\*\*",
        compute=c_s0b_base_exact_match,
    ),
    dict(
        label="Quail table: first token rank 1, real engine",
        readme_pattern=r"first token at rank 1, real engine \| ([\d.]+) \|",
        compute=c_s0b_rank1_engine,
    ),
    dict(
        label="Quail table: second seed s1b, exact answer greedy",
        readme_pattern=r"real engine, second seed \(`s1b`\) \| ([\d.]+) \|",
        compute=c_s1b_exact_match,
    ),
    dict(
        label="Quail table: replica free/pinned rank 1",
        readme_pattern=r"free routing / pinned routing \| ([\d.]+ / [\d.]+) \|",
        compute=c_s0b_replica_free_pinned,
    ),
    dict(
        label="Quail table: base model's own prior, pinned routing",
        readme_pattern=r"own prior, first token at rank 1, pinned routing \| ([\d.]+) \|",
        compute=c_s0b_base_prior_pinned,
    ),
    dict(
        label="Quail table: collateral damage mean KL",
        readme_pattern=r"mean KL to the base model \| ([\d.]+) \|",
        compute=c_s0b_damage_kl_mean,
    ),
    dict(
        label="Quail table: composition probes",
        readme_pattern=r"both answers right / at least one \| ([\d]+ / [\d]+, [\d]+ / [\d]+) \|",
        compute=c_s0b_probes,
    ),
    dict(
        label="Quail limitations: composition probes are truncated",
        readme_pattern=(
            r"every\s+one\s+of\s+the\s+(\d+)\s+generations\s+stops\s+at\s+the\s+probe\s+"
            r"tool's\s+default\s+budget\s+of\s+(\d+)\s+new\s+tokens,\s+and\s+(\d+)\s+of\s+them"
        ),
        compute=c_s0b_probe_truncation,
    ),
    dict(
        label="FAQ RAG: overlay size per fact, KB",
        readme_pattern=r"overlay\s+is\s+about\s+(\d+)\s+KB\s+per\s+fact",
        compute=c_overlay_kb_per_fact,
    ),
    dict(
        label="FAQ RAG: training tokens per fact",
        readme_pattern=r"come\s+to\s+(\d+)\s+tokens\s+per\s+fact",
        compute=c_train_tokens_per_fact,
    ),
    dict(
        label="FAQ agent memory: failures lost to the heavier sibling",
        readme_pattern=r"in\s+(\d+)\s+of\s+those\s+(\d+)\s+cases\s+to\s+the\s+sibling",
        compute=c_row_sharing_heavier_sibling,
    ),
    dict(
        label="Damage paragraph: quantization yardstick",
        readme_pattern=r"re-quantizes the same rows,\n([\d.]+), and",
        compute=c_yardstick_floor,
    ),
    dict(
        label="Damage paragraph: 24-fact overlay, seed 0",
        readme_pattern=r"24-fact overlay measured ([\d.]+) \(seed 0\)",
        compute=c_24fact_seed0,
    ),
    dict(
        label="Damage paragraph: 24-fact overlay, seed 1",
        readme_pattern=r"\(seed 0\) and ([\d.]+) \(seed 1\)",
        compute=c_24fact_seed1,
    ),
    dict(
        label="Damage paragraph: 100-fact vs. yardstick ratio",
        readme_pattern=r"about (\d+)× that yardstick",
        compute=c_100fact_vs_yardstick_ratio,
    ),
    dict(
        label="Capacity table: n=24 row (top-1 | KL)",
        readme_pattern=r"\| 24 \| ([\d.]+) \| ([\d.]+) \(seed 1\)",
        compute=lambda: tuple(c_capacity_row(24, "damage_n24_seed1.json").split(" | ")),
    ),
    dict(
        label="Capacity table: n=100 row (top-1 | KL)",
        readme_pattern=r"\| 100 \| ([\d.]+) \| ([\d.]+) \|",
        compute=lambda: tuple(c_capacity_row(100, "damage_n100.json").split(" | ")),
    ),
    dict(
        label="Capacity table: n=300 row (top-1 | KL)",
        readme_pattern=r"\| 300 \| ([\d.]+) \| ([\d.]+) \|",
        compute=lambda: tuple(c_capacity_row(300, "damage_n300.json").split(" | ")),
    ),
    dict(
        label="Capacity prose: damage/yardstick ratio, n=24/100/300",
        readme_pattern=r"grows sublinearly \(([\d.]+)×, ([\d.]+)× and ([\d.]+)× the 24-fact\nyardstick\)",
        compute=lambda: tuple(
            c_capacity_ratio_vs_yardstick(f)
            for f in ("damage_n24_seed1.json", "damage_n100.json", "damage_n300.json")
        ),
    ),
    dict(
        label="Languages table: exact answer with the overlay (it | en | zh)",
        readme_pattern=r"exact answer, greedy, with the overlay \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",
        compute=lambda: c_lang_row("em"),
    ),
    dict(
        label="Languages table: exact answer, base model (it | en | zh)",
        readme_pattern=r"exact answer, greedy, base model \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",
        compute=lambda: c_lang_row("em_base"),
    ),
    dict(
        label="Languages table: first token at rank 1 (it | en | zh)",
        readme_pattern=r"first answer token at rank 1, with the overlay \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",
        compute=lambda: c_lang_row("rank1"),
    ),
    dict(
        label="Languages table: damage on own-language text (it | en | zh)",
        readme_pattern=r"damage on neutral text of its own language, mean KL \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",
        compute=lambda: c_lang_row("damage"),
    ),
    dict(
        label="Languages table: number of test sentences (it | en | zh)",
        readme_pattern=r"test sentences \| (\d+) \| (\d+) \| (\d+) \|",
        compute=lambda: c_lang_row("n"),
    ),
    dict(
        label="Languages: facts covered by the English corpus",
        readme_pattern=r"English corpus covers (\d+) of the 100 facts",
        compute=c_en_facts_covered,
    ),
    dict(
        label="Languages: 95% CI on the language coefficient",
        readme_pattern=r"95 % interval is (\[.\d\.\d+, \+\d\.\d+\]),",
        compute=c_zh_language_coef_ci,
    ),
    dict(
        label="Specular zh-on-it: exact answer, overlay vs base",
        readme_pattern=r"exact answer ([\d.]+), against the base model's ([\d.]+);",
        compute=c_specular_em,
    ),
    dict(
        label="Specular zh-on-it: first token at rank 1, overlay vs base",
        readme_pattern=r"first answer token at rank 1 ([\d.]+), against the base model's ([\d.]+);",
        compute=c_specular_rank1,
    ),
    dict(
        label="Specular zh-on-it: sentences identical to the base model",
        readme_pattern=r"- (\d+) of (\d+) sentences identical to the base model",
        compute=c_specular_identical,
    ),
    dict(
        label="Specular zh-on-it: overlay rows actually read",
        readme_pattern=r"did read (\d+) overlay rows",
        compute=c_specular_overlay_hits,
    ),
    dict(
        label="Specular zh-on-it: composition probes",
        readme_pattern=r"(\d+) of (\d+) with both answers, (\d+) of (\d+) with\n",
        compute=c_specular_probes,
    ),
    dict(
        label="Specular zh-on-it: mixed-script probes",
        readme_pattern=r"do no better: (\d+) of (\d+) and (\d+) of (\d+)\.",
        compute=lambda: tuple(x for x in c_specular_mixed()),
    ),
    dict(
        label="Row sharing: rows written through more than one window",
        readme_pattern=r"Of the ([\d,]+) rows in\n  the Quail overlay, (\d+) are written through more than one token window",
        compute=c_row_sharing_windows,
    ),
    dict(
        label="Row sharing: pure-collision AUC",
        readme_pattern=r"separates successes from failures with AUC ([\d.]+)",
        compute=c_row_sharing_collision_auc,
    ),
    dict(
        label="Row sharing: share of read slots shared by content (successes)",
        readme_pattern=r"([\d.]+) of the slots a successful\n  prompt reads",
        compute=c_row_sharing_content_share,
    ),
    dict(
        label="Row sharing: failures lost to a sibling of the same subject",
        readme_pattern=r"of the (\d+) failures whose first token belongs to\n  another grafted fact, (\d+) are facts about the same subject",
        compute=c_row_sharing_siblings,
    ),
]


def main() -> int:
    text = README.read_text()
    n_fail = 0
    for check in CHECKS:
        matches = list(re.finditer(check["readme_pattern"], text))
        if len(matches) != 1:
            print(f"FAIL  {check['label']}: README pattern matched {len(matches)} times (expected 1)")
            n_fail += 1
            continue
        groups = matches[0].groups()
        readme_value = groups[0] if len(groups) == 1 else groups
        try:
            computed = check["compute"]()
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole run
            print(f"FAIL  {check['label']}: compute() raised {exc!r}")
            n_fail += 1
            continue
        same = (computed == readme_value)
        status = "same" if same else "DIFFERENT"
        if not same:
            n_fail += 1
        print(f"{'uguale' if same else 'diverso':<8} {check['label']}: README={readme_value!r} "
              f"computed={computed!r} ({status})")

    print(f"\n{len(CHECKS)} checks, {len(CHECKS) - n_fail} uguale, {n_fail} diverso")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

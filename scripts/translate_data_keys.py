"""Deterministic Italian -> English translation of the Quail data-schema keys
and enum-like values that `engraft/eval.py`, `damage.py`, `engine_check.py`,
`probes.py`, `descend_corpus.py` and `teacher.py` expect (see
`docs/formats.md`). Applied to `data/quail/results/` and `data/quail/corpus/`.

Only dict KEYS and specific SCHEMA VALUES (family/template_id/column/stratum/
routing_mode enums) are touched. Free text (corpus sentences, probe text,
greedy output, README prose, `_comment`/`production_note`/etc.) is data and
is left byte-for-byte alone. Numbers, token ids and booleans are never
touched.

Also applied, the same way, to `results/2026-09-12/` (the pre-`engraft/`
run of record: `s0/`, `s1/`, `s0-bf16/`, `controls/`, `engine_check/`,
`corpus/census_it.json`, `corpus/usage_corpus_resolved.json` — all produced
by the same code paths as the `data/quail/` schema above).

The map is derived directly from the public code, not guessed:

- `allievo`/`maestro` -> `student`/`teacher`: the column names
  `engraft.eval.write_eval_report`, `engraft.engine_check` and
  `engraft.probes.column_from_overlay` use everywhere (`p_first.student`,
  `rank_first.student`, `correct.teacher`, `rec["student"]`,
  `greedy["student"]`, ...).
- `colonna` -> `column`: the field `engraft.probes.run_probe` writes
  (`rec["column"]`), with its value translated the same way
  (`allievo`->`student`; `maestro` never occurs as a column value in this
  corpus's probes, but is mapped defensively).
- `risposta_a`/`risposta_b` -> `answer_a`/`answer_b`, `uscita` -> `output`:
  the fields `engraft.probes.run_probe`/`score` read and write.
- `kd_allievo_maestro(_mean)` -> `kd_student_teacher(_mean)`,
  `p_allievo_free`/`rank_allievo_free` -> `p_student_free`/`rank_student_free`,
  `top3_allievo` -> `top3_student`: exact field names from
  `engraft.eval.evaluate_fragment`/`_aggregate` and `engraft.damage.cmd_run`.
- `family`/`template_id` values (`affermazione`/`parafrasi`/`domanda`) ->
  (`statement`/`paraphrase`/`question`): the vocabulary
  `engraft.eval.FAMILY_TO_CENSUS_CTX` requires (`cloze`/`chat` are already
  English and pass through unchanged; `FAMILY_TO_CENSUS_CTX` has no entry
  for `statement` itself -- see NOTE below, reported instead of invented).
- `stratum`/`by_stratum` value+key `coda` -> `tail`: the enum
  `engraft.damage.cmd_run`/`_stratum_summary` uses (`for stratum in
  ("top", "tail")`).
- `routing_mode` value `libero` -> `free`: the enum
  `engraft.damage.cmd_run` writes (`"free" (default) | "locked"`).

NOTE (not invented, reported): `engraft.eval.FAMILY_TO_CENSUS_CTX` maps
`"statement"` nowhere -- only `"paraphrase"`, `"question"`, `"cloze"`,
`"chat"`, `"statement"` are keys in that dict actually (all five families
are present), so this *is* covered; no gap here. If a future run of this
script ever sees a `family`/`template_id` suffix outside
{affermazione, parafrasi, domanda, cloze, chat}, it refuses to guess and
raises instead of silently leaving it (so an unmapped Italian value is never
mistaken for an intentional pass-through).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# The map (see module docstring for provenance of every entry)
# ---------------------------------------------------------------------

# Dict KEY renames, applied everywhere in the tree (these names occur only
# in this one data schema -- no collision risk with unrelated keys).
KEY_RENAME: dict[str, str] = {
    "allievo": "student",
    "maestro": "teacher",
    "colonna": "column",
    "risposta_a": "answer_a",
    "risposta_b": "answer_b",
    "uscita": "output",
    "kd_allievo_maestro": "kd_student_teacher",
    "kd_allievo_maestro_mean": "kd_student_teacher_mean",
    "p_allievo_free": "p_student_free",
    "rank_allievo_free": "rank_student_free",
    "top3_allievo": "top3_student",
}

# family / template_id suffix vocabulary (engraft.eval.FAMILY_TO_CENSUS_CTX).
FAMILY_VALUE_MAP: dict[str, str] = {
    "affermazione": "statement",
    "parafrasi": "paraphrase",
    "domanda": "question",
    # cloze, chat: already English, identity (kept explicit for the assert
    # below that every observed family value is accounted for).
    "cloze": "cloze",
    "chat": "chat",
}

# column / colonna value vocabulary (engraft.probes.column_from_overlay).
COLUMN_VALUE_MAP: dict[str, str] = {
    "allievo": "student",
    "maestro": "teacher",
    "base": "base",
    "student": "student",
    "teacher": "teacher",
}

# stratum value/key vocabulary (engraft.damage: "top"/"tail").
STRATUM_VALUE_MAP: dict[str, str] = {
    "coda": "tail",
    "top": "top",
    "tail": "tail",
}

# routing_mode value vocabulary (engraft.damage: "free"/"locked").
ROUTING_MODE_VALUE_MAP: dict[str, str] = {
    "libero": "free",
    "free": "free",
    "locked": "locked",
}

# ---------------------------------------------------------------------
# Second map: `data/quail/results/languages/fact_balance.json` and
# `mass_ols.json` (the mass-controlled language-comparison data, not
# produced by any `engraft.*` module -- these keys come from private ad-hoc
# scripts referenced in `results/languages/README.md`). Applied by the same
# KEY_RENAME mechanism, merged into it in `translate()`.
LANGUAGE_KEY_RENAME: dict[str, str] = {
    "cella": "cell",
    "celle": "cells",
    "per_cella": "per_cell",
    "differenze": "differences",
    "collisione_na": "collision_na",
    "direzione_na": "direction_na",
    "concordanza": "agreement",
    "eval_fonte": "eval_source",
    "fact_mass_max_abs_delta_vs_ricalcolata": "fact_mass_max_abs_delta_vs_recomputed",
    "motore_em_greedy": "engine_em_greedy",
    "motore_primo_token": "engine_first_token",
    "n_comuni": "n_common",
    "n_fatti_comuni": "n_facts_common",
    "quartili": "quartiles",
    "quartili_fatti": "fact_quartiles",
    "spearman_massa_tasso": "spearman_mass_rate",
    "tasso": "rate",
    "tasso_macro": "rate_macro",
    "tasso_micro": "rate_micro",
    "tasso_Q1_quartili_A": "rate_Q1_quartiles_A",
    "tasso_Q2_quartili_A": "rate_Q2_quartiles_A",
    "tasso_Q3_quartili_A": "rate_Q3_quartiles_A",
    "tasso_Q4_quartili_A": "rate_Q4_quartiles_A",
    "tasso_Q1_quartili_B": "rate_Q1_quartiles_B",
    "tasso_Q2_quartili_B": "rate_Q2_quartiles_B",
    "tasso_Q3_quartili_B": "rate_Q3_quartiles_B",
    "tasso_Q4_quartili_B": "rate_Q4_quartiles_B",
    "media_f": "mean_f",
    "media_r": "mean_r",
}

# `mass_ols.json`'s `metric` field carries a fixed descriptive sentence
# (not a coded enum) that used the superseded "RBR (blocked routing)" term
# this repository's README now calls "pinned routing"; fixed by exact-value
# lookup like the other VALUE_MAP_BY_KEY entries, not a substring rewrite,
# so an unrecognized future sentence is refused rather than mis-edited.
METRIC_VALUE_MAP: dict[str, str] = {
    "fraction rank-1, RBR (blocked routing)": "fraction rank-1, pinned routing",
}

# Keys whose scalar VALUE gets translated (in addition to the key rename
# pass above). Keyed by the ORIGINAL (pre-rename) key name.
VALUE_MAP_BY_KEY: dict[str, dict[str, str]] = {
    "family": FAMILY_VALUE_MAP,
    "colonna": COLUMN_VALUE_MAP,
    "column": COLUMN_VALUE_MAP,
    "stratum": STRATUM_VALUE_MAP,
    "routing_mode": ROUTING_MODE_VALUE_MAP,
    "metric": METRIC_VALUE_MAP,
}

# Parent-key contexts whose own dict KEYS are family values (eval.py's
# `by_family`, built as `{fragment["family"]: stats}`).
FAMILY_KEYED_PARENTS = {"by_family"}

# ---------------------------------------------------------------------
# Third map: `data/quail/results/capacity-curve/capacity_curve.json`'s
# `by_class_template.by_class` fact-class taxonomy (`luogo`/`studioso`/
# `strumento`, the invented world's three fact classes -- not produced by
# any `engraft.*` module either) and its composite per-template-id keys
# (e.g. "luogo:shared:cloze", "luogo:usage:u_scop_affermazione_heldout").
# ---------------------------------------------------------------------

CLASS_PREFIX_MAP: dict[str, str] = {
    "luogo": "place",
    "studioso": "scholar",
    "strumento": "instrument",
}

# Parent-key contexts whose own dict KEYS are class names (capacity_curve's
# `by_class_template.by_class`, built as `{class_name: {template_id: stats}}`).
CLASS_KEYED_PARENTS = {"by_class"}
# The next nesting level down from a CLASS_KEYED_PARENTS dict: keys there are
# composite "<class>:<scope>:<...>" template ids, not class names alone.
CLASS_TEMPLATE_KEYED_PARENTS = {"class_templates"}


def _translate_class_template_key(key: str) -> str:
    """Translate a `by_class`/`by_class_template` key: either a bare class
    name ("luogo") or a composite "<class>:<scope>:<...>" template id
    ("luogo:usage:u_scop_affermazione_heldout"). Only the leading class
    segment (via CLASS_PREFIX_MAP) and any standalone `_`-delimited
    "affermazione" segment (the same value FAMILY_VALUE_MAP maps to
    "statement" elsewhere in this corpus) are translated. Every other
    segment (`shared`/`unique`/`usage`/`cloze`/`paraphrase_other_tail`/
    `paraphrase_same_tail`/`chat`/`heldout`/`u`/`scop`/`sede`) is already
    English or an unexplained abbreviation from the private pipeline this
    script does not guess at, and is left untouched.
    """
    parts = key.split(":")
    if parts[0] not in CLASS_PREFIX_MAP:
        raise UnmappedValue(f"class-template key {key!r} has unmapped class prefix {parts[0]!r}")
    parts[0] = CLASS_PREFIX_MAP[parts[0]]
    new_parts = []
    for part in parts:
        segs = part.split("_")
        segs = ["statement" if s == "affermazione" else s for s in segs]
        new_parts.append("_".join(segs))
    return ":".join(new_parts)


class UnmappedValue(Exception):
    pass


def _translate_template_id(value: str) -> str:
    """`"quail:<family>"` -> `"quail:<english family>"`."""
    if not value.startswith("quail:"):
        return value
    suffix = value[len("quail:"):]
    if suffix not in FAMILY_VALUE_MAP:
        raise UnmappedValue(f"template_id suffix {suffix!r} not in FAMILY_VALUE_MAP: {value!r}")
    return "quail:" + FAMILY_VALUE_MAP[suffix]


def translate(obj, parent_key: str | None = None, counts: dict | None = None):
    if counts is None:
        counts = {"keys_renamed": 0, "values_translated": 0}
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            new_k = k
            in_class_context = parent_key in CLASS_KEYED_PARENTS or parent_key in CLASS_TEMPLATE_KEYED_PARENTS
            if parent_key in FAMILY_KEYED_PARENTS and k in FAMILY_VALUE_MAP:
                new_k = FAMILY_VALUE_MAP[k]
            elif in_class_context:
                new_k = _translate_class_template_key(k)
            elif k in KEY_RENAME:
                new_k = KEY_RENAME[k]
            elif k in LANGUAGE_KEY_RENAME:
                new_k = LANGUAGE_KEY_RENAME[k]
            if new_k != k:
                if new_k in obj and new_k != k:
                    raise ValueError(f"rename collision: {k!r} -> {new_k!r} already present in {list(obj.keys())}")
                counts["keys_renamed"] += 1

            # A key renamed under CLASS_KEYED_PARENTS ("by_class") holds a
            # dict of composite template-id keys as its value -- recurse
            # with the synthetic marker CLASS_TEMPLATE_KEYED_PARENTS names,
            # not the translated class name itself, so those template-id
            # keys are recognized and translated on the next level down.
            recurse_parent_key = "class_templates" if parent_key in CLASS_KEYED_PARENTS else new_k

            new_v = v
            if k == "template_id" and isinstance(v, str):
                translated = _translate_template_id(v)
                if translated != v:
                    counts["values_translated"] += 1
                new_v = translated
            elif k in VALUE_MAP_BY_KEY and isinstance(v, str):
                vmap = VALUE_MAP_BY_KEY[k]
                if v not in vmap:
                    raise UnmappedValue(f"key {k!r} has unmapped value {v!r} (known: {sorted(vmap)})")
                translated = vmap[v]
                if translated != v:
                    counts["values_translated"] += 1
                new_v = translated
            else:
                new_v = translate(v, parent_key=recurse_parent_key, counts=counts)

            new[new_k] = new_v
        return new
    elif isinstance(obj, list):
        return [translate(item, parent_key=parent_key, counts=counts) for item in obj]
    else:
        return obj


# ---------------------------------------------------------------------
# Numeric-leaf invariance check (path-independent: renames move paths)
# ---------------------------------------------------------------------


def numeric_leaves(obj) -> list[float]:
    """Sorted multiset of every non-bool int/float leaf. Path-independent on
    purpose -- a rename changes every path under the renamed key, so a
    path-keyed diff would flag everything as changed even when no number
    moved. Order-independent comparison plus a leaf COUNT is the correct
    invariant here."""
    out: list[float] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            out.append(float(o))

    walk(obj)
    return out


def check_numeric_invariant(before, after, label: str) -> None:
    nb, na = numeric_leaves(before), numeric_leaves(after)
    if len(nb) != len(na):
        raise AssertionError(f"{label}: numeric leaf COUNT changed {len(nb)} -> {len(na)}")
    nb_sorted, na_sorted = sorted(nb), sorted(na)
    for i, (a, b) in enumerate(zip(nb_sorted, na_sorted)):
        if a != b and not (a != a and b != b):  # nan-safe equality
            raise AssertionError(f"{label}: numeric leaf multiset differs at rank {i}: {a!r} vs {b!r}")
    print(f"  numeric leaves invariant OK: {len(nb)} leaves, multiset identical")


def process_file(path: Path, apply: bool) -> dict:
    text = path.read_text()
    before = json.loads(text)
    counts: dict = {"keys_renamed": 0, "values_translated": 0}
    after = translate(before, parent_key=None, counts=counts)
    check_numeric_invariant(before, after, str(path))
    changed = counts["keys_renamed"] > 0 or counts["values_translated"] > 0
    if changed and apply:
        path.write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n")
    return {"path": str(path), "changed": changed, **counts}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="JSON files to translate")
    p.add_argument("--apply", action="store_true", help="write changes (default: dry-run, check only)")
    args = p.parse_args(argv)

    reports = []
    for raw in args.paths:
        path = Path(raw)
        print(f"== {path} ==")
        try:
            report = process_file(path, apply=args.apply)
        except UnmappedValue as e:
            print(f"  UNMAPPED VALUE, refusing to guess: {e}", file=sys.stderr)
            return 2
        reports.append(report)
        print(f"  keys_renamed={report['keys_renamed']} values_translated={report['values_translated']} "
              f"changed={report['changed']} apply={args.apply}")

    n_changed = sum(1 for r in reports if r["changed"])
    print(f"\n{len(reports)} files checked, {n_changed} with changes"
          f"{' (written)' if args.apply else ' (dry-run, not written)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

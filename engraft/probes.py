"""Composition probes for the Quail corpus: questions that combine TWO facts
of the same subject. Each probe lives in a `probes.json`-shaped file (key
`"probes"`) -- an eval-only artifact, never part of the corpus resolver or
compiler. This module runs a greedy continuation from the engine, for ONE
column only (base or student, never both at once -- unlike
`engraft.engine_check`, which measures both when an overlay is given), and
scores binary success per probe when BOTH facts' `answer` strings appear as
case-sensitive substrings of the whole greedy output.

Reused via import (never copied):
- `engraft.table.PleTokenizer` to tokenize the probes.
- `engraft.engine.ENGINE_CFG`/`LensClient` -- same q8 column, same engine
  client.
- `engraft.engine_check._greedy_continuation_local` -- same sequential
  greedy (one job per step, `overlay: None` stays `null` in JSON for the
  base column, never the literal string `"None"`), unchanged.

Declared choice -- **the probe's opening EOS**: `PleTokenizer` exposes no EOS
of its own (a thin wrapper over the HF tokenizer, no `eos_token_id` field);
the only source is `PleTable.eos_token_id` (a GGUF metadata scalar), but
loading the table here would add a GGUF/mmap dependency this module's
contract does not otherwise need (no `--table` flag). `--eos` is therefore an
explicit, required CLI argument (the caller reads it from the same table
metadata used to build the usage corpus's fragments, whose `tokens[0]`
already carries this convention) -- never a hardcoded constant, unlike the
private original, which hardcoded a value verified for one specific model
build.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from pathlib import Path

from engraft.engine import ENGINE_CFG, LensClient
from engraft.engine_check import _greedy_continuation_local
from engraft.table import PleTokenizer


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
    tmp.replace(path)


# --------------------------------------------------------------------------
# Inputs: probes.json + a fact registry
# --------------------------------------------------------------------------


def load_probes(path: str) -> list[dict]:
    """The `"probes"` list (ignores any other top-level key such as raw
    extraction logs or violations)."""
    data = json.loads(Path(path).read_text())
    return data["probes"]


def load_facts(path: str) -> dict[str, dict]:
    """`{fact_id: record}` from a fact registry file (key `"facts"`)."""
    data = json.loads(Path(path).read_text())
    return {f["fact_id"]: f for f in data["facts"]}


def resolve_answers(probe: dict, facts_by_id: dict[str, dict]) -> tuple[str, str]:
    """`(answer_a, answer_b)` of the probe's two `fact_ids`, in the order
    they appear -- a clear error (not an opaque `KeyError`) if a `fact_id`
    is missing from the registry, or if a probe has a number of `fact_ids`
    other than two (the "two facts of the same subject" schema)."""
    fact_ids = probe["fact_ids"]
    if len(fact_ids) != 2:
        raise ValueError(f"resolve_answers: probe {probe.get('id')!r} has {len(fact_ids)} fact_ids (expected 2)")
    answers = []
    for fid in fact_ids:
        fact = facts_by_id.get(fid)
        if fact is None:
            raise ValueError(
                f"resolve_answers: probe {probe.get('id')!r} references fact_id {fid!r} "
                "absent from the registry (--facts)"
            )
        answers.append(fact["answer"])
    return answers[0], answers[1]


# --------------------------------------------------------------------------
# Prompt and greedy (one column only: base or student)
# --------------------------------------------------------------------------


def build_prompt(probe: dict, tok: PleTokenizer, eos: int) -> list[int]:
    """`tokens = [eos] + tok.encode(text)` -- the same opening convention as
    the already-resolved usage-corpus fragments."""
    return [eos] + tok.encode(probe["text"])


def column_from_overlay(overlay: str | None) -> str:
    return "base" if overlay is None else "student"


def score(output: str, answer_a: str, answer_b: str) -> dict:
    """Binary success: both `answer` strings as CASE-SENSITIVE substrings of
    the whole greedy output."""
    hit_a = answer_a in output
    hit_b = answer_b in output
    return {"hit_a": hit_a, "hit_b": hit_b, "both": hit_a and hit_b}


def run_probe(client, raw_dir, tok, log_, probe: dict, facts_by_id: dict[str, dict],
              overlay: str | None, eos: int, max_new_tokens: int) -> dict:
    answer_a, answer_b = resolve_answers(probe, facts_by_id)
    prefix = build_prompt(probe, tok, eos)
    column = column_from_overlay(overlay)
    gen, output, _degenerate = _greedy_continuation_local(
        client, raw_dir, tok, log_, prefix, overlay, max_new_tokens, f"gr_{probe['id']}_{column}",
    )
    outcome = score(output, answer_a, answer_b)
    return {
        "id": probe["id"], "fact_ids": probe["fact_ids"], "subject": probe.get("subject"),
        "text": probe["text"], "answer_a": answer_a, "answer_b": answer_b,
        "output": output, "tokens": gen, "hit_a": outcome["hit_a"], "hit_b": outcome["hit_b"],
        "both": outcome["both"], "column": column, "overlay": overlay,
    }


def run_probes(client, raw_dir, tok, log_, probes: list[dict], facts_by_id: dict[str, dict],
                overlay: str | None, eos: int, max_new_tokens: int) -> list[dict]:
    records = []
    for probe in probes:
        rec = run_probe(client, raw_dir, tok, log_, probe, facts_by_id, overlay, eos, max_new_tokens)
        records.append(rec)
        log_.info("%s measured (both=%s)", rec["id"], rec["both"])
    return records


# --------------------------------------------------------------------------
# --dry-run: one job per probe (the greedy's first step), the two targets
# --------------------------------------------------------------------------


def build_jobs(probes: list[dict], facts_by_id: dict[str, dict], tok: PleTokenizer,
               overlay: str | None, eos: int, max_new_tokens: int) -> dict:
    """The greedy is sequential (every step past the first depends on the
    engine-generated token, as in `engraft.engine_check.build_jobs`): dry
    enumeration lists only the first step's job per probe, with the two
    targets (`answer_a`/`answer_b`) alongside for readability."""
    column = column_from_overlay(overlay)
    jobs = []
    for probe in probes:
        answer_a, answer_b = resolve_answers(probe, facts_by_id)
        prefix = build_prompt(probe, tok, eos)
        first_step_job = {
            "id": f"gr_{probe['id']}_{column}_greedy0", "text": "", "tokens": prefix,
            "overlay": overlay, "capture": [], "logits": "last",
        }
        jobs.append({
            "id": probe["id"], "fact_ids": probe["fact_ids"], "subject": probe.get("subject"),
            "column": column, "answer_a": answer_a, "answer_b": answer_b,
            "first_step_job": first_step_job,
        })
    n_probes = len(jobs)
    n_expected_calls = n_probes * max_new_tokens
    return {"jobs": jobs, "n_probes": n_probes, "n_expected_calls": n_expected_calls}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(records: list[dict]) -> str:
    n = len(records)
    n_both = sum(1 for r in records if r["both"])
    n_at_least_one = sum(1 for r in records if r["hit_a"] or r["hit_b"])

    lines = ["# engraft.probes report", ""]
    lines.append(f"Probes measured: {n}")
    lines.append(f"Probes with both answers (`both`): {n_both}")
    lines.append(f"Probes with at least one answer: {n_at_least_one}")

    by_subject: dict[str, list[dict]] = {}
    for r in records:
        by_subject.setdefault(r.get("subject") or "?", []).append(r)
    lines.append("\n## By subject")
    lines.append("| subject | n | both |")
    lines.append("|---|---|---|")
    for subject in sorted(by_subject):
        rs = by_subject[subject]
        lines.append(f"| {subject} | {len(rs)} | {sum(1 for r in rs if r['both'])} |")

    lines.append("\n## Probes with `both` true")
    hits = [r for r in records if r["both"]]
    if hits:
        lines.append("| id | output |")
        lines.append("|---|---|")
        for r in hits:
            output_truncated = r["output"][:120]
            lines.append(f"| {r['id']} | {output_truncated} |")
    else:
        lines.append("(none)")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probes", required=False, help="probes.json (key 'probes')")
    p.add_argument("--facts", required=False, help="fact registry json (key 'facts')")
    p.add_argument("--out", required=True)
    p.add_argument("--overlay", required=False, default=None,
                   help="merged.pleo; absent = base column, present = student column")
    p.add_argument("--lens-cmd", default=None)
    p.add_argument("--eos", type=int, default=None,
                   help="opening EOS token id (required for a real/dry run, matches the usage "
                        "corpus's fragments' tokens[0])")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--config", default=None, help="engraft.toml path (tokenizer resolution)")
    args = p.parse_args(argv)

    out_dir = Path(args.out).resolve()  # streaming-mode lens requires absolute paths, like engine_check
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        results_path = out_dir / "probe_results.json"
        if not results_path.exists():
            print(f"--render-only: {results_path} not found", file=sys.stderr)
            return 2
        records = json.loads(results_path.read_text())
        (out_dir / "report.md").write_text(build_report(records))
        print(f"report regenerated (--render-only, no engine): {out_dir / 'report.md'}")
        return 0

    if not args.probes or not args.facts:
        print("--probes and --facts required (absent only with --render-only)", file=sys.stderr)
        return 2
    if args.eos is None:
        print("--eos required (absent only with --render-only)", file=sys.stderr)
        return 2
    overlay = str(Path(args.overlay).resolve()) if args.overlay else None

    probes = load_probes(args.probes)
    facts_by_id = load_facts(args.facts)
    from engraft.config import load as load_config
    engraft_cfg = load_config(args.config)
    tok = PleTokenizer(engraft_cfg.get_path("model.tokenizer"))

    if args.dry_run:
        jobs = build_jobs(probes, facts_by_id, tok, overlay, args.eos, args.max_new_tokens)
        _atomic_write_json(out_dir / "jobs.json", jobs)
        print(f"n_probes={jobs['n_probes']}")
        print(f"n_expected_calls={jobs['n_expected_calls']}")
        return 0

    if not args.lens_cmd:
        print("--lens-cmd required (absent only with --dry-run/--render-only)", file=sys.stderr)
        return 2

    log_ = logging.getLogger("engraft.probes")
    log_.setLevel(logging.INFO)
    log_.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log_.addHandler(h)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    argv_engine = shlex.split(args.lens_cmd) + ["--jobs", "-", "--out", str(raw_dir)] + ENGINE_CFG["q8"]["args"]
    client = LensClient(argv_engine, raw_dir, out_dir / "engine.log", env=ENGINE_CFG["q8"]["env"])
    try:
        records = run_probes(client, raw_dir, tok, log_, probes, facts_by_id, overlay, args.eos, args.max_new_tokens)
    finally:
        client.close()

    _atomic_write_json(out_dir / "probe_results.json", records)
    (out_dir / "report.md").write_text(build_report(records))
    print(f"probe_results.json/report.md written to {out_dir}, n_probes={len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

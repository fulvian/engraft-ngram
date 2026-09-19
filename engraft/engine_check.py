"""Real-engine verification of the corpus's fragments: redoes ONLY the
**base** and **student** columns (no document in the prefix -- the teacher
does not exist here, `doc_tokens` is not available in the usage corpus) on
the real llama.cpp fork, to compare them against the replica
(`engraft.eval`).

Row measured (never recomputed here): imported from `engraft.eval.answer_row`
(`tokens = [EOS] + text + answer_tokens`, `answer_spans` = indices of EVERY
answer token, the row predicting the FIRST answer token is
`min(answer_spans) - 1`). For an engine job with `logits: "last"` this is
exactly the last position when the prefix sent to the engine is
`tokens[: row + 1]` (the same prefix the replica uses for that row) -- the
prefix used both for the p_first/rank_first jobs and as the greedy's
starting point.

Reused (never copied): `run_job`/`logsoftmax64`/`rank_of`/`LensClient`/
`LensError`/`ENGINE_CFG`/`_decode_ids` from `engraft.engine`; `answer_row`
from `engraft.eval`.

Declared choices:
- **Fixed seed for the family-balanced greedy selection**: `GREEDY_SEED`
  below. Selection: for every family present in the split, shuffles the ids
  with `np.random.default_rng(GREEDY_SEED)` and takes one fragment per
  family round-robin until `--greedy-n` fragments are reached or every
  family is exhausted.
- **Greedy length**: exactly `len(answer_spans)` steps (no +2 slack --
  that only serves detecting degenerate continuations past the expected
  answer; here the exact textual comparison on the answer is enough).
- **`--dry-run` and the greedy**: the greedy is sequential -- every step past
  the first depends on the token the engine generated at the previous step,
  so it is NOT enumerable dry. `jobs.json` carries the 2 p_first/rank_first
  jobs per fragment IN FULL (`pfirst_jobs`, the jobs actually sent to the
  engine) and, for the greedy, only one descriptor per fragment/column
  (`greedy_descriptors`: id, column, n_steps, the first step's job in full).
- **Greedy without an overlay (base column)**: `engraft.engine.greedy_continuation`
  always does `str(overlay_path)`, so `overlay_path=None` would become the
  literal string `"None"` in the job, never `null`. `_greedy_continuation_local`
  below fixes this: same logic, same `run_job` (reused, unchanged), but
  `overlay: None` stays `None`.
- **Rank definition differs between engine and replica (irrelevant in
  practice, declared for the contract)**: `rank_of` here uses a stable
  `argsort` (ties: lower token index wins); `engraft.eval._rank_and_p`
  (replica) counts strictly-greater logits (ties share a rank). At real
  float logits an exact tie is nearly impossible; the engine/replica
  agreement below compares only the binary outcome rank==1, not the rank
  number, so this convention difference does not touch it.
- **Engine/replica agreement on rank 1**: the fraction of fragments (among
  those with `--replica-eval` resolved) where the engine's `rank_first==1`
  (student column) coincides with the replica's `rank_first.student==1`
  (both correct or both not) -- does not require identical ranks, only the
  binary "first place" outcome.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from engraft.engine import ENGINE_CFG, LensClient, _decode_ids, logsoftmax64, rank_of, run_job
from engraft.eval import answer_row
from engraft.table import PleTokenizer

GREEDY_SEED = 20260912


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
    tmp.replace(path)


# --------------------------------------------------------------------------
# Usage corpus / replica
# --------------------------------------------------------------------------


def load_fragments(usage_corpus_path: str, split: str) -> list[dict]:
    """Fragments of the requested split, WITHOUT the ones lacking
    `answer_spans` (a declared drop is safer than a crash mid-measurement --
    `answer_row` would otherwise raise `ValueError` partway through)."""
    data = json.loads(Path(usage_corpus_path).read_text())
    fragments_all = data.get("fragments", data)
    if split != "all":
        fragments_all = [f for f in fragments_all if f.get("split") == split]
    good, dropped = [], []
    for f in fragments_all:
        (good if f.get("answer_spans") else dropped).append(f)
    if dropped:
        print(f"load_fragments: {len(dropped)} fragments without answer_spans skipped: "
              f"{[f.get('id') for f in dropped]}", file=sys.stderr)
    return good


def load_replica_eval(path: str | None) -> dict[str, dict]:
    """`{id: record}` from `eval.json` (`engraft.eval.write_eval_report`);
    `{}` if `path` is `None` (`--replica-eval` is optional, the per-fragment
    comparison simply becomes absent, never an error)."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {r["id"]: r for r in data.get("fragments", [])}


# --------------------------------------------------------------------------
# Balanced greedy selection (fixed seed, round-robin per family)
# --------------------------------------------------------------------------


def select_greedy_fragments(fragments: list[dict], n: int, seed: int = GREEDY_SEED) -> list[dict]:
    """`n` fragments (or fewer if the split has fewer), round-robin over the
    families present (first-appearance order), each shuffled with a fixed
    `seed`."""
    by_family: dict[str, list[dict]] = {}
    order: list[str] = []
    for f in fragments:
        fam = f.get("family")
        if fam not in by_family:
            by_family[fam] = []
            order.append(fam)
        by_family[fam].append(f)
    rng = np.random.default_rng(seed)
    for fam in order:
        rng.shuffle(by_family[fam])
    picked: list[dict] = []
    idx_by_family = {fam: 0 for fam in order}
    while len(picked) < n and any(idx_by_family[fam] < len(by_family[fam]) for fam in order):
        for fam in order:
            if len(picked) >= n:
                break
            i = idx_by_family[fam]
            if i < len(by_family[fam]):
                picked.append(by_family[fam][i])
                idx_by_family[fam] = i + 1
    return picked


# --------------------------------------------------------------------------
# p_first/rank_first job (base + student), one fragment
# --------------------------------------------------------------------------


def pfirst_job(frag: dict, overlay: str | None, kind: str) -> tuple[dict, int]:
    """Job `logits: "last"` on the prefix `tokens[: row + 1]` (the same row
    as `engraft.eval.answer_row`): returns `(job, y)` with
    `y = tokens[row + 1]` (the first answer token, the target compared
    against argmax/rank)."""
    row = answer_row(frag)
    tokens = list(frag["tokens"])
    if not (0 <= row < len(tokens) - 1):
        raise ValueError(f"pfirst_job: {frag.get('id')!r} -- answer row {row} outside [0, {len(tokens) - 1})")
    y = tokens[row + 1]
    prefix = tokens[: row + 1]
    job = {"id": f"pf_{frag['id']}_{kind}", "text": "", "tokens": prefix, "overlay": overlay,
           "capture": [], "logits": "last"}
    return job, y


def measure_pfirst(client, raw_dir, log_, frag: dict, overlay: str | None, kind: str) -> dict:
    job, y = pfirst_job(frag, overlay, kind)
    result, row, _meta = run_job(client, raw_dir, job, log_)
    logp = logsoftmax64(row)
    return {
        "p_first": float(np.exp(logp[y])), "rank_first": rank_of(row, y),
        "argmax": int(np.argmax(row)), "overlay_hits": result.get("overlay_hits"),
    }


# --------------------------------------------------------------------------
# Greedy (full answer), base + student, selected fragments
# --------------------------------------------------------------------------


def greedy_prefix_and_answer(frag: dict) -> tuple[list[int], list[int]]:
    """`(prefix_tokens, answer_tokens)`: `prefix_tokens` = `tokens[: row + 1]`
    (the same prefix as the p_first jobs -- the greedy resumes exactly where
    the first answer token is predicted), `answer_tokens` = the tokens from
    `answer_spans`, in order."""
    row = answer_row(frag)
    tokens = list(frag["tokens"])
    spans = sorted(int(s) for s in frag["answer_spans"])
    answer_tokens = [tokens[s] for s in spans]
    return tokens[: row + 1], answer_tokens


def _greedy_continuation_local(client, raw_dir, tok, log_, prefix_tokens: list[int], overlay: str | None,
                                n: int, tag: str) -> tuple[list[int], str, bool]:
    """Like `engraft.engine.greedy_continuation` (same `run_job`, unchanged),
    but without that function's unconditional `str(overlay_path)`: calling it
    with `overlay_path=None` would produce `job["overlay"] = "None"` (a
    literal string, not JSON null). Here the base column also needs no
    overlay: `job["overlay"]` stays `None` (JSON null) when `overlay` is
    `None`, still reusing `run_job` without copying its logic."""
    seq = list(prefix_tokens)
    gen: list[int] = []
    for i in range(n):
        job = {"id": f"{tag}_greedy{i}", "text": "", "tokens": seq,
               "overlay": (str(overlay) if overlay is not None else None), "capture": [], "logits": "last"}
        _result, row, _meta = run_job(client, raw_dir, job, log_)
        nxt = int(np.argmax(row))
        gen.append(nxt)
        seq.append(nxt)
    text = _decode_ids(tok, gen)
    degenerate = (len(gen) >= 3 and gen[-1] == gen[-2] == gen[-3]) or text.strip() == ""
    return gen, text, degenerate


def measure_greedy(client, raw_dir, tok, log_, frag: dict, overlay: str | None, kind: str) -> dict:
    prefix, answer_tokens = greedy_prefix_and_answer(frag)
    gen, gen_text, degenerate = _greedy_continuation_local(
        client, raw_dir, tok, log_, prefix, overlay, len(answer_tokens), f"gr_{frag['id']}_{kind}",
    )
    answer_text = _decode_ids(tok, answer_tokens)
    return {"tokens": gen, "text": gen_text, "degenerate": degenerate, "exact_match": gen_text == answer_text,
            "answer_text": answer_text}


# --------------------------------------------------------------------------
# Job descriptors for --dry-run
# --------------------------------------------------------------------------


def _kinds(overlay: str | None) -> tuple[tuple[str, str | None], ...]:
    """Columns to measure: without an overlay (a base cell) ONLY `base`;
    with an overlay `base` + `student`."""
    return (("base", None),) if overlay is None else (("base", None), ("student", overlay))


def build_jobs(fragments: list[dict], overlay: str | None, greedy_fragments: list[dict]) -> dict:
    """Jobs to write to `jobs.json` dry (`pfirst_jobs`: the REAL jobs, in
    full, per fragment x {base, student}; `greedy_descriptors`: the greedy is
    sequential -- only the first step's job is dry-enumerable).
    `overlay=None` -> base column only."""
    pfirst_jobs = []
    for frag in fragments:
        for kind, ov in _kinds(overlay):
            job, y = pfirst_job(frag, ov, kind)
            pfirst_jobs.append({**job, "_target_token": y, "_fragment_id": frag["id"], "_family": frag.get("family")})
    greedy_descriptors = []
    for frag in greedy_fragments:
        prefix, answer_tokens = greedy_prefix_and_answer(frag)
        for kind, ov in _kinds(overlay):
            first_job = {"id": f"gr_{frag['id']}_{kind}_greedy0", "text": "", "tokens": prefix,
                         "overlay": ov, "capture": [], "logits": "last"}
            greedy_descriptors.append({
                "fragment_id": frag["id"], "family": frag.get("family"), "kind": kind,
                "n_steps": len(answer_tokens), "first_step_job": first_job,
                "note": "later steps not dry-enumerable (depend on the engine-generated token)",
            })
    n_greedy_engine_calls = sum(d["n_steps"] for d in greedy_descriptors)
    return {
        "pfirst_jobs": pfirst_jobs, "greedy_descriptors": greedy_descriptors,
        "n_pfirst_jobs": len(pfirst_jobs), "n_greedy_descriptors": len(greedy_descriptors),
        "n_greedy_engine_calls_expected": n_greedy_engine_calls,
    }


# --------------------------------------------------------------------------
# Per-fragment results + report
# --------------------------------------------------------------------------


def run_check(client, raw_dir, tok, log_, fragments: list[dict], overlay: str | None,
              greedy_fragments: list[dict], replica_by_id: dict) -> list[dict]:
    """`overlay=None` -> base cell: record with only `base` keys (and
    `greedy.base`), no `student`, no comparison against the replica."""
    greedy_ids = {f["id"] for f in greedy_fragments}
    records = []
    for frag in fragments:
        base = measure_pfirst(client, raw_dir, log_, frag, None, "base")
        rec = {"id": frag["id"], "family": frag.get("family"), "fact_ids": frag.get("fact_ids", []),
               "base": base}
        student = None
        if overlay is not None:
            student = measure_pfirst(client, raw_dir, log_, frag, overlay, "student")
            rec["student"] = student
        if frag["id"] in greedy_ids:
            rec["greedy"] = {"base": measure_greedy(client, raw_dir, tok, log_, frag, None, "base")}
            if overlay is not None:
                rec["greedy"]["student"] = measure_greedy(client, raw_dir, tok, log_, frag, overlay, "student")
        replica_rec = replica_by_id.get(frag["id"])
        if replica_rec is not None and student is not None:
            rep_p = replica_rec["p_first"]["student"]
            rep_rank = replica_rec["rank_first"]["student"]
            rec["replica"] = {"p_first": rep_p, "rank_first": rep_rank}
            rec["delta_p_first"] = abs(student["p_first"] - rep_p)
        records.append(rec)
        log_.info("%s measured (base rank=%s student rank=%s)", frag["id"], base["rank_first"],
                  student["rank_first"] if student is not None else "-")
    return records


def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def _p90(xs: list[float]) -> float | None:
    return float(np.percentile(xs, 90)) if xs else None


def build_report(records: list[dict]) -> str:
    by_family: dict[str, list[dict]] = {}
    for r in records:
        by_family.setdefault(r.get("family") or "?", []).append(r)

    lines = ["# engraft.engine_check report", ""]
    lines.append(f"Fragments measured: {len(records)}")
    lines.append("\n## By family")
    lines.append("| family | n | correct base | correct student (engine) | correct student (replica) |")
    lines.append("|---|---|---|---|---|")

    def _frac_correct(rs: list[dict], key: str) -> str:
        sub = [r for r in rs if key in r]
        if not sub:
            return "-"
        return f"{sum(1 for r in sub if r[key]['rank_first'] == 1) / len(sub):.2f}"

    def _frac_replica_correct(rs: list[dict]) -> str:
        sub = [r for r in rs if "replica" in r]
        if not sub:
            return "-"
        return f"{sum(1 for r in sub if r['replica']['rank_first'] == 1) / len(sub):.2f}"

    for fam in sorted(by_family):
        rs = by_family[fam]
        lines.append(
            f"| {fam} | {len(rs)} | {_frac_correct(rs, 'base')} | {_frac_correct(rs, 'student')} | "
            f"{_frac_replica_correct(rs)} |"
        )
    lines.append(
        f"| **total** | {len(records)} | {_frac_correct(records, 'base')} | "
        f"{_frac_correct(records, 'student')} | {_frac_replica_correct(records)} |"
    )

    lines.append("\n## Engine/replica agreement on rank 1 (student column)")
    with_replica = [r for r in records if "replica" in r]
    if with_replica:
        agree = sum(
            1 for r in with_replica
            if (r["student"]["rank_first"] == 1) == (r["replica"]["rank_first"] == 1)
        )
        lines.append(f"- agreement: {agree}/{len(with_replica)} = {agree / len(with_replica):.3f}")
        deltas = [r["delta_p_first"] for r in with_replica]
        lines.append(f"- median |Δp_first|: {_median(deltas):.4g}, p90: {_p90(deltas):.4g}")
    else:
        lines.append("- no `--replica-eval` given (or no common fragment): not computed")

    greedy_recs = [r for r in records if "greedy" in r]
    lines.append("\n## Greedy (full answer, exact match)")
    if greedy_recs:
        by_family_greedy: dict[str, list[dict]] = {}
        for r in greedy_recs:
            by_family_greedy.setdefault(r.get("family") or "?", []).append(r)
        lines.append("| family | n | exact match base | exact match student |")
        lines.append("|---|---|---|---|")
        for fam in sorted(by_family_greedy):
            rs = by_family_greedy[fam]
            em_base = sum(1 for r in rs if r["greedy"]["base"]["exact_match"]) / len(rs)
            with_student = [r for r in rs if "student" in r["greedy"]]
            em_student = (f"{sum(1 for r in with_student if r['greedy']['student']['exact_match']) / len(with_student):.2f}"
                          if with_student else "-")
            lines.append(f"| {fam} | {len(rs)} | {em_base:.2f} | {em_student} |")
    else:
        lines.append("(no fragment with a greedy -- `--greedy-n 0` or an empty split)")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usage-corpus", required=False, help="usage_corpus_resolved.json")
    p.add_argument("--overlay", required=False, default=None,
                   help="merged.pleo (engraft.descend_corpus); absent = base cell (base column only)")
    p.add_argument("--split", default="heldout", choices=["heldout", "train", "all"])
    p.add_argument("--replica-eval", default=None, help="eval.json from engraft.eval (optional)")
    p.add_argument("--out", required=True)
    p.add_argument("--lens-cmd", default=None)
    p.add_argument("--greedy-n", type=int, default=24)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--config", default=None, help="engraft.toml path (tokenizer resolution)")
    args = p.parse_args(argv)

    out_dir = Path(args.out).resolve()  # streaming-mode lens requires absolute paths
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        results_path = out_dir / "results.json"
        if not results_path.exists():
            print(f"--render-only: {results_path} not found", file=sys.stderr)
            return 2
        records = json.loads(results_path.read_text())
        (out_dir / "report.md").write_text(build_report(records))
        print(f"report regenerated (--render-only, no engine): {out_dir / 'report.md'}")
        return 0

    if not args.usage_corpus:
        print("--usage-corpus required (absent only with --render-only)", file=sys.stderr)
        return 2
    overlay = str(Path(args.overlay).resolve()) if args.overlay else None

    fragments = load_fragments(args.usage_corpus, args.split)
    if not fragments:
        print(f"no fragment with split={args.split!r}", file=sys.stderr)
        return 2
    replica_by_id = load_replica_eval(args.replica_eval)
    greedy_fragments = select_greedy_fragments(fragments, args.greedy_n)

    if args.dry_run:
        jobs = build_jobs(fragments, overlay, greedy_fragments)
        _atomic_write_json(out_dir / "jobs.json", jobs)
        print(
            f"jobs.json: {jobs['n_pfirst_jobs']} full p_first/rank_first jobs + "
            f"{jobs['n_greedy_descriptors']} greedy descriptors "
            f"({jobs['n_greedy_engine_calls_expected']} expected engine calls for the greedy, "
            "sequential, not dry-enumerable)"
        )
        return 0

    if not args.lens_cmd:
        print("--lens-cmd required (absent only with --dry-run/--render-only)", file=sys.stderr)
        return 2

    log_ = logging.getLogger("engraft.engine_check")
    log_.setLevel(logging.INFO)
    log_.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log_.addHandler(h)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    from engraft.config import load as load_config
    engraft_cfg = load_config(args.config)
    tok = PleTokenizer(engraft_cfg.get_path("model.tokenizer"))

    argv_engine = shlex.split(args.lens_cmd) + ["--jobs", "-", "--out", str(raw_dir)] + ENGINE_CFG["q8"]["args"]
    client = LensClient(argv_engine, raw_dir, out_dir / "engine.log", env=ENGINE_CFG["q8"]["env"])
    try:
        t0 = time.time()
        records = run_check(client, raw_dir, tok, log_, fragments, overlay, greedy_fragments, replica_by_id)
        log_.info("measurement completed in %.1fs", time.time() - t0)
    finally:
        client.close()

    _atomic_write_json(out_dir / "results.json", records)
    (out_dir / "report.md").write_text(build_report(records))
    print(f"results.json/report.md written to {out_dir}, n_fragments={len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

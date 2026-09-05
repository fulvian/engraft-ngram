"""Engine-side check of the chained grafts, against the real (or fake) engine.

Reuse (never modified here): `run_job`, `run_job_all`, `check_expected_hits`,
`greedy_continuation`, `corpus_nll_mean`, `logsoftmax64`, `rank_of`, `RowSet`,
`write_pleo`, `read_pleo`, `read_plert1`, `LensClient`, `LensError`,
`ENGINE_CFG`, `_decode_ids` from `engraft.engine` and `engraft.lens`. Reads
`facts/facts_resolved.json` and `keys.json` (never recomputed here: tokenizing
and key checks are already done by `engraft.facts`) and
`results/<date>/{facts/<fid>/<fid>.pleo,.json,merged.pleo,merged_manifest.json}`
(written by `engraft.run`).

Q8 phase (default engine): for every fact, with its **own** overlay
(`<fid>.pleo`): `p_first`/`rank_first` of the answer's first token, a greedy
run of n+2 tokens (`answer_reproduced`), sisters (argmax, delta-logp, expected
8 hits), 16 expected hits on the trigger. With `merged.pleo`: the same
measurements for every included fact (per the manifest), plus paraphrases,
it/en corpus (mean delta-NLL), and documents (delta-logp per position, split
into answer positions and other positions, with a per-position `positions`
list -- `fid`/`answer_position` at each response position).

F32 phase (full-precision engine): for every non-excluded graft (fid, i), a
base job on the tokens `trigger+answer[:i]` (`logp_base_f32`; the comparison
against `logp_y` at jsonl step 0 is asserted here) -- with no overlay for
i == 0, or with the union of `ckpt_<fid>_<j>_final.pleo` for j < i for i >= 1
(the replica's own prefix at position i already includes the earlier grafts;
the base job must read the same prefix, not the untouched table) -- and a free
job with the **fact's** overlay (`<fid>.pleo`) on the same tokens, with
`routing_record`: `p_y_free`, compared against `final_p_free` from `<fid>.json`
(`delta_p_q5`, `q5_pass` <= 0.05) and `prefix_routing_diverging` (a list of
layer/position pairs) against the graft's `.plert1`
(`ckpt_<fid>_<i>_final.plert1`).

`engine_check.json` is written incrementally and atomically after every fact
(Q8) and every graft (F32): an interrupted engine window resumes from the
facts/grafts already measured.

Dry-run switches (each recorded with `skipped_reason`, verified by `--report`):
`--target-token-map <path.json>` ({fid: {pos: token}}, a token within the fake
engine's vocabulary of 64), `--no-assert-overlay-hits`, `--skip-corpus`,
`--skip-docs`. No switch is enabled in a real run.

`--render-only` regenerates `report.md` from an existing `engine_check.json`
alone (no `--lens-cmd`, no engine, no config): use it to re-render after a
`build_report` change without re-running the engine window.

Usage (real run):
  uv run engraft-check 2026-01-01 --lens-cmd "<engine> ..."

Usage (dry run, fake engine):
  uv run engraft-check 2026-01-01-dryrun --lens-cmd \
      "uv run python -m engraft.testing.fake_lens --fake-table" \
      --target-token-map results/2026-01-01-dryrun/target_token_map.json \
      --no-assert-overlay-hits --skip-corpus --skip-docs \
      --results-dir results/2026-01-01-dryrun
  (the token map is written by `engraft-run 2026-01-01-dryrun --fake`)
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

from engraft.config import load as load_config
from engraft.engine import (
    ENGINE_CFG,
    LensClient,
    LensError,
    _decode_ids,
    check_expected_hits,
    corpus_nll_mean,
    greedy_continuation,
    logsoftmax64,
    rank_of,
    run_job,
    run_job_all,
)
from engraft.lens import RowSet, read_pleo, read_plert1, write_pleo
from engraft.table import PleTable, PleTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTS_DIR = REPO_ROOT / "facts"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus"
DEFAULT_DOCS_DIR = FACTS_DIR / "docs"
Q5_TOLERANCE = 0.05
CONSISTENCY_TOLERANCE_NAT = 1e-3


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _setup_log(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("engraft.check")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(h)
    return log


def _load_resolved() -> dict:
    return json.loads((FACTS_DIR / "facts_resolved.json").read_text())["facts"]


def _load_keys() -> dict:
    return json.loads((FACTS_DIR / "keys.json").read_text())


def _target_token_map(args) -> dict[str, dict[str, int]]:
    if not args.target_token_map:
        return {}
    return json.loads(Path(args.target_token_map).read_text())


def _y_for(args, ttmap: dict, fid: str, pos: int, real_token: int) -> int:
    if not args.target_token_map:
        return real_token
    entry = ttmap.get(fid, {}).get(str(pos))
    if entry is None:
        raise RuntimeError(f"--target-token-map: no entry for {fid!r} position {pos}")
    return int(entry)


def _diverging_pairs(free: dict[int, np.ndarray], replica: dict[int, np.ndarray]) -> list[dict]:
    """Q5: position-by-position, layer-by-layer comparison (unlike a
    last-position-only comparison)."""
    out = []
    for il in sorted(set(free.keys()) & set(replica.keys())):
        a = np.asarray(free[il])
        b = np.asarray(replica[il])
        n = min(a.shape[0], b.shape[0])
        for pos in range(n):
            if frozenset(int(x) for x in a[pos]) != frozenset(int(x) for x in b[pos]):
                out.append({"layer": il, "position": pos})
    return out


# --------------------------------------------------------------------------
# Fase Q8
# --------------------------------------------------------------------------


def _measure_with_overlay(
    client, raw_dir, tok, log, fid: str, overlay_path: Path, resolved_fact: dict,
    tag_prefix: str, args, ttmap: dict, sisters_base: dict,
) -> dict:
    trigger_tokens = resolved_fact["trigger_tokens"]
    answer_tokens = resolved_fact["answer_tokens"]
    y0 = _y_for(args, ttmap, fid, 0, answer_tokens[0])

    job = {
        "id": f"{tag_prefix}_{fid}_trigger", "text": "", "tokens": trigger_tokens,
        "overlay": str(overlay_path), "capture": [], "logits": "last",
    }
    result, row, _meta = run_job(client, raw_dir, job, log)
    logp = logsoftmax64(row)
    entry: dict = {
        "p_first": float(np.exp(logp[y0])), "rank_first": rank_of(row, y0),
        "argmax": int(np.argmax(row)), "hits": result.get("overlay_hits"),
    }
    if args.assert_overlay_hits:
        check_expected_hits(job["id"], result, 16)

    n_greedy = len(answer_tokens) + 2
    gen, gen_text, degenerate = greedy_continuation(
        client, raw_dir, tok, log, trigger_tokens, overlay_path, n_greedy, f"{tag_prefix}_{fid}",
    )
    if args.target_token_map:
        # The fake engine does not know the real tokens: a greedy comparison is
        # meaningless, only p_first/rank_first (via --target-token-map) remain evidence.
        entry["answer_reproduced"] = None
        entry["skipped_reason_answer_reproduced"] = "fake engine: greedy not comparable with --target-token-map"
    else:
        entry["answer_reproduced"] = gen[: len(answer_tokens)] == answer_tokens
    entry["greedy"] = {"tokens": gen, "text": gen_text, "degenerate": degenerate}

    sisters_entry = {}
    for sidx, sister in enumerate(resolved_fact.get("sisters", [])):
        sid = f"s{sidx}"
        job_s = {
            "id": f"{tag_prefix}_{fid}_sib_{sid}", "text": "", "tokens": sister["tokens"],
            "overlay": str(overlay_path), "capture": [], "logits": "last",
        }
        result_s, row_s, _m = run_job(client, raw_dir, job_s, log)
        base_argmax, base_logp = sisters_base[sid]
        logp_s = logsoftmax64(row_s)
        argmax_s = int(np.argmax(row_s))
        sisters_entry[sid] = {
            "argmax": argmax_s, "argmax_unchanged": argmax_s == base_argmax,
            "delta_logp_argmax_base": float(logp_s[base_argmax] - base_logp[base_argmax]),
            "hits": result_s.get("overlay_hits"),
        }
        if args.assert_overlay_hits:
            check_expected_hits(job_s["id"], result_s, 8)
    entry["sisters"] = sisters_entry
    return entry


def run_q8_phase(args, run_dir: Path, raw_dir: Path, log, tok, facts_resolved: dict, keys: dict, results_dir: Path) -> dict:
    argv = shlex.split(args.lens_cmd) + ["--jobs", "-", "--out", str(raw_dir)] + ENGINE_CFG["q8"]["args"]
    client = LensClient(argv, raw_dir, run_dir / "engine_q8.log", env=ENGINE_CFG["q8"]["env"])
    ttmap = _target_token_map(args)
    out: dict = {"facts": {}, "merged": {"facts": {}, "corpus": {}, "docs": {}}, "sisters_base": {}}
    try:
        # basi (senza overlay) per ogni sorella, parafrasi -- serve per Δlogp/argmax_unchanged
        sisters_base: dict[str, dict[str, tuple[int, np.ndarray]]] = {}
        paraphrase_base: dict[str, dict[str, tuple[int, np.ndarray]]] = {}
        for fid, rf in facts_resolved.items():
            sisters_base[fid] = {}
            for sidx, sister in enumerate(rf.get("sisters", [])):
                sid = f"s{sidx}"
                job = {"id": f"q8_base_{fid}_sib_{sid}", "text": "", "tokens": sister["tokens"], "overlay": None, "capture": [], "logits": "last"}
                _r, row, _m = run_job(client, raw_dir, job, log)
                sisters_base[fid][sid] = (int(np.argmax(row)), logsoftmax64(row))
            paraphrase_base[fid] = {}
            for key in ("paraphrase_same_tail", "paraphrase_other_tail"):
                tokens = rf[key]["tokens"]
                job = {"id": f"q8_base_{fid}_{key}", "text": "", "tokens": tokens, "overlay": None, "capture": [], "logits": "last"}
                _r, row, _m = run_job(client, raw_dir, job, log)
                paraphrase_base[fid][key] = (int(np.argmax(row)), logsoftmax64(row))
        out["sisters_base"] = {fid: list(v.keys()) for fid, v in sisters_base.items()}

        for fid, rf in facts_resolved.items():
            fact_pleo = results_dir / "facts" / fid / f"{fid}.pleo"
            if not fact_pleo.exists():
                out["facts"][fid] = {"skipped_reason": f"{fact_pleo} missing (fact not closed)"}
                _atomic_write_json(run_dir / "engine_check.json", out)
                continue
            out["facts"][fid] = _measure_with_overlay(
                client, raw_dir, tok, log, fid, fact_pleo, rf, "q8", args, ttmap, sisters_base[fid],
            )
            _atomic_write_json(run_dir / "engine_check.json", out)
            log.info("Q8 fact %s measured", fid)

        merged_pleo = results_dir / "merged.pleo"
        manifest = json.loads((results_dir / "merged_manifest.json").read_text())
        if merged_pleo.exists():
            for fid in manifest["included"]:
                rf = facts_resolved[fid]
                entry = _measure_with_overlay(
                    client, raw_dir, tok, log, fid, merged_pleo, rf, "q8m", args, ttmap, sisters_base[fid],
                )
                # Q4: parafrasi sull'overlay unico
                paraphrases = {}
                for key in ("paraphrase_same_tail", "paraphrase_other_tail"):
                    tokens = rf[key]["tokens"]
                    job = {
                        "id": f"q8m_{fid}_{key}", "text": "", "tokens": tokens,
                        "overlay": str(merged_pleo), "capture": [], "logits": "last",
                    }
                    result_p, row_p, _m = run_job(client, raw_dir, job, log)
                    base_argmax, base_logp = paraphrase_base[fid][key]
                    logp_p = logsoftmax64(row_p)
                    y0 = _y_for(args, ttmap, fid, 0, rf["answer_tokens"][0])
                    paraphrases[key] = {
                        "p_first": float(np.exp(logp_p[y0])), "rank_first": rank_of(row_p, y0),
                        "argmax": int(np.argmax(row_p)), "argmax_unchanged_vs_base": int(np.argmax(row_p)) == base_argmax,
                        "delta_logp_argmax_base": float(logp_p[base_argmax] - base_logp[base_argmax]),
                        "hits": result_p.get("overlay_hits"),
                    }
                entry["paraphrases"] = paraphrases
                out["merged"]["facts"][fid] = entry
                _atomic_write_json(run_dir / "engine_check.json", out)
                log.info("Q2/Q4 fact %s measured (merged)", fid)

            if args.skip_corpus:
                out["merged"]["corpus"] = {"skipped_reason": "fake engine vocabulary too small for the corpus"}
            else:
                out["merged"]["corpus"] = _measure_corpus(client, raw_dir, tok, log, args, merged_pleo)
            _atomic_write_json(run_dir / "engine_check.json", out)

            if args.skip_docs:
                out["merged"]["docs"] = {"skipped_reason": "fake engine vocabulary too small for the documents"}
            else:
                out["merged"]["docs"] = _measure_docs(client, raw_dir, tok, log, args, merged_pleo, facts_resolved)
            _atomic_write_json(run_dir / "engine_check.json", out)
    finally:
        client.close()
    return out


def _measure_corpus(client, raw_dir, tok, log, args, merged_pleo: Path) -> dict:
    out = {}
    for lang in ("it", "en"):
        text = (Path(args.corpus_dir) / f"{lang}.txt").read_text()
        c_tokens = tok.encode(text)
        job_base = {"id": f"q8m_corpus_{lang}_base", "text": "", "tokens": c_tokens, "overlay": None, "capture": [], "logits": "all"}
        _r, all_base, meta = run_job_all(client, raw_dir, job_base, log)
        job_m = {"id": f"q8m_corpus_{lang}_merged", "text": "", "tokens": c_tokens, "overlay": str(merged_pleo), "capture": [], "logits": "all"}
        result_m, all_m, _meta_m = run_job_all(client, raw_dir, job_m, log)
        targets = np.asarray(c_tokens[1:])
        n_vocab = int(meta["n_vocab"])
        if int(targets.max()) >= n_vocab:
            out[lang] = {"skipped_reason": f"n_vocab={n_vocab} too small for the real corpus"}
            continue
        nll_base = corpus_nll_mean(all_base[:-1], targets)
        nll_merged = corpus_nll_mean(all_m[:-1], targets)
        out[lang] = {
            "nll_base": nll_base, "nll_merged": nll_merged, "delta_nll": nll_merged - nll_base,
            "overlay_hits": result_m.get("overlay_hits"),
        }
    return out


def _measure_docs(client, raw_dir, tok, log, args, merged_pleo: Path, facts_resolved: dict) -> dict:
    out = {}
    for lang in ("it", "en"):
        doc_path = DEFAULT_DOCS_DIR / f"{lang}.txt"
        if not doc_path.exists():
            out[lang] = {"skipped_reason": f"{doc_path} missing"}
            continue
        doc_tokens = tok.encode(doc_path.read_text())
        # For every response position (t_pred) of the document, the fact and its
        # chain position (`i` in `doc_positions.positions`) that wrote it.
        response_map: dict[int, dict] = {}
        for fid, rf in facts_resolved.items():
            if rf.get("doc_id") != lang or rf.get("doc_positions") is None:
                continue
            for p in rf["doc_positions"].get("positions", []):
                response_map[int(p["t_pred"])] = {"fid": fid, "answer_position": int(p["i"])}
        response_positions = set(response_map.keys())

        job_base = {"id": f"q8m_doc_{lang}_base", "text": "", "tokens": doc_tokens, "overlay": None, "capture": [], "logits": "all"}
        _r, all_base, meta = run_job_all(client, raw_dir, job_base, log)
        job_m = {"id": f"q8m_doc_{lang}_merged", "text": "", "tokens": doc_tokens, "overlay": str(merged_pleo), "capture": [], "logits": "all"}
        result_m, all_m, _meta_m = run_job_all(client, raw_dir, job_m, log)
        targets = np.asarray(doc_tokens[1:])
        n_vocab = int(meta["n_vocab"])
        if int(targets.max()) >= n_vocab:
            out[lang] = {"skipped_reason": f"n_vocab={n_vocab} too small for the real document"}
            continue

        deltas_response, deltas_other = [], []
        positions: list[dict] = []
        for p in range(len(targets)):
            target_token = int(targets[p])
            lp_base = float(logsoftmax64(all_base[p])[target_token])
            lp_merged = float(logsoftmax64(all_m[p])[target_token])
            delta = lp_merged - lp_base
            is_response = p in response_positions
            (deltas_response if is_response else deltas_other).append(delta)

            entry = {
                "t_pred": p,
                "target_token": target_token,
                "is_response": is_response,
                "logp_base": lp_base,
                "logp_merged": lp_merged,
                "delta": delta,
            }
            if tok is not None:
                entry["target_str"] = tok.decode_token(target_token)
            resp = response_map.get(p)
            if resp is not None:
                entry["fid"] = resp["fid"]
                entry["answer_position"] = resp["answer_position"]
            positions.append(entry)

        def _stats(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0, "mean": None, "min": None, "max": None}
            arr = np.asarray(xs)
            return {"n": len(xs), "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}

        out[lang] = {
            "response_positions": _stats(deltas_response),
            "other_positions": _stats(deltas_other),
            "overlay_hits": result_m.get("overlay_hits"),
            "positions": positions,
        }
    return out


# --------------------------------------------------------------------------
# Fase F32
# --------------------------------------------------------------------------


def run_f32_phase(args, run_dir: Path, raw_dir: Path, log, facts_resolved: dict, results_dir: Path) -> dict:
    routing_dir = run_dir / "routing"
    routing_dir.mkdir(parents=True, exist_ok=True)
    base_argv = shlex.split(args.lens_cmd) + ["--jobs", "-", "--out", str(raw_dir)]
    argv = base_argv + ENGINE_CFG["f32"]["args"]
    try:
        client = LensClient(argv, raw_dir, run_dir / "engine_f32.log", env=ENGINE_CFG["f32"]["env"])
    except LensError:
        fb = ENGINE_CFG["f32"].get("fallback_args")
        if fb is None:
            raise
        argv = base_argv + fb
        client = LensClient(argv, raw_dir, run_dir / "engine_f32.log", env=ENGINE_CFG["f32"]["env"])

    ttmap = _target_token_map(args)
    out: dict = {"grafts": {}}
    try:
        for fid, rf in facts_resolved.items():
            fact_json_path = results_dir / "facts" / fid / f"{fid}.json"
            fact_pleo = results_dir / "facts" / fid / f"{fid}.pleo"
            if not fact_json_path.exists() or not fact_pleo.exists():
                continue
            fact_summary = json.loads(fact_json_path.read_text())
            innesti_by_pos = {it["position"]: it for it in fact_summary["grafts"]}
            trigger_tokens = rf["trigger_tokens"]
            answer_tokens = rf["answer_tokens"]

            for i in sorted(innesti_by_pos.keys()):  # solo le posizioni davvero chiuse (spec: legge <fid>.json)
                tokens_i = trigger_tokens + answer_tokens[:i]
                y = _y_for(args, ttmap, fid, i, answer_tokens[i])
                key = f"{fid}_{i}"

                # For i >= 1 the replica's own prefix (graft.prepare_innesto) already
                # includes the overlay of grafts < i: the base job must read the same
                # prefix, not the untouched table, or this compares two different
                # bases. Written under `run_dir` (this invocation's own output), never
                # under `results_dir`: in a dry run `results_dir` is the read-only real
                # directory (`--results-dir`); in a real run `run_dir == results_dir`,
                # so the file still lands in `results/<date>/facts/<fid>/` as expected.
                # For i == 0 there is no earlier graft: overlay=None, unchanged.
                base_overlay_path = None
                if i > 0:
                    prefix_dir = run_dir / "facts" / fid
                    prefix_dir.mkdir(parents=True, exist_ok=True)
                    base_overlay_path = prefix_dir / f"{fid}_prefix{i}.pleo"
                    parts = [
                        read_pleo(results_dir / "facts" / fid / f"ckpt_{fid}_{j}_final.pleo")
                        for j in range(i)
                    ]
                    rows_prefix, data_prefix = RowSet.build_overlay(parts)
                    write_pleo(base_overlay_path, rows_prefix, data_prefix)

                job_base = {
                    "id": f"f32_{key}_base", "text": "", "tokens": tokens_i,
                    "overlay": str(base_overlay_path) if base_overlay_path is not None else None,
                    "capture": [], "logits": "last",
                }
                _r, row_base, _m = run_job(client, raw_dir, job_base, log)
                logp_base_f32 = float(logsoftmax64(row_base)[y])

                free_routing_path = routing_dir / f"{key}_free.bin"
                job_free = {
                    "id": f"f32_{key}_free", "text": "", "tokens": tokens_i,
                    "overlay": str(fact_pleo), "capture": [], "logits": "last",
                    "routing_record": str(free_routing_path),
                }
                _r2, row_free, _m2 = run_job(client, raw_dir, job_free, log)
                p_y_free = float(np.exp(logsoftmax64(row_free)[y]))

                entry = {"logp_base_f32": logp_base_f32, "p_y_free": p_y_free}

                stem_final = f"ckpt_{fid}_{i}_final"
                plert1_path = results_dir / "facts" / fid / f"{stem_final}.plert1"
                if plert1_path.exists():
                    replica_routing = read_plert1(plert1_path)
                    free_routing = read_plert1(free_routing_path)
                    diverging = _diverging_pairs(free_routing, replica_routing)
                    entry["prefix_routing_diverging"] = diverging

                    innesto = innesti_by_pos.get(i, {})
                    p_replica_free = innesto.get("final_p_free")
                    if p_replica_free is not None:
                        delta_q5 = p_y_free - p_replica_free
                        entry["p_replica_free"] = p_replica_free
                        entry["delta_p_q5"] = delta_q5
                        entry["q5_pass"] = abs(delta_q5) <= Q5_TOLERANCE

                    # consistency check, second half: |logp_y at jsonl step 0 -
                    # logp_base_f32| <= 1e-3 nat, only for a non-perturbed graft
                    # (no `_seed` in the fact name here: main facts never have one).
                    log_path = results_dir / "facts" / fid / f"descend_{fid}_{i}.jsonl"
                    if log_path.exists():
                        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
                        if lines:
                            rec0 = json.loads(lines[0])
                            diff = abs(rec0["logp_y"] - logp_base_f32)
                            entry["diff_logp_y_step0_vs_base_f32"] = diff
                            entry["consistency_pass"] = diff <= CONSISTENCY_TOLERANCE_NAT
                            if not args.target_token_map and not entry["consistency_pass"]:
                                raise RuntimeError(
                                    f"{key}: consistency check failed, |logp_y step0 - logp_base_f32| "
                                    f"= {diff} > {CONSISTENCY_TOLERANCE_NAT}"
                                )

                out["grafts"][key] = entry
        return out
    finally:
        client.close()


def build_report(q8: dict, f32: dict) -> str:
    lines = ["# engine check report", "", "## Q1 (own overlay, per-fact)"]
    for fid, entry in q8.get("facts", {}).items():
        if "skipped_reason" in entry:
            lines.append(f"- {fid}: SKIPPED ({entry['skipped_reason']})")
            continue
        lines.append(
            f"- {fid}: p_first={entry.get('p_first')} rank_first={entry.get('rank_first')} "
            f"answer_reproduced={entry.get('answer_reproduced')} hits={entry.get('hits')}"
        )
    lines.append("\n## Q2/Q4 (merged overlay, corpus of all facts)")
    for fid, entry in q8.get("merged", {}).get("facts", {}).items():
        lines.append(
            f"- {fid}: p_first={entry.get('p_first')} rank_first={entry.get('rank_first')} "
            f"answer_reproduced={entry.get('answer_reproduced')} hits={entry.get('hits')}"
        )
        # The paraphrases (Q4) were already in engine_check.json (`entry["paraphrases"]`,
        # written by run_q8_phase) but never rendered here -- same tail / other tail,
        # only the fields present.
        paraphrases = entry.get("paraphrases") or {}
        for pkey, p in paraphrases.items():
            lines.append(
                f"    - {pkey}: p_first={p.get('p_first')} rank_first={p.get('rank_first')} "
                f"argmax_unchanged_vs_base={p.get('argmax_unchanged_vs_base')} "
                f"delta_logp_argmax_base={p.get('delta_logp_argmax_base')}"
            )
    lines.append(f"\ncorpus: {q8.get('merged', {}).get('corpus')}")
    docs = q8.get("merged", {}).get("docs", {}) or {}
    lines.append("\ndocs (Q3, aggregate statistics):")
    if "skipped_reason" in docs:
        lines.append(f"- SKIPPED ({docs['skipped_reason']})")
    else:
        for lang, entry in docs.items():
            if not isinstance(entry, dict):
                continue
            if "skipped_reason" in entry:
                lines.append(f"- {lang}: SKIPPED ({entry['skipped_reason']})")
            else:
                lines.append(
                    f"- {lang}: response={entry.get('response_positions')} "
                    f"other={entry.get('other_positions')} overlay_hits={entry.get('overlay_hits')}"
                )
    # Table of response positions only (which fact generalizes to the document and
    # which does not), not the whole per-position dictionary.
    lines.append("\n### docs -- response positions")
    lines.append("| lang | fid | position | target | logp_base | logp_merged | delta |")
    lines.append("|---|---|---|---|---|---|---|")
    for lang, entry in docs.items():
        if not isinstance(entry, dict):
            continue
        for p in entry.get("positions", []):
            if not p.get("is_response"):
                continue
            target_label = p.get("target_str", p.get("target_token"))
            lines.append(
                f"| {lang} | {p.get('fid')} | {p.get('answer_position')} | {target_label} | "
                f"{p['logp_base']:.4f} | {p['logp_merged']:.4f} | {p['delta']:.4f} |"
            )
    lines.append("\n## Q5 (F32 fidelity)")
    for key, entry in f32.get("grafts", {}).items():
        lines.append(
            f"- {key}: logp_base_f32={entry.get('logp_base_f32')} p_y_free={entry.get('p_y_free')} "
            f"q5_pass={entry.get('q5_pass')} diverging={len(entry.get('prefix_routing_diverging', []))} "
            f"consistency_pass={entry.get('consistency_pass')}"
        )
    return "\n".join(lines) + "\n"


def _check_report(run_dir: Path, results_dir: Path, facts_resolved: dict, skipped_flags: dict, log) -> int:
    ec_path = run_dir / "engine_check.json"
    if not ec_path.exists():
        log.error("--report: %s not found", ec_path)
        return 2
    data = json.loads(ec_path.read_text())

    for switch, active in skipped_flags.items():
        if active and not data.get(f"skipped_reason_{switch}") and switch not in ("target_token_map",):
            pass  # i flag skip_corpus/skip_docs si verificano sotto via i campi merged

    facts_required = ["p_first", "rank_first", "answer_reproduced", "sisters", "hits"]
    for fid in facts_resolved:
        entry = data.get("q8", {}).get("facts", {}).get(fid)
        if entry is None:
            log.error("--report: missing Q8 fact: %r", fid)
            return 2
        if "skipped_reason" in entry:
            continue
        for field in facts_required:
            if field not in entry:
                log.error("--report: Q8 fact %r: missing field %r", fid, field)
                return 2

    manifest_path = results_dir / "merged_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        merged_facts = data.get("q8", {}).get("merged", {}).get("facts", {})
        for fid in manifest.get("included", []):
            entry = merged_facts.get(fid)
            if entry is None:
                log.error("--report: missing merged fact: %r", fid)
                return 2
            for field in facts_required + ["paraphrases"]:
                if field not in entry:
                    log.error("--report: merged fact %r: missing field %r", fid, field)
                    return 2

        merged = data.get("q8", {}).get("merged", {})
        for section in ("corpus", "docs"):
            val = merged.get(section)
            if val is None:
                log.error("--report: missing merged section %r", section)
                return 2
            if isinstance(val, dict) and "skipped_reason" in val and not val["skipped_reason"]:
                log.error("--report: merged section %r skipped without skipped_reason", section)
                return 2
            if isinstance(val, dict):
                for lang, lang_val in val.items():
                    if isinstance(lang_val, dict) and lang_val.get("skipped_reason") == "":
                        log.error("--report: %r/%r skipped without skipped_reason", section, lang)
                        return 2

    f32_required = ["logp_base_f32", "p_y_free", "q5_pass", "prefix_routing_diverging"]
    f32_entries = data.get("f32", {}).get("grafts", {})
    for fid, rf in facts_resolved.items():
        for i in range(len(rf["answer_tokens"])):
            key = f"{fid}_{i}"
            entry = f32_entries.get(key)
            if entry is None:
                continue  # fatto non chiuso: nessun job F32 previsto
            for field in f32_required:
                if field not in entry:
                    log.error("--report: F32 graft %r: missing field %r", key, field)
                    return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--config", default=None, help="path to engraft.toml (default: ./engraft.toml)")
    p.add_argument("--lens-cmd", required=False)
    p.add_argument("--results-dir", default=None, help="directory with facts/<fid>/<fid>.pleo and merged.pleo (default: results/<date>)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "results"))
    p.add_argument("--target-token-map", default=None)
    p.add_argument("--skip-corpus", action="store_true")
    p.add_argument("--skip-docs", action="store_true")
    p.add_argument("--no-assert-overlay-hits", dest="assert_overlay_hits", action="store_false", default=True)
    p.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    p.add_argument("--report", action="store_true")
    p.add_argument("--render-only", action="store_true", help="regenerate report.md from an existing engine_check.json, no engine involved")
    args = p.parse_args(argv)

    run_dir = Path(args.out_root) / args.data
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_log(run_dir)

    if args.render_only:
        ec_path = run_dir / "engine_check.json"
        if not ec_path.exists():
            log.error("--render-only: %s not found", ec_path)
            return 2
        data = json.loads(ec_path.read_text())
        report = build_report(data.get("q8", {}), data.get("f32", {}))
        (run_dir / "report.md").write_text(report)
        log.info("report: %s", run_dir / "report.md")
        return 0

    results_dir = Path(args.results_dir) if args.results_dir else run_dir
    facts_resolved = _load_resolved()
    keys = _load_keys()

    if args.report:
        skipped_flags = {
            "skip_corpus": args.skip_corpus, "skip_docs": args.skip_docs,
            "no_assert_overlay_hits": not args.assert_overlay_hits,
            "target_token_map": bool(args.target_token_map),
        }
        return _check_report(run_dir, results_dir, facts_resolved, skipped_flags, log)

    if not args.lens_cmd:
        log.error("--lens-cmd required (only omitted with --report)")
        return 2

    engraft_cfg = load_config(args.config)
    tok = PleTokenizer(engraft_cfg.get_path("model.tokenizer"))
    fork_commit = engraft_cfg.get("engine.fork_commit")

    t0 = time.time()
    q8 = run_q8_phase(args, run_dir, raw_dir, log, tok, facts_resolved, keys, results_dir)
    log.info("Q8 phase: %.1fs", time.time() - t0)

    t0 = time.time()
    f32 = run_f32_phase(args, run_dir, raw_dir, log, facts_resolved, results_dir)
    log.info("F32 phase: %.1fs", time.time() - t0)

    engine_check = {
        "q8": q8, "f32": f32, "fork_commit": fork_commit,
        "skip_corpus": bool(args.skip_corpus), "skip_docs": bool(args.skip_docs),
        "assert_overlay_hits": bool(args.assert_overlay_hits),
        "target_token_map": args.target_token_map,
    }
    _atomic_write_json(run_dir / "engine_check.json", engine_check)

    report = build_report(q8, f32)
    (run_dir / "report.md").write_text(report)
    log.info("report: %s", run_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

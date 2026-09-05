"""CPU runner: chained fact grafting over a replica of the PLE layer.

Single load (weights and table loaded once): facts are processed in a fixed
order (memory facts, language-alternated, then counterfactuals), then two
perturbed-seed repeats of two designated facts. For each fact,
`engraft.replica.graft.graft_fact` (resumable per graft via `state.json`,
clearing the jsonl of any graft left open before relaunching: `descend` opens
it in append mode). At the end: `merged.pleo` (union of all non-excluded
facts, per `keys.json`), `merged_manifest.json`, `summary.json` (repeat
concordance, timing/RSS/conflicts).

`on_step` is an optional hook, called after every gradient step is logged: it
lets the caller interrupt a graft that is running far slower than expected
(`StepBudgetExceeded`), propagated up to `main`, which writes a blocking
report instead of insisting or silently arbitrating a scientific choice.

Usage (real run):
  uv run engraft-run 2026-01-01
Usage (dry run, fake replica/table, no GGUF):
  uv run engraft-run 2026-01-01-dryrun --fake
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from engraft.config import load as load_config
from engraft.lens import RowSet, read_pleo, write_pleo
import engraft.replica.graft as G
import engraft.replica.descend as D

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTS_DIR = REPO_ROOT / "facts"
RESULTS_ROOT = REPO_ROOT / "results"

PERTURB_REL = 0.01
PERTURB_SEED = 1
Q6_TARGETS = [("it_gatto", 0), ("en_dog", 0)]  # the two perturbed-seed repeats

GUARDIAN_ESTIMATE_S_PER_STEP = 7.0
GUARDIAN_FACTOR = 3.0
GUARDIAN_CONSECUTIVE_STEPS = 20

MEMORY_CEILING_BYTES = 70 * (1 << 30)
MEMORY_MARGIN_BYTES = 10 * (1 << 30)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class StepBudgetExceeded(RuntimeError):
    """A graft exceeded 3x the estimated s/step for 20 consecutive steps."""


def make_guardian(tag: str, estimate_s_per_step: float = GUARDIAN_ESTIMATE_S_PER_STEP):
    """`on_step` for `descend`: raises (a declared stop, not a silent arbitration)
    if the time between two consecutive steps exceeds `GUARDIAN_FACTOR` times
    the estimate, for `GUARDIAN_CONSECUTIVE_STEPS` steps in a row."""
    threshold = GUARDIAN_FACTOR * estimate_s_per_step
    state = {"consecutive": 0, "prev_t": None}

    def on_step(record: dict) -> None:
        t = record["t_s"]
        prev = state["prev_t"]
        state["prev_t"] = t
        if prev is None:
            return
        dt = t - prev
        if dt > threshold:
            state["consecutive"] += 1
        else:
            state["consecutive"] = 0
        if state["consecutive"] >= GUARDIAN_CONSECUTIVE_STEPS:
            raise StepBudgetExceeded(
                f"{tag}: {state['consecutive']} consecutive steps over {threshold:.1f}s "
                f"({GUARDIAN_FACTOR:g}x the {estimate_s_per_step:.1f} s/step estimate) -- stopped"
            )

    return on_step


def order_facts(facts_by_id: dict[str, dict]) -> list[dict]:
    """Memory facts (language-alternated, in facts.json order within each
    language), then counterfactuals."""
    memoria = [f for f in facts_by_id.values() if f["kind"] == "memory"]
    contro = [f for f in facts_by_id.values() if f["kind"] == "counterfactual"]
    it_mem = [f for f in memoria if f["lang"] == "it"]
    en_mem = [f for f in memoria if f["lang"] == "en"]
    alternated: list[dict] = []
    for pair in itertools.zip_longest(it_mem, en_mem):
        for f in pair:
            if f is not None:
                alternated.append(f)
    return alternated + contro


def _clear_open_graft_logs(fdir: Path, fid: str, n: int, state: dict) -> None:
    """`descend` opens the jsonl in append mode: a graft not closed in
    `state.json` must restart from an empty file (otherwise an interrupted run
    mixes with the repeat)."""
    done = state.get("done", {})
    for i in range(n):
        if str(i) not in done:
            log_path = fdir / f"descend_{fid}_{i}.jsonl"
            if log_path.exists():
                log_path.unlink()


def run_fact(replica, table, tok, fact: dict, cfg: dict, out_root: Path) -> dict:
    fid = fact["id"]
    fdir = out_root / "facts" / fid
    fdir.mkdir(parents=True, exist_ok=True)
    state_path = fdir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}}
    n = len(fact["answer_tokens"])
    _clear_open_graft_logs(fdir, fid, n, state)

    guardian = make_guardian(fid)
    t0 = time.time()
    summary = G.graft_fact(
        replica, table, tok, fact, cfg, out_dir=fdir, state_path=state_path, on_step=guardian,
    )
    summary["wall_s"] = time.time() - t0
    summary["peak_rss_gb_so_far"] = rss_gb()
    _assert_sanity_step0(fdir, fid, n)
    log(f"{fid}: {summary['wall_s']:.0f}s, p_free_product={summary.get('p_free_product')}, "
        f"RSS {rss_gb():.1f} GB")
    return summary


def _assert_sanity_step0(fdir: Path, fid: str, n: int) -> None:
    """For every non-perturbed graft, `routing_changed_vs_base` at jsonl step 0
    must be 0 relative to the `routing_trigger` built by `graft_fact`
    (replica-vs-replica consistency, asserted here by the runner; the second
    half of the check, the comparison against `logp_base_f32` from the real
    engine, is asserted by `engraft.check` in the engine window)."""
    for i in range(n):
        log_path = fdir / f"descend_{fid}_{i}.jsonl"
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        rec0 = json.loads(lines[0])
        changed = rec0.get("routing_changed_vs_base") or 0
        if changed != 0:
            raise RuntimeError(
                f"{fid} graft {i}: consistency check failed, routing_changed_vs_base at "
                f"step 0 = {changed} (expected 0): {rec0}"
            )


def run_q6_perturbed(replica, table, fid: str, position: int, facts_resolved: dict, out_root: Path) -> dict:
    """Relaunches the graft (fid, position) from perturbed rows (fixed seed and
    relative perturbation) and compares against the already-closed
    unperturbed run in `facts/<fid>/state.json`."""
    fact = facts_resolved[fid]
    out_dir = out_root / "facts" / fid
    seed_dir = out_root / "facts" / f"{fid}_{position}_seed{PERTURB_SEED}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    prep = G.prepare_innesto(replica, table, fact, position, overlay_map={}, out_dir=seed_dir)
    rows_true = prep["rows_true"]
    row_mask = G.ROW_MASK_T8
    rng = np.random.default_rng(PERTURB_SEED)
    row_norms = rows_true.norm(dim=1).numpy()
    sigma = PERTURB_REL * row_norms / math.sqrt(rows_true.shape[1])  # math.sqrt (not np.sqrt): stays float32
    delta = rng.normal(0.0, 1.0, size=rows_true.shape).astype(np.float32) * sigma[:, None]
    delta[~row_mask] = 0.0
    np.save(seed_dir / "perturbation.npy", delta)
    rows_start = rows_true + torch.from_numpy(delta)

    tag = f"{fid}_{position}_seed{PERTURB_SEED}"
    log_path = seed_dir / f"descend_{tag}.jsonl"
    if log_path.exists():
        log_path.unlink()
    y = fact["answer_tokens"][position]
    guardian = make_guardian(tag)
    t0 = time.time()
    summary = D.descend(
        replica, prep["tokens"], prep["prefix_state"], prep["routing_trigger"], rows_true, y,
        {}, prep["rowset"].rows_global, 0.0, seed_dir, log_path,
        row_mask=row_mask, refresh_every=1, thresholds=[], p_stop=0.95, plateau_steps=150,
        tag=tag, rows_start=rows_start, on_step=guardian,
    )
    summary["wall_s"] = time.time() - t0

    base_summary_path = out_dir / f"{fid}.json"
    base_innesto = None
    if base_summary_path.exists():
        base_data = json.loads(base_summary_path.read_text())
        base_innesto = next((it for it in base_data["grafts"] if it["position"] == position), None)

    concordance = None
    if base_innesto is not None and base_innesto.get("n_steps"):
        n_base = base_innesto["n_steps"]
        n_seed = summary.get("n_steps") or 0
        steps_within_20pct = abs(n_seed - n_base) <= 0.2 * n_base
        same_stop_reason = summary.get("stop_reason") == base_innesto.get("stop_reason")
        p_base = base_innesto.get("final_p_free")
        p_seed = summary.get("final_p_free")
        p_within = (
            p_base is not None and p_seed is not None and abs(p_seed - p_base) <= 0.02
        )
        concordance = {
            "n_steps_base": n_base, "n_steps_seed1": n_seed, "steps_within_20pct": steps_within_20pct,
            "stop_reason_base": base_innesto.get("stop_reason"), "stop_reason_seed1": summary.get("stop_reason"),
            "same_stop_reason": same_stop_reason,
            "final_p_free_base": p_base, "final_p_free_seed1": p_seed, "p_within_0_02": p_within,
            "concordant": bool(steps_within_20pct and same_stop_reason and p_within),
        }

    out = {
        "fid": fid, "position": position, "seed": PERTURB_SEED,
        "n_steps": summary.get("n_steps"), "stop_reason": summary.get("stop_reason"),
        "final_p_free": summary.get("final_p_free"), "wall_s": summary["wall_s"],
        "concordance": concordance,
    }
    (seed_dir / f"{tag}_summary.json").write_text(json.dumps(out, indent=2, default=str))
    log(f"repeat {tag}: n_steps={out['n_steps']} stop={out['stop_reason']} "
        f"final_p_free={out['final_p_free']} concordant={concordance and concordance['concordant']}")
    return out


def build_merged(facts_resolved: dict, keys: dict, order: list[dict], out_root: Path) -> dict:
    excluded = set(keys.get("excluded_facts", []))
    fact_pleo_paths = {}
    for fact in order:
        fid = fact["id"]
        pleo_path = out_root / "facts" / fid / f"{fid}.pleo"
        if pleo_path.exists():
            fact_pleo_paths[fid] = pleo_path
        else:
            excluded = excluded | {fid}  # fact never closed (interrupted run): not in the union
    order_ids = [f["id"] for f in order]
    rows_all, data_all, manifest = G.merge_fact_overlays(order_ids, excluded, fact_pleo_paths)
    write_pleo(out_root / "merged.pleo", rows_all, data_all)
    (out_root / "merged_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"merged.pleo: {manifest['n_rows']} rows, included={manifest['included']}, "
        f"excluded={manifest['excluded']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("--config", default=None, help="path to engraft.toml (default: ./engraft.toml)")
    parser.add_argument("--fake", action="store_true", help="fake replica/table (no GGUF)")
    args = parser.parse_args(argv)

    out_root = RESULTS_ROOT / args.data
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = dict(G.DEFAULT_CFG)

    if args.fake:
        from engraft.testing.fake_table import FakeTable
        from engraft.testing.fake_replica import FakeGraftReplica

        table = FakeTable(seed=42)
        replica = FakeGraftReplica(table, seed=0, vocab=64)
        tok = None
        facts_resolved = _fake_facts_resolved(table)
        keys = _fake_keys(facts_resolved)
        cfg["plateau_steps"] = 3
        cfg["p_stop"] = 0.05
    else:
        engraft_cfg = load_config(args.config)
        from engraft.table import PleTable, PleTokenizer
        from engraft.replica.hparams import Hparams
        from engraft.replica.weights import GgufWeights
        from engraft.replica.model import Replica, check_memory

        mem = check_memory(ceiling_bytes=MEMORY_CEILING_BYTES, margin_bytes=MEMORY_MARGIN_BYTES)
        (out_root / "memory.json").write_text(json.dumps(mem, indent=2))
        log(f"check_memory: {mem}")
        if not mem["ok"]:
            log("INSUFFICIENT MEMORY: stopping (declared, not worked around).")
            return 3

        shards = engraft_cfg.get_list("model.shards")
        shard1, shard2, shard3 = shards[0], shards[1], shards[2]
        ram_cache_bytes = engraft_cfg.get_int("run.ram_cache_gb") * (1 << 30)
        disk_cache_bytes = engraft_cfg.get_int("run.disk_cache_gb") * (1 << 30)
        cache_dir = engraft_cfg.get_path("run.cache_dir")

        hp = Hparams.from_gguf_paths(shard1, shard2)
        w = GgufWeights(
            [shard1, shard2, shard3], ram_cache_bytes=ram_cache_bytes,
            disk_cache_dir=str(cache_dir), disk_cache_bytes=disk_cache_bytes,
        )
        table = PleTable(engraft_cfg.get_path("model.table"))
        tok = PleTokenizer(engraft_cfg.get_path("model.tokenizer"))
        replica = Replica(hp, w, table)

        facts_resolved = json.loads((FACTS_DIR / "facts_resolved.json").read_text())["facts"]
        keys = json.loads((FACTS_DIR / "keys.json").read_text())

    (out_root / "graft_config.json").write_text(json.dumps(
        {**{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()},
         "MAX_STEPS": D.MAX_STEPS}, indent=2,
    ))

    order = order_facts(facts_resolved)
    log(f"fact order: {[f['id'] for f in order]}")

    fact_summaries: dict[str, dict] = {}
    blocked = None
    for fact in order:
        fid = fact["id"]
        try:
            fact_summaries[fid] = run_fact(replica, table, tok, fact, cfg, out_root)
        except (G.PreconditionError, StepBudgetExceeded) as e:
            log(f"BLOCKED on {fid}: {e}")
            blocked = {"fid": fid, "error": str(e), "type": type(e).__name__}
            break

    q6_results = []
    if blocked is None:
        for fid, position in Q6_TARGETS:
            if fid in fact_summaries:
                q6_results.append(run_q6_perturbed(replica, table, fid, position, facts_resolved, out_root))
            else:
                log(f"repeat: {fid} not closed, perturbed repeat skipped")

        build_merged(facts_resolved, keys, order, out_root)

    summary = {
        "order": [f["id"] for f in order],
        "facts": fact_summaries,
        "q6": q6_results,
        "keys": keys,
        "peak_rss_gb": rss_gb(),
        "blocked": blocked,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"done. Peak RSS {rss_gb():.2f} GB. Blocked: {blocked is not None}")
    return 0 if blocked is None else 1


def _fake_facts_resolved(table) -> dict[str, dict]:
    """Synthetic facts for `--fake` (dry run): same shape as a real
    `facts_resolved.json`, but with small tokens (< the fake replica's
    vocabulary) in place of real tokenizer tokens -- this is not a test of
    `engraft.facts` (already covered by `tests/test_facts.py`), only of the
    runner."""

    def make(fid: str, lang: str, kind: str, trigger_tokens: list[int], answer_tokens: list[int]) -> dict:
        trigger_rows = RowSet.from_position(table, trigger_tokens, len(trigger_tokens) - 1).rows_global.tolist()
        chain_rows = {}
        for i in range(1, len(answer_tokens)):
            prefix = trigger_tokens + answer_tokens[:i]
            chain_rows[str(i)] = RowSet.from_position(table, prefix, len(prefix) - 1).rows_global.tolist()
        return {
            "id": fid, "lang": lang, "kind": kind, "trigger_tokens": trigger_tokens,
            "answer_tokens": answer_tokens, "trigger_rows_global": trigger_rows, "chain_rows_global": chain_rows,
        }

    return {
        "it_gatto": make("it_gatto", "it", "memory", [1, 2, 3, 4, 5], [10, 11]),
        "en_dog": make("en_dog", "en", "memory", [6, 7, 8, 9, 10], [12]),
        "it_capitale": make("it_capitale", "it", "counterfactual", [21, 22, 23], [13, 14]),
    }


def _fake_keys(facts_resolved: dict[str, dict]) -> dict:
    return {"n_conflicts": 0, "conflicts": [], "excluded_facts": []}


if __name__ == "__main__":
    sys.exit(main())

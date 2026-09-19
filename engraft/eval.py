"""Replica-side evaluation of a corpus overlay: three columns per fragment --
**base** (no overlay, no document), **student** (the overlay produced by
`engraft.descend_corpus`, no document), **teacher** (the source document in
the prefix, true rows) -- computed in the replica (`seq_forward*`), never in
the real engine (see `engraft.engine_check` for that).

Overlap bands (`overlap_from_census`): reduces a census dict's
`per_fact_overlap[fact_id][ctx]["hit_all_B"]` to one integer per fragment,
`ctx` chosen from the fragment's `family` (see `FAMILY_TO_CENSUS_CTX`).
`evaluate_corpus` accepts either `census` (the whole dict, per-fragment
reduction via `overlap_from_census`) or `overlap_by_fact_id` (an integer
already reduced per `fact_id`) -- `census` takes precedence when both are
given.

`--ple-gate` (additive, default off -> `eval.json` byte-identical to the
flag-off behavior): wraps each of the `base`/`student`/`student_free`
forwards (never the teacher) in its own `engraft.replica.seq.capture_ple_gate`
-- a diagnostic on the PLE gate's internal signals (`s`, `gate`,
`gated_norm`, `hidden_norm`, `value_norm`) at the answer row and its causal
window. The private original also had a `--ple-contesa` flag (an
"interference" measurement requiring an intermediate per-sub-block capture
inside `Replica.run_layer` that this reference's public interface does not
expose); it is not part of this port -- see `docs/formats.md` for why.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from engraft.lens import local_to_global, read_pleo
from engraft.table import ROW_LEN
from engraft.replica.seq import capture_ple_gate
import engraft.replica.distill as D

OVERLAP_BANDS = [(0, 0, "0"), (1, 8, "1-8"), (9, 24, "9-24"), (25, None, ">24")]


def band_of(n: int) -> str:
    """Overlap band ("0 / 1-8 / 9-24 / >24 shared rows")."""
    for lo, hi, label in OVERLAP_BANDS:
        if hi is None:
            if n >= lo:
                return label
        elif lo <= n <= hi:
            return label
    return "?"  # n < 0: never expected, not a fatal error for a diagnostic aggregate


FAMILY_TO_CENSUS_CTX = {
    "paraphrase": "Hb_paraphrase",
    "question": "Hc_question",
    "cloze": "Hd_cloze",
    "chat": "Hc_question",  # the form farthest from the trigger
    "statement": "Hc_question",
}


def overlap_from_census(census: dict, fragment: dict) -> int | None:
    """`hit_all_B` from `census["per_fact_overlap"][fact_id][ctx]` for the
    fragment's `fact_ids` (`ctx` from `FAMILY_TO_CENSUS_CTX[fragment["family"]]`):
    `None` if the family is not mapped, if `per_fact_overlap` is missing, or
    if none of the fragment's `fact_ids` appear in the census under that
    class -- never an error (a fragment with no overlap data simply stays
    out of the banding). With more than one `fact_id` (fragments shared
    across facts) takes the MAXIMUM."""
    ctx = FAMILY_TO_CENSUS_CTX.get(fragment.get("family"))
    if ctx is None:
        return None
    per_fact = census.get("per_fact_overlap") or {}
    counts: list[int] = []
    for fid in fragment.get("fact_ids", []):
        entry = per_fact.get(fid)
        if not entry:
            continue
        ctx_entry = entry.get(ctx)
        if not ctx_entry:
            continue
        val = ctx_entry.get("hit_all_B")
        if val is not None:
            counts.append(int(val))
    return max(counts) if counts else None


def answer_row(frag: dict) -> int:
    """The logit row of the FIRST position of `answer_spans` (a multi-token
    answer has several rows admissible by the training loss, but only the
    FIRST is the position measured for `p_first`/`rank_first`). The row that
    predicts the token at index `s` is `s-1` (`Replica.prefix`: row `r`
    predicts `tokens[r+1]`). Same convention as
    `engraft.descend_corpus.answer_rows` (the full list); this is its
    `min(...) - 1`."""
    spans = frag.get("answer_spans") or []
    if not spans:
        raise ValueError(f"answer_row: fragment {frag.get('id')!r} has no answer_spans")
    return min(int(s) for s in spans) - 1


def _rank_and_p(logits_row: torch.Tensor, y: int) -> tuple[int, float]:
    """`(rank, p)` of token `y` in `logits_row` `[V]`: `rank=1` = argmax (no
    strictly greater logit); `p` = softmax probability (F32, fixed scale)."""
    logits32 = logits_row.detach().to(torch.float32)
    logp = torch.log_softmax(logits32, dim=-1)
    p = float(torch.exp(logp[y]).item())
    rank = int((logits32 > logits32[y]).sum().item()) + 1
    return rank, p


def window_rows(replica, r: int) -> list[int]:
    """PLE-gate measurement: the positions the PLE layer's dilated causal
    conv reads to contribute to row `r` -- `r, r-n, ..., r-(K-1)n` with
    `n = hp.ple_ngram_size`, `K = hp.ple_conv_kernel`, filtered to `>= 0`."""
    hp = replica.hp
    n = hp.ple_ngram_size
    kern = hp.ple_conv_kernel
    return [r - i * n for i in range(kern) if r - i * n >= 0]


def _rows_global_at(table, tokens: list[int], t: int) -> list[int]:
    """Global (per-head) addresses read into `emb` at position `t`, offline,
    without a forward or dequantization: `table.ngram_addresses(tokens)[t]`
    (LOCAL rows per head) + `local_to_global`."""
    addr_t = table.ngram_addresses(tokens)[t]
    return [local_to_global(table, h, addr_t[h]) for h in range(len(addr_t))]


def overlay_hits(table, tokens: list[int], rows: list[int], row_map: dict[int, int]) -> int:
    """PLE-gate measurement: how many of the global rows read at `rows` (one
    or more positions, union without double counting) are in `row_map` (the
    forward's actual overlay, indexed by global row). Offline, no forward."""
    hit: set[int] = set()
    for t in rows:
        hit.update(g for g in _rows_global_at(table, tokens, t) if g in row_map)
    return len(hit)


def evaluate_fragment(
    replica, w, table, frag: dict, doc_tokens: list[int], eos: int,
    row_map: dict[int, int], rows_var_overlay: torch.Tensor, forward_fn, k: int,
    doc_state=None, routing_base: "tuple[dict[str, np.ndarray], np.ndarray] | None" = None,
    skip_base: bool = False, skip_teacher: bool = False,
    ple_gate: bool = False, ple_arrays: "dict[str, np.ndarray] | None" = None,
) -> dict:
    """The three columns for ONE fragment: `base` (empty `rows_var`, no
    document), `student` (`row_map`/`rows_var_overlay`, no document),
    `teacher` (document in the prefix via `base_state`, empty `rows_var` --
    the teacher reads the TRUE rows). `doc_state` (optional, reused across
    fragments of the same document): if `None` and `doc_tokens` non-empty,
    computed here once.

    `routing_base` (`(by_frag, layers)` from `engraft.replica.distill.load_routing_base`,
    optional): with it given, **base** and **student** are forwarded at
    LOCKED routing (RBR resolved for `frag["id"]`, the same for both -- only
    this way does the rank comparison isolate the overlay's effect, not
    routing noise); the **teacher** always stays at FREE routing. In
    addition, a SECOND forward of the student at free routing produces
    `p_student_free`/`rank_student_free` -- the cost of locking.

    `skip_base`/`skip_teacher` (additive, default `False`): skip the
    corresponding forward -- no fabricated value, the `p_first`/`rank_first`/
    `correct` keys for the skipped column are `None` (never an implicit
    `0.0`/`False`). With `skip_teacher`, `kd_student_teacher` is `None` (it
    depends on the teacher's logits -- not a column constant with respect to
    the overlay) and NO `doc_prefix_state` is computed (the document forward
    serves ONLY the teacher): unlike the base/teacher columns (independent
    of the overlay), the document forward serves only the teacher.
    `state_teacher` stays explicitly `None` (never an alias of `state_base`)
    when `skip_teacher` is given.

    `ple_gate`/`ple_arrays` (PLE-gate measurement, additive, default
    `False`/`None` -> identical behavior, flag off, `eval.json`
    byte-identical): with `ple_gate=True`, EACH of the `base`/`student`/
    `student_free` forwards (never the teacher) is wrapped in its OWN
    `capture_ple_gate` with its OWN `out` (a single `capture_ple_gate`
    around several forwards would merge the columns into one array). The
    record gains the `"ple_gate"` key; if `ple_arrays` is a dict (not
    `None`), it accumulates the `[T,hc]`/`[T]`/`[T]` arrays per column under
    the keys `"<id>/<column>/gate"`, `"<id>/<column>/s"`,
    `"<id>/<column>/value_norm"`."""
    tokens = list(frag["tokens"])
    row = answer_row(frag)
    if not (0 <= row < len(tokens) - 1):
        raise ValueError(
            f"evaluate_fragment: {frag.get('id')!r} -- answer row {row} outside "
            f"[0, {len(tokens) - 1}) (n_frag={len(tokens) - 1})"
        )
    y = tokens[row + 1]
    dev = getattr(getattr(replica, "backend", None), "device", "cpu")  # real: cuda:0; fake: cpu
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32, device=dev)
    rows_var_overlay = rows_var_overlay.to(dev)

    routing_init = None
    if routing_base is not None:
        by_frag, layers = routing_base
        n_frag_expected = len(tokens) - 1
        rinit_np = D.routing_init_for_fragment(by_frag, layers, frag["id"], n_frag_expected)
        routing_init = {il: torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
                        for il, arr in rinit_np.items()}

    n_prefix = len(tokens) - 1

    def _forward_maybe_gated(rows, rmap, **extra):
        """One forward, with its own `capture_ple_gate` capture if
        `ple_gate` (never shared across columns). Returns `(state, gate_out|None)`."""
        if not ple_gate:
            state, _extra = forward_fn(
                replica, w, tokens, rows, rmap, return_logits=True, grad_proxy=False, **extra,
            )
            return state, None
        gate_out: dict = {}
        with capture_ple_gate(replica, gate_out):
            state, _extra = forward_fn(
                replica, w, tokens, rows, rmap, return_logits=True, grad_proxy=False, **extra,
            )
        if gate_out["gate"].shape[0] != n_prefix:
            raise ValueError(
                f"evaluate_fragment: {frag.get('id')!r} -- capture_ple_gate captured "
                f"T={gate_out['gate'].shape[0]} rows, expected n_prefix={n_prefix} "
                "(len(tokens)-1, never len(tokens))"
            )
        return state, gate_out

    state_base = gate_base = None
    if not skip_base:
        state_base, gate_base = _forward_maybe_gated(empty_rows, {}, routing_init=routing_init)
    state_student, gate_student = _forward_maybe_gated(rows_var_overlay, row_map, routing_init=routing_init)
    state_student_free = gate_student_free = None
    if routing_init is not None:
        state_student_free, gate_student_free = _forward_maybe_gated(rows_var_overlay, row_map)
    state_teacher = None
    if not skip_teacher:
        if doc_tokens:
            if doc_state is None:
                doc_state, _hash = D.doc_prefix_state(replica, w, doc_tokens, eos, forward_fn)
            full_tokens = list(doc_tokens) + tokens
            state_teacher, _ = forward_fn(
                replica, w, full_tokens, empty_rows, {}, return_logits=True, grad_proxy=False,
                base_state=doc_state,
            )
        else:
            state_teacher = state_base if state_base is not None else forward_fn(
                replica, w, tokens, empty_rows, {}, return_logits=True, grad_proxy=False,
                routing_init=routing_init,
            )[0]

    rank_base, p_base = (None, None) if state_base is None else _rank_and_p(state_base.logits[row], y)
    rank_student, p_student = _rank_and_p(state_student.logits[row], y)
    rank_teacher, p_teacher = (None, None) if state_teacher is None else _rank_and_p(state_teacher.logits[row], y)

    # Adherence: KD student/teacher on the fragment's admitted positions
    # (n_excl=0 here -- evaluation covers EVERY position of the fragment,
    # unlike the training loss which excludes the first n_excl; no separator
    # position to exclude inside a single fragment). `kd_student_teacher`
    # depends on the TEACHER's logits -- with skip_teacher there is nothing
    # to compute, `None` (never a fabricated zero).
    kd_student_teacher = None
    if state_teacher is not None:
        n_frag = state_student.logits.shape[0]
        loss_mask = D.loss_mask_for_fragment(n_frag, n_excl=0).to(state_student.logits.device)
        ids_m, logp_m, tail_m = D.make_targets(state_teacher.logits, k)
        kd_val, _per_pos = D.kd_loss(state_student.logits, ids_m, logp_m, tail_m, loss_mask)
        kd_student_teacher = float(kd_val.item())

    result = {
        "id": frag["id"], "family": frag.get("family"), "split": frag.get("split"),
        "fact_ids": frag.get("fact_ids", []),
        "p_first": {"base": p_base, "student": p_student, "teacher": p_teacher},
        "rank_first": {"base": rank_base, "student": rank_student, "teacher": rank_teacher},
        "correct": {
            "base": None if rank_base is None else rank_base == 1,
            "student": rank_student == 1,
            "teacher": None if rank_teacher is None else rank_teacher == 1,
        },
        "kd_student_teacher": kd_student_teacher,
    }
    if state_student_free is not None:
        rank_student_free, p_student_free = _rank_and_p(state_student_free.logits[row], y)
        result["p_student_free"] = p_student_free
        result["rank_student_free"] = rank_student_free

    if ple_gate:
        hc = replica.hp.hc_mult
        win = window_rows(replica, row)
        hits_answer = overlay_hits(table, tokens, [row], row_map)
        hits_window = overlay_hits(table, tokens, win, row_map)
        win_idx = torch.tensor(win, dtype=torch.long)

        def _col_stats(g: dict) -> dict:
            return {
                "gate_answer": g["gate"][row].tolist(),
                "s_answer": g["s"][row].tolist(),
                "gate_window_mean": g["gate"].index_select(0, win_idx).mean(dim=0).tolist(),
                "gate_frag_mean": g["gate"].mean(dim=0).tolist(),
                "gated_over_hidden_answer": (
                    g["gated_norm"][row] / g["hidden_norm"][row].clamp_min(1e-12)
                ).tolist(),
            }

        ple_gate_record: dict = {
            "hc": hc, "answer_row": row, "window_rows": win,
            "overlay_hits_answer": hits_answer, "overlay_hits_window": hits_window,
        }
        if gate_base is not None:
            ple_gate_record["base"] = _col_stats(gate_base)
        ple_gate_record["student"] = _col_stats(gate_student)
        if gate_student_free is not None:
            ple_gate_record["student_free"] = _col_stats(gate_student_free)
        result["ple_gate"] = ple_gate_record

        if ple_arrays is not None:
            fid = frag["id"]
            for col_name, g in (("base", gate_base), ("student", gate_student),
                                 ("student_free", gate_student_free)):
                if g is None:
                    continue
                ple_arrays[f"{fid}/{col_name}/gate"] = g["gate"].numpy()
                ple_arrays[f"{fid}/{col_name}/s"] = g["s"].numpy()
                ple_arrays[f"{fid}/{col_name}/value_norm"] = g["value_norm"].numpy()

    return result


def _aggregate(records: list[dict], key_fn, skip_base: bool = False, skip_teacher: bool = False) -> dict:
    """`correct_frac` computed ONLY for the non-skipped columns (never a
    fabricated `0.0` from a falsy `None` -- `skip_base`/`skip_teacher`
    exclude the column from the dict instead of passing off an empty count
    as a zero) and `kd_student_teacher_mean` omitted entirely with
    `skip_teacher` (every value would be `None`, `np.mean` would raise)."""
    cols = [c for c, skip in (("base", skip_base), ("student", False), ("teacher", skip_teacher)) if not skip]
    buckets: dict = {}
    for r in records:
        buckets.setdefault(key_fn(r), []).append(r)
    out = {}
    for key, rs in buckets.items():
        entry = {
            "n": len(rs),
            "correct_frac": {col: sum(1 for r in rs if r["correct"][col]) / len(rs) for col in cols},
        }
        if not skip_teacher:
            entry["kd_student_teacher_mean"] = float(np.mean([r["kd_student_teacher"] for r in rs]))
        out[str(key)] = entry
    return out


def evaluate_corpus(
    replica, w, table, fragments: list[dict], doc_tokens: list[int], eos: int,
    row_map: dict[int, int], rows_var_overlay: torch.Tensor, forward_fn, k: int,
    overlap_by_fact_id: dict[str, int] | None = None, census: dict | None = None,
    routing_base: "tuple[dict[str, np.ndarray], np.ndarray] | None" = None,
    skip_base: bool = False, skip_teacher: bool = False,
    ple_gate: bool = False, ple_arrays: "dict[str, np.ndarray] | None" = None,
) -> dict:
    """Evaluates `fragments` (typically the `heldout` split, plus a sample of
    `train`) and aggregates by `family` and by overlap band
    (`overlap_band`). Overlap: `census` (the whole `census_<lang>.json` dict
    -- `overlap_from_census` per fragment, uses `family`+`fact_ids`) takes
    PRECEDENCE over `overlap_by_fact_id` (an integer already reduced per
    `fact_id`); neither given -> no band, never an error.

    `routing_base` (optional): forwarded unchanged to `evaluate_fragment` --
    see its docstring.

    `skip_base`/`skip_teacher` (additive, default `False`): forwarded to
    `evaluate_fragment` and `_aggregate`; with `skip_teacher` NO
    `doc_prefix_state` is computed here (the document forward serves ONLY
    the teacher -- without skip_teacher the behavior stays IDENTICAL, same
    `eval.json`).

    `ple_gate`/`ple_arrays` (PLE-gate measurement, additive, default
    `False`/`None`): forwarded unchanged to `evaluate_fragment`."""
    doc_state = None
    if doc_tokens and not skip_teacher:
        doc_state, _hash = D.doc_prefix_state(replica, w, doc_tokens, eos, forward_fn)

    per_fragment = [
        evaluate_fragment(
            replica, w, table, frag, doc_tokens, eos, row_map, rows_var_overlay, forward_fn, k,
            doc_state=doc_state, routing_base=routing_base,
            skip_base=skip_base, skip_teacher=skip_teacher,
            ple_gate=ple_gate, ple_arrays=ple_arrays,
        )
        for frag in fragments
    ]

    for rec, frag in zip(per_fragment, fragments):
        overlap = None
        if census is not None:
            overlap = overlap_from_census(census, frag)
        elif overlap_by_fact_id and frag.get("fact_ids"):
            counts = [overlap_by_fact_id[fid] for fid in frag["fact_ids"] if fid in overlap_by_fact_id]
            if counts:
                overlap = max(counts)
        rec["overlap_count"] = overlap
        rec["overlap_band"] = band_of(overlap) if overlap is not None else None

    return {
        "n_fragments": len(per_fragment),
        "fragments": per_fragment,
        "by_family": _aggregate(per_fragment, lambda r: r["family"], skip_base=skip_base, skip_teacher=skip_teacher),
        "by_overlap_band": _aggregate(
            [r for r in per_fragment if r["overlap_band"] is not None], lambda r: r["overlap_band"],
            skip_base=skip_base, skip_teacher=skip_teacher,
        ),
    }


def write_eval_report(result: dict, out_json: Path, out_md: Path) -> None:
    out_json.write_text(json.dumps(result, indent=2, default=str))

    def _fmt(cf: dict, col: str) -> str:
        # A skipped column (`--skip-base`/`--skip-teacher`) is absent from
        # `correct_frac` -- "n/a" instead of formatting `None` with `:.2f`.
        return f"{cf[col]:.2f}" if col in cf else "n/a"

    def _table(agg: dict) -> list[str]:
        lines = ["| key | n | correct (base) | correct (student) | correct (teacher) | KD student/teacher |",
                  "|---|---|---|---|---|---|"]
        for key, a in sorted(agg.items()):
            cf = a["correct_frac"]
            kd_txt = f"{a['kd_student_teacher_mean']:.4f}" if "kd_student_teacher_mean" in a else "n/a"
            lines.append(
                f"| {key} | {a['n']} | {_fmt(cf, 'base')} | {_fmt(cf, 'student')} | {_fmt(cf, 'teacher')} | "
                f"{kd_txt} |"
            )
        return lines

    lines = [
        "# Evaluation (replica-side)", "",
        f"Fragments evaluated: {result['n_fragments']}", "",
        "## By family", "",
        *_table(result["by_family"]), "",
        "## By overlap band", "",
    ]
    if result["by_overlap_band"]:
        lines += _table(result["by_overlap_band"])
    else:
        lines.append("(no band -- neither `census` nor `overlap_by_fact_id` given/resolved)")
    out_md.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--usage-corpus", required=True, help="usage_corpus_resolved.json")
    parser.add_argument("--doc-tokens", default=None, help="json: list of the source document's tokens")
    parser.add_argument("--overlay", required=True, help="merged.pleo (engraft.descend_corpus)")
    parser.add_argument("--census", default=None, help="census_<lang>.json (per_fact_overlap -> overlap bands)")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--split", default="heldout", choices=["heldout", "train", "all"])
    parser.add_argument("--routing-base", default=None,
                         help="locked routing (RBR): routing_base_*.npz from engraft.teacher --routing-out -- "
                              "base and student at locked routing (per id), teacher ALWAYS free; also produces "
                              "p_student_free/rank_student_free (the cost of locking)")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--working-set-gb", type=float, default=2.0)
    parser.add_argument("--dequant-cache-gb", type=float, default=24.0)
    parser.add_argument("--dense-dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--head-dtype", default="f32", choices=["f32", "bf16"],
                         help="GEMM input dtype of the vocabulary head; if different from what is recorded "
                              "in merged_manifest.json next to --overlay, only a WARNING (never an error)")
    parser.add_argument("--wdot", default="split", choices=["bf16", "split", "f32"])
    parser.add_argument("--delta-chunk-size", type=int, default=64)
    parser.add_argument("--min-avail-gb", type=float, default=100.0)
    parser.add_argument("--table-path", default=None, help="GGUF n-gram table shard path (real run only)")
    parser.add_argument("--shard-paths", nargs="+", default=None, help="GGUF weight shard paths (real run only)")
    parser.add_argument("--routing-config-mismatch-ok", action="store_true", default=False,
                         help="proceeds even if dense_dtype/wdot of --routing-base do not match this run's "
                              "(the deviation must be DECLARED in the report)")
    parser.add_argument("--skip-base", action="store_true", default=False,
                         help="skips the 'base' forward (a column independent of the overlay) -- "
                              "correct_frac.base/p_first.base/rank_first.base absent (None per fragment, "
                              "never a fabricated zero)")
    parser.add_argument("--skip-teacher", action="store_true", default=False,
                         help="skips the 'teacher' forward AND doc_prefix_state (the document forward "
                              "serves only the teacher) -- correct_frac.teacher and "
                              "kd_student_teacher_mean absent (kd_student_teacher DEPENDS on the overlay, "
                              "it is not a column constant: lost by choice, not by constancy)")
    parser.add_argument("--ple-gate", action="store_true", default=False,
                         help="PLE-gate measurement: adds the 'ple_gate' key to every record and writes "
                              "ple_gate.npz next to eval.json -- default off, eval.json byte-identical to today")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.fake:
        from engraft.descend_corpus import _fake_replica_and_table
        replica, w, table, _step_fn, forward_fn = _fake_replica_and_table(head_dtype=args.head_dtype)
    else:  # pragma: no cover -- requires a real GGUF/CUDA device
        from engraft.descend_corpus import load_real_backend
        if not args.table_path or not args.shard_paths:
            raise SystemExit("a real run requires --table-path and --shard-paths")
        replica, w, table, _step_fn, forward_fn = load_real_backend(args)

    fragments_all = json.loads(Path(args.usage_corpus).read_text())
    fragments_all = fragments_all.get("fragments", fragments_all)
    fragments = fragments_all if args.split == "all" else [
        f for f in fragments_all if f.get("split") == args.split
    ]
    if not fragments:
        raise SystemExit(f"engraft.eval: no fragment with split={args.split!r}")

    doc_tokens = json.loads(Path(args.doc_tokens).read_text()) if args.doc_tokens else []
    eos = fragments[0]["tokens"][0]

    rows_g, data = read_pleo(args.overlay)
    row_map = {int(g): i for i, g in enumerate(rows_g.tolist())}
    rows_var_overlay = torch.from_numpy(np.asarray(data, dtype=np.float32))

    # WARNING (never an error -- historical overlays lack this file) if this
    # run's head_dtype/dense_dtype differ from those recorded in the graft
    # manifest next to --overlay.
    overlay_manifest_path = Path(args.overlay).parent / "merged_manifest.json"
    if overlay_manifest_path.exists():
        overlay_manifest = json.loads(overlay_manifest_path.read_text())
        for field, arg_value in (("head_dtype", args.head_dtype), ("dense_dtype", args.dense_dtype)):
            overlay_value = overlay_manifest.get(field)
            if overlay_value is not None and overlay_value != arg_value:
                print(f"WARNING: --{field.replace('_', '-')}={arg_value!r} differs from {field}="
                      f"{overlay_value!r} recorded in {overlay_manifest_path} (the overlay was produced "
                      "with a different head/dense dtype than this evaluation)")

    census = json.loads(Path(args.census).read_text()) if args.census else None

    routing_base = None
    if args.routing_base:
        rbr_by_frag, rbr_layers, rbr_cfg = D.load_routing_base(
            args.routing_base, dense_dtype=args.dense_dtype, wdot=args.wdot,
            mismatch_ok=args.routing_config_mismatch_ok,
        )
        routing_base = (rbr_by_frag, rbr_layers)
        print(f"--routing-base: {args.routing_base} loaded, {len(rbr_by_frag)} fragments, "
              f"{rbr_layers.shape[0]} layers, RBR config={rbr_cfg}, "
              f"run config=(dense_dtype={args.dense_dtype!r}, wdot={args.wdot!r}), "
              f"mismatch_ok={args.routing_config_mismatch_ok}")

    ple_arrays: dict[str, np.ndarray] | None = {} if args.ple_gate else None
    result = evaluate_corpus(
        replica, w, table, fragments, doc_tokens, eos, row_map, rows_var_overlay, forward_fn, args.k,
        census=census, routing_base=routing_base,
        skip_base=args.skip_base, skip_teacher=args.skip_teacher,
        ple_gate=args.ple_gate, ple_arrays=ple_arrays,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_eval_report(result, out_dir / "eval.json", out_dir / "eval.md")
    if args.ple_gate:
        np.savez(out_dir / "ple_gate.npz", **ple_arrays)
        print(f"ple_gate.npz written to {out_dir} ({len(ple_arrays)} arrays)")
    print(f"eval.json/eval.md written to {out_dir}, n_fragments={result['n_fragments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

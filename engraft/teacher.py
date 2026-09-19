"""Teacher/base-model target capture: a CLI around
`engraft.replica.distill.teacher_targets`/`save_routing_base`, called once
per corpus. With `--doc-tokens` it produces the teacher's targets (the
document in the prefix); without it, the SAME function with `doc_tokens=[]`
produces the BASE's targets.

Backend/EOS: reuses `engraft.descend_corpus.load_real_backend`/
`_fake_replica_and_table` (never copied) -- same backend arguments as
`engraft.descend_corpus`. `eos` always comes from `table.eos_token_id`, never
from a CLI argument (the document and the fragments share the same EOS by
construction).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import engraft.replica.distill as D
from engraft.descend_corpus import _fake_replica_and_table, load_real_backend


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_fragments_all(usage_corpus_resolved_path: Path) -> list[dict]:
    data = json.loads(Path(usage_corpus_resolved_path).read_text())
    return data["fragments"] if "fragments" in data else data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--usage-corpus", required=True, help="usage_corpus_resolved.json")
    parser.add_argument("--doc-tokens", default=None,
                         help="json list of the source document's tokens; omitted -> BASE targets (doc_tokens=[])")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--sample-full", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output .npz (TeacherTargets.save)")
    parser.add_argument("--bias-out", default=None, help="optional json: truncation bias_report")
    parser.add_argument("--routing-out", default=None,
                         help="locked-routing (RBR) output .npz, ONLY without --doc-tokens -- "
                              "raises if combined with it")
    parser.add_argument("--fake", action="store_true")
    # Backend (same arguments as engraft.descend_corpus)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--working-set-gb", type=float, default=2.0)
    parser.add_argument("--dequant-cache-gb", type=float, default=24.0)
    parser.add_argument("--dense-dtype", default="f32", choices=["f32", "bf16"])
    parser.add_argument("--wdot", default="split", choices=["bf16", "split", "f32"])
    parser.add_argument("--table-path", default=None, help="GGUF n-gram table shard path (real run only)")
    parser.add_argument("--shard-paths", nargs="+", default=None, help="GGUF weight shard paths (real run only)")
    parser.add_argument("--min-avail-gb", type=float, default=100.0)
    args = parser.parse_args(argv)

    if args.fake:
        replica, w, table, _step_fn, forward_fn = _fake_replica_and_table()
    else:  # pragma: no cover -- requires a real GGUF/CUDA device
        if not args.table_path or not args.shard_paths:
            raise SystemExit("a real run requires --table-path and --shard-paths")
        replica, w, table, _step_fn, forward_fn = load_real_backend(args)

    eos = int(table.eos_token_id)
    fragments_all = _load_fragments_all(args.usage_corpus)
    doc_tokens = json.loads(Path(args.doc_tokens).read_text()) if args.doc_tokens else []

    if args.routing_out and doc_tokens:
        raise SystemExit(
            "--routing-out requires the ABSENCE of --doc-tokens (locked-routing capture "
            "is offered only for the BASE run; the conditioned teacher stays at free routing)"
        )

    log(f"teacher_targets: {len(fragments_all)} fragments, k={args.k}, "
        f"document={'yes' if doc_tokens else 'no (base targets)'}, eos={eos}")
    routing_out: dict | None = {} if args.routing_out else None
    targets, bias_report = D.teacher_targets(
        replica, w, doc_tokens, fragments_all, args.k, forward_fn, eos,
        sample_full=args.sample_full, seed=args.seed, routing_out=routing_out,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets.save(out_path)
    log(f"targets written to {out_path} (n_positions={targets.ids.shape[0]}, k={args.k})")

    if routing_out is not None:
        routing_meta = D.save_routing_base(
            Path(args.routing_out), routing_out,
            dense_dtype=args.dense_dtype, moe_kernel="per_expert", wdot=args.wdot,
        )
        log(f"locked routing written to {args.routing_out}: {routing_meta}, "
            f"config=(dense_dtype={args.dense_dtype!r}, wdot={args.wdot!r})")

    if args.bias_out:
        bias_path = Path(args.bias_out)
        bias_path.parent.mkdir(parents=True, exist_ok=True)
        bias_path.write_text(json.dumps(bias_report, indent=2))
        log(f"bias_report written to {bias_path}: {bias_report}")
    elif bias_report is not None:
        log(f"bias_report (not saved, --bias-out not given): {bias_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

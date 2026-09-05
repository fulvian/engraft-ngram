"""Fake engine standing in for the fork's `llama-ple-lens` in tests and dry runs.

Speaks the fork's streaming protocol: prints `PLE_READY`, then for each JSON
line read from stdin performs a "job" and emits `PLE_RESULT {...}`; a blank
line or EOF terminates. Not a model: `logits(emb) = A @ emb + b` with `A`/`b`
fixed from a seed, and nothing positional (no real n-gram head, no history):
the only position that matters is the last one, and the same logit/tensor row
is repeated for every position requested by `logits: "all"` or a tensor
captured with a T axis.

Embedding construction: starts from zero (this toy engine's "true rows" are
null: there is no real n-gram table behind it) and writes into it, head by
head, the overlay vectors of the job (if present) -- the same "true row =
zero, overlay = written value" convention used throughout. Each overlay
entry's global row is translated to a head using only a table's addressing
metadata (`n_heads`, `head_offsets`, `head_vocab_sizes`), never its row
values: with `--fake-table`, `engraft.testing.fake_table.FakeTable` supplies
that metadata and no GGUF is read at all; without it, the table path is
resolved from `engraft.config` (`model.table`) if `--table` is not given.

`routing_record`/`routing_freeze` per job: a fake per-layer router uses the
first 10 experts of `_moe_probs(emb)` rotated by `il` (depends on `emb`, so a
free probe can change which expert is picked); `ne1` per layer follows the
same rule as the real engine (1 on the last layer with `logits: "last"`,
`n_token` elsewhere); `routing_freeze` substitutes the indices used with those
from the file (fail-fast on a shape mismatch or missing layer);
`ffn_moe_weights-<il>`, if captured, is `probs[ids]` with the indices actually
used (frozen or free).

uv run python -m engraft.testing.fake_lens --jobs - --out DIR [--fake-table] [--die-after N] [--table PATH]
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

from engraft.lens import read_pleo, read_plert1, write_plert1
from engraft.table import PleTable, ROW_LEN

N_VOCAB = 64
N_EMBD = 2560
N_HC = 4


N_EXPERTS = 512
N_EXPERT_USED = 10
N_LAYER = 48  # same 48 layers as the real model


@functools.lru_cache(maxsize=None)
def _toy_weights(seed: int = 1234) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((N_VOCAB, N_EMBD)).astype(np.float64) * 0.1
    b = rng.standard_normal(N_VOCAB).astype(np.float64) * 0.1
    return a, b


@functools.lru_cache(maxsize=None)
def _toy_moe_weights(seed: int = 5678) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_EXPERTS, N_EMBD)).astype(np.float64) * 0.1


def _moe_probs(emb: np.ndarray) -> np.ndarray:
    """Fake router: a deterministic softmax([512]) as a function of `emb`, via
    a fixed seeded projection, so an overlay perturbation deterministically
    changes the "top 10 experts"."""
    w = _toy_moe_weights()
    logits = w @ np.asarray(emb, dtype=np.float64)
    z = logits - logits.max()
    p = np.exp(z)
    return (p / p.sum()).astype(np.float32)


def _layer_ne1(il: int, t_len: int, logits_mode: str) -> int:
    """ne1=1 on the last layer with `logits: "last"` (build_inp_out_ids cuts to
    n_outputs=1 before the last MoE layer), n_token elsewhere -- including the
    last layer when `logits: "all"` (n_outputs=n_token, no cut)."""
    if il == N_LAYER - 1 and logits_mode == "last":
        return 1
    return t_len


def _build_routing(
    job: dict, emb: np.ndarray, t_len: int, logits_mode: str,
) -> tuple[dict[int, np.ndarray], str | None, int, int]:
    """Applies routing_record/routing_freeze to the fake engine.

    Ritorna (used_ids, error, routing_frozen_layers, routing_recorded_layers):
    `used_ids[il]` e' l'array [ne1, 10] int32 effettivamente usato per quello strato
    (congelato se routing_freeze e' attivo, altrimenti libero) -- letto da
    `ffn_moe_weights-<il>` quando catturato. Un `error` non vuoto significa job fallito
    (fail-fast, stesso contratto del fork C++): il chiamante non deve scrivere meta.json.
    """
    record_active = job.get("routing_record") is not None
    freeze_active = job.get("routing_freeze") is not None

    freeze_map: dict[int, np.ndarray] | None = None
    if freeze_active:
        try:
            freeze_map = read_plert1(job["routing_freeze"])
        except (OSError, ValueError) as e:
            return {}, f"routing_freeze: {e}", 0, 0

    used_ids: dict[int, np.ndarray] = {}
    record_map: dict[int, np.ndarray] = {}
    frozen_layers = 0
    # `_moe_probs(emb)` (quindi `_toy_moe_weights`) e l'argsort dipendono solo da `emb`,
    # non da `il`: calcolati una volta per lavoro invece che 48 volte (regressione di
    # prestazioni segnalata dal conduttore -- stesso risultato numerico di
    # `_free_ids_for_layer` chiamata per strato, solo il `np.roll` resta per strato).
    top10 = np.argsort(-_moe_probs(emb))[:N_EXPERT_USED].copy()
    for il in range(N_LAYER):
        ne1 = _layer_ne1(il, t_len, logits_mode)
        free_row = np.roll(top10, il).astype(np.int32)
        free_ids = np.broadcast_to(free_row, (ne1, N_EXPERT_USED)).copy()

        if record_active:
            record_map[il] = free_ids  # "registra il valore corrente (routing libero)" prima della riscrittura

        if freeze_active:
            want = freeze_map.get(il)
            if want is None:
                return {}, f"routing_freeze: layer {il} absent from file", 0, 0
            if want.shape != (ne1, N_EXPERT_USED):
                return (
                    {}, f"routing_freeze: layer {il} shape mismatch: file {want.shape} tensor {(ne1, N_EXPERT_USED)}",
                    0, 0,
                )
            used_ids[il] = want
            frozen_layers += 1
        else:
            used_ids[il] = free_ids

    if freeze_active and frozen_layers != N_LAYER:
        return {}, f"routing_freeze: froze {frozen_layers} layer(s), expected {N_LAYER}", 0, 0

    recorded_layers = 0
    if record_active:
        if len(record_map) != N_LAYER:
            return {}, f"routing_record: recorded {len(record_map)} layer(s), expected {N_LAYER}", 0, 0
        write_plert1(job["routing_record"], record_map)
        recorded_layers = N_LAYER

    return used_ids, None, frozen_layers, recorded_layers


def _head_for_row(table: PleTable, row_global: int) -> int | None:
    for h in range(table.n_heads):
        off = table.head_offsets[h]
        if off <= row_global < off + table.head_vocab_sizes[h]:
            return h
    return None


def _build_emb(table: PleTable, overlay_path: str | None) -> tuple[np.ndarray, int]:
    emb = np.zeros(N_EMBD, dtype=np.float64)
    if overlay_path is None:
        return emb, 0
    rows, data = read_pleo(overlay_path)
    hits = 0
    for row_g, vec in zip(rows.tolist(), data):
        h = _head_for_row(table, int(row_g))
        if h is None:
            continue
        emb[h * ROW_LEN:(h + 1) * ROW_LEN] = vec
        hits += 1
    return emb, hits


def _capture_tensor_data(
    name: str, t_len: int, emb: np.ndarray, logits_mode: str, used_ids: dict[int, np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """ne (ordine ggml: ne0 piu' veloce) e dati, per la convenzione di lettura di
    `ple_lens.load_job` (`reshape(ne[3], ne[2], ne[1], ne[0])`, sempre in ordine C)."""
    if name == "ple_embd-1":
        ne = [N_EMBD, t_len, 1, 1]
        arr = np.broadcast_to(emb.astype(np.float32), (1, 1, t_len, N_EMBD)).copy()
    elif name == "l_last-0":
        ne = [N_EMBD, N_HC, t_len, 1]
        arr = np.zeros((1, t_len, N_HC, N_EMBD), dtype=np.float32)
    elif name == "ple_gate-1":
        ne = [1, N_HC, t_len, 1]
        arr = np.full((1, t_len, N_HC, 1), 0.5, dtype=np.float32)
    elif name in ("ple_gated_value-1", "ple_conv_out-1"):
        ne = [N_EMBD, N_HC, t_len, 1]
        arr = np.zeros((1, t_len, N_HC, N_EMBD), dtype=np.float32)
    elif name.startswith("ffn_moe_probs-"):
        # [512, ne1] to the callback: ne1=1 on the last layer with logits:"last",
        # n_token elsewhere -- same probs row repeated for every position, like
        # the other captures of this fake engine.
        il = int(name.split("-")[1])
        ne1 = _layer_ne1(il, t_len, logits_mode)
        ne = [N_EXPERTS, ne1, 1, 1]
        probs = _moe_probs(emb)
        arr = np.broadcast_to(probs, (1, 1, ne1, N_EXPERTS)).copy()
    elif name.startswith("ffn_moe_weights-"):
        # [n_expert_used, ne1] (spec 2d §3.1 "Motore finto": probs[ids], ids = quelli
        # effettivamente usati per lo strato -- congelati se routing_freeze e' attivo).
        il = int(name.split("-")[1])
        ne1 = _layer_ne1(il, t_len, logits_mode)
        ids = used_ids[il]  # [ne1, 10]
        probs = _moe_probs(emb)
        weights = probs[ids]  # [ne1, 10]
        ne = [N_EXPERT_USED, ne1, 1, 1]
        arr = weights.reshape(1, 1, ne1, N_EXPERT_USED).astype(np.float32)
    else:
        raise ValueError(f"tensor not expected by the fake engine: {name!r}")
    return ne, arr


def _write_f32(path: Path, arr: np.ndarray) -> None:
    np.asarray(arr, dtype=np.float32).tofile(path)


def _write_calls(out_dir: Path, n_calls: int) -> None:
    (out_dir / "_calls.json").write_text(json.dumps({"n_calls": n_calls}))


def _emit(obj: dict) -> None:
    print("PLE_RESULT " + json.dumps(obj))
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--die-after", type=int, default=None)
    # Resolved lazily below (never at import time, and never required unless
    # this module actually runs): default None means "read engraft.toml".
    parser.add_argument("--table", default=None)
    parser.add_argument(
        "--fake-table", action="store_true",
        help="use engraft.testing.fake_table instead of a real GGUF (no table read at all)",
    )
    # per-group engine args (-fa/-ctk/-ctv): accepted and ignored by the fake
    # engine, but recorded together with the relevant env vars in
    # `engine_meta.json` so a caller can verify `LensClient`/engine switching
    # actually passed them through.
    parser.add_argument("-fa", dest="fa", default=None)
    parser.add_argument("-ctk", dest="ctk", default=None)
    parser.add_argument("-ctv", dest="ctv", default=None)
    args = parser.parse_args(argv)

    if args.jobs != "-":
        print(f"fake_lens: streaming mode only (--jobs -), got {args.jobs!r}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fake_table:
        from engraft.testing.fake_table import FakeTable

        table = FakeTable(seed=42)
    else:
        table_path = args.table
        if table_path is None:
            from engraft.config import load as load_config

            table_path = load_config().get("model.table")
        table = PleTable(table_path)
    a, b = _toy_weights()

    (out_dir / "engine_meta.json").write_text(json.dumps({
        "fa": args.fa, "ctk": args.ctk, "ctv": args.ctv,
        "env": {
            "GGML_CUDA_FORCE_CUBLAS_RT": os.environ.get("GGML_CUDA_FORCE_CUBLAS_RT"),
            "GGML_CUDA_CUBLAS_COMPUTE_TYPE": os.environ.get("GGML_CUDA_CUBLAS_COMPUTE_TYPE"),
        },
    }))

    n_calls = 0
    _write_calls(out_dir, n_calls)

    print("PLE_READY")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.strip() == "":
            break
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"id": "?", "status": "error", "error": f"invalid JSON: {e}"})
            continue

        job_id = job.get("id", "?")
        job_dir = out_dir / job_id
        meta_path = job_dir / "meta.json"

        if meta_path.exists():
            _emit({"id": job_id, "status": "skipped"})
            continue

        if "__fail__" in job_id:
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "error.txt").write_text("fake_lens: fallimento richiesto dal job id")
            _emit({"id": job_id, "status": "error", "error": "fake_lens: fallimento richiesto dal job id"})
            continue

        tokens = job["tokens"]
        t_len = len(tokens)
        overlay_path = job.get("overlay")
        emb, hits = _build_emb(table, overlay_path)
        logits_vec = (a @ emb + b).astype(np.float32)  # [n_vocab]

        used_ids, routing_error, frozen_layers, recorded_layers = _build_routing(
            job, emb, t_len, job["logits"],
        )
        if routing_error is not None:
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "error.txt").write_text(routing_error)
            _emit({"id": job_id, "status": "error", "error": routing_error})
            continue

        job_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": job_id,
            "tokens": tokens,
            "n_vocab": N_VOCAB,
            "logits_mode": job["logits"],
            "overlay_hits": hits,
            "overlay_rows": hits,
            "t_decode_ms": 0.0,
            "fork_commit": "fake",
            "tensors": {},
            "routing_frozen_layers": frozen_layers,
            "routing_recorded_layers": recorded_layers,
        }
        for name in job.get("capture", []):
            ne, arr = _capture_tensor_data(name, t_len, emb, job["logits"], used_ids)
            _write_f32(job_dir / f"{name}.f32", arr)
            meta["tensors"][name] = {"ne": ne, "nb": [], "type": "f32", "file": f"{name}.f32"}

        n_pos = t_len if job["logits"] == "all" else 1
        logits_out = np.tile(logits_vec, (n_pos, 1))
        _write_f32(job_dir / "logits.f32", logits_out)

        meta_tmp = job_dir / "meta.json.tmp"
        meta_tmp.write_text(json.dumps(meta))
        meta_tmp.replace(meta_path)

        n_calls += 1
        _write_calls(out_dir, n_calls)

        _emit({
            "id": job_id, "status": "ok", "overlay_hits": hits, "t_decode_ms": 0.0,
            "routing_frozen_layers": frozen_layers, "routing_recorded_layers": recorded_layers,
        })

        if args.die_after is not None and n_calls >= args.die_after:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

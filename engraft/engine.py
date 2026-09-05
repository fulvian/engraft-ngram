"""Client for the fork's streaming lens protocol, plus small measurement helpers.

Extracted from the research driver: only the engine-facing plumbing (process
management, the streaming JSON protocol, per-engine launch flags) and a
handful of pure measurement functions used against the resulting logits. None
of the fact- or experiment-specific orchestration logic lives here; that
belongs to `engraft.facts`, `engraft.run`, and `engraft.check`.

Tokenizer resolution is deliberately not a filesystem search: the tokenizer
path is an explicit configuration key (`model.tokenizer`), resolved by the
caller through `engraft.config` and passed in. No path here is fixed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from engraft.lens import logits as job_logits
from engraft.table import PleTokenizer

# Per-engine launch flags: "q8" is the default quantized engine, "f32" forces
# full-precision KV cache/compute for a fidelity check. `-fa` is appended by
# the caller per engine, not baked into `--lens-cmd`. `fallback_args` is used
# only if the process dies before PLE_READY under "f32" (the model likely
# rejects `-fa off -ctk f32 -ctv f32`).
ENGINE_CFG = {
    "q8": {"env": {}, "args": ["-fa", "on"]},
    "f32": {
        "env": {"GGML_CUDA_FORCE_CUBLAS_RT": "1", "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"},
        "args": ["-fa", "off", "-ctk", "f32", "-ctv", "f32"],
        "fallback_args": ["-fa", "on"],
    },
}


class LensError(RuntimeError):
    """A job failed (status=error from the tool) or the engine process died."""


class LensClient:
    """Talks the fork's streaming lens protocol over stdin/stdout.

    Starts the engine (or a fake standing in for it), waits for the
    ``PLE_READY`` line, then exchanges one JSON job per line for a
    ``PLE_RESULT ...`` line back.
    """

    def __init__(self, cmd: list[str], out_dir: Path, log_path: Path, env: dict[str, str] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env = dict(env or {})
        self._log_f = open(log_path, "a", buffering=1)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_f,
            text=True,
            bufsize=1,
            env={**os.environ, **self.env},
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise LensError("engine exited before PLE_READY (see log)")
            if line.strip() == "PLE_READY":
                return

    def run(self, job: dict) -> dict:
        self._proc.stdin.write(json.dumps(job) + "\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise LensError(f"engine exited during job {job.get('id')!r} (see log)")
            line = line.strip()
            if line.startswith("PLE_RESULT "):
                result = json.loads(line[len("PLE_RESULT "):])
                break
        if result.get("status") == "error":
            raise LensError(f"{result.get('id')}: {result.get('error')}")
        if result.get("status") == "skipped":
            meta = json.loads((self.out_dir / result["id"] / "meta.json").read_text())
            result["overlay_hits"] = meta.get("overlay_hits")
            result["t_decode_ms"] = meta.get("t_decode_ms")
        return result

    def close(self) -> None:
        try:
            self._proc.stdin.write("\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._log_f.close()


def _decode_ids(tok: PleTokenizer, ids: list[int]) -> str:
    """Decodes via the underlying HF tokenizer (PleTokenizer exposes only encode)."""
    return tok._tok.decode(ids)  # noqa: SLF001 -- no other way without touching table.py


# --------------------------------------------------------------------------
# Pure measurement helpers on logits/results
# --------------------------------------------------------------------------


def logsoftmax64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return x - m - np.log(np.sum(np.exp(x - m)))


def rank_of(row: np.ndarray, y: int) -> int:
    """1-based rank of `y` in descending logit order (ties: stable order)."""
    order = np.argsort(-row, kind="stable")
    return int(np.nonzero(order == y)[0][0]) + 1


def _read_meta(raw_dir: Path, job_id: str) -> dict:
    return json.loads((raw_dir / job_id / "meta.json").read_text())


def _rmtree_raw(raw_dir: Path, job_id: str) -> None:
    shutil.rmtree(raw_dir / job_id, ignore_errors=True)


def run_job(client: LensClient, raw_dir: Path, job: dict, log: logging.Logger) -> tuple[dict, np.ndarray, dict]:
    """Runs `job`, returns (result, last-position logit row float64, meta.json).

    Cleans the raw dump before returning: only for `logits: "last"` jobs, or
    when only the last position is needed. Use `run_job_all` for
    `logits: "all"` (corpus/document) jobs.
    """
    result, all_rows, meta = run_job_all(client, raw_dir, job, log)
    return result, all_rows[-1], meta


def run_job_all(client: LensClient, raw_dir: Path, job: dict, log: logging.Logger) -> tuple[dict, np.ndarray, dict]:
    """Like `run_job` but returns all positions (needed by `corpus_nll_mean`)."""
    t0 = time.time()
    result = client.run(job)
    dt = (time.time() - t0) * 1000.0
    meta = _read_meta(raw_dir, job["id"])
    all_rows = np.asarray(job_logits(raw_dir, job["id"]), dtype=np.float64).copy()
    log.info(
        "%s status=%s overlay_hits=%s t_decode_ms=%.1f (rtt=%.1fms)",
        job["id"], result.get("status"), result.get("overlay_hits"), result.get("t_decode_ms", 0.0), dt,
    )
    _rmtree_raw(raw_dir, job["id"])
    return result, all_rows, meta


def check_expected_hits(job_id: str, result: dict, expected: int) -> None:
    """Fail-fast: a trigger-position job with overlay_hits != expected is an
    error, not a data point."""
    got = result.get("overlay_hits")
    if got != expected:
        raise RuntimeError(
            f"{job_id}: overlay_hits={got!r}, expected {expected} (fail-fast, not a data point)"
        )


def greedy_continuation(
    client: LensClient, raw_dir: Path, tok: PleTokenizer, log: logging.Logger,
    prefix_tokens: list[int], overlay_path: Path, n: int, tag: str,
) -> tuple[list[int], str, bool]:
    seq = list(prefix_tokens)
    gen: list[int] = []
    for i in range(n):
        job = {
            "id": f"{tag}_greedy{i}", "text": "", "tokens": seq,
            "overlay": str(overlay_path), "capture": [], "logits": "last",
        }
        _result, row, _meta = run_job(client, raw_dir, job, log)
        nxt = int(np.argmax(row))
        gen.append(nxt)
        seq.append(nxt)
    text = _decode_ids(tok, gen)
    degenerate = (len(gen) >= 3 and gen[-1] == gen[-2] == gen[-3]) or text.strip() == ""
    return gen, text, degenerate


def corpus_nll_mean(logits_all: np.ndarray, targets: np.ndarray) -> float:
    """logits_all: [n_pos, n_vocab] float, targets: [n_pos-1] next tokens.
    Mean NLL in nats, per-row log-softmax (float64)."""
    lp = np.empty(len(targets), dtype=np.float64)
    for i in range(len(targets)):
        lp[i] = logsoftmax64(logits_all[i])[targets[i]]
    return float(np.mean(-lp))

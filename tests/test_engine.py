"""Tests for engraft.engine, the functions extracted from the research driver.

Runs a minimal fake engine (defined in this file, not `engraft.testing.fake_lens`:
that one needs a real GGUF for row-to-head addressing metadata, which this test
suite must never touch) that speaks the same streaming protocol as the real
engine but computes logits from a fixed linear map over the overlay vectors,
with no n-gram table at all. This is enough to exercise every function
extracted into `engraft.engine`: none of them touch table addressing, only the
client protocol and the resulting logits/results.

uv run pytest tests/test_engine.py
"""
from __future__ import annotations

import json
import logging
import sys
import textwrap

import numpy as np
import pytest

from engraft.engine import (
    check_expected_hits,
    corpus_nll_mean,
    greedy_continuation,
    logsoftmax64,
    rank_of,
    run_job,
    run_job_all,
)
from engraft.lens import write_pleo
from engraft.table import ROW_LEN

N_VOCAB = 16

_FAKE_ENGINE_SRC = textwrap.dedent(f"""
    import argparse, json, sys
    from pathlib import Path
    import numpy as np

    N_VOCAB = {N_VOCAB}
    N_EMBD = 4 * {ROW_LEN}

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--jobs", required=True)
        p.add_argument("--out", required=True)
        args = p.parse_args()
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(0)
        a = rng.standard_normal((N_VOCAB, N_EMBD)).astype(np.float64) * 0.1
        b = rng.standard_normal(N_VOCAB).astype(np.float64) * 0.1

        print("PLE_READY", flush=True)
        for line in sys.stdin:
            line = line.rstrip("\\n")
            if line.strip() == "":
                break
            job = json.loads(line)
            job_id = job["id"]
            tokens = job["tokens"]
            overlay = job.get("overlay")
            emb = np.zeros(N_EMBD, dtype=np.float64)
            hits = 0
            if overlay:
                magic_len = 4
                with open(overlay, "rb") as f:
                    f.read(magic_len)
                    n, dim = np.frombuffer(f.read(8), dtype="<u4")
                    rows = np.frombuffer(f.read(int(n) * 4), dtype="<i4")
                    data = np.frombuffer(f.read(int(n) * int(dim) * 4), dtype="<f4").reshape(int(n), int(dim))
                hits = int(n)
                for i in range(min(int(n), N_EMBD // {ROW_LEN})):
                    lo, hi = i * {ROW_LEN}, (i + 1) * {ROW_LEN}
                    emb[lo:hi] = data[i].astype(np.float64)
            logits_vec = (a @ emb + b).astype(np.float32)
            n_pos = len(tokens) if job["logits"] == "all" else 1
            job_dir = out_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            np.tile(logits_vec, (n_pos, 1)).tofile(job_dir / "logits.f32")
            meta = {{"id": job_id, "tokens": tokens, "n_vocab": N_VOCAB, "logits_mode": job["logits"], "overlay_hits": hits, "t_decode_ms": 0.0}}
            (job_dir / "meta.json").write_text(json.dumps(meta))
            print("PLE_RESULT " + json.dumps({{"id": job_id, "status": "ok", "overlay_hits": hits, "t_decode_ms": 0.0}}), flush=True)

    if __name__ == "__main__":
        main()
""")


@pytest.fixture()
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture()
def engine_script(tmp_path):
    path = tmp_path / "fake_engine_no_table.py"
    path.write_text(_FAKE_ENGINE_SRC)
    return path


@pytest.fixture()
def client(tmp_path, raw_dir, engine_script):
    from engraft.engine import LensClient

    log_path = tmp_path / "engine.log"
    c = LensClient([sys.executable, str(engine_script), "--jobs", "-", "--out", str(raw_dir)], raw_dir, log_path)
    yield c
    c.close()


@pytest.fixture()
def log():
    logger = logging.getLogger("test_engine")
    logger.addHandler(logging.NullHandler())
    return logger


def _overlay_path(tmp_path, n_heads: int = 4):
    rows = np.arange(1000, 1000 + n_heads, dtype=np.int32)
    data = np.ones((n_heads, ROW_LEN), dtype=np.float32)
    path = tmp_path / "overlay.pleo"
    write_pleo(path, rows, data)
    return path


def test_run_job_returns_expected_shape(client, raw_dir, tmp_path, log):
    overlay = _overlay_path(tmp_path)
    job = {"id": "j1", "text": "", "tokens": [1, 2, 3], "overlay": str(overlay), "capture": [], "logits": "last"}
    result, row, meta = run_job(client, raw_dir, job, log)
    assert result["status"] == "ok"
    assert row.shape == (N_VOCAB,)
    assert row.dtype == np.float64
    assert meta["n_vocab"] == N_VOCAB
    assert not (raw_dir / "j1").exists()  # run_job cleans the raw dump


def test_run_job_all_returns_every_position(client, raw_dir, tmp_path, log):
    overlay = _overlay_path(tmp_path)
    job = {"id": "j2", "text": "", "tokens": [1, 2, 3, 4], "overlay": str(overlay), "capture": [], "logits": "all"}
    result, all_rows, meta = run_job_all(client, raw_dir, job, log)
    assert result["status"] == "ok"
    assert all_rows.shape == (4, N_VOCAB)


def test_run_job_overlay_changes_logits(client, raw_dir, tmp_path, log):
    job_base = {"id": "base", "text": "", "tokens": [1, 2], "overlay": None, "capture": [], "logits": "last"}
    _r, row_base, _m = run_job(client, raw_dir, job_base, log)
    overlay = _overlay_path(tmp_path)
    job_ov = {"id": "ov", "text": "", "tokens": [1, 2], "overlay": str(overlay), "capture": [], "logits": "last"}
    _r2, row_ov, _m2 = run_job(client, raw_dir, job_ov, log)
    assert not np.array_equal(row_base, row_ov)


def test_logsoftmax64_sums_to_one():
    x = np.array([1.0, 2.0, 3.0, -1.0])
    lp = logsoftmax64(x)
    assert lp.dtype == np.float64
    np.testing.assert_allclose(np.sum(np.exp(lp)), 1.0, atol=1e-12)


def test_rank_of_orders_descending():
    row = np.array([0.1, 5.0, 3.0, -2.0])
    assert rank_of(row, 1) == 1  # largest
    assert rank_of(row, 2) == 2
    assert rank_of(row, 0) == 3
    assert rank_of(row, 3) == 4  # smallest


def test_rank_of_stable_on_ties():
    row = np.array([1.0, 1.0, 0.0])
    assert rank_of(row, 0) == 1  # first of the tied pair wins the tie (stable order)


def test_check_expected_hits_passes_on_match():
    check_expected_hits("job", {"overlay_hits": 4}, 4)  # must not raise


def test_check_expected_hits_raises_on_mismatch():
    with pytest.raises(RuntimeError):
        check_expected_hits("job", {"overlay_hits": 3}, 4)


def test_corpus_nll_mean_matches_hand_computation():
    logits_all = np.array([[0.0, 0.0], [1.0, -1.0]])
    targets = np.array([1])
    got = corpus_nll_mean(logits_all, targets)
    expected = -logsoftmax64(logits_all[0])[1]
    assert got == pytest.approx(expected)


def test_greedy_continuation_generates_n_tokens(client, raw_dir, tmp_path, log):
    class _FakeTok:
        def __init__(self):
            self._tok = self

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    overlay = _overlay_path(tmp_path)
    tok = _FakeTok()
    gen, text, degenerate = greedy_continuation(client, raw_dir, tok, log, [1, 2], overlay, 3, "tag")
    assert len(gen) == 3
    assert text == " ".join(str(i) for i in gen)
    assert isinstance(degenerate, bool)

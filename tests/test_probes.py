"""Tests for engraft.probes: Quail composition probes. No real engine, no
GGUF -- a fake client writes the raw files like the real engine (`run_job`,
unchanged, reused via `engraft.engine_check._greedy_continuation_local`) and
produces controlled generated text, so the binary scoring (both `answer`
strings as case-sensitive substrings) is checkable without the engine.

uv run pytest tests/test_probes.py -q
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import engraft.probes as Q

EOS = 42


# --------------------------------------------------------------------------
# Fixture: probes + a fake fact registry (no real tokenizer)
# --------------------------------------------------------------------------


def _probes():
    return [
        {"id": "probe_a", "fact_ids": ["f1", "f2"], "text": "question about mars and sun",
         "subject": "Subject One"},
        {"id": "probe_b", "fact_ids": ["f3", "f4"], "text": "question about moon and star",
         "subject": "Subject One"},
        {"id": "probe_c", "fact_ids": ["f5", "f6"], "text": "question about river and mountain",
         "subject": "Subject Two"},
    ]


def _facts():
    return [
        {"fact_id": "f1", "answer": "Mars"},
        {"fact_id": "f2", "answer": "Sun"},
        {"fact_id": "f3", "answer": "Moon"},
        {"fact_id": "f4", "answer": "Star"},
        {"fact_id": "f5", "answer": "Nile"},
        {"fact_id": "f6", "answer": "Everest"},
    ]


# --------------------------------------------------------------------------
# Fake tokenizer/decoder: encode by word (deterministic), decode by
# space-joined words -- needed to produce a readable greedy output (unlike
# test_engine_check.py's `_FakeTok`, which only decodes numbers: here we
# need to produce "Mars"/"mars" for the case-sensitive comparison).
# --------------------------------------------------------------------------


class _WordVocab:
    """Small fake vocabulary: every known word has a fixed id; unknown ids
    decode to `"<w{id}>"`."""

    def __init__(self):
        self._id_to_word = {
            0: "question", 1: "about", 2: "mars", 3: "and", 4: "sun", 5: "moon", 6: "star",
            7: "river", 8: "mountain", 100: "Mars", 101: "Sun", 102: "mars", 103: "Moon",
            104: "Star", 105: "Nile", 106: "Everest", 200: "other",
        }
        self._word_to_id = {w: i for i, w in self._id_to_word.items()}

    def encode(self, text: str) -> list[int]:
        return [self._word_to_id.get(w, 999) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_word.get(i, f"<w{i}>") for i in ids)


class _FakeTok:
    """Like `PleTokenizer`: exposes `encode` and, via `_tok.decode`, the
    decoding `engraft.engine._decode_ids` uses."""

    def __init__(self, *_a, **_k):
        self._tok = _WordVocab()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)


# --------------------------------------------------------------------------
# FakeLensClient: like test_engine_check.py's, adapted to the job id
# `gr_{probe_id}_{column}_greedy{i}` (no `pf_*` job here: probes have no
# single-token target, only a generation sequence). Every probe carries a
# fixed token sequence to generate, taken from the `sequences` seed.
# --------------------------------------------------------------------------


_JOB_ID_RE = re.compile(r"^gr_(.+)_(base|student)_greedy(\d+)$")


class FakeLensClient:
    def __init__(self, raw_dir: Path, sequences: dict[str, list[int]], n_vocab: int = 256):
        self.raw_dir = Path(raw_dir)
        self.sequences = sequences
        self.n_vocab = n_vocab
        self.calls: list[dict] = []

    def run(self, job: dict) -> dict:
        self.calls.append(job)
        job_id = job["id"]
        m = _JOB_ID_RE.match(job_id)
        if not m:
            raise ValueError(f"unexpected job id in the fake client: {job_id!r}")
        probe_id, _kind, step = m.groups()
        step = int(step)
        job_dir = self.raw_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        digest = int(hashlib.sha256(job_id.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(1000 + digest)
        arr = (rng.standard_normal((1, self.n_vocab)) * 0.01).astype(np.float32)

        seq = self.sequences[probe_id]
        y = seq[min(step, len(seq) - 1)]
        arr[-1, y] += 12.0  # this step's target pushed to the top (greedy = argmax)

        arr.tofile(job_dir / "logits.f32")
        (job_dir / "meta.json").write_text(json.dumps({
            "n_vocab": self.n_vocab, "tokens": job["tokens"], "logits_mode": job["logits"],
        }))
        return {"status": "ok", "id": job_id, "overlay_hits": 0, "t_decode_ms": 0.1}

    def close(self) -> None:
        pass


def _write_fixture_files(tmp_path):
    probes_path = tmp_path / "probes.json"
    probes_path.write_text(json.dumps({"probes": _probes(), "raw": [], "probe_attempt": 1}))
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps({"facts": _facts()}))
    return probes_path, facts_path


# --------------------------------------------------------------------------
# (1) --dry-run: one job per probe, EOS at the front, overlay matches the flag
# --------------------------------------------------------------------------


def test_dry_run_writes_one_job_per_probe(tmp_path, monkeypatch):
    probes_path, facts_path = _write_fixture_files(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setenv("ENGRAFT_MODEL_TOKENIZER", "unused")
    monkeypatch.setattr(Q, "PleTokenizer", _FakeTok)
    monkeypatch.setattr(Q, "LensClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no engine")))

    rc = Q.main(["--probes", str(probes_path), "--facts", str(facts_path), "--out", str(out_dir),
                 "--eos", str(EOS), "--dry-run"])
    assert rc == 0
    jobs = json.loads((out_dir / "jobs.json").read_text())
    assert jobs["n_probes"] == 3
    assert jobs["n_expected_calls"] == 3 * 40  # default --max-new-tokens
    assert len(jobs["jobs"]) == 3
    for j in jobs["jobs"]:
        assert j["first_step_job"]["tokens"][0] == EOS
        assert j["first_step_job"]["overlay"] is None
        assert j["column"] == "base"
        assert j["first_step_job"]["logits"] == "last"
        assert j["first_step_job"]["capture"] == []


def test_dry_run_with_overlay_sets_student_column_and_path(tmp_path, monkeypatch):
    probes_path, facts_path = _write_fixture_files(tmp_path)
    overlay_path = tmp_path / "merged.pleo"
    overlay_path.write_text("fake")
    out_dir = tmp_path / "out"

    monkeypatch.setenv("ENGRAFT_MODEL_TOKENIZER", "unused")
    monkeypatch.setattr(Q, "PleTokenizer", _FakeTok)
    monkeypatch.setattr(Q, "LensClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no engine")))

    rc = Q.main(["--probes", str(probes_path), "--facts", str(facts_path), "--out", str(out_dir),
                 "--overlay", str(overlay_path), "--eos", str(EOS), "--dry-run"])
    assert rc == 0
    jobs = json.loads((out_dir / "jobs.json").read_text())
    for j in jobs["jobs"]:
        assert j["column"] == "student"
        assert j["first_step_job"]["overlay"] == str(overlay_path.resolve())


# --------------------------------------------------------------------------
# (2) Scoring: both -> both True; only one -> False with correct hit_a/hit_b;
# case-sensitive comparison.
# --------------------------------------------------------------------------


def test_score_both_hits():
    outcome = Q.score("Mars and Sun in the sky", "Mars", "Sun")
    assert outcome == {"hit_a": True, "hit_b": True, "both": True}


def test_score_only_one_hit():
    outcome = Q.score("Mars and nothing else", "Mars", "Sun")
    assert outcome == {"hit_a": True, "hit_b": False, "both": False}


def test_score_case_sensitive_no_match():
    # lowercase "mars" does not count for "Mars" (case-sensitive comparison).
    outcome = Q.score("mars without a good answer", "Mars", "Sun")
    assert outcome == {"hit_a": False, "hit_b": False, "both": False}


# --------------------------------------------------------------------------
# (3) missing fact_id from the registry -> a clear error
# --------------------------------------------------------------------------


def test_resolve_answers_raises_on_missing_fact_id():
    probe = {"id": "probe_x", "fact_ids": ["f1", "fXXX"]}
    facts_by_id = {"f1": {"answer": "Mars"}}
    with pytest.raises(ValueError, match="fXXX"):
        Q.resolve_answers(probe, facts_by_id)


def test_resolve_answers_raises_on_wrong_fact_id_count():
    probe = {"id": "probe_y", "fact_ids": ["f1"]}
    with pytest.raises(ValueError, match="probe_y"):
        Q.resolve_answers(probe, {"f1": {"answer": "Mars"}})


# --------------------------------------------------------------------------
# (4) --render-only reproduces the report from a saved probe_results.json
# --------------------------------------------------------------------------


def _records_for_report():
    return [
        {"id": "probe_a", "fact_ids": ["f1", "f2"], "subject": "Subject One", "text": "t",
         "answer_a": "Mars", "answer_b": "Sun", "output": "Mars and Sun", "tokens": [100, 3, 101],
         "hit_a": True, "hit_b": True, "both": True, "column": "base", "overlay": None},
        {"id": "probe_b", "fact_ids": ["f3", "f4"], "subject": "Subject One", "text": "t",
         "answer_a": "Moon", "answer_b": "Star", "output": "only other", "tokens": [200],
         "hit_a": False, "hit_b": False, "both": False, "column": "base", "overlay": None},
        {"id": "probe_c", "fact_ids": ["f5", "f6"], "subject": "Subject Two", "text": "t",
         "answer_a": "Nile", "answer_b": "Everest", "output": "nothing here", "tokens": [200],
         "hit_a": False, "hit_b": False, "both": False, "column": "base", "overlay": None},
    ]


def test_render_only_reproduces_report_from_saved_results(tmp_path):
    records = _records_for_report()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "probe_results.json").write_text(json.dumps(records))
    direct_report = Q.build_report(records)

    rc = Q.main(["--out", str(out_dir), "--render-only"])
    assert rc == 0
    rendered = (out_dir / "report.md").read_text()
    assert rendered == direct_report
    assert "Probes with both answers" in rendered
    assert "probe_a" in rendered  # the only probe with both=True, listed


def test_render_only_fails_clearly_without_results_json(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = Q.main(["--out", str(out_dir), "--render-only"])
    assert rc == 2


def test_report_renders_with_zero_hits():
    records = [r for r in _records_for_report() if not r["both"]]
    report = Q.build_report(records)
    assert "Probes with both answers (`both`): 0" in report
    assert "(none)" in report


# --------------------------------------------------------------------------
# (5) Full run with the fake client on 2 probes: probe_results.json + report.md
# coherent (one probe answers both correctly, the other only one).
# --------------------------------------------------------------------------


def test_full_run_with_fake_client_produces_coherent_results_and_report(tmp_path, monkeypatch):
    probes_two = _probes()[:2]  # probe_a (f1 Mars / f2 Sun), probe_b (f3 Moon / f4 Star)
    probes_path = tmp_path / "probes.json"
    probes_path.write_text(json.dumps({"probes": probes_two, "raw": [], "probe_attempt": 1}))
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps({"facts": _facts()}))
    out_dir = tmp_path / "out"

    monkeypatch.setenv("ENGRAFT_MODEL_TOKENIZER", "unused")
    monkeypatch.setattr(Q, "PleTokenizer", _FakeTok)

    # probe_a: generates "Mars" (100) then "Sun" (101) -> both True.
    # probe_b: generates "Moon" (103) then an unrelated token (200="other") -> only hit_a.
    sequences = {"probe_a": [100, 101], "probe_b": [103, 200]}
    client = FakeLensClient(out_dir / "raw", sequences)
    monkeypatch.setattr(Q, "LensClient", lambda *a, **k: client)

    rc = Q.main(["--probes", str(probes_path), "--facts", str(facts_path), "--out", str(out_dir),
                 "--lens-cmd", "fake-engine", "--eos", str(EOS), "--max-new-tokens", "2"])
    assert rc == 0
    records = json.loads((out_dir / "probe_results.json").read_text())
    assert len(records) == 2
    by_id = {r["id"]: r for r in records}
    assert by_id["probe_a"]["both"] is True
    assert by_id["probe_a"]["output"] == "Mars Sun"
    assert by_id["probe_a"]["tokens"] == [100, 101]
    assert by_id["probe_b"]["both"] is False
    assert by_id["probe_b"]["hit_a"] is True
    assert by_id["probe_b"]["hit_b"] is False
    assert all(r["column"] == "base" and r["overlay"] is None for r in records)

    report = (out_dir / "report.md").read_text()
    assert "Probes measured: 2" in report
    assert "Probes with both answers (`both`): 1" in report
    assert "probe_a" in report

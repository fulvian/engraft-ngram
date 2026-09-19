"""Tests for engraft.engine_check: real-engine verification of usage-corpus
fragments. No real engine, no GGUF: a FAKE `LensClient` (`FakeLensClient`
below) writes the raw files `engraft.lens.logits`/`run_job_all` expect and
produces deterministic seeded logits, pushed toward the target token only
when the job carries an overlay (simulates the graft that brings the answer
to the top).

uv run pytest tests/test_engine_check.py -q
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import engraft.engine_check as E
from engraft.eval import answer_row

EOS = 99


# --------------------------------------------------------------------------
# Fake fragments: 3 families, 1- and 2-token answers
# --------------------------------------------------------------------------


def _frag(id_, family, split, prefix, answer_tokens, fact_ids=None):
    """`tokens = [EOS] + prefix + answer_tokens`, `answer_spans` = indices of
    the last `len(answer_tokens)` tokens -- same shape as the resolved usage
    corpus, see `engraft.eval.answer_row`'s docstring."""
    tokens = [EOS] + list(prefix) + list(answer_tokens)
    n = len(tokens)
    spans = list(range(n - len(answer_tokens), n))
    return {
        "id": id_, "family": family, "split": split, "fact_ids": fact_ids or [f"{id_}_fact"],
        "tokens": tokens, "answer_spans": spans,
    }


def _fragments():
    return [
        _frag("f0", "paraphrase", "heldout", [1, 2, 3], [10]),
        _frag("f1", "paraphrase", "heldout", [1, 2, 4], [11]),
        _frag("f2", "question", "heldout", [5, 6], [12]),
        _frag("f3", "question", "heldout", [5, 7], [13, 14]),  # 2-token answer
        _frag("f4", "cloze", "heldout", [8, 9], [15]),
        _frag("f5", "cloze", "heldout", [8, 20], [16]),
        _frag("f6", "paraphrase", "train", [1, 2, 3], [10]),  # train: excluded by split=heldout
    ]


# --------------------------------------------------------------------------
# (1) generated prefixes match engraft.eval.answer_row
# --------------------------------------------------------------------------


def test_pfirst_job_prefix_matches_answer_row():
    for frag in _fragments():
        row = answer_row(frag)
        job, y = E.pfirst_job(frag, None, "base")
        assert job["tokens"] == frag["tokens"][: row + 1]
        assert y == frag["tokens"][row + 1]


def test_greedy_prefix_matches_pfirst_prefix_and_full_answer():
    frag = next(f for f in _fragments() if f["id"] == "f3")  # 2-token answer
    job, y = E.pfirst_job(frag, None, "base")
    prefix, answer_tokens = E.greedy_prefix_and_answer(frag)
    assert prefix == job["tokens"]
    assert answer_tokens == [13, 14]
    assert answer_tokens[0] == y


def test_load_fragments_filters_by_split(tmp_path):
    corpus_path = tmp_path / "usage_corpus.json"
    corpus_path.write_text(json.dumps({"fragments": _fragments()}))
    heldout = E.load_fragments(str(corpus_path), "heldout")
    assert len(heldout) == 6
    assert all(f["split"] == "heldout" for f in heldout)
    train = E.load_fragments(str(corpus_path), "train")
    assert [f["id"] for f in train] == ["f6"]
    everything = E.load_fragments(str(corpus_path), "all")
    assert len(everything) == 7


def test_load_fragments_drops_fragments_without_answer_spans(tmp_path, capsys):
    frags = _fragments()
    frags.append({"id": "bad0", "family": "paraphrase", "split": "heldout",
                  "tokens": [EOS, 1, 2], "answer_spans": []})
    corpus_path = tmp_path / "usage_corpus.json"
    corpus_path.write_text(json.dumps({"fragments": frags}))
    heldout = E.load_fragments(str(corpus_path), "heldout")
    assert "bad0" not in [f["id"] for f in heldout]
    assert len(heldout) == 6
    assert "bad0" in capsys.readouterr().err


# --------------------------------------------------------------------------
# FakeLensClient: deterministic seeded logits, pushed toward the target only
# with an overlay (simulates the graft) -- pfirst and greedy.
# --------------------------------------------------------------------------


_JOB_ID_RE = re.compile(r"^(pf|gr)_(.+)_(base|student)(?:_greedy(\d+))?$")


def _parse_job_id(job_id: str) -> tuple[str, str, int | None]:
    m = _JOB_ID_RE.match(job_id)
    if not m:
        raise ValueError(f"unexpected job id in the fake client: {job_id!r}")
    _tag, frag_id, kind, step = m.groups()
    return frag_id, kind, (int(step) if step is not None else None)


class FakeLensClient:
    """Writes <raw_dir>/<job_id>/{meta.json,logits.f32} like the real
    engine, so `run_job`/`run_job_all` (unchanged, from `engraft.engine`)
    read them without knowing. `answers_by_frag`: {frag_id: [answer_token_0,
    answer_token_1, ...]} -- the target token of position `step` (0 for
    pfirst jobs) is pushed to the TOP (+12) when the job carries an overlay,
    and deliberately sunk (-6, with another token pushed to the top) when it
    does not -- so base always fails, student always succeeds, by
    construction."""

    def __init__(self, raw_dir: Path, answers_by_frag: dict[str, list[int]], n_vocab: int = 64,
                 seed: int = 2026, overlay_hits: int = 16):
        self.raw_dir = Path(raw_dir)
        self.answers_by_frag = answers_by_frag
        self.n_vocab = n_vocab
        self.seed = seed
        self.overlay_hits = overlay_hits
        self.calls: list[dict] = []

    def run(self, job: dict) -> dict:
        self.calls.append(job)
        job_id = job["id"]
        job_dir = self.raw_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        mode = job["logits"]
        n_pos = 1 if mode == "last" else len(job["tokens"])
        digest = int(hashlib.sha256(job_id.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(self.seed + digest)
        arr = (rng.standard_normal((n_pos, self.n_vocab)) * 0.01).astype(np.float32)

        frag_id, _kind, step = _parse_job_id(job_id)
        answers = self.answers_by_frag.get(frag_id)
        if answers is not None:
            pos = step if step is not None else 0
            y = answers[min(pos, len(answers) - 1)]
            if job.get("overlay") is not None:
                arr[-1, y] += 12.0
            else:
                arr[-1, y] -= 6.0
                arr[-1, (y + 1) % self.n_vocab] += 6.0  # base: another token to the top
        arr.tofile(job_dir / "logits.f32")
        (job_dir / "meta.json").write_text(json.dumps({
            "n_vocab": self.n_vocab, "tokens": job["tokens"], "logits_mode": mode,
        }))
        overlay_hits = self.overlay_hits if job.get("overlay") is not None else 0
        return {"status": "ok", "id": job_id, "overlay_hits": overlay_hits, "t_decode_ms": 0.1}

    def close(self) -> None:
        pass


class _FakeTok:
    """Fake `PleTokenizer`: only `_decode_ids` (via `_tok.decode`) is needed here."""

    class _Inner:
        def decode(self, ids: list[int]) -> str:
            return " ".join(str(i) for i in ids)

    def __init__(self):
        self._tok = self._Inner()


def _answers_by_frag(fragments: list[dict]) -> dict[str, list[int]]:
    out = {}
    for f in fragments:
        _prefix, answer_tokens = E.greedy_prefix_and_answer(f)
        out[f["id"]] = answer_tokens
    return out


def _null_logger():
    import logging
    lg = logging.getLogger("test_engine_check_null")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


# --------------------------------------------------------------------------
# measure_pfirst / measure_greedy with the fake client
# --------------------------------------------------------------------------


def test_measure_pfirst_student_beats_base_with_fake_client(tmp_path):
    frags = _fragments()
    client = FakeLensClient(tmp_path, _answers_by_frag(frags))
    log_ = _null_logger()
    for frag in frags:
        if frag["split"] != "heldout":
            continue
        base = E.measure_pfirst(client, tmp_path, log_, frag, None, "base")
        student = E.measure_pfirst(client, tmp_path, log_, frag, "overlay.pleo", "student")
        assert base["rank_first"] != 1  # sunk on purpose
        assert student["rank_first"] == 1  # pushed to the top on purpose
        assert student["overlay_hits"] == 16
        assert base["overlay_hits"] == 0


def test_measure_greedy_student_exact_match_base_not(tmp_path):
    frag = next(f for f in _fragments() if f["id"] == "f3")  # 2-token answer
    client = FakeLensClient(tmp_path, _answers_by_frag([frag]))
    tok = _FakeTok()
    log_ = _null_logger()
    g_student = E.measure_greedy(client, tmp_path, tok, log_, frag, "overlay.pleo", "student")
    g_base = E.measure_greedy(client, tmp_path, tok, log_, frag, None, "base")
    assert g_student["exact_match"] is True
    assert g_student["tokens"] == [13, 14]
    assert g_base["exact_match"] is False


# --------------------------------------------------------------------------
# (2) --dry-run: jobs.json with 2 jobs per fragment, never touches the engine
# --------------------------------------------------------------------------


def test_dry_run_writes_two_jobs_per_fragment_and_never_touches_the_engine(tmp_path, monkeypatch):
    frags = _fragments()
    corpus_path = tmp_path / "usage_corpus.json"
    corpus_path.write_text(json.dumps({"fragments": frags}))
    overlay_path = tmp_path / "merged.pleo"
    overlay_path.write_text("fake")
    out_dir = tmp_path / "out"

    def _boom(*a, **k):
        raise AssertionError("--dry-run must not build a LensClient")

    monkeypatch.setattr(E, "LensClient", _boom)

    rc = E.main([
        "--usage-corpus", str(corpus_path), "--overlay", str(overlay_path), "--split", "heldout",
        "--out", str(out_dir), "--dry-run", "--greedy-n", "3",
    ])
    assert rc == 0
    jobs = json.loads((out_dir / "jobs.json").read_text())
    assert jobs["n_pfirst_jobs"] == 2 * 6  # 6 heldout fragments (f6 is train, excluded)
    assert len(jobs["pfirst_jobs"]) == jobs["n_pfirst_jobs"]
    ids = [j["_fragment_id"] for j in jobs["pfirst_jobs"]]
    assert ids.count("f0") == 2 and ids.count("f6") == 0  # train excluded from the count
    kinds = {j["id"].rsplit("_", 1)[-1] for j in jobs["pfirst_jobs"] if j["_fragment_id"] == "f0"}
    assert kinds == {"base", "student"}


# --------------------------------------------------------------------------
# (3) per-family report sums to the total; expected engine/replica agreement
# --------------------------------------------------------------------------


def _records_for_report() -> list[dict]:
    """6 fragments (3 families x 2), `replica` given for all: expected exact
    agreement 4/6 (built by hand: f0/f2/f4 agree -- both rank 1; f1/f3/f5
    disagree, except f1 where both agree at rank>1, to reach 4/6 agreeing)."""
    def rec(id_, fam, student_rank, replica_rank):
        return {
            "id": id_, "family": fam, "fact_ids": [f"{id_}_fact"],
            "base": {"p_first": 0.001, "rank_first": 500, "argmax": 0, "overlay_hits": 0},
            "student": {"p_first": 0.9 if student_rank == 1 else 0.01, "rank_first": student_rank,
                        "argmax": 1, "overlay_hits": 16},
            "replica": {"p_first": 0.8 if replica_rank == 1 else 0.02, "rank_first": replica_rank},
            "delta_p_first": abs((0.9 if student_rank == 1 else 0.01) - (0.8 if replica_rank == 1 else 0.02)),
        }
    return [
        rec("f0", "paraphrase", 1, 1),   # agree (both correct)
        rec("f1", "paraphrase", 5, 7),   # agree (both incorrect)
        rec("f2", "question", 1, 1),     # agree
        rec("f3", "question", 1, 3),     # disagree (engine yes, replica no)
        rec("f4", "cloze", 1, 1),        # agree
        rec("f5", "cloze", 2, 1),        # disagree (engine no, replica yes)
    ]


def test_report_per_family_sums_to_total_and_agreement_is_exact():
    records = _records_for_report()
    report = E.build_report(records)
    assert "| paraphrase | 2 |" in report
    assert "| question | 2 |" in report
    assert "| cloze | 2 |" in report
    assert "| **total** | 6 |" in report
    assert "agreement: 4/6 = 0.667" in report


def test_report_lists_greedy_exact_match_when_present():
    records = _records_for_report()
    records[0]["greedy"] = {
        "base": {"tokens": [0], "text": "0", "degenerate": False, "exact_match": False, "answer_text": "10"},
        "student": {"tokens": [10], "text": "10", "degenerate": False, "exact_match": True, "answer_text": "10"},
    }
    report = E.build_report(records)
    assert "## Greedy" in report
    assert "| paraphrase | 1 | 0.00 | 1.00 |" in report


def test_report_without_replica_declares_agreement_not_computed():
    records = [{
        "id": "f0", "family": "paraphrase", "fact_ids": [],
        "base": {"p_first": 0.01, "rank_first": 10, "argmax": 0, "overlay_hits": 0},
        "student": {"p_first": 0.9, "rank_first": 1, "argmax": 1, "overlay_hits": 16},
    }]
    report = E.build_report(records)
    assert "not computed" in report


# --------------------------------------------------------------------------
# (4) --render-only reproduces the same report.md from saved results.json
# --------------------------------------------------------------------------


def test_render_only_reproduces_report_from_saved_results(tmp_path):
    records = _records_for_report()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "results.json").write_text(json.dumps(records))
    direct_report = E.build_report(records)

    rc = E.main(["--out", str(out_dir), "--render-only"])
    assert rc == 0
    rendered = (out_dir / "report.md").read_text()
    assert rendered == direct_report


def test_render_only_fails_clearly_without_results_json(tmp_path, capsys):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = E.main(["--out", str(out_dir), "--render-only"])
    assert rc == 2


# --------------------------------------------------------------------------
# select_greedy_fragments: family-balanced, fixed seed -> deterministic
# --------------------------------------------------------------------------


def test_select_greedy_fragments_is_balanced_and_deterministic():
    frags = [f for f in _fragments() if f["split"] == "heldout"]
    picked1 = E.select_greedy_fragments(frags, 3)
    picked2 = E.select_greedy_fragments(frags, 3)
    assert [f["id"] for f in picked1] == [f["id"] for f in picked2]  # same seed -> same choice
    families_picked = {f["family"] for f in picked1}
    assert len(families_picked) == 3  # one per family, round-robin


def test_select_greedy_fragments_caps_at_available_fragments():
    frags = [f for f in _fragments() if f["split"] == "heldout"]
    picked = E.select_greedy_fragments(frags, 1000)
    assert len(picked) == len(frags)


# --------------------------------------------------------------------------
# (5) base cell: without --overlay only the base column
# --------------------------------------------------------------------------


def test_dry_run_without_overlay_enumerates_only_base_jobs(tmp_path, monkeypatch):
    frags = _fragments()
    corpus_path = tmp_path / "usage_corpus.json"
    corpus_path.write_text(json.dumps({"fragments": frags}))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(E, "LensClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no engine")))
    rc = E.main(["--usage-corpus", str(corpus_path), "--split", "heldout", "--out", str(out_dir),
                 "--dry-run", "--greedy-n", "3"])
    assert rc == 0
    jobs = json.loads((out_dir / "jobs.json").read_text())
    assert jobs["n_pfirst_jobs"] == 6  # one job (base) per heldout fragment
    assert {j["id"].rsplit("_", 1)[-1] for j in jobs["pfirst_jobs"]} == {"base"}
    assert all(j["overlay"] is None for j in jobs["pfirst_jobs"])
    assert {d["kind"] for d in jobs["greedy_descriptors"]} == {"base"}
    assert jobs["n_greedy_descriptors"] == 3


def test_run_check_without_overlay_records_base_only_and_report_renders(tmp_path):
    frags = [f for f in _fragments() if f["split"] == "heldout"]
    client = FakeLensClient(tmp_path, _answers_by_frag(frags))
    records = E.run_check(client, tmp_path, _FakeTok(), _null_logger(), frags, None, frags[:2], {})
    assert len(records) == len(frags)
    assert all("student" not in r and "replica" not in r for r in records)
    assert all(r["base"]["overlay_hits"] == 0 for r in records)
    assert sum(1 for r in records if "greedy" in r) == 2
    assert all(set(r["greedy"]) == {"base"} for r in records if "greedy" in r)
    report = E.build_report(records)
    assert f"| **total** | {len(frags)} | 0.00 | - | - |" in report
    assert "exact match base | exact match student" in report
    assert "| 0.00 | - |" in report

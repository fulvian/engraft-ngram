"""Tests for engraft.check: the F32-phase prefix overlay for chained grafts, the
`_measure_docs` per-position `positions` list, paraphrases in `build_report`, and
`--render-only`. The engine is replaced by stubs (fake `run_job`/`run_job_all`): no
`LensClient` or `engraft.testing.fake_lens` needed to isolate `run_f32_phase` /
`_measure_docs` logic on their own.

uv run pytest tests/test_check.py
"""
from __future__ import annotations

import json
import logging
import types
from pathlib import Path

import numpy as np
import pytest

import engraft.check as m
from engraft.lens import read_pleo, write_pleo, write_plert1
from engraft.table import ROW_LEN

LOG = logging.getLogger("test_check")


class _StubClient:
    def close(self):
        pass


def _row(n_vocab: int, y: int) -> np.ndarray:
    row = np.zeros(n_vocab, dtype=np.float64)
    row[y] = 5.0
    return row


EXPECTED_LOGP_Y = float(m.logsoftmax64(_row(64, 9))[9])  # what the stub run_job always produces


def _make_fixture(tmp_path, n_answer=2):
    """A fact `f1` with `n_answer` already-closed positions: ckpt_f1_<i>_final.{pleo,plert1}
    for every i, descend_f1_<i>.jsonl with one step 0, f1.json with the graft list
    (position, final_p_free)."""
    results_dir = tmp_path / "results"
    fdir = results_dir / "facts" / "f1"
    fdir.mkdir(parents=True)

    grafts = []
    rows_all = []
    for i in range(n_answer):
        rows_i = np.array([1000 + i], dtype=np.int32)  # one T8 row per position, all different
        data_i = np.full((1, ROW_LEN), float(i + 1), dtype=np.float32)
        write_pleo(fdir / f"ckpt_f1_{i}_final.pleo", rows_i, data_i)
        write_plert1(fdir / f"ckpt_f1_{i}_final.plert1", {0: np.zeros((1, 1), dtype=np.int32)})
        (fdir / f"descend_f1_{i}.jsonl").write_text(json.dumps({"step": 0, "logp_y": EXPECTED_LOGP_Y}) + "\n")
        grafts.append({"position": i, "final_p_free": 0.9})
        rows_all.append((rows_i, data_i))

    fact_rows, fact_data = m.RowSet.build_overlay(rows_all)
    write_pleo(fdir / "f1.pleo", fact_rows, fact_data)
    (fdir / "f1.json").write_text(json.dumps({"id": "f1", "grafts": grafts}))

    facts_resolved = {"f1": {"trigger_tokens": [1, 2, 3], "answer_tokens": [9] * n_answer}}
    return results_dir, facts_resolved


def _run_phase(monkeypatch, results_dir, facts_resolved, tmp_path, captured_jobs, target_token_map=None):
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(m, "LensClient", lambda *a, **k: _StubClient())

    def fake_run_job(client, raw_dir, job, log):
        captured_jobs.append(job)
        y = 9  # fixed target in the fake engine
        row = _row(64, y)
        if job.get("routing_record"):
            write_plert1(Path(job["routing_record"]), {0: np.zeros((1, 1), dtype=np.int32)})
        return {}, row, {}

    monkeypatch.setattr(m, "run_job", fake_run_job)

    args = types.SimpleNamespace(lens_cmd="ignored", target_token_map=target_token_map)
    return m.run_f32_phase(args, run_dir, raw_dir, LOG, facts_resolved, results_dir), run_dir


# --------------------------------------------------------------------------
# 1) i>=1: the base job uses the prefix overlay (union of the ckpts < i), written
#    under run_dir/facts/<fid>/, never under results_dir (read-only in a dry run)
# --------------------------------------------------------------------------


def test_base_job_uses_prefix_overlay_for_i_ge_1_and_none_for_i_eq_0(tmp_path, monkeypatch):
    results_dir, facts_resolved = _make_fixture(tmp_path, n_answer=2)
    captured_jobs: list[dict] = []
    out, run_dir = _run_phase(monkeypatch, results_dir, facts_resolved, tmp_path, captured_jobs)

    base0 = next(j for j in captured_jobs if j["id"] == "f32_f1_0_base")
    base1 = next(j for j in captured_jobs if j["id"] == "f32_f1_1_base")

    assert base0["overlay"] is None

    expected_prefix_path = run_dir / "facts" / "f1" / "f1_prefix1.pleo"
    assert base1["overlay"] == str(expected_prefix_path)
    assert expected_prefix_path.exists()
    # only files already present in the fixture (ckpt/pleo/plert1/json/jsonl) live
    # under results_dir (read-only): no prefix* file there.
    assert not any((results_dir / "facts" / "f1").glob("f1_prefix*.pleo"))

    # the prefix overlay's content is the union of the ckpts < i (here only i=0)
    rows_written, data_written = read_pleo(expected_prefix_path)
    rows_ckpt0, data_ckpt0 = read_pleo(results_dir / "facts" / "f1" / "ckpt_f1_0_final.pleo")
    assert set(rows_written.tolist()) == set(rows_ckpt0.tolist())
    for r, v in zip(rows_written.tolist(), data_written):
        idx = list(rows_ckpt0).index(r)
        assert np.array_equal(v, data_ckpt0[idx])

    assert out["grafts"]["f1_0"]["consistency_pass"] is True  # diff 0.0 (deterministic stub)
    assert out["grafts"]["f1_1"]["consistency_pass"] is True


def test_consistency_check_still_raises_when_diff_is_large(tmp_path, monkeypatch):
    """Non-regression check: a large diff must still raise (the consistency check
    stays alive for ordinary grafts)."""
    results_dir, facts_resolved = _make_fixture(tmp_path, n_answer=1)
    (results_dir / "facts" / "f1" / "descend_f1_0.jsonl").write_text(
        json.dumps({"step": 0, "logp_y": -50.0}) + "\n"
    )
    captured_jobs: list[dict] = []
    with pytest.raises(RuntimeError, match="consistency check failed"):
        _run_phase(monkeypatch, results_dir, facts_resolved, tmp_path, captured_jobs)


def test_target_token_map_mode_does_not_raise(tmp_path, monkeypatch):
    """Dry-run switch: with --target-token-map the consistency check is recorded but
    does not raise (existing behavior, unchanged)."""
    results_dir, facts_resolved = _make_fixture(tmp_path, n_answer=1)
    (results_dir / "facts" / "f1" / "descend_f1_0.jsonl").write_text(
        json.dumps({"step": 0, "logp_y": -50.0}) + "\n"
    )
    ttmap_path = tmp_path / "target_token_map.json"
    ttmap_path.write_text(json.dumps({"f1": {"0": 9}}))
    captured_jobs: list[dict] = []
    out, _run_dir = _run_phase(
        monkeypatch, results_dir, facts_resolved, tmp_path, captured_jobs,
        target_token_map=str(ttmap_path),
    )
    assert out["grafts"]["f1_0"]["consistency_pass"] is False


# --------------------------------------------------------------------------
# _measure_docs: "positions" list per document position
# --------------------------------------------------------------------------


class _StubTok:
    """`.encode` ignores the document's actual text (deterministic, independent of
    the real docs/it.txt content): 5 tokens -> 4 target positions."""

    def encode(self, text):
        return [1, 2, 3, 4, 5]

    def decode_token(self, token_id):
        return f"<{token_id}>"


def test_measure_docs_positions_list_flags_response_and_others(tmp_path, monkeypatch):
    facts_resolved = {
        "fA": {"doc_id": "it", "doc_positions": {"positions": [{"i": 0, "t_pred": 2}]}},
    }

    def fake_run_job_all(client, raw_dir, job, log):
        n_vocab = 8
        targets = [2, 3, 4, 5]  # doc_tokens[1:] for _StubTok.encode -> [1,2,3,4,5]
        all_logits = np.zeros((len(targets), n_vocab), dtype=np.float64)
        for p, y in enumerate(targets):
            all_logits[p, y] = 5.0
        return {"overlay_hits": 7}, all_logits, {"n_vocab": n_vocab}

    monkeypatch.setattr(m, "run_job_all", fake_run_job_all)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    out = m._measure_docs(
        client=None, raw_dir=raw_dir, tok=_StubTok(), log=LOG, args=None,
        merged_pleo=tmp_path / "merged.pleo", facts_resolved=facts_resolved,
    )

    it = out["it"]
    positions = it["positions"]
    assert len(positions) == 4  # 4 target positions

    response = [p for p in positions if p["is_response"]]
    other = [p for p in positions if not p["is_response"]]
    assert len(response) == 1
    assert len(other) == 3

    r = response[0]
    assert r["t_pred"] == 2
    assert r["fid"] == "fA"
    assert r["answer_position"] == 0
    assert r["target_token"] == 4  # targets[2] = 4
    assert r["target_str"] == "<4>"
    assert r["logp_base"] == r["logp_merged"]  # same fake engine for base and merged
    assert r["delta"] == pytest.approx(0.0)

    for p in other:
        assert "fid" not in p
        assert "answer_position" not in p
        assert "target_str" in p  # tok.decode_token available at every position

    # aggregate statistics unchanged (one list per group, not altered)
    assert it["response_positions"]["n"] == 1
    assert it["other_positions"]["n"] == 3
    assert it["overlay_hits"] == 7


# --------------------------------------------------------------------------
# build_report: paraphrases under Q2/Q4, --render-only
# --------------------------------------------------------------------------


def test_build_report_emits_paraphrases_under_q2_q4():
    q8 = {
        "merged": {
            "facts": {
                "f1": {
                    "p_first": 0.9, "rank_first": 1, "answer_reproduced": True, "hits": 16,
                    "paraphrases": {
                        "paraphrase_same_tail": {
                            "p_first": 0.8, "rank_first": 1, "argmax": 5,
                            "argmax_unchanged_vs_base": True, "delta_logp_argmax_base": -0.01,
                            "hits": 16,
                        },
                        "paraphrase_other_tail": {
                            "p_first": 0.001, "rank_first": 40, "argmax": 3,
                            "argmax_unchanged_vs_base": True, "delta_logp_argmax_base": -0.0002,
                            "hits": 0,
                        },
                    },
                },
            },
            "corpus": {}, "docs": {},
        },
        "facts": {},
    }
    report = m.build_report(q8, {"grafts": {}})
    assert "## Q2/Q4 (merged overlay, corpus of all facts)" in report
    idx_q2 = report.index("## Q2/Q4")
    idx_corpus = report.index("\ncorpus:")
    section = report[idx_q2:idx_corpus]
    assert "paraphrase_same_tail" in section
    assert "paraphrase_other_tail" in section
    assert "argmax_unchanged_vs_base=True" in section
    assert "delta_logp_argmax_base=-0.01" in section
    assert "rank_first=40" in section


def test_render_only_regenerates_report_from_engine_check_json_without_lens_cmd(tmp_path):
    run_dir = tmp_path / "dryrun"
    run_dir.mkdir()
    engine_check = {
        "q8": {
            "facts": {}, "sisters_base": {},
            "merged": {
                "facts": {
                    "f1": {
                        "p_first": 0.9, "rank_first": 1, "answer_reproduced": True, "hits": 16,
                        "paraphrases": {
                            "paraphrase_same_tail": {
                                "p_first": 0.8, "rank_first": 1,
                                "argmax_unchanged_vs_base": True, "delta_logp_argmax_base": -0.01,
                            },
                        },
                    },
                },
                "corpus": {}, "docs": {},
            },
        },
        "f32": {"grafts": {}},
    }
    (run_dir / "engine_check.json").write_text(json.dumps(engine_check))
    old_report = run_dir / "report.md"
    old_report.write_text("old, to overwrite")

    rc = m.main(["dryrun", "--out-root", str(tmp_path), "--render-only"])
    assert rc == 0
    new_report = old_report.read_text()
    assert "paraphrase_same_tail" in new_report
    assert new_report != "old, to overwrite"


def test_render_only_fails_cleanly_without_engine_check_json(tmp_path):
    run_dir = tmp_path / "dryrun"
    run_dir.mkdir()
    rc = m.main(["dryrun", "--out-root", str(tmp_path), "--render-only"])
    assert rc == 2

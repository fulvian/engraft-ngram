"""Tests for engraft.eval: the replica-side base/student/teacher evaluation,
on the CPU fake path.

uv run pytest tests/test_eval.py -q
"""
from __future__ import annotations

import json

import pytest
import torch

import engraft.eval as EV
import engraft.replica.distill as D
from engraft.lens import RowSet
from engraft.replica.backend import Backend
from engraft.replica.model import Replica
from engraft.table import ROW_LEN
from engraft.testing.fake_full_weights import FakeFullWeights, tiny_hparams
from engraft.testing.fake_table import FakeTable

torch.set_num_threads(1)

EOS = 19


def _fresh_replica(seed: int = 31):
    hp = tiny_hparams()
    w = FakeFullWeights(hp)
    table = FakeTable(seed=seed)
    return Replica(hp, w, table, backend=Backend.cpu_f32()), table


def _fragments():
    return [
        {"id": "f0", "split": "heldout", "family": "statement", "fact_ids": ["fact_a"],
         "tokens": [EOS, 1, 2, 3], "answer_spans": [3]},
        {"id": "f1", "split": "heldout", "family": "question", "fact_ids": ["fact_b"],
         "tokens": [EOS, 4, 5], "answer_spans": [2]},
    ]


# --------------------------------------------------------------------------
# band_of / answer_row
# --------------------------------------------------------------------------


def test_band_of_matches_declared_thresholds():
    assert EV.band_of(0) == "0"
    assert EV.band_of(1) == "1-8"
    assert EV.band_of(8) == "1-8"
    assert EV.band_of(9) == "9-24"
    assert EV.band_of(24) == "9-24"
    assert EV.band_of(25) == ">24"
    assert EV.band_of(500) == ">24"


def test_answer_row_is_min_span_minus_one():
    frag = {"id": "f", "answer_spans": [5, 3, 4]}
    assert EV.answer_row(frag) == 2


def test_answer_row_rejects_missing_spans():
    with pytest.raises(ValueError):
        EV.answer_row({"id": "f", "answer_spans": []})


# --------------------------------------------------------------------------
# three columns: student == base with an empty overlay; teacher != base with a document
# --------------------------------------------------------------------------


def test_evaluate_fragment_student_equals_base_when_overlay_empty():
    replica, table = _fresh_replica()
    frag = _fragments()[0]
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)

    rec = EV.evaluate_fragment(
        replica, None, table, frag, [], EOS, empty_row_map, empty_rows, D.fake_seq_forward, k=6,
    )
    assert rec["p_first"]["student"] == rec["p_first"]["base"]
    assert rec["rank_first"]["student"] == rec["rank_first"]["base"]
    # no document: teacher == base by construction (no prefix extension)
    assert rec["p_first"]["teacher"] == rec["p_first"]["base"]
    assert abs(rec["kd_student_teacher"]) < 1e-6


def test_evaluate_fragment_teacher_differs_from_base_with_document():
    replica, table = _fresh_replica(seed=32)
    frag = _fragments()[0]
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)
    doc_tokens = [10, 11, 12]

    rec = EV.evaluate_fragment(
        replica, None, table, frag, doc_tokens, EOS, empty_row_map, empty_rows, D.fake_seq_forward, k=6,
    )
    assert rec["p_first"]["teacher"] != rec["p_first"]["base"], (
        "the document must change the teacher's logits relative to the base (different context)"
    )


def test_evaluate_fragment_student_differs_from_base_with_nonempty_overlay():
    replica, table = _fresh_replica(seed=33)
    frag = _fragments()[1]

    rs = RowSet.from_position(table, frag["tokens"], EV.answer_row(frag))
    g0 = int(rs.rows_global[0])
    row_map = {g0: 0}
    rows_var = torch.from_numpy(rs.data[0:1].copy() + 5.0)  # shift a read row by a lot

    rec = EV.evaluate_fragment(
        replica, None, table, frag, [], EOS, row_map, rows_var, D.fake_seq_forward, k=6,
    )
    assert rec["p_first"]["student"] != rec["p_first"]["base"]


def test_evaluate_fragment_skip_base_and_skip_teacher_leave_none():
    replica, table = _fresh_replica(seed=35)
    frag = _fragments()[0]
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)

    rec = EV.evaluate_fragment(
        replica, None, table, frag, [10, 11], EOS, empty_row_map, empty_rows, D.fake_seq_forward, k=6,
        skip_base=True, skip_teacher=True,
    )
    assert rec["p_first"]["base"] is None
    assert rec["rank_first"]["base"] is None
    assert rec["correct"]["base"] is None
    assert rec["p_first"]["teacher"] is None
    assert rec["kd_student_teacher"] is None
    assert rec["p_first"]["student"] is not None


# --------------------------------------------------------------------------
# evaluate_corpus: per-family/per-band aggregates, eval.json/eval.md written
# --------------------------------------------------------------------------


def test_evaluate_corpus_aggregates_and_writes_report(tmp_path):
    replica, table = _fresh_replica(seed=34)
    fragments = _fragments()
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)
    overlap_by_fact_id = {"fact_a": 0, "fact_b": 30}  # bands "0" and ">24"

    result = EV.evaluate_corpus(
        replica, None, table, fragments, [], EOS, empty_row_map, empty_rows, D.fake_seq_forward, k=6,
        overlap_by_fact_id=overlap_by_fact_id,
    )
    assert result["n_fragments"] == 2
    assert set(result["by_family"].keys()) == {"statement", "question"}
    assert set(result["by_overlap_band"].keys()) == {"0", ">24"}
    assert result["by_overlap_band"]["0"]["n"] == 1
    assert result["by_overlap_band"][">24"]["n"] == 1

    out_json = tmp_path / "eval.json"
    out_md = tmp_path / "eval.md"
    EV.write_eval_report(result, out_json, out_md)
    assert out_json.exists()
    assert out_md.exists()
    loaded = json.loads(out_json.read_text())
    assert loaded["n_fragments"] == 2
    md_text = out_md.read_text()
    assert "By family" in md_text
    assert "By overlap band" in md_text


def test_evaluate_corpus_census_overrides_overlap_by_fact_id():
    replica, table = _fresh_replica(seed=36)
    fragments = _fragments()
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)
    census = {"per_fact_overlap": {"fact_a": {"Hc_question": {"hit_all_B": 3}}}}

    result = EV.evaluate_corpus(
        replica, None, table, fragments, [], EOS, empty_row_map, empty_rows, D.fake_seq_forward, k=6,
        overlap_by_fact_id={"fact_a": 99}, census=census,
    )
    frag_a = next(r for r in result["fragments"] if r["id"] == "f0")
    assert frag_a["overlap_count"] == 3  # from census, not overlap_by_fact_id=99


# --------------------------------------------------------------------------
# --ple-gate: additive, off-by-default behavior unchanged
# --------------------------------------------------------------------------


def test_ple_gate_off_produces_no_ple_gate_key():
    replica, table = _fresh_replica(seed=37)
    frag = _fragments()[0]
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)

    rec = EV.evaluate_fragment(
        replica, None, table, frag, [], EOS, empty_row_map, empty_rows, D.fake_seq_forward, k=6,
    )
    assert "ple_gate" not in rec


def test_ple_gate_on_adds_key_and_fills_arrays():
    replica, table = _fresh_replica(seed=38)
    frag = _fragments()[0]
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, ROW_LEN, dtype=torch.float32)
    ple_arrays: dict = {}

    rec = EV.evaluate_fragment(
        replica, None, table, frag, [], EOS, empty_row_map, empty_rows, D.fake_seq_forward, k=6,
        ple_gate=True, ple_arrays=ple_arrays,
    )
    assert "ple_gate" in rec
    assert rec["ple_gate"]["answer_row"] == EV.answer_row(frag)
    assert "base" in rec["ple_gate"]
    assert "student" in rec["ple_gate"]
    assert f"{frag['id']}/base/gate" in ple_arrays
    assert f"{frag['id']}/student/gate" in ple_arrays


def test_window_rows_and_overlay_hits_offline():
    replica, table = _fresh_replica(seed=39)
    win = EV.window_rows(replica, 5)
    n = replica.hp.ple_ngram_size
    kern = replica.hp.ple_conv_kernel
    assert win == [5 - i * n for i in range(kern) if 5 - i * n >= 0]

    frag = _fragments()[0]
    tokens = frag["tokens"]
    rs = RowSet.from_position(table, tokens, 2)
    row_map = {int(g): 0 for g in rs.rows_global[:1]}
    hits = EV.overlay_hits(table, tokens, [2], row_map)
    assert hits == 1

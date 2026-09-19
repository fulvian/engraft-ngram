"""Tests for `engraft.descend_corpus` and `engraft.teacher`: the corpus-level
descent CLI, end to end on the CPU `--fake` path (weak-form reproducibility,
see `docs/formats.md`).

uv run pytest tests/test_descend_corpus.py
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import engraft.descend_corpus as DC
import engraft.replica.distill as D
import engraft.teacher as T
from engraft.lens import read_pleo
from engraft.testing.fake_full_weights import tiny_hparams


def _eos() -> int:
    return tiny_hparams().n_vocab - 1


def _fragment(fid: str, fact: str, toks: list[int], answer_idx: int, split: str) -> dict:
    return {
        "id": fid, "fact_ids": [fact], "tokens": [_eos()] + toks,
        "answer_spans": [answer_idx], "split": split,
    }


def _synthetic_fragments() -> list[dict]:
    return [
        _fragment("f0", "fact_a", [3, 5, 7, 2], 3, "train"),
        _fragment("f1", "fact_a", [3, 5, 9, 2], 3, "train"),
        _fragment("f2", "fact_a", [3, 5, 4, 2], 3, "test"),
        _fragment("f3", "fact_b", [8, 1, 6], 2, "train"),
        _fragment("f4", "fact_b", [8, 1, 10, 6], 3, "train"),
        _fragment("f5", "fact_b", [8, 1, 11, 6], 3, "test"),
    ]


@pytest.fixture
def corpus_dir(tmp_path):
    fragments = _synthetic_fragments()
    path = tmp_path / "usage_corpus_resolved.json"
    path.write_text(json.dumps({"fragments": fragments}))
    return tmp_path, path


def test_compute_fact_weights_matches_mass_ratio():
    fragments = _synthetic_fragments()
    train = [f for f in fragments if f["split"] == "train"]
    weights, mass, stats = DC.compute_fact_weights(train, n_excl=1)
    assert set(weights) == {"fact_a", "fact_b"}
    assert stats["n_facts"] == 2
    assert stats["mass_median"] == pytest.approx((mass["fact_a"] + mass["fact_b"]) / 2.0)
    for fid, m in mass.items():
        assert weights[fid] == pytest.approx(stats["mass_median"] / m)


def test_compute_fact_weights_rejects_multi_fact_fragment():
    bad = [{"id": "x", "fact_ids": ["a", "b"], "tokens": [_eos(), 1, 2]}]
    with pytest.raises(ValueError):
        DC.compute_fact_weights(bad, n_excl=1)


def test_resolve_routing_regime_accepts_misto_alias():
    assert DC.resolve_routing_regime("locked") == "locked"
    assert DC.resolve_routing_regime("mixed") == "mixed"
    assert DC.resolve_routing_regime("misto") == "mixed"
    assert DC.resolve_routing_regime("bloccato") == "locked"
    with pytest.raises(ValueError):
        DC.resolve_routing_regime("nonsense")


def test_teacher_cli_produces_targets_and_routing_base(corpus_dir):
    tmp_path, corpus_path = corpus_dir
    targets_path = tmp_path / "targets.npz"
    routing_path = tmp_path / "routing.npz"
    rc = T.main([
        "--usage-corpus", str(corpus_path), "--out", str(targets_path),
        "--routing-out", str(routing_path), "--k", "10", "--sample-full", "0", "--fake",
    ])
    assert rc == 0
    assert targets_path.exists()
    assert routing_path.exists()

    targets = D.TeacherTargets.load(targets_path)
    assert targets.ids.shape[1] == 10
    by_frag, layers, cfg = D.load_routing_base(routing_path)
    assert layers.shape[0] == tiny_hparams().n_layer
    assert cfg["dense_dtype"] == "f32"


def test_teacher_rejects_routing_out_with_doc_tokens(corpus_dir, tmp_path):
    tmp_path2, corpus_path = corpus_dir
    doc_path = tmp_path2 / "doc.json"
    doc_path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SystemExit):
        T.main([
            "--usage-corpus", str(corpus_path), "--out", str(tmp_path2 / "t.npz"),
            "--doc-tokens", str(doc_path), "--routing-out", str(tmp_path2 / "r.npz"),
            "--k", "10", "--fake",
        ])


@pytest.fixture
def targets_and_routing(corpus_dir):
    tmp_path, corpus_path = corpus_dir
    targets_path = tmp_path / "targets.npz"
    routing_path = tmp_path / "routing.npz"
    rc = T.main([
        "--usage-corpus", str(corpus_path), "--out", str(targets_path),
        "--routing-out", str(routing_path), "--k", "10", "--sample-full", "0", "--fake",
    ])
    assert rc == 0
    return tmp_path, corpus_path, targets_path, routing_path


def test_descend_corpus_cli_fake_locked_writes_pleo(targets_and_routing):
    tmp_path, corpus_path, targets_path, _routing_path = targets_and_routing
    out_dir = tmp_path / "run_locked"
    rc = DC.main([
        "--usage-corpus", str(corpus_path), "--targets", str(targets_path), "--out", str(out_dir),
        "--policy", "all-read", "--arm", "kd", "--k", "10", "--pack-len", "64", "--n-excl", "1",
        "--max-steps", "5", "--fake",
    ])
    assert rc == 0
    manifest = json.loads((out_dir / "merged_manifest.json").read_text())
    assert manifest["stop_reason"] == "max_steps"
    assert manifest["routing_regime"] == "locked"
    rows, data = read_pleo(out_dir / "merged.pleo")
    assert rows.shape[0] == data.shape[0] > 0


def test_descend_corpus_cli_fake_mass_weight_and_mixed_regime_writes_pleo(targets_and_routing):
    """The corpus-level CLI, capture -> descent with `--fact-weight mass` and
    `--routing-regime mixed` -> stop -> `.pleo`, entirely on CPU synthetic
    data (weak-form C1 reproducibility, run through the CLI itself)."""
    tmp_path, corpus_path, targets_path, routing_path = targets_and_routing
    out_dir = tmp_path / "run_mixed"
    rc = DC.main([
        "--usage-corpus", str(corpus_path), "--targets", str(targets_path), "--out", str(out_dir),
        "--policy", "all-read", "--arm", "kd", "--k", "10", "--pack-len", "64", "--n-excl", "1",
        "--fact-weight", "mass", "--routing-base", str(routing_path), "--routing-regime", "mixed",
        "--stop-criterion", "acc_heldout_rate", "--eval-every-free", "1", "--max-steps", "8", "--fake",
    ])
    assert rc == 0
    manifest = json.loads((out_dir / "merged_manifest.json").read_text())
    assert manifest["routing_regime"] == "mixed"
    assert manifest["fact_weight"] == "mass"
    assert manifest["fact_weight_stats"]["n_facts"] == 2
    assert manifest["phases"][0]["state"] == "LOCKED"
    rows, data = read_pleo(out_dir / "merged.pleo")
    assert rows.shape[0] == data.shape[0] > 0
    # every variable row's global id round-trips through the manifest's row_map
    summary = json.loads((out_dir / "summary.json").read_text())
    assert set(int(g) for g in summary["row_map"]) <= set(int(r) for r in rows)


def test_descend_corpus_cli_fake_census_overrides_recompute(targets_and_routing):
    """`--census` short-circuits `build_all_read_row_set`: an obviously-wrong
    row set from the census file must show up verbatim in the run's output,
    proving the CLI used it instead of recomputing (docs/formats.md)."""
    tmp_path, corpus_path, targets_path, _routing_path = targets_and_routing
    census_path = tmp_path / "census.json"
    fake_row_sets = {"all_read": [0, 1, 2], "entity": [0]}
    census_path.write_text(json.dumps({"schema": "engraft-census/v1", "row_sets": fake_row_sets}))

    out_dir = tmp_path / "run_census"
    rc = DC.main([
        "--usage-corpus", str(corpus_path), "--targets", str(targets_path), "--out", str(out_dir),
        "--policy", "all-read", "--arm", "kd", "--k", "10", "--pack-len", "64", "--n-excl", "1",
        "--census", str(census_path), "--max-steps", "3", "--fake",
    ])
    assert rc == 0
    manifest = json.loads((out_dir / "merged_manifest.json").read_text())
    # policy "all-read" makes the variable-row candidate set == row_sets["all_read"]
    # (engraft.replica.distill.build_rows_var); with the census override that must be
    # exactly the 3 fake ids (minus any excluded separator position), never the far
    # larger set build_all_read_row_set would have recomputed from the synthetic corpus.
    assert manifest["n_rows_variable"] <= len(fake_row_sets["all_read"])
    rows, _data = read_pleo(out_dir / "merged.pleo")
    assert set(int(r) for r in rows) <= set(fake_row_sets["all_read"])


def test_descend_corpus_cli_fake_checkpoint_is_not_supported(targets_and_routing):
    """The `--fake` CPU path never needs whole-layer checkpointing (its graph
    is tiny) and does not support it -- see `docs/formats.md`."""
    tmp_path, corpus_path, targets_path, _routing_path = targets_and_routing
    out_dir = tmp_path / "run_ckpt"
    with pytest.raises(NotImplementedError):
        DC.main([
            "--usage-corpus", str(corpus_path), "--targets", str(targets_path), "--out", str(out_dir),
            "--policy", "all-read", "--arm", "kd", "--k", "10", "--pack-len", "64", "--n-excl", "1",
            "--checkpoint", "--max-steps", "3", "--fake",
        ])

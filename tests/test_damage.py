"""Tests for engraft.damage: the collateral-damage plan/run subcommands, on
the CPU fake path.

uv run pytest tests/test_damage.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import engraft.damage as SD
from engraft.lens import local_to_global
from engraft.testing.fake_table import FakeTable


def _synthetic_corpus(n: int, seed: int, eos: int, vocab: int = 50, every: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=n, dtype=np.int32)
    tokens[every - 1::every] = eos  # one EOS every ~`every` tokens
    return tokens


def _pick_rows(tokens, table):
    rows_big, rows_tri = SD.ngram_rows(tokens, table)
    picks = [(100, 0, rows_big), (150, 3, rows_tri), (300, 7, rows_big), (777, 1, rows_tri), (5000, 5, rows_big)]
    overlay_rows = [int(arr[pos, g]) for pos, g, arr in picks]
    return np.array(sorted(set(overlay_rows)), dtype=np.int64)


# --------------------------------------------------------------------------
# ngram_rows / scan_hits: exact against the scalar reference
# --------------------------------------------------------------------------


def test_scan_hits_exact_against_scalar_ngram_addresses():
    table = FakeTable(seed=7)
    eos = table.eos_token_id
    tokens = _synthetic_corpus(20_000, seed=1, eos=eos)

    overlay_rows = _pick_rows(tokens, table)
    dnorm_fake = np.arange(1, len(overlay_rows) + 1, dtype=np.float32)

    hits = SD.scan_hits(tokens, table, overlay_rows, dnorm_fake, block=4_000)

    overlay_set = set(int(x) for x in overlay_rows.tolist())
    tokens_list = tokens[:2000].tolist()
    addr = table.ngram_addresses(tokens_list)
    expected_hit_pos = set()
    for pos in range(len(tokens_list) - 1):  # last position excluded, like scan_hits
        rows_g = [local_to_global(table, h, addr[pos][h]) for h in range(table.n_heads)]
        if overlay_set & set(rows_g):
            expected_hit_pos.add(pos)

    got_hit_pos = set(int(p) for p in hits["pos"].tolist() if p < 2000)
    assert got_hit_pos == expected_hit_pos

    # no hit on the corpus's last position (no following token)
    assert (len(tokens) - 1) not in set(hits["pos"].tolist())


def test_sample_chunks_hit_centered_at_quarter_offset():
    table = FakeTable(seed=7)
    eos = table.eos_token_id
    tokens = _synthetic_corpus(20_000, seed=1, eos=eos)
    overlay_rows = _pick_rows(tokens, table)
    dnorm_fake = np.arange(1, len(overlay_rows) + 1, dtype=np.float32)  # all CHANGED (>0)
    cos_fake = np.ones(len(overlay_rows), dtype=np.float32)

    hits = SD.scan_hits(tokens, table, overlay_rows, dnorm_fake, block=4_000)
    assert hits["pos"].size > 0

    chunk_len = 256
    chunks = SD.sample_chunks(hits, table, tokens, overlay_rows, dnorm_fake, cos_fake,
                               chunk_len, n_random=4, n_hit=4, seed=0)
    hit_chunks = [c for c in chunks if c["kind"] == "hit"]
    assert len(hit_chunks) > 0
    for c in hit_chunks:
        center = chunk_len // 4
        assert c["start"] == max(0, min(20_000 - chunk_len - 1, c["center_pos"] - center))
        rel = c["center_pos"] - c["start"]
        rel_entries = {hp["pos_rel"] for hp in c["hit_positions"]}
        assert rel in rel_entries


def test_sample_chunks_never_targets_unchanged_rows_and_targets_are_distinct():
    """Half of the overlay's rows are UNCHANGED (dnorm=0, coincide with the
    true row) -- `sample_chunks` must NEVER pick one of these as a hit
    block's target, and no target row may repeat across blocks."""
    table = FakeTable(seed=11)
    eos = table.eos_token_id
    tokens = _synthetic_corpus(20_000, seed=2, eos=eos)
    rows_big, rows_tri = SD.ngram_rows(tokens, table)

    picks = [
        (100, 0, rows_big), (150, 3, rows_tri), (300, 7, rows_big), (777, 1, rows_tri),
        (1200, 2, rows_big), (2200, 4, rows_tri), (3300, 6, rows_big), (4400, 0, rows_tri),
        (5500, 5, rows_big), (6600, 1, rows_tri),
    ]
    overlay_rows = np.array(sorted({int(arr[pos, g]) for pos, g, arr in picks}), dtype=np.int64)
    n = overlay_rows.shape[0]
    dnorm = np.zeros(n, dtype=np.float32)
    dnorm[1::2] = 1.0  # odd indices CHANGED, even indices UNCHANGED (dnorm=0)
    cos = np.ones(n, dtype=np.float32)

    hits = SD.scan_hits(tokens, table, overlay_rows, dnorm, block=4_000)
    changed_rows = set(overlay_rows[dnorm > 0].tolist())

    chunks = SD.sample_chunks(hits, table, tokens, overlay_rows, dnorm, cos,
                               chunk_len=256, n_random=2, n_hit=6, seed=0)
    hit_chunks = [c for c in chunks if c["kind"] == "hit"]
    assert len(hit_chunks) > 0
    targets = [c["target_row"] for c in hit_chunks]
    assert all(t in changed_rows for t in targets)
    assert len(targets) == len(set(targets))


# --------------------------------------------------------------------------
# compute_rows_delta / rows_delta_summary
# --------------------------------------------------------------------------


def test_compute_rows_delta_zero_for_true_rows():
    table = FakeTable(seed=5)
    rows_global = np.array([0, 1, 2], dtype=np.int64)
    from engraft.replica.distill import _read_true_rows

    true = _read_true_rows(table, rows_global.tolist())
    delta = SD.compute_rows_delta(table, rows_global, true)
    assert np.allclose(delta["dnorm"], 0.0, atol=1e-5)
    assert np.allclose(delta["cos"], 1.0, atol=1e-4)

    summary = SD.rows_delta_summary(delta)
    assert summary["n_rows"] == 3
    assert summary["dnorm_quantiles"]["max"] < 1e-4


def test_compute_rows_delta_nonzero_for_shifted_rows():
    table = FakeTable(seed=6)
    rows_global = np.array([0, 1], dtype=np.int64)
    from engraft.replica.distill import _read_true_rows

    true = _read_true_rows(table, rows_global.tolist())
    shifted = true + 5.0
    delta = SD.compute_rows_delta(table, rows_global, shifted)
    assert np.all(delta["dnorm"] > 4.0)


# --------------------------------------------------------------------------
# CLI plan+run, fake end to end
# --------------------------------------------------------------------------


def test_cli_plan_then_run_fake(tmp_path):
    from engraft.testing.fake_full_weights import tiny_hparams

    n_vocab = tiny_hparams().n_vocab
    rng = np.random.default_rng(0)
    neutral = rng.integers(0, n_vocab, size=3000).astype(np.int64)
    neutral_path = tmp_path / "neutral_tokens.npy"
    np.save(neutral_path, neutral)

    from engraft.lens import write_pleo

    table = SD._load_table(fake=True, table_path=None)
    # a small overlay: 4 rows shifted from their true value
    rows_global = np.array([0, 1, 2, 3], dtype=np.int64)
    from engraft.replica.distill import _read_true_rows

    true = _read_true_rows(table, rows_global.tolist())
    overlay_data = true + 2.0
    overlay_path = tmp_path / "overlay.pleo"
    write_pleo(overlay_path, rows_global, overlay_data)

    plan_out = tmp_path / "plan"
    rc = SD.main(["plan", "--overlay", str(overlay_path), "--neutral-tokens", str(neutral_path),
                  "--chunk-len", "32", "--n-random", "2", "--n-hit", "2", "--seed", "0",
                  "--out", str(plan_out), "--fake"])
    assert rc == 0
    assert (plan_out / "chunks.json").exists()
    assert (plan_out / "rows_delta.npz").exists()
    assert (plan_out / "hits.npz").exists()

    run_out = tmp_path / "run"
    rc = SD.main(["run", "--plan", str(plan_out), "--overlay", str(overlay_path),
                  "--neutral-tokens", str(neutral_path), "--out", str(run_out), "--fake"])
    assert rc == 0
    damage = json.loads((run_out / "damage.json").read_text())
    assert damage["n_chunks_done"] > 0
    assert damage["routing_mode"] == "free"
    assert (run_out / "damage.md").exists()


def test_cli_run_rbr_gives_zero_flips_and_declares_locked(tmp_path):
    from engraft.testing.fake_full_weights import tiny_hparams

    n_vocab = tiny_hparams().n_vocab
    rng = np.random.default_rng(1)
    neutral = rng.integers(0, n_vocab, size=3000).astype(np.int64)
    neutral_path = tmp_path / "neutral_tokens.npy"
    np.save(neutral_path, neutral)

    from engraft.lens import write_pleo
    from engraft.replica.distill import _read_true_rows

    table = SD._load_table(fake=True, table_path=None)
    rows_global = np.array([0, 1, 2, 3], dtype=np.int64)
    true = _read_true_rows(table, rows_global.tolist())
    overlay_path = tmp_path / "overlay.pleo"
    write_pleo(overlay_path, rows_global, true + 2.0)

    plan_out = tmp_path / "plan"
    rc = SD.main(["plan", "--overlay", str(overlay_path), "--neutral-tokens", str(neutral_path),
                  "--chunk-len", "32", "--n-random", "2", "--n-hit", "2", "--seed", "0",
                  "--out", str(plan_out), "--fake"])
    assert rc == 0

    run_out = tmp_path / "run"
    rc = SD.main(["run", "--plan", str(plan_out), "--overlay", str(overlay_path),
                  "--neutral-tokens", str(neutral_path), "--out", str(run_out), "--fake", "--rbr"])
    assert rc == 0
    damage = json.loads((run_out / "damage.json").read_text())
    assert damage["routing_mode"] == "locked"

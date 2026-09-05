"""Correctness tests for engraft.lens.

Tests that read a real GGUF (PleTable, PleReplica) are marked `real` and
excluded by default (`-m "not real"`); run them explicitly once
`engraft.toml` points at real files.

uv run pytest tests/test_lens.py
uv run pytest tests/test_lens.py -m real   # requires engraft.toml
"""
from __future__ import annotations

import gguf
import numpy as np
import pytest

from engraft.config import load as load_config
from engraft.lens import (
    PleReplica,
    RowSet,
    local_to_global,
    read_pleo,
    read_plert1,
    write_pleo,
    write_plert1,
)
from engraft.table import ROW_LEN, PleTable

real = pytest.mark.real


@pytest.fixture(scope="module")
def table_path():
    return load_config().get_path("model.table")


@pytest.fixture(scope="module")
def shard2_path():
    # The PLE-bearing shard (blk.1.ple_*) of the split model, index 1 of 3.
    return load_config().get_list("model.shards")[1]


@pytest.fixture(scope="module")
def table(table_path):
    return PleTable(table_path)


@pytest.fixture(scope="module")
def replica(shard2_path):
    return PleReplica(shard2_path)


# --------------------------------------------------------------------------
# .pleo
# --------------------------------------------------------------------------


def test_pleo_roundtrip(tmp_path):
    rows = np.array([5, 100000, 3, 999999999], dtype=np.int32)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, ROW_LEN)).astype(np.float32)
    path = tmp_path / "x.pleo"
    write_pleo(path, rows, data)
    got_rows, got_data = read_pleo(path)
    np.testing.assert_array_equal(got_rows, rows)
    np.testing.assert_array_equal(got_data, data)


def test_pleo_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.pleo"
    path.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(ValueError):
        read_pleo(path)


# --------------------------------------------------------------------------
# PLERT1: round-trip of the format read/written by the fork (routing_record
# writes, routing_freeze reads) -- exercised here purely in Python, no engine
# involved.
# --------------------------------------------------------------------------


def test_plert1_roundtrip_ne1_varies_per_layer(tmp_path):
    """Round-trip with `ne1` different on the last layer (`logits: "last"`
    gives ne1=1 only on the output layer, the others have ne1=n_token)."""
    rng = np.random.default_rng(20260904)
    n_expert_used = 10
    n_layer = 48
    n_token = 7
    layers = {}
    for il in range(n_layer):
        ne1 = 1 if il == n_layer - 1 else n_token
        layers[il] = rng.integers(0, 512, size=(ne1, n_expert_used)).astype(np.int32)

    path = tmp_path / "routing.bin"
    write_plert1(path, layers)
    got = read_plert1(path)

    assert set(got.keys()) == set(layers.keys())
    for il in layers:
        np.testing.assert_array_equal(got[il], layers[il])
    assert got[n_layer - 1].shape == (1, n_expert_used)
    assert got[0].shape == (n_token, n_expert_used)


def test_plert1_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOPE!!" + b"\x00" * 20)
    with pytest.raises(ValueError):
        read_plert1(path)


# --------------------------------------------------------------------------
# local_to_global
# --------------------------------------------------------------------------


@real
def test_local_to_global_matches_head_offsets(table):
    for h in (0, 3, 8, 15):
        assert local_to_global(table, h, 0) == table.head_offsets[h]
        assert local_to_global(table, h, 7) == table.head_offsets[h] + 7


# --------------------------------------------------------------------------
# RowSet
# --------------------------------------------------------------------------


@real
def test_rowset_from_position_reads_true_rows(table):
    tokens = [10, 20, 30, 40, 50]
    rs = RowSet.from_position(table, tokens, 4)
    addr = table.ngram_addresses(tokens)[4]
    for h in range(table.n_heads):
        expected = table.read_rows(h, addr[h], 1)[0]
        np.testing.assert_array_equal(rs.data[h], expected)
        assert rs.rows_global[h] == local_to_global(table, h, addr[h])


@real
def test_isolate_keeps_only_head_h(table):
    tokens = [10, 20, 30, 40, 50]
    rs = RowSet.from_position(table, tokens, 4)
    for h in (0, 5, 15):
        out = rs.isolate(h)
        assert out.shape == (table.n_heads, ROW_LEN)
        np.testing.assert_array_equal(out[h], rs.data[h])
        for other in range(table.n_heads):
            if other != h:
                assert np.all(out[other] == 0.0)


@real
def test_random_matched_preserves_norm(table):
    tokens = [10, 20, 30, 40, 50]
    rs = RowSet.from_position(table, tokens, 4)
    rng = np.random.default_rng(20260903)
    out = rs.random_matched(rng)
    true_norms = np.linalg.norm(rs.data, axis=1)
    out_norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(out_norms, true_norms, rtol=1e-5, atol=1e-6)
    # must not be a copy (unless a zero-norm row, not expected here)
    assert not np.allclose(out, rs.data)


@real
def test_zero_and_scale(table):
    tokens = [10, 20, 30, 40, 50]
    rs = RowSet.from_position(table, tokens, 4)
    assert np.all(rs.zero() == 0.0)
    np.testing.assert_allclose(rs.scale(10.0), rs.data * 10.0)


@real
def test_swap_returns_other_data(table):
    tokens_a = [10, 20, 30, 40, 50]
    tokens_b = [99, 199, 299, 399, 499]
    rs_a = RowSet.from_position(table, tokens_a, 4)
    rs_b = RowSet.from_position(table, tokens_b, 4)
    swapped = rs_a.swap(rs_b)
    np.testing.assert_array_equal(swapped, rs_b.data)


def test_build_overlay_dedups_same_vector():
    rows1 = np.array([5, 10], dtype=np.int32)
    data1 = np.ones((2, ROW_LEN), dtype=np.float32)
    rows2 = np.array([10, 20], dtype=np.int32)
    data2 = np.stack([np.ones(ROW_LEN, dtype=np.float32), np.full(ROW_LEN, 2.0, dtype=np.float32)])
    rows, data = RowSet.build_overlay([(rows1, data1), (rows2, data2)])
    assert rows.tolist() == [5, 10, 20]
    np.testing.assert_array_equal(data[rows.tolist().index(5)], np.ones(ROW_LEN))
    np.testing.assert_array_equal(data[rows.tolist().index(10)], np.ones(ROW_LEN))
    np.testing.assert_array_equal(data[rows.tolist().index(20)], np.full(ROW_LEN, 2.0))


def test_build_overlay_rejects_conflicting_vectors():
    rows1 = np.array([5], dtype=np.int32)
    data1 = np.ones((1, ROW_LEN), dtype=np.float32)
    rows2 = np.array([5], dtype=np.int32)
    data2 = np.full((1, ROW_LEN), 2.0, dtype=np.float32)
    with pytest.raises(ValueError):
        RowSet.build_overlay([(rows1, data1), (rows2, data2)])


# --------------------------------------------------------------------------
# PleReplica
# --------------------------------------------------------------------------


@real
def test_replica_gate_zero_emb_exact_half(replica):
    t = 3
    emb = np.zeros((t, 2560), dtype=np.float32)
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((t, 4, 2560)).astype(np.float32)
    gate = replica.gate(emb, hidden)
    assert gate.shape == (t, 4)
    np.testing.assert_array_equal(gate, np.full((t, 4), 0.5, dtype=np.float32))
    gated = replica.gated(emb, hidden)
    assert np.all(gated == 0.0)


@real
def test_replica_gate_scale_invariant_gated_linear(replica):
    rng = np.random.default_rng(2)
    t = 5
    emb = rng.standard_normal((t, 2560)).astype(np.float32)
    hidden = rng.standard_normal((t, 4, 2560)).astype(np.float32)

    gate1 = replica.gate(emb, hidden)
    gate10 = replica.gate(emb * 10.0, hidden)
    np.testing.assert_allclose(gate1, gate10, atol=1e-6)

    gated1 = replica.gated(emb, hidden)
    gated10 = replica.gated(emb * 10.0, hidden)
    np.testing.assert_allclose(gated10, gated1 * 10.0, rtol=1e-4, atol=1e-5)


@real
def test_replica_conv_out_length_one_uses_only_tap3(replica):
    rng = np.random.default_rng(3)
    gated = rng.standard_normal((1, 4, 2560)).astype(np.float32)
    out = replica.conv_out(gated)
    assert out.shape == (1, 4, 2560)

    # hand computation: only tap 3 (current position, back=0) contributes.
    normed = replica._rmsnorm_per_stream(gated, replica.norm_conv, replica.eps)
    w3 = replica.conv1d[:, :, 3]  # current tap
    pre_silu = normed[0] * w3
    expected = pre_silu / (1.0 + np.exp(-pre_silu))
    np.testing.assert_allclose(out[0], expected, rtol=1e-5, atol=1e-6)


def _mask_heads(heads: list[int]) -> np.ndarray:
    """Boolean mask [2560]: True on the coordinates of the listed heads (160 each)."""
    mask = np.zeros(2560, dtype=bool)
    for h in heads:
        mask[h * ROW_LEN : (h + 1) * ROW_LEN] = True
    return mask


_T8 = _mask_heads(list(range(8, 16)))
_T1 = _mask_heads([8])


@real
def test_ple_out_equals_gated_plus_conv(replica):
    rng = np.random.default_rng(10)
    t_len = 12
    emb_seq = rng.standard_normal((t_len, 2560)).astype(np.float32)
    emb_seq = emb_seq / np.linalg.norm(emb_seq, axis=1, keepdims=True) * 0.1
    hidden_seq = rng.standard_normal((t_len, 4, 2560)).astype(np.float32)

    out = replica.ple_out(emb_seq, hidden_seq)
    gated_seq = replica.gated(emb_seq, hidden_seq)
    expected = gated_seq + replica.conv_out(gated_seq)
    np.testing.assert_array_equal(out, expected)


@real
def test_ple_out_at_matches_ple_out(replica):
    rng = np.random.default_rng(11)
    t_len = 12
    emb_seq = rng.standard_normal((t_len, 2560)).astype(np.float32)
    emb_seq = emb_seq / np.linalg.norm(emb_seq, axis=1, keepdims=True) * 0.1
    hidden_seq = rng.standard_normal((t_len, 4, 2560)).astype(np.float32)

    full = replica.ple_out(emb_seq, hidden_seq)
    for t in (11, 0):
        at = replica.ple_out_at(emb_seq[t][None, :], emb_seq, hidden_seq, t)
        np.testing.assert_allclose(at[0], full[t], rtol=1e-5, atol=1e-6)


@real
def test_jacobian_out_matches_finite_differences(replica):
    # seed 2: |s| (the argument of the gate's square root, per flow) stays far from 0 at
    # every probed coordinate — near s=0 sqrt(|s|) has curvature that breaks the
    # central difference even at a small step (this is not a bug in the replica).
    rng = np.random.default_rng(2)
    t_len = 12
    t = 7
    emb_seq = rng.standard_normal((t_len, 2560)).astype(np.float32)
    emb_seq = emb_seq / np.linalg.norm(emb_seq, axis=1, keepdims=True) * 0.1
    hidden_seq = rng.standard_normal((t_len, 4, 2560)).astype(np.float32)

    # precondition: no flow near s=0 (gate's square root) at this position,
    # otherwise sqrt(|s|) is not smooth and the central difference becomes ill-conditioned
    # regardless of the step — a property of the gate, not a defect of jacobian_out.
    k = replica.key(emb_seq[t : t + 1])
    q = replica._rmsnorm_per_stream(hidden_seq[t : t + 1], replica.norm_query, replica.eps)
    s = np.sum(k * q, axis=-1) / np.sqrt(2560)
    assert np.min(np.abs(s)) > 0.1

    coords = rng.choice(2560, size=8, replace=False)
    jac = replica.jacobian_out(emb_seq, hidden_seq, t, coords=coords, step=1e-3)
    assert jac.shape == (4 * 2560, 8)

    h = 2e-3
    for i, c in enumerate(coords):
        plus = emb_seq[t].copy()
        plus[c] += h
        minus = emb_seq[t].copy()
        minus[c] -= h
        out_plus = replica.ple_out_at(plus[None, :], emb_seq, hidden_seq, t)[0].reshape(-1)
        out_minus = replica.ple_out_at(minus[None, :], emb_seq, hidden_seq, t)[0].reshape(-1)
        fd = (out_plus.astype(np.float64) - out_minus.astype(np.float64)) / (2 * h)
        col = jac[:, i]
        rel_err = np.linalg.norm(col - fd) / np.linalg.norm(fd)
        assert rel_err <= 1e-3, f"coord {c}: rel_err={rel_err}"


@real
def test_jacobian_out_matches_analytic_direct_channel(replica, shard2_path):
    rng = np.random.default_rng(13)
    t_len = 5
    t = 2
    emb_seq = rng.standard_normal((t_len, 2560)).astype(np.float32)
    emb_seq = emb_seq / np.linalg.norm(emb_seq, axis=1, keepdims=True) * 0.1
    hidden_seq = np.zeros((t_len, 4, 2560), dtype=np.float32)

    zeroed = PleReplica(shard2_path)
    zeroed.conv1d[:] = 0.0

    coords = rng.choice(2560, size=6, replace=False)
    jac = zeroed.jacobian_out(emb_seq, hidden_seq, t, coords=coords, step=1e-3)

    expected_block = 0.5 * zeroed.w_value[:, coords]  # [2560, n_coords]
    expected = np.tile(expected_block, (4, 1))  # [10240, n_coords]
    rel_err = np.linalg.norm(jac - expected) / np.linalg.norm(expected)
    assert rel_err <= 1e-3, f"rel_err={rel_err}"


@real
def test_solve_emb_recovers_known_target(replica):
    rng = np.random.default_rng(14)
    t_len = 6
    t = 3
    emb_seq = rng.standard_normal((t_len, 2560)).astype(np.float32)
    emb_seq = emb_seq / np.linalg.norm(emb_seq, axis=1, keepdims=True) * 0.1
    hidden_seq = rng.standard_normal((t_len, 4, 2560)).astype(np.float32)

    v = rng.standard_normal(2560).astype(np.float32) * _T8
    v = v / np.linalg.norm(v)
    emb_known = emb_seq[t] + 0.5 * v
    target = replica.ple_out_at(emb_known[None, :], emb_seq, hidden_seq, t)[0]

    emb_sol, resid = replica.solve_emb(target, emb_seq, hidden_seq, t, mask=_T8, mu=1e-6)
    assert resid <= 1e-2, f"resid={resid}"
    np.testing.assert_array_equal(emb_sol[~_T8], emb_seq[t][~_T8])


@real
def test_solve_emb_tikhonov_pulls_toward_orig(replica):
    rng = np.random.default_rng(15)
    t_len = 6
    t = 3
    emb_seq = rng.standard_normal((t_len, 2560)).astype(np.float32)
    emb_seq = emb_seq / np.linalg.norm(emb_seq, axis=1, keepdims=True) * 0.1
    hidden_seq = rng.standard_normal((t_len, 4, 2560)).astype(np.float32)

    v = rng.standard_normal(2560).astype(np.float32) * _T8
    v = v / np.linalg.norm(v)
    emb_known = emb_seq[t] + 0.5 * v
    target = replica.ple_out_at(emb_known[None, :], emb_seq, hidden_seq, t)[0]

    emb_loose, resid_loose = replica.solve_emb(target, emb_seq, hidden_seq, t, mask=_T8, mu=1e-6)
    emb_tight, resid_tight = replica.solve_emb(target, emb_seq, hidden_seq, t, mask=_T8, mu=1.0)

    dist_loose = np.linalg.norm(emb_loose - emb_seq[t])
    dist_tight = np.linalg.norm(emb_tight - emb_seq[t])
    assert dist_tight < dist_loose, f"dist_tight={dist_tight} dist_loose={dist_loose}"
    assert resid_tight > resid_loose, f"resid_tight={resid_tight} resid_loose={resid_loose}"


@real
def test_solve_emb_respects_mask(replica):
    rng = np.random.default_rng(16)
    t_len = 6
    t = 3
    emb_seq = rng.standard_normal((t_len, 2560)).astype(np.float32)
    emb_seq = emb_seq / np.linalg.norm(emb_seq, axis=1, keepdims=True) * 0.1
    hidden_seq = rng.standard_normal((t_len, 4, 2560)).astype(np.float32)

    target = rng.standard_normal((4, 2560)).astype(np.float32)
    emb_sol, _ = replica.solve_emb(target, emb_seq, hidden_seq, t, mask=_T1, mu=1e-2, n_iter=5)

    changed = emb_sol != emb_seq[t]
    assert changed.sum() > 0, "solve_emb did not move any coordinate (test is not discriminating)"
    assert np.all(changed <= _T1)


@real
def test_replica_tensors_byte_identical_to_ple_split(replica, shard2_path, table_path):
    r_shard2 = gguf.GGUFReader(str(shard2_path))
    r_split = gguf.GGUFReader(str(table_path))
    by_shard2 = {t.name: t for t in r_shard2.tensors}
    by_split = {t.name: t for t in r_split.tensors}
    names = [
        "blk.1.ple_key.weight",
        "blk.1.ple_value.weight",
        "blk.1.ple_norm_key.weight",
        "blk.1.ple_norm_query.weight",
        "blk.1.ple_norm_conv.weight",
        "blk.1.ple_conv1d.weight",
    ]
    for name in names:
        assert name in by_shard2, f"{name} missing from shard 2"
        assert name in by_split, f"{name} absent from the split table GGUF"
        assert by_shard2[name].tensor_type == by_split[name].tensor_type
        np.testing.assert_array_equal(by_shard2[name].data, by_split[name].data)

"""Tests for engraft.replica.graft and the overlay extension of engraft.replica.model.

Fake replica (no GGUF, no real weights): same principle as
tests/test_replica_descend.py (a live router dependent on `rows`/trained rows,
gradient through the softmax weights), extended with `prefix()` and
`ple_true_emb()`, which use `engraft.testing.fake_table` for addressing (same
bigram/trigram structure as the real table).

uv run pytest tests/test_replica_graft.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engraft.testing.fake_table import FakeTable
from engraft.testing.fake_replica import FakeGraftReplica
from engraft.lens import RowSet, read_pleo, read_plert1, write_pleo
from engraft.table import ROW_LEN

import engraft.replica.graft as G
from engraft.replica.model import Replica, check_precondition

torch.manual_seed(0)

N_LAYER = 3
N_EXPERT_USED = 2
N_EXPERT_TOTAL = 6
VOCAB = 6
N_HEADS = 16
DIM = N_HEADS * ROW_LEN


# --------------------------------------------------------------------------
# (i) model.py: ple_true_emb/prefix con overlay
# --------------------------------------------------------------------------


def test_ple_true_emb_overlay_substitutes_present_rows_only():
    table = FakeTable(seed=7)
    r = Replica.__new__(Replica)  # nessun hp/pesi: ple_true_emb usa solo self.table
    r.table = table
    tokens = [1, 2, 3, 4, 5]
    t = 3
    rs = RowSet.from_position(table, tokens, t)
    replaced_row_global = int(rs.rows_global[0])
    fake_vec = np.full(ROW_LEN, 99.0, dtype=np.float32)
    overlay = {replaced_row_global: fake_vec}

    out = r.ple_true_emb(tokens, t, overlay=overlay).numpy().reshape(N_HEADS, ROW_LEN)
    assert np.array_equal(out[0], fake_vec)
    for h in range(1, N_HEADS):
        assert np.array_equal(out[h], rs.data[h]), f"testata {h} alterata senza overlay"


def test_ple_true_emb_no_overlay_matches_bit_exact():
    table = FakeTable(seed=8)
    r = Replica.__new__(Replica)
    r.table = table
    tokens = [10, 20, 30, 40]
    t = 2
    rs = RowSet.from_position(table, tokens, t)
    out_plain = r.ple_true_emb(tokens, t).numpy()
    out_overlay_none = r.ple_true_emb(tokens, t, overlay=None).numpy()
    assert np.array_equal(out_plain, rs.data.reshape(1, -1))
    assert np.array_equal(out_plain, out_overlay_none)


# --------------------------------------------------------------------------
# Replica finta per graft_fact (routing dal vivo, gradiente dai pesi softmax)
# --------------------------------------------------------------------------


def _make_fact(table, seed=0, n_answer=3):
    trigger_tokens = [1, 2, 3, 4, 5]
    answer_tokens = list(range(n_answer))  # < VOCAB: usati anche come y (indice di classe) in descend
    trigger_rows = RowSet.from_position(table, trigger_tokens, len(trigger_tokens) - 1).rows_global.tolist()
    chain_rows = {}
    for i in range(1, n_answer):
        prefix = trigger_tokens + answer_tokens[:i]
        chain_rows[str(i)] = RowSet.from_position(table, prefix, len(prefix) - 1).rows_global.tolist()
    fact = {
        "id": "f1", "trigger_tokens": trigger_tokens, "answer_tokens": answer_tokens,
        "trigger_rows_global": trigger_rows, "chain_rows_global": chain_rows,
    }
    return fact


FAST_CFG = {"p_stop": 0.02, "plateau_steps": 2, "refresh_every": 1, "thresholds": [], "lam": 0.0}


# --------------------------------------------------------------------------
# (ii) graft_fact chiama prefix con l'overlay degli innesti precedenti e descend
# con tag <fid>_i, refresh_every=1, row_mask T8
# --------------------------------------------------------------------------


def test_graft_fact_calls_prefix_with_prior_overlay_and_descend_tag(tmp_path):
    table = FakeTable(seed=11)
    replica = FakeGraftReplica(table, seed=1)
    fact = _make_fact(table, n_answer=2)
    out_dir = tmp_path / "f1"
    state_path = out_dir / "state.json"

    summary = G.graft_fact(replica, table, tok=None, fact=fact, cfg=FAST_CFG, out_dir=out_dir, state_path=state_path)

    assert len(replica.prefix_calls) == 2
    assert replica.prefix_calls[0]["overlay"] == {}
    # al passo 1 l'overlay deve contenere le righe scritte dal ckpt finale del passo 0
    rows0, data0 = read_pleo(out_dir / "ckpt_f1_0_final.pleo")
    overlay1 = replica.prefix_calls[1]["overlay"]
    for r, v in zip(rows0.tolist(), data0):
        assert int(r) in overlay1
        assert np.array_equal(overlay1[int(r)], v)

    assert (out_dir / "descend_f1_0.jsonl").exists()
    assert (out_dir / "descend_f1_1.jsonl").exists()
    assert (out_dir / "ckpt_f1_0_final.plert1").exists()  # refresh_every=1 -> plert1 scritto
    assert (out_dir / "ckpt_f1_1_final.plert1").exists()
    assert (out_dir / "f1.pleo").exists()
    assert summary["id"] == "f1"
    assert summary["n_positions"] == 2
    assert [it["position"] for it in summary["grafts"]] == [0, 1]

    # row_mask T8: le righe bigram (0-7) restano identiche alla riga vera in ogni ckpt
    for i in range(2):
        rows_g, data_g = read_pleo(out_dir / f"ckpt_f1_{i}_final.pleo")
        tokens_i = fact["trigger_tokens"] + fact["answer_tokens"][:i]
        rs_true = RowSet.from_position(table, tokens_i, len(tokens_i) - 1)
        rows_map = {int(r): v for r, v in zip(rows_g.tolist(), data_g)}
        for h in range(8):  # bigram
            rg = int(rs_true.rows_global[h])
            assert rg in rows_map
            assert np.array_equal(rows_map[rg], rs_true.data[h])


# --------------------------------------------------------------------------
# (iii) state.json fa saltare gli innesti chiusi alla ripresa
# --------------------------------------------------------------------------


def test_state_json_skips_closed_grafts_on_resume(tmp_path):
    table = FakeTable(seed=12)
    fact = _make_fact(table, n_answer=3)
    out_dir = tmp_path / "f1"
    state_path = out_dir / "state.json"

    replica1 = FakeGraftReplica(table, seed=2)
    G.graft_fact(replica1, table, tok=None, fact=fact, cfg=FAST_CFG, out_dir=out_dir, state_path=state_path)
    assert len(replica1.prefix_calls) == 3

    state = json.loads(state_path.read_text())
    assert set(state["done"].keys()) == {"0", "1", "2"}

    replica2 = FakeGraftReplica(table, seed=3)
    summary2 = G.graft_fact(replica2, table, tok=None, fact=fact, cfg=FAST_CFG, out_dir=out_dir, state_path=state_path)
    assert replica2.prefix_calls == [], "no graft must be rerun: all already closed in state.json"
    assert summary2["n_positions"] == 3
    assert [it["position"] for it in summary2["grafts"]] == [0, 1, 2]


# --------------------------------------------------------------------------
# (iv) merge_fact_overlays: esclude per intero il fatto perdente (keys.json)
# --------------------------------------------------------------------------


def test_merge_fact_overlays_excludes_fact_per_keys_json(tmp_path):
    rows_a = np.array([100, 108, 109], dtype=np.int32)  # riga 108/109 in T8 (>=8 in questo schema di test)
    data_a = np.stack([np.full(ROW_LEN, 1.0, dtype=np.float32) for _ in rows_a])
    rows_b = np.array([200, 108, 210], dtype=np.int32)  # 108 in comune con "a"
    data_b = np.stack([np.full(ROW_LEN, 2.0, dtype=np.float32) for _ in rows_b])

    path_a = tmp_path / "a.pleo"
    path_b = tmp_path / "b.pleo"
    write_pleo(path_a, rows_a, data_a)
    write_pleo(path_b, rows_b, data_b)

    rows_all, data_all, manifest = G.merge_fact_overlays(
        order=["a", "b"], excluded_facts={"b"}, fact_pleo_paths={"a": path_a, "b": path_b},
    )
    assert manifest["included"] == ["a"]
    assert manifest["excluded"] == ["b"]
    assert set(rows_all.tolist()) == set(rows_a.tolist())
    for r, v in zip(rows_all.tolist(), data_all):
        assert np.array_equal(v, np.full(ROW_LEN, 1.0, dtype=np.float32))


# --------------------------------------------------------------------------
# (vi) routing_trigger a 48 (qui n_layer) strati; check_precondition prima di descend
# --------------------------------------------------------------------------


def test_build_routing_trigger_shapes_and_last_row():
    n_layer = 4
    n_prefix = 3
    cap_prefix = {il: np.arange(n_prefix * 10).reshape(n_prefix, 10).astype(np.int32) for il in range(n_layer - 1)}
    cap_last = {il: np.arange(10).reshape(1, 10).astype(np.int32) + 1000 * il for il in range(n_layer)}

    routing_trigger = G._build_routing_trigger(cap_prefix, cap_last, n_layer)

    assert set(routing_trigger.keys()) == set(range(n_layer))
    for il in range(n_layer - 1):
        assert routing_trigger[il].shape == (n_prefix + 1, 10)
        assert np.array_equal(routing_trigger[il][:-1], cap_prefix[il])
        assert np.array_equal(routing_trigger[il][-1], cap_last[il][0])
    assert routing_trigger[n_layer - 1].shape == (1, 10)
    assert np.array_equal(routing_trigger[n_layer - 1], cap_last[n_layer - 1])


def test_graft_fact_calls_check_precondition_before_descend_and_fails_fast(tmp_path, monkeypatch):
    table = FakeTable(seed=13)
    replica = FakeGraftReplica(table, seed=4)
    fact = _make_fact(table, n_answer=2)
    out_dir = tmp_path / "f1"
    state_path = out_dir / "state.json"

    calls = {"precondition": 0}

    def _fake_precondition(*args, **kwargs):
        calls["precondition"] += 1
        return {"ok": False, "hits": [{"stub": True}], "rows_global": [], "prompts": {}}

    monkeypatch.setattr(G, "check_precondition", _fake_precondition)

    with pytest.raises(G.PreconditionError):
        G.graft_fact(replica, table, tok=None, fact=fact, cfg=FAST_CFG, out_dir=out_dir, state_path=state_path)

    assert calls["precondition"] == 1
    # fail-fast: nessun descend eseguito (nessun jsonl/ckpt scritto per la posizione 0)
    assert not (out_dir / "descend_f1_0.jsonl").exists()
    assert not (out_dir / "ckpt_f1_0_final.pleo").exists()

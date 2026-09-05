"""Tests for engraft.replica.descend.

Fake replica (no GGUF, no heavy torch): a toy model with `n_layer` layers,
`n_expert_used` experts chosen by top-k over a router that depends on `rows`
(so a row perturbation can change live routing, as in the real engine).
`last_step(tokens, state, rows, routing_source, persist_experts,
capture_routing, diag)` has the same signature as `Replica.last_step`; the
gradient flows through the softmax weights of the selected experts (same
principle as `moe_ffn`), not through the indices.

uv run pytest tests/test_replica_descend.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import engraft.replica.descend as D
from engraft.lens import read_plert1, read_pleo
from engraft.table import ROW_LEN

torch.manual_seed(0)

N_LAYER = 3
N_EXPERT_USED = 2
N_EXPERT_TOTAL = 6
VOCAB = 6
N_ROWS = 16
N_PREFIX = 4  # positions 0..N_PREFIX-1 in the routing_trigger of layers 0..N_LAYER-2


class FakeReplica:
    """Live router = topk(softmax(x @ gate_w[il])); differentiable combination
    (probs[idx] weighs the selected experts, same gather scheme as `moe_ffn`,
    so the gradient does NOT flow through the indices, only through the
    softmax weights)."""

    def __init__(self, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_layer = N_LAYER
        self.gate_w = [
            torch.tensor(rng.standard_normal((dim, N_EXPERT_TOTAL)) * 0.3, dtype=torch.float64)
            for _ in range(N_LAYER)
        ]
        self.expert_w = [
            torch.tensor(rng.standard_normal((N_EXPERT_TOTAL, dim)) * 0.3, dtype=torch.float64)
            for _ in range(N_LAYER)
        ]
        self.out_w = torch.tensor(rng.standard_normal((dim, VOCAB)) * 0.3, dtype=torch.float64)
        self.routing_source_log: list[bool] = []  # True if routing_source was None (refresh)

    def last_step(
        self, tokens, state, rows, routing_source=None, persist_experts=True,
        capture_routing=None, diag=None,
    ):
        self.routing_source_log.append(routing_source is None)
        x = rows.reshape(-1).to(torch.float64)
        for il in range(self.n_layer):
            gate_logits = x @ self.gate_w[il]
            probs = torch.softmax(gate_logits, dim=-1)
            if routing_source is not None and il in routing_source:
                arr = np.asarray(routing_source[il])
                row = arr[0] if arr.shape[0] == 1 else arr[-1]
                idx = torch.from_numpy(np.asarray(row, dtype=np.int64))
            else:
                idx = torch.topk(probs, N_EXPERT_USED).indices
            if capture_routing is not None:
                capture_routing[il] = idx.detach().numpy().reshape(1, -1).astype(np.int32)
            w_sel = probs[idx]
            contrib = (w_sel.unsqueeze(-1) * self.expert_w[il][idx]).sum(dim=0)
            x = x + contrib
        return x @ self.out_w


def _make_fixture(seed: int = 0):
    dim = N_ROWS * ROW_LEN
    replica = FakeReplica(dim, seed=seed)
    rng = np.random.default_rng(seed + 100)
    rows_true = torch.tensor(rng.standard_normal((N_ROWS, ROW_LEN)) * 0.05, dtype=torch.float32)

    # routing_trigger: layers 0..N_LAYER-2 -> [N_PREFIX,k] (prefix, independent
    # of the rows: the prefix does not read them); layer N_LAYER-1 -> [1,k]
    # (only the last position, as in the real engine).
    routing_trigger: dict[int, np.ndarray] = {}
    for il in range(N_LAYER - 1):
        routing_trigger[il] = rng.integers(0, N_EXPERT_TOTAL, size=(N_PREFIX, N_EXPERT_USED)).astype(np.int32)
    routing_trigger[N_LAYER - 1] = rng.integers(0, N_EXPERT_TOTAL, size=(1, N_EXPERT_USED)).astype(np.int32)

    trigger_tokens = [1, 2, 3, 4, 5]
    trigger_rows_global = np.arange(1000, 1000 + N_ROWS, dtype=np.int32)
    y = 0
    return replica, trigger_tokens, routing_trigger, rows_true, trigger_rows_global, y


def _recompute_p(replica, trigger_tokens, rows_np, routing_source, y):
    rows_t = torch.from_numpy(rows_np.copy())
    with torch.no_grad():
        logits = replica.last_step(trigger_tokens, None, rows_t, routing_source=routing_source, persist_experts=False)
    logp = torch.log_softmax(logits.to(torch.float64), dim=-1)
    return float(torch.exp(logp[y]).item())


# --------------------------------------------------------------------------
# (i) routing_source=None exactly at multiples of k
# --------------------------------------------------------------------------


def test_refresh_calls_live_routing_at_multiples_of_k(tmp_path):
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=1)
    k = 2
    out_dir = tmp_path
    log_path = tmp_path / "descend.jsonl"
    summary = D.descend(
        replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, out_dir, log_path, refresh_every=k, thresholds=[], p_stop=2.0, plateau_steps=100000,
    )
    assert summary["n_steps"] >= 6, "at least 6 steps are needed to check the first refreshes"
    for step in range(6):
        expect_refresh = (step % k == 0)
        assert replica.routing_source_log[step] == expect_refresh, (
            f"step {step}: expected routing_source=None -> {expect_refresh}"
        )


# --------------------------------------------------------------------------
# (ii) the rows saved in each .npy reproduce the recorded p (pre-update rows)
# --------------------------------------------------------------------------


def test_checkpoint_npy_reproduces_recorded_p():
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=2)
    out_dir = None
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        log_path = out_dir / "descend.jsonl"
        summary = D.descend(
            replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
            0.0, out_dir, log_path, refresh_every=None, thresholds=[0.05, 0.1], p_stop=2.0,
        )
        ck = None
        for key in ("p0.1", "p0.05"):
            if key in summary["checkpoints"]:
                ck = key
                break
        assert ck is not None, f"no threshold reached: checkpoints={summary['checkpoints']}"
        rows_np = np.load(out_dir / f"ckpt_0.0_{ck}.npy")
        recomputed_p = _recompute_p(replica, trigger_tokens, rows_np, routing_trigger, y)
        assert recomputed_p == pytest.approx(summary["checkpoints"][ck]["p"], abs=1e-6)


# --------------------------------------------------------------------------
# (iii) .plert1 reads back: shape [T,k] at 0..n-2, [1,k] at the last one, last row =
# captured routing, prefix rows = routing_trigger
# --------------------------------------------------------------------------


def test_plert1_roundtrip_shape_and_content(tmp_path):
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=3)
    log_path = tmp_path / "descend.jsonl"
    summary = D.descend(
        replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path, log_path, refresh_every=1, thresholds=[0.05], p_stop=2.0, plateau_steps=100000,
    )
    ck_key = "p0.05"
    assert ck_key in summary["checkpoints"], summary["checkpoints"]
    step = summary["checkpoints"][ck_key]["step"]
    plert1_path = tmp_path / f"ckpt_0.0_{ck_key}.plert1"
    assert plert1_path.exists()
    layers = read_plert1(plert1_path)
    assert set(layers.keys()) == set(range(N_LAYER))
    for il in range(N_LAYER - 1):
        assert layers[il].shape == (N_PREFIX, N_EXPERT_USED)
        assert np.array_equal(layers[il][:-1], routing_trigger[il][:-1])
    assert layers[N_LAYER - 1].shape == (1, N_EXPERT_USED)

    # independent recomputation of the routing captured at that step (refresh_every=1: every
    # step is a refresh, so the routing used for step `step` depends only
    # on that step's pre-update rows, deterministic).
    rows_np = np.load(tmp_path / f"ckpt_0.0_{ck_key}.npy")
    rows_t = torch.from_numpy(rows_np.copy())
    cap2: dict[int, np.ndarray] = {}
    with torch.no_grad():
        replica.last_step(trigger_tokens, None, rows_t, routing_source=None, persist_experts=False, capture_routing=cap2)
    for il in range(N_LAYER):
        assert np.array_equal(layers[il][-1], cap2[il][0]), f"strato {il}: ultima riga plert1 diversa dal routing ricalcolato"


# --------------------------------------------------------------------------
# (iv) row_mask: rows outside the mask stay bit-identical
# --------------------------------------------------------------------------


def test_row_mask_freezes_rows_outside_mask(tmp_path):
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=4)
    row_mask = np.array([False] * 8 + [True] * 8)
    log_path = tmp_path / "descend.jsonl"
    summary = D.descend(
        replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path, log_path, row_mask=row_mask, refresh_every=2, thresholds=[0.05],
        p_stop=2.0, plateau_steps=100000,
    )
    ck_key = "final"
    rows_np = np.load(tmp_path / f"ckpt_0.0_{ck_key}.npy")
    rows_true_np = rows_true.numpy()
    assert np.array_equal(rows_np[~row_mask], rows_true_np[~row_mask]), "rows outside the mask were altered"


# --------------------------------------------------------------------------
# (v) refresh_every=None: no .plert1, routing_source always routing_trigger
# --------------------------------------------------------------------------


def test_refresh_every_none_matches_2f_behavior(tmp_path):
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=5)
    log_path = tmp_path / "descend.jsonl"
    summary = D.descend(
        replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path, log_path, refresh_every=None, thresholds=[0.05, 0.1], p_stop=0.5,
    )
    assert not list(tmp_path.glob("*.plert1")), "no .plert1 expected with refresh_every=None"
    # the last n_steps calls are the descent's (one per step); the
    # subsequent call (out of scope here) is the 2f final free-routing
    # diagnostic, unchanged and not part of this check.
    assert all(not is_none for is_none in replica.routing_source_log[: summary["n_steps"]]), (
        "routing_source must always be given (routing_trigger), never None, with refresh_every=None"
    )
    assert summary["refresh_every"] is None
    assert summary["n_refreshes"] == 0


# --------------------------------------------------------------------------
# (vi) plateau stop: stop_reason and plateau_refreshes = max(3, ceil(50/k))
# --------------------------------------------------------------------------


def test_plateau_stop(tmp_path):
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=6)
    k = 3
    log_path = tmp_path / "descend.jsonl"
    summary = D.descend(
        replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path, log_path, refresh_every=k, thresholds=[], p_stop=2.0, plateau_steps=50,
    )
    assert summary["plateau_refreshes"] == max(3, -(-50 // k))
    assert summary["stop_reason"] == "plateau"


# --------------------------------------------------------------------------
# (vii) rows_start != rows_true: step 0 starts from rows_start, the regularizer
# stays referred to rows_true
# --------------------------------------------------------------------------


def test_rows_start_used_for_step0_regularizer_vs_rows_true(tmp_path):
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=7)
    rng = np.random.default_rng(999)
    delta = rng.standard_normal(rows_true.shape).astype(np.float32) * 0.01
    rows_start = rows_true + torch.from_numpy(delta)

    log_path = tmp_path / "descend.jsonl"
    summary = D.descend(
        replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path, log_path, refresh_every=1, thresholds=[], p_stop=2.0,
        plateau_steps=100000, rows_start=rows_start,
    )
    with open(log_path) as f:
        rec0 = __import__("json").loads(f.readline())

    # step 0's forward pass (refresh always active) must come from rows_start,
    # not from rows_true: direct recomputation for comparison.
    p0_from_start = _recompute_p(replica, trigger_tokens, rows_start.numpy(), None, y)
    p0_from_true = _recompute_p(replica, trigger_tokens, rows_true.numpy(), None, y)
    assert rec0["p"] == pytest.approx(p0_from_start, abs=1e-6)
    assert p0_from_start != pytest.approx(p0_from_true, abs=1e-9)

    # the regularizer (MU * ||rows-rows_true||^2/||rows_true||^2) is measured against
    # rows_true, not rows_start: at step 0 it equals MU * ||delta||^2/||rows_true||^2
    # restricted to the rows (no mask here: all 16 rows).
    reg_expected = D.MU * float((delta ** 2).sum() / (rows_true.numpy() ** 2).sum())
    reg_from_record = rec0["loss"] - (-rec0["logp_y"])  # lam=0.0, kl_total=0
    assert reg_from_record == pytest.approx(reg_expected, abs=1e-6)


# --------------------------------------------------------------------------
# (viii) k that does not divide the last step: final with extra forward pass
# --------------------------------------------------------------------------


def test_final_refreshed_by_extra_forward_when_last_step_not_a_refresh(tmp_path):
    replica, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=8)
    k = 7  # 300 (MAX_STEPS) is not a multiple of 7: the last step (299) is not a refresh
    assert D.MAX_STEPS % k != 0
    log_path = tmp_path / "descend.jsonl"
    summary = D.descend(
        replica, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path, log_path, refresh_every=k, thresholds=[], p_stop=2.0, plateau_steps=10**9,
    )
    assert summary["n_steps"] == D.MAX_STEPS
    assert summary["stop_reason"] == "max_steps"
    assert summary["final_refreshed_by_extra_forward"] is True
    assert summary["final_p_free"] is not None
    plert1_path = tmp_path / "ckpt_0.0_final.plert1"
    assert plert1_path.exists()
    layers = read_plert1(plert1_path)
    assert set(layers.keys()) == set(range(N_LAYER))

    # the extra forward pass is on the last step's rows_pre (without gradient, live
    # routing): final p_free must match the direct recomputation.
    rows_np = np.load(tmp_path / "ckpt_0.0_final.npy")
    recomputed = _recompute_p(replica, trigger_tokens, rows_np, None, y)
    assert recomputed == pytest.approx(summary["final_p_free"], abs=1e-6)


# --------------------------------------------------------------------------
# (ix) plateau_metric: default "logp" (0.05 nat margin), "p_free" (0.01 absolute
# margin) selectable for compatibility.
# --------------------------------------------------------------------------


def test_plateau_metric_default_is_logp_bit_exact(tmp_path):
    replica_a, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=6)
    replica_b, _, routing_trigger_b, rows_true_b, _, _ = _make_fixture(seed=6)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    kwargs = dict(
        refresh_every=3, thresholds=[], p_stop=2.0, plateau_steps=50,
    )
    summary_default = D.descend(
        replica_a, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path / "a", tmp_path / "a" / "d.jsonl", **kwargs,
    )
    summary_explicit = D.descend(
        replica_b, trigger_tokens, None, routing_trigger_b, rows_true_b, y, {}, rows_global,
        0.0, tmp_path / "b", tmp_path / "b" / "d.jsonl", plateau_metric="logp", **kwargs,
    )
    assert summary_default["plateau_metric"] == "logp"
    assert summary_default["n_steps"] == summary_explicit["n_steps"]
    assert summary_default["stop_reason"] == summary_explicit["stop_reason"]
    assert summary_default["final_logp_y"] == summary_explicit["final_logp_y"]


def test_plateau_metric_logp_survives_false_plateau(tmp_path):
    """seed=1: starting point logp0 ~ -12.95, p_free0 ~ 2e-6 -- below p 0.01 an
    absolute improvement of 0.01 is impossible by construction: with
    plateau_metric="p_free" and plateau_steps=5 the counter saturates
    immediately (a false plateau), while with plateau_metric="logp" the descent
    (in progress: logp improves by more than 0.05 nat per refresh) continues
    much further."""
    replica_p, trigger_tokens, routing_trigger, rows_true, rows_global, y = _make_fixture(seed=1)
    replica_l, _, routing_trigger_l, rows_true_l, _, _ = _make_fixture(seed=1)
    (tmp_path / "p").mkdir()
    (tmp_path / "l").mkdir()

    summary_pfree = D.descend(
        replica_p, trigger_tokens, None, routing_trigger, rows_true, y, {}, rows_global,
        0.0, tmp_path / "p", tmp_path / "p" / "d.jsonl", refresh_every=1, thresholds=[],
        p_stop=2.0, plateau_steps=5, plateau_metric="p_free",
    )
    summary_logp = D.descend(
        replica_l, trigger_tokens, None, routing_trigger_l, rows_true_l, y, {}, rows_global,
        0.0, tmp_path / "l", tmp_path / "l" / "d.jsonl", refresh_every=1, thresholds=[],
        p_stop=2.0, plateau_steps=5, plateau_metric="logp",
    )
    assert summary_pfree["stop_reason"] == "plateau"
    assert summary_logp["stop_reason"] == "plateau"
    assert summary_pfree["plateau_metric"] == "p_free"
    assert summary_logp["plateau_metric"] == "logp"
    # same exact starting point (seed=1) and same descent dynamics: only the stop
    # criterion changes, and "logp" runs many more steps before stopping.
    assert summary_logp["n_steps"] > summary_pfree["n_steps"] + 10

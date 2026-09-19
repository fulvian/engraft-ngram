"""Tests for `engraft.replica.seq`: the CPU reference sequence forward
(`seq_forward`), the packed multi-fragment overlay descent, and the
whole-layer checkpoint's math-preserving property
(`run_layer_checkpointed`).

uv run pytest tests/test_seq.py
"""
from __future__ import annotations

import numpy as np
import torch

import engraft.replica.backend as Bk
from engraft.lens import RowSet, read_pleo, write_pleo
from engraft.replica.model import Replica, head_dtype_from_name
from engraft.replica.pack import pack_fragments
from engraft.replica.seq import capture_ple_gate, run_layer_checkpointed, seq_forward
from engraft.testing.fake_full_weights import FakeFullWeights, tiny_hparams
from engraft.testing.fake_table import FakeTable

torch.manual_seed(0)


def _fake_replica():
    hp = tiny_hparams()
    w = FakeFullWeights(hp)
    table = FakeTable(seed=1)
    table.eos_token_id = hp.n_vocab - 1
    replica = Replica(hp, w, table, backend=Bk.Backend.cpu_f32(), head_dtype=head_dtype_from_name("f32"))
    return replica, table


def test_seq_forward_single_fragment_produces_expected_shapes_and_gradient():
    replica, table = _fake_replica()
    tokens = [table.eos_token_id, 3, 5, 7, 2, 9]
    row_map: dict[int, int] = {}
    rows_var = torch.zeros(0, 160, dtype=torch.float32, requires_grad=True)

    state, stats = seq_forward(replica, tokens, rows_var, row_map, return_logits=True, grad_proxy=True)
    assert stats.n_ple_calls == len(tokens) - 1
    assert state.logits.shape == (len(tokens) - 1, replica.hp.n_vocab)

    loss = state.logits.sum()
    loss.backward()
    assert state.emb_leaf.grad is not None


def test_packed_multi_fragment_overlay_gradient_and_pleo_roundtrip(tmp_path):
    """The core technique end to end: a packed multi-fragment sequence
    (`pack.pack_fragments`, `segment_ids`), a differentiable overlay row,
    a few descent steps with a decreasing loss, then a `.pleo` write/reread
    round trip (bit-identical)."""
    replica, table = _fake_replica()
    frags = [[3, 5, 7, 2], [9, 4, 6]]
    packed_list = pack_fragments(frags, table.eos_token_id, max_len=64)
    packed = packed_list[0]
    assert packed.segment_ids.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]

    rs0 = RowSet.from_position(table, packed.tokens, 0)
    g0 = int(rs0.rows_global[0])
    row_map = {g0: 0}
    rows_var = torch.zeros(1, 160, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([rows_var], lr=0.05)

    y = torch.tensor(packed.tokens[1 : len(packed.tokens)], dtype=torch.int64)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        state, stats = seq_forward(
            replica, packed.tokens, rows_var, row_map, return_logits=True, grad_proxy=True,
            segment_ids=packed.segment_ids, positions=packed.positions,
        )
        assert stats.n_ple_calls == len(packed.tokens) - 1
        loss = torch.nn.functional.cross_entropy(state.logits, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    out_path = tmp_path / "smoke.pleo"
    rows_all = np.array([g0], dtype=np.int32)
    data_all = rows_var.detach().numpy().astype(np.float32)
    write_pleo(out_path, rows_all, data_all)
    rows_read, data_read = read_pleo(out_path)
    assert rows_read.tolist() == [g0]
    assert float(np.abs(data_all - data_read).max()) == 0.0


def test_run_layer_checkpointed_bit_identical():
    """`run_layer_checkpointed` (whole-layer `torch.utils.checkpoint`) never
    changes the math: same logits, same gradient, `torch.equal` (not just
    `allclose`), with vs. without the checkpoint context -- see
    `docs/formats.md`'s note on `--checkpoint`."""
    replica, table = _fake_replica()
    tokens = [table.eos_token_id, 3, 5, 7, 2, 9]

    state1 = replica.prefix(tokens, grad_proxy=True, return_logits=True)
    loss1 = state1.logits.sum()
    loss1.backward()
    grad1 = state1.emb_leaf.grad.clone()

    with run_layer_checkpointed(replica):
        state2 = replica.prefix(tokens, grad_proxy=True, return_logits=True)
        loss2 = state2.logits.sum()
        loss2.backward()
    grad2 = state2.emb_leaf.grad.clone()

    assert torch.equal(state1.logits, state2.logits)
    assert torch.equal(loss1.detach(), loss2.detach())
    assert torch.equal(grad1, grad2)


def test_import_without_triton():
    """`engraft.replica.seq` must import cleanly without Triton installed
    (the one published Triton kernel is used only when the real backend
    actually runs the per-expert step -- never at import time)."""
    import sys
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name in ("triton", "triton.language") or name.startswith("triton."):
            raise ImportError(f"blocked for this test: {name}")
        return real_import(name, *args, **kwargs)

    saved = {k: v for k, v in sys.modules.items() if k.startswith("engraft.replica.seq") or k.startswith("engraft.replica.triton_experts")}
    for k in list(saved):
        del sys.modules[k]
    builtins.__import__ = blocking_import
    try:
        import engraft.replica.seq as seq_mod  # noqa: F401
        assert seq_mod.TritonExpertMatmul is None
    finally:
        builtins.__import__ = real_import
        for k in list(sys.modules):
            if k.startswith("engraft.replica.seq") or k.startswith("engraft.replica.triton_experts"):
                del sys.modules[k]
        sys.modules.update(saved)


def test_capture_ple_gate_populates_out_and_restores_run_layer():
    replica, table = _fake_replica()
    eos = table.eos_token_id
    tokens = [eos, 1, 2, 3, 4]
    empty_row_map: dict[int, int] = {}
    empty_rows = torch.zeros(0, 160, dtype=torch.float32)
    orig_run_layer = replica.run_layer

    out: dict = {}
    with capture_ple_gate(replica, out):
        state, _extra = seq_forward(
            replica, tokens, empty_rows, empty_row_map, return_logits=True, grad_proxy=False,
        )
    n_prefix = len(tokens) - 1
    assert out["gate"].shape[0] == n_prefix
    assert out["s"].shape[0] == n_prefix
    assert out["value_norm"].shape[0] == n_prefix
    # restored, not left patched
    assert replica.run_layer is orig_run_layer or replica.__dict__.get("run_layer") is None


def test_capture_ple_gate_rejects_non_ple_layer():
    replica, _table = _fake_replica()
    non_ple_layer = 0
    assert not replica.hp.is_ple(non_ple_layer)
    out: dict = {}
    try:
        with capture_ple_gate(replica, out, layer=non_ple_layer):
            pass
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

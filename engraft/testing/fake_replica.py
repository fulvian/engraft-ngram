"""Fake replica for graft_fact, shared by tests/test_replica_graft.py and by
`engraft.run --fake` (import guarded, only in the dry-run branch: no dependency
of `tests/` on production code).

Live router dependent on `rows` (softmax of router logits, top-k), gradient
from the softmax weights of the selected experts (same gather scheme as
`moe_ffn`): same principle as `tests/test_replica_descend.py`, extended with
`prefix()` and `ple_true_emb()`, which use a `FakeTable`
(engraft.testing.fake_table) for addressing (same bigram/trigram structure as
the real table).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from engraft.lens import RowSet
from engraft.table import ROW_LEN


class FakeGraftReplica:
    def __init__(
        self, table, seed: int = 0, n_layer: int = 3, n_expert_used: int = 2,
        n_expert_total: int = 6, vocab: int = 6, n_heads: int = 16,
    ):
        self.table = table
        self.n_layer = n_layer
        self.hp = SimpleNamespace(n_layer=n_layer)
        self.n_expert_used = n_expert_used
        dim = n_heads * ROW_LEN
        rng = np.random.default_rng(seed)
        self.gate_w = [
            torch.tensor(rng.standard_normal((dim, n_expert_total)) * 0.3, dtype=torch.float64)
            for _ in range(n_layer)
        ]
        self.expert_w = [
            torch.tensor(rng.standard_normal((n_expert_total, dim)) * 0.3, dtype=torch.float64)
            for _ in range(n_layer)
        ]
        self.out_w = torch.tensor(rng.standard_normal((dim, vocab)) * 0.3, dtype=torch.float64)
        self.prefix_calls: list[dict] = []  # spia (test 6.1 (ii))

    def ple_true_emb(self, tokens, t, overlay=None):
        rs = RowSet.from_position(self.table, tokens, t)
        data = rs.data.copy()
        if overlay:
            for h in range(rs.rows_global.shape[0]):
                rg = int(rs.rows_global[h])
                if rg in overlay:
                    data[h] = np.asarray(overlay[rg], dtype=np.float32)
        return data.reshape(-1)

    def prefix(self, tokens, routing_source=None, capture_routing=None, overlay=None, diag=None):
        self.prefix_calls.append({"tokens": list(tokens), "overlay": dict(overlay or {})})
        n_prefix = len(tokens) - 1
        if n_prefix <= 0:
            return {"n_prefix": 0}
        for il in range(self.n_layer - 1):  # l'ultimo strato non consuma routing nel prefisso
            rows_idx = []
            for t in range(n_prefix):
                x = torch.from_numpy(self.ple_true_emb(tokens, t, overlay=overlay)).to(torch.float64)
                if routing_source is not None and il in routing_source:
                    arr = np.asarray(routing_source[il])
                    row = arr[0] if arr.shape[0] == 1 else arr[t]
                    idx = np.asarray(row, dtype=np.int32)
                else:
                    logits = x @ self.gate_w[il]
                    idx = torch.topk(torch.softmax(logits, dim=-1), self.n_expert_used).indices.numpy().astype(np.int32)
                rows_idx.append(idx)
            if capture_routing is not None:
                capture_routing[il] = np.stack(rows_idx, axis=0)
        return {"n_prefix": n_prefix}

    def last_step(self, tokens, state, rows, routing_source=None, persist_experts=True, capture_routing=None, diag=None):
        x = rows.reshape(-1).to(torch.float64)
        for il in range(self.n_layer):
            gate_logits = x @ self.gate_w[il]
            probs = torch.softmax(gate_logits, dim=-1)
            if routing_source is not None and il in routing_source:
                arr = np.asarray(routing_source[il])
                row = arr[0] if arr.shape[0] == 1 else arr[-1]
                idx = torch.from_numpy(np.asarray(row, dtype=np.int64))
            else:
                idx = torch.topk(probs, self.n_expert_used).indices
            if capture_routing is not None:
                capture_routing[il] = idx.detach().numpy().reshape(1, -1).astype(np.int32)
            w_sel = probs[idx]
            contrib = (w_sel.unsqueeze(-1) * self.expert_w[il][idx]).sum(dim=0)
            x = x + contrib
        return x @ self.out_w

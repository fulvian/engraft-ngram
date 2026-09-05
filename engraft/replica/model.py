"""Replica orchestration: constant prefix, differentiable last position.

`Replica.prefix(tokens)` runs a full forward (no gradient) over positions
0..T-2, producing for every layer the state `last_step` needs for position
T-1: K/V cache (attention layers), (conv history, recurrence state) (delta net
layers), PLE conv history (layer 1). Routing is imposed by `routing_source`
when given (PLERT1, `{il: array[T_tot,10]}`), otherwise computed live
(softmax + top-k over the router logits). At the final layer (`n_layer-1`)
the prefix skips the MoE computation (it feeds nothing downstream).

`Replica.last_step(state, rows, routing_source)` runs the whole 48-layer stack
for just position T-1, substituting the given `rows` (differentiable when they
require a gradient) for the PLE table gather at layer `ple_layer`.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from engraft.lens import RowSet, local_to_global
from engraft.table import PleTable
from engraft.replica.hparams import Hparams
from engraft.replica.weights import GgufWeights
from engraft.replica.layers import (
    AttnWeights,
    DeltaNetState,
    DeltaNetWeights,
    PleWeights,
    attention_full,
    delta_net_init_state,
    hc_combine,
    hc_mix,
    linear_attn_layer,
    moe_ffn,
    ple_forward,
)


# --------------------------------------------------------------------------
# Loading weights per layer
# --------------------------------------------------------------------------


def load_attn_weights(w: GgufWeights, il: int) -> AttnWeights:
    p = f"blk.{il}."
    return AttnWeights(
        wq=torch.from_numpy(w.tensor(p + "attn_q.weight")),
        wk=torch.from_numpy(w.tensor(p + "attn_k.weight")),
        wv=torch.from_numpy(w.tensor(p + "attn_v.weight")),
        wo=torch.from_numpy(w.tensor(p + "attn_output.weight")),
        q_norm=torch.from_numpy(w.tensor(p + "attn_q_norm.weight")),
        k_norm=torch.from_numpy(w.tensor(p + "attn_k_norm.weight")),
    )


def load_delta_net_weights(w: GgufWeights, il: int) -> DeltaNetWeights:
    p = f"blk.{il}."
    return DeltaNetWeights(
        wqkv=torch.from_numpy(w.tensor(p + "attn_qkv.weight")),
        wqkv_gate=torch.from_numpy(w.tensor(p + "attn_gate.weight")),
        ssm_conv1d=torch.from_numpy(w.tensor(p + "ssm_conv1d.weight")),
        ssm_dt_bias=torch.from_numpy(w.tensor(p + "ssm_dt.bias")),
        ssm_a=torch.from_numpy(w.tensor(p + "ssm_a")),
        ssm_beta=torch.from_numpy(w.tensor(p + "ssm_beta.weight")),
        ssm_alpha=torch.from_numpy(w.tensor(p + "ssm_alpha.weight")),
        ssm_norm=torch.from_numpy(w.tensor(p + "ssm_norm.weight")),
        ssm_out=torch.from_numpy(w.tensor(p + "ssm_out.weight")),
    )


def load_ple_weights(w: GgufWeights, il: int) -> PleWeights:
    p = f"blk.{il}."
    return PleWeights(
        w_key=torch.from_numpy(w.tensor(p + "ple_key.weight")),
        w_value=torch.from_numpy(w.tensor(p + "ple_value.weight")),
        norm_key=torch.from_numpy(w.tensor(p + "ple_norm_key.weight")),
        norm_query=torch.from_numpy(w.tensor(p + "ple_norm_query.weight")),
        norm_conv=torch.from_numpy(w.tensor(p + "ple_norm_conv.weight")),
        conv1d=torch.from_numpy(w.tensor(p + "ple_conv1d.weight")),
    )


def load_hc(w: GgufWeights, il: int, slot: str):
    p = f"blk.{il}.hc_{slot}_"
    norm = torch.from_numpy(w.tensor(p + "norm.weight"))
    down = torch.from_numpy(w.tensor(p + "down.weight"))
    up = torch.from_numpy(w.tensor(p + "up.weight"))
    inject = torch.from_numpy(w.tensor(p + "inject.weight"))
    return norm, down, up, inject


@dataclasses.dataclass
class MoeNonExpertWeights:
    gate_inp: torch.Tensor
    gate_inp_shexp: torch.Tensor
    up_shexp: torch.Tensor
    gate_shexp: torch.Tensor
    down_shexp: torch.Tensor


def load_moe_nonexpert(w: GgufWeights, il: int) -> MoeNonExpertWeights:
    p = f"blk.{il}."
    return MoeNonExpertWeights(
        gate_inp=torch.from_numpy(w.tensor(p + "ffn_gate_inp.weight")),
        gate_inp_shexp=torch.from_numpy(w.tensor(p + "ffn_gate_inp_shexp.weight")),
        up_shexp=torch.from_numpy(w.tensor(p + "ffn_up_shexp.weight")),
        gate_shexp=torch.from_numpy(w.tensor(p + "ffn_gate_shexp.weight")),
        down_shexp=torch.from_numpy(w.tensor(p + "ffn_down_shexp.weight")),
    )


# --------------------------------------------------------------------------
# Per-layer state, carried from the prefix to the last position
# --------------------------------------------------------------------------


@dataclasses.dataclass
class LayerState:
    attn: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # (k,v,positions)
    delta: DeltaNetState | None = None
    ple_hist: torch.Tensor | None = None


@dataclasses.dataclass
class PrefixState:
    layers: dict[int, LayerState]
    n_prefix: int  # T-1 (number of positions already consumed)


# --------------------------------------------------------------------------
# Replica
# --------------------------------------------------------------------------


class Replica:
    def __init__(self, hp: Hparams, w: GgufWeights, table: PleTable):
        self.hp = hp
        self.w = w
        self.table = table

    def embed(self, tokens: list[int]) -> torch.Tensor:
        tok_embd = self.w.tensor("token_embd.weight")  # [n_vocab, n_embd]
        idx = np.asarray(tokens, dtype=np.int64)
        return torch.from_numpy(tok_embd[idx].copy())  # [T, n_embd]

    def ple_true_emb(
        self, tokens: list[int], t: int, overlay: dict[int, np.ndarray] | None = None,
    ) -> torch.Tensor:
        """emb [1, n_embd] from the table's true gather for position `t` of `tokens`.

        If `overlay` is given ({global_row: vector [160]}), the position's
        global rows present in the overlay replace the true gather (a graft
        from an earlier chain position, read from the prefix); absent rows
        keep the true gather. `overlay=None` (default) is identical to the
        prior behavior."""
        rs = RowSet.from_position(self.table, tokens, t)
        if not overlay:
            return torch.from_numpy(rs.data.reshape(1, -1).copy())  # [1, n_heads*head_dim] = [1,n_embd]
        data = rs.data.copy()
        for h in range(rs.rows_global.shape[0]):
            row_g = int(rs.rows_global[h])
            if row_g in overlay:
                data[h] = np.asarray(overlay[row_g], dtype=np.float32)
        return torch.from_numpy(data.reshape(1, -1).copy())

    def _expert_fns(self, il: int, persist: bool):
        p = f"blk.{il}."

        def gate_fn(e: int) -> torch.Tensor:
            return torch.from_numpy(self.w.expert(p + "ffn_gate_exps.weight", e, persist=persist))

        def up_fn(e: int) -> torch.Tensor:
            return torch.from_numpy(self.w.expert(p + "ffn_up_exps.weight", e, persist=persist))

        def down_fn(e: int) -> torch.Tensor:
            return torch.from_numpy(self.w.expert(p + "ffn_down_exps.weight", e, persist=persist))

        return gate_fn, up_fn, down_fn

    def _routing_for(
        self,
        il: int,
        positions: torch.Tensor,
        x: torch.Tensor,
        gate_inp: torch.Tensor,
        routing_source: dict[int, np.ndarray] | None,
        diag: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> torch.Tensor:
        """`diag`, if given, receives {il: (top11_idx[T,11], top11_val[T,11])} when the
        routing is computed live (for the Q1b relative margin: tenth chosen
        vs eleventh excluded, without having to rerun the forward pass)."""
        if routing_source is not None and il in routing_source:
            arr = routing_source[il]  # [T_tot,10] (0..46) or [1,10] only the last position (47)
            pos_idx = positions.to(torch.int64).numpy()
            if arr.shape[0] == 1:
                # layer 47: the file records only the last position (row 0);
                # here it must come from the single last_step call, one position.
                if len(pos_idx) != 1:
                    raise ValueError(
                        f"routing_source[{il}] has a single row (last position only) "
                        f"but {len(pos_idx)} positions were requested"
                    )
                sel = arr[0:1]
            else:
                sel = arr[pos_idx]
            return torch.from_numpy(np.asarray(sel, dtype=np.int64))
        logits = x @ gate_inp.T
        probs = torch.softmax(logits, dim=-1)
        k = self.hp.n_expert_used
        if diag is not None:
            top = torch.topk(probs, k + 1, dim=-1)
            diag[il] = (top.indices.numpy(), top.values.numpy())
            return top.indices[:, :k]
        return torch.topk(probs, k, dim=-1).indices

    def run_layer(
        self,
        il: int,
        x: torch.Tensor,  # [T,hc,n_embd]
        positions: torch.Tensor,
        state: LayerState,
        routing_source: dict[int, np.ndarray] | None,
        ple_emb: torch.Tensor | None,
        need_ffn_output: bool = True,
        persist_experts: bool = False,
        diag: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[torch.Tensor, LayerState, torch.Tensor | None]:
        hp = self.hp
        eps = hp.f_norm_rms_eps
        hc = hp.hc_mult
        new_ple_hist = state.ple_hist

        if hp.is_ple(il):
            ple_w = load_ple_weights(self.w, il)
            hist = state.ple_hist if state.ple_hist is not None else torch.zeros(0, hc, hp.n_embd)
            x, new_ple_hist = ple_forward(ple_emb, x, ple_w, hist, hp)

        an, ad, au, ai = load_hc(self.w, il, "attn")
        mixed, inject = hc_mix(x, an, ad, au, ai, eps, hc)

        if hp.is_recr(il):
            dn_w = load_delta_net_weights(self.w, il)
            dstate = state.delta if state.delta is not None else delta_net_init_state(hp)
            block_out, new_delta = linear_attn_layer(mixed, dn_w, dstate, hp)
            new_attn = None
        else:
            attn_w = load_attn_weights(self.w, il)
            if state.attn is not None:
                k_cache, v_cache, cache_pos = state.attn
            else:
                k_cache = v_cache = cache_pos = None
            block_out, k_new, v_new = attention_full(mixed, attn_w, positions, hp, k_cache, v_cache, cache_pos)
            if k_cache is not None:
                new_attn = (
                    torch.cat([k_cache, k_new], dim=0),
                    torch.cat([v_cache, v_new], dim=0),
                    torch.cat([cache_pos, positions], dim=0),
                )
            else:
                new_attn = (k_new, v_new, positions)
            new_delta = None

        x = hc_combine(x, block_out, inject, hc)

        routing_used = None
        if need_ffn_output:
            fn, ad2, au2, ai2 = load_hc(self.w, il, "ffn")
            mixed2, inject2 = hc_mix(x, fn, ad2, au2, ai2, eps, hc)
            moe_w = load_moe_nonexpert(self.w, il)
            routing_used = self._routing_for(il, positions, mixed2, moe_w.gate_inp, routing_source, diag)
            gate_fn, up_fn, down_fn = self._expert_fns(il, persist=persist_experts)
            ffn_out = moe_ffn(
                mixed2, moe_w.gate_inp, gate_fn, up_fn, down_fn, routing_used,
                moe_w.up_shexp, moe_w.gate_shexp, moe_w.down_shexp, moe_w.gate_inp_shexp,
                hp.n_expert_used,
            )
            x = hc_combine(x, ffn_out, inject2, hc)

        return x, LayerState(attn=new_attn, delta=new_delta, ple_hist=new_ple_hist), routing_used

    # -- prefisso -------------------------------------------------------------

    def prefix(
        self,
        tokens: list[int],
        routing_source: dict[int, np.ndarray] | None = None,
        capture_routing: dict[int, np.ndarray] | None = None,
        diag: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        overlay: dict[int, np.ndarray] | None = None,
    ) -> PrefixState:
        """Full forward (no gradient) over positions 0..T-2.

        If `capture_routing` is given (an empty dict passed by the caller), it
        is populated with the routing actually used at each layer (0..n_layer-2,
        the last layer consumes no routing in the prefix): used for Q1b.

        `overlay` ({global_row: vector [160]}) is forwarded to `ple_true_emb`
        for every prefix position -- grafts from earlier chain positions
        (already-descended rows) replace the true gather where present.
        `overlay=None` (default) is identical to the prior behavior, bit for
        bit."""
        n_prefix = len(tokens) - 1
        layers: dict[int, LayerState] = {il: LayerState() for il in range(self.hp.n_layer)}
        if n_prefix <= 0:
            return PrefixState(layers=layers, n_prefix=0)

        with torch.no_grad():
            emb = self.embed(tokens[:n_prefix])  # [n_prefix, n_embd]
            x = emb.unsqueeze(1).repeat(1, self.hp.hc_mult, 1)  # [T,hc,n_embd]
            positions = torch.arange(n_prefix, dtype=torch.float64)

            for il in range(self.hp.n_layer):
                ple_emb = None
                if self.hp.is_ple(il):
                    rows = [self.ple_true_emb(tokens, t, overlay=overlay) for t in range(n_prefix)]
                    ple_emb = torch.cat(rows, dim=0)  # [n_prefix, n_embd]
                need_ffn = not (il == self.hp.n_layer - 1)
                x, layers[il], routing_used = self.run_layer(
                    il, x, positions, layers[il], routing_source, ple_emb, need_ffn_output=need_ffn, diag=diag
                )
                if capture_routing is not None and routing_used is not None:
                    capture_routing[il] = routing_used.numpy()

        return PrefixState(layers=layers, n_prefix=n_prefix)

    # -- last position (differentiable in `rows`) ------------------------------

    def last_step(
        self,
        tokens: list[int],
        state: PrefixState,
        rows: torch.Tensor,  # [16,160], requires_grad when the gradient is needed
        routing_source: dict[int, np.ndarray] | None = None,
        persist_experts: bool = True,
        capture_routing: dict[int, np.ndarray] | None = None,
        diag: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> torch.Tensor:
        """Returns the [n_vocab] logits of position T-1, differentiable in `rows`."""
        hp = self.hp
        t_last = len(tokens) - 1
        emb_last = self.embed([tokens[t_last]])  # [1,n_embd], fisso
        x = emb_last.unsqueeze(1).repeat(1, hp.hc_mult, 1)  # [1,hc,n_embd]
        positions = torch.tensor([float(t_last)], dtype=torch.float64)
        rows_emb = rows.reshape(1, -1)  # [1, n_heads*head_dim] = [1,n_embd]

        for il in range(hp.n_layer):
            ple_emb = rows_emb if hp.is_ple(il) else None
            x, _, routing_used = self.run_layer(
                il, x, positions, state.layers[il], routing_source, ple_emb,
                need_ffn_output=True, persist_experts=persist_experts, diag=diag,
            )
            if capture_routing is not None and routing_used is not None:
                capture_routing[il] = routing_used.numpy()

        hc_norm = torch.from_numpy(self.w.tensor("output_hc_norm.weight"))
        hc_down = torch.from_numpy(self.w.tensor("output_hc_down.weight"))
        hc_up = torch.from_numpy(self.w.tensor("output_hc_up.weight"))
        mixed, _ = hc_mix(x, hc_norm, hc_down, hc_up, None, hp.f_norm_rms_eps, hp.hc_mult)  # [1,n_embd]

        output_w = torch.from_numpy(self.w.tensor("output.weight"))  # [n_vocab,n_embd]
        logits = mixed @ output_w.T  # [1, n_vocab]
        return logits[0]

    def routing_free_full(
        self, tokens: list[int], rows: torch.Tensor
    ) -> tuple[dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]]]:
        """Fully free-routing run over prefix + last position, captured live at
        every layer (Q1b: layers 0-46 all positions, layer 47 only the last).
        Reruns prefix() from scratch (routing_source=None): never reuses state
        computed with a different routing.

        Returns (full_routing, diag_prefix, diag_last_position); the two diag
        dicts are {il: (top11_idx, top11_val)}, used for the Q1b relative
        margin on disagreeing cases."""
        prefix_captured: dict[int, np.ndarray] = {}
        last_captured: dict[int, np.ndarray] = {}
        diag_prefix: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        diag_last: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with torch.no_grad():
            state = self.prefix(
                tokens, routing_source=None, capture_routing=prefix_captured, diag=diag_prefix
            )
            self.last_step(
                tokens, state, rows, routing_source=None, persist_experts=False,
                capture_routing=last_captured, diag=diag_last,
            )
        captured: dict[int, np.ndarray] = {}
        for il in range(self.hp.n_layer):
            if il in prefix_captured and il in last_captured:
                captured[il] = np.concatenate([prefix_captured[il], last_captured[il]], axis=0)
            elif il in last_captured:
                captured[il] = last_captured[il]  # layer n_layer-1: only the last position
            elif il in prefix_captured:
                captured[il] = prefix_captured[il]
        return captured, diag_prefix, diag_last

    def logits_full(self, tokens: list[int], rows: torch.Tensor, routing_source=None) -> torch.Tensor:
        state = self.prefix(tokens, routing_source)
        return self.last_step(tokens, state, rows, routing_source)


# --------------------------------------------------------------------------
# Precondizione e memoria (T3)
# --------------------------------------------------------------------------


def check_precondition(table: PleTable, prompts: dict[str, list[int]], rows_global: np.ndarray) -> dict:
    """Checks that `rows_global` (the 16 trigger rows) do not appear in the prefix
    (positions 0..T-2) of any of the given prompts. Returns a dict ready for
    precondition.json."""
    rows_set = set(int(r) for r in rows_global)
    hits = []
    for name, tokens in prompts.items():
        addr = table.ngram_addresses(tokens)
        t_len = len(tokens)
        for t in range(t_len - 1):  # prefisso: esclude l'ultima posizione
            for h in range(table.n_heads):
                g = local_to_global(table, h, addr[t][h])
                if g in rows_set:
                    hits.append({"prompt": name, "t": t, "head": h, "row_global": g})
    return {
        "rows_global": [int(r) for r in rows_global],
        "prompts": {k: v for k, v in prompts.items()},
        "hits": hits,
        "ok": len(hits) == 0,
    }


def check_memory(ceiling_bytes: int, margin_bytes: int) -> dict:
    meminfo = Path("/proc/meminfo").read_text()
    avail_kb = None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
            break
    if avail_kb is None:
        raise RuntimeError("MemAvailable not found in /proc/meminfo")
    avail_bytes = avail_kb * 1024
    ok = avail_bytes >= ceiling_bytes + margin_bytes
    return {
        "mem_available_bytes": avail_bytes,
        "ceiling_bytes": ceiling_bytes,
        "margin_bytes": margin_bytes,
        "ok": ok,
    }

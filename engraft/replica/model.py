"""Replica orchestration: constant prefix, differentiable last position, and a
differentiable multi-position prefix forward used by the corpus-level descent
(`prefix(grad_proxy=True, return_logits=True)`).

`Replica.prefix(tokens)` runs a full forward (no gradient by default) over
positions 0..T-2, producing for every layer the state `last_step` needs for
position T-1: K/V cache (attention layers), (conv history, recurrence state)
(delta net layers), PLE conv history (layer 1). Routing is imposed by
`routing_source` when given (a per-layer array of imposed expert indices),
otherwise computed live (softmax + top-k over the router logits). At the
final layer (`n_layer-1`) the prefix skips the MoE computation unless
`return_logits=True` (which needs the FFN output at every layer, including
the last, to project to vocabulary logits).

`Replica.last_step(state, rows, routing_source)` runs the whole layer stack
for just position T-1, substituting the given `rows` (differentiable when
they require a gradient) for the PLE table gather at layer `ple_layer`.

`Replica.prefix(grad_proxy=True, return_logits=True, base_state=..., cache=...,
positions=...)` is the differentiable, logit-producing, incrementally
extensible forward the corpus-level descent (`engraft.replica.seq`) uses:
`grad_proxy` keeps the graph instead of running under `torch.no_grad()`,
`return_logits` projects every new position to vocabulary logits (not just
the last one), `base_state`/`cache` extend an already-computed prefix instead
of recomputing it from scratch, and `positions` lets a packed multi-fragment
batch (`engraft.replica.pack`) override the default `arange` RoPE positions.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.profiler import record_function

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


def _as_tensor(x, device=None, dtype=None) -> torch.Tensor:
    """`w.tensor(name)`/`w.expert(name,e)` return either a numpy array (CPU,
    the plain `GgufWeights` store) or an already-resident torch tensor
    (`DeviceIQ4`): this is the one place `model.py` decides how to wrap them,
    so the loaders below stay agnostic to which store is in use. `device`/
    `dtype` (default `None`): when given, the tensor is moved there with
    `.to()` -- this is the single choke point through which every weight
    enters the active phase's dtype. With `device=None, dtype=None` (default)
    the behavior is unchanged: `torch.from_numpy` for `GgufWeights`,
    passthrough for `DeviceIQ4`."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    if kwargs:
        t = t.to(**kwargs)
    return t


def _fetch_dense(w: GgufWeights, name: str, device, dtype) -> torch.Tensor:
    """Choke point for the five per-layer loaders (`load_attn_weights`,
    `load_delta_net_weights`, `load_ple_weights`, `load_hc`,
    `load_moe_nonexpert`). Each of them calls `_as_tensor(w.tensor(name),
    device, dtype)`: `w.tensor(name)` always dequantizes in `self.dtype`
    (f32) and `_as_tensor` then COPIES to `dtype` at every call -- even at a
    warm dense cache, because `tensor()` returns the f32 copy, not the cast
    one. With `DeviceIQ4.tensor_as`, the cast copy is the one that enters the
    cache: used here only when `dtype` is given and `w` exposes it
    (`GgufWeights`, CPU/numpy, does not: unchanged behavior). `_as_tensor`
    afterwards is a no-op when `t` is already on the right device in the
    right dtype (same object, no extra copy)."""
    if dtype is not None and hasattr(w, "tensor_as"):
        t = w.tensor_as(name, dtype)
    else:
        t = w.tensor(name)
    return _as_tensor(t, device, dtype)


_HEAD_DTYPE_BY_NAME = {"f32": torch.float32, "bf16": torch.bfloat16}


def head_dtype_from_name(name: str) -> torch.dtype:
    """Converts a configuration name (CLI `--head-dtype`, manifest
    `head_dtype`/`head_out_dtype`) to the `torch.dtype` of `Replica.head_dtype`.
    Closed set `{f32, bf16}`, narrower than `backend.PhaseDtypes` (no int8 in
    the head)."""
    td = _HEAD_DTYPE_BY_NAME.get(name)
    if td is None:
        raise ValueError(f"head_dtype={name!r} not recognized (valid: {sorted(_HEAD_DTYPE_BY_NAME)})")
    return td


_HEAD_DTYPE_NAME_BY_TORCH = {v: k for k, v in _HEAD_DTYPE_BY_NAME.items()}


def head_dtype_to_name(dtype: torch.dtype) -> str:
    """Inverse of `head_dtype_from_name`: used to write `head_dtype`/
    `head_out_dtype` as strings in a run manifest, never as `torch.dtype`
    (not JSON-serializable without `default=str`, which would give
    'torch.float32' instead of 'f32')."""
    name = _HEAD_DTYPE_NAME_BY_TORCH.get(dtype)
    if name is None:
        raise ValueError(f"head_dtype_to_name: dtype {dtype!r} not among the known names {sorted(_HEAD_DTYPE_BY_NAME)}")
    return name


def probe_mm_out_dtype(device) -> bool:
    """A device-side probe to decide whether `torch.mm(a_bf16, b_bf16,
    out_dtype=torch.float32)` is usable in place of the default explicit
    cast (`(mixed.to(bf16) @ W.T).to(f32)`). Without an explicit cast on the
    output, the `ToCopyBackward` node that today guarantees a bf16/WMMA dgrad
    does not exist: this must VERIFY that `out_dtype` does not fall back to
    an f32 GEMM anyway. Three checks, all required: (1) the forward is
    accepted; (2) `.sum().backward()` succeeds AND the inputs' gradient stays
    bf16 (not promoted to f32); (3) a GEMM profile shows no f32 rocBLAS
    kernel on the dgrad. Returns `True` only if all three pass; `False`
    otherwise (including `TypeError` if `out_dtype` is not a recognized
    argument in this torch/ROCm build).

    Never called on CPU: CPU does not go through the same kernel path as a
    real accelerator, so check (3) is meaningless and (2) would not be
    representative -- raises immediately, no silent attempt."""
    dev = torch.device(device)
    if dev.type == "cpu":
        raise RuntimeError(
            "probe_mm_out_dtype: never on CPU -- requires a real CUDA/ROCm "
            "device to verify the dgrad kernel"
        )
    a = torch.randn(8, 16, dtype=torch.bfloat16, device=dev, requires_grad=True)
    b = torch.randn(16, 8, dtype=torch.bfloat16, device=dev, requires_grad=True)
    try:
        out = torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return False
    if out.dtype != torch.float32:
        return False
    try:
        out.sum().backward()
    except RuntimeError:
        return False
    if a.grad is None or a.grad.dtype != torch.bfloat16:
        return False
    with torch.profiler.profile() as prof:
        torch.mm(a.detach(), b.detach(), out_dtype=torch.float32)
    kernel_names = " ".join(e.key.lower() for e in prof.key_averages())
    if "sb_mt" in kernel_names or ("rocblas" in kernel_names and "wmma" not in kernel_names):
        return False
    return True


# --------------------------------------------------------------------------
# Loading weights per layer
# --------------------------------------------------------------------------


def load_attn_weights(w: GgufWeights, il: int, device=None, dtype=None) -> AttnWeights:
    p = f"blk.{il}."
    with record_function("replica/load_attn"):
        return AttnWeights(
            wq=_fetch_dense(w, p + "attn_q.weight", device, dtype),
            wk=_fetch_dense(w, p + "attn_k.weight", device, dtype),
            wv=_fetch_dense(w, p + "attn_v.weight", device, dtype),
            wo=_fetch_dense(w, p + "attn_output.weight", device, dtype),
            q_norm=_fetch_dense(w, p + "attn_q_norm.weight", device, dtype),
            k_norm=_fetch_dense(w, p + "attn_k_norm.weight", device, dtype),
        )


def load_delta_net_weights(w: GgufWeights, il: int, device=None, dtype=None) -> DeltaNetWeights:
    p = f"blk.{il}."
    with record_function("replica/load_delta_net"):
        return DeltaNetWeights(
            wqkv=_fetch_dense(w, p + "attn_qkv.weight", device, dtype),
            wqkv_gate=_fetch_dense(w, p + "attn_gate.weight", device, dtype),
            ssm_conv1d=_fetch_dense(w, p + "ssm_conv1d.weight", device, dtype),
            ssm_dt_bias=_fetch_dense(w, p + "ssm_dt.bias", device, dtype),
            ssm_a=_fetch_dense(w, p + "ssm_a", device, dtype),
            ssm_beta=_fetch_dense(w, p + "ssm_beta.weight", device, dtype),
            ssm_alpha=_fetch_dense(w, p + "ssm_alpha.weight", device, dtype),
            ssm_norm=_fetch_dense(w, p + "ssm_norm.weight", device, dtype),
            ssm_out=_fetch_dense(w, p + "ssm_out.weight", device, dtype),
        )


def load_ple_weights(w: GgufWeights, il: int, device=None, dtype=None) -> PleWeights:
    p = f"blk.{il}."
    with record_function("replica/load_ple"):
        return PleWeights(
            w_key=_fetch_dense(w, p + "ple_key.weight", device, dtype),
            w_value=_fetch_dense(w, p + "ple_value.weight", device, dtype),
            norm_key=_fetch_dense(w, p + "ple_norm_key.weight", device, dtype),
            norm_query=_fetch_dense(w, p + "ple_norm_query.weight", device, dtype),
            norm_conv=_fetch_dense(w, p + "ple_norm_conv.weight", device, dtype),
            conv1d=_fetch_dense(w, p + "ple_conv1d.weight", device, dtype),
        )


def load_hc(w: GgufWeights, il: int, slot: str, device=None, dtype=None):
    p = f"blk.{il}.hc_{slot}_"
    with record_function("replica/load_hc"):
        norm = _fetch_dense(w, p + "norm.weight", device, dtype)
        down = _fetch_dense(w, p + "down.weight", device, dtype)
        up = _fetch_dense(w, p + "up.weight", device, dtype)
        inject = _fetch_dense(w, p + "inject.weight", device, dtype)
        return norm, down, up, inject


@dataclasses.dataclass
class MoeNonExpertWeights:
    gate_inp: torch.Tensor
    gate_inp_shexp: torch.Tensor
    up_shexp: torch.Tensor
    gate_shexp: torch.Tensor
    down_shexp: torch.Tensor


def load_moe_nonexpert(w: GgufWeights, il: int, device=None, dtype=None) -> MoeNonExpertWeights:
    p = f"blk.{il}."
    with record_function("replica/load_moe_nonexpert"):
        return MoeNonExpertWeights(
            gate_inp=_fetch_dense(w, p + "ffn_gate_inp.weight", device, dtype),
            gate_inp_shexp=_fetch_dense(w, p + "ffn_gate_inp_shexp.weight", device, dtype),
            up_shexp=_fetch_dense(w, p + "ffn_up_shexp.weight", device, dtype),
            gate_shexp=_fetch_dense(w, p + "ffn_gate_shexp.weight", device, dtype),
            down_shexp=_fetch_dense(w, p + "ffn_down_shexp.weight", device, dtype),
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
    n_prefix: int  # T-1 (positions already consumed)
    # Additive, default None: identical behavior when not requested.
    # `emb_leaf`: the new positions' input embedding, made a leaf with
    # `requires_grad_(True)` when `Replica.prefix(grad_proxy=True)` -- used
    # only to measure the cost of a backward pass "at every position" (a
    # proxy, not the scientific gradient with respect to the PLE rows).
    # `logits`: [n_new, n_vocab] logits of every new position, populated only
    # with `Replica.prefix(return_logits=True)` (the same final projection as
    # `last_step`, applied to every position instead of only the last one).
    emb_leaf: torch.Tensor | None = None
    logits: torch.Tensor | None = None


# --------------------------------------------------------------------------
# PrefixCache
# --------------------------------------------------------------------------


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _prefix_state_nbytes(state: PrefixState) -> int:
    """Byte estimate of the per-layer state (feeds a memory-guard budget) --
    sum of the live tensors (K/V, delta-net state/history, PLE conv
    history)."""
    total = 0
    for ls in state.layers.values():
        if ls.attn is not None:
            k, v, pos = ls.attn
            total += _tensor_nbytes(k) + _tensor_nbytes(v) + _tensor_nbytes(pos)
        if ls.delta is not None:
            total += _tensor_nbytes(ls.delta.conv_hist) + _tensor_nbytes(ls.delta.s)
        if ls.ple_hist is not None:
            total += _tensor_nbytes(ls.ple_hist)
    return total


def _overlay_dicts_equal(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> bool:
    """Value comparison (never a direct `==`/`!=` on a dict with numpy
    values: that raises on the array's ambiguous truth value)."""
    if set(a.keys()) != set(b.keys()):
        return False
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def _rows_read_in_prefix(table, tokens: list[int]) -> set[int]:
    """**Every** global row read by positions 0..len(tokens)-2 of `tokens`.
    Deliberately **not** filtered against a candidate row set (unlike
    `check_precondition`, which looks for "hits" against a given target): the
    cache key must notice a future overlay on any row read by the prefix,
    not only rows already known at `put` time -- filtering by the keys of an
    overlay map taken at one instant would miss rows not yet overlaid at that
    moment (a correctness bug, not the intended optimization; the table is
    small by construction: 16 rows per position, so correctness of the key
    is the goal here, not speed)."""
    t_len = len(tokens)
    addr = table.ngram_addresses(tokens)
    out: set[int] = set()
    for t in range(t_len - 1):
        for h in range(table.n_heads):
            out.add(local_to_global(table, h, addr[t][h]))
    return out


class PrefixCache:
    """RAM cache of `PrefixState`: key = **consumed** tokens (`tokens[:-1]`,
    positions 0..n_prefix-1) + overlay **restricted to the rows read** at
    those positions; lookup by **longest matching prefix** (a longer but
    non-identical key does not hit -- it extends the closest matching base).
    Every entry also carries `cap_prefix` (captured routing), because
    `graft.prepare_graft` reconstructs `routing_trigger` from it. Byte cap:
    LRU eviction (never the last entry, so a single `put` larger than the cap
    does not empty the cache)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._entries: list[dict] = []
        self._used = 0

    def lookup(
        self, tokens_consumed: list[int], overlay_map: dict[int, np.ndarray],
    ) -> tuple[PrefixState, dict[int, np.ndarray]] | None:
        tokens_t = tuple(tokens_consumed)
        best = None
        best_len = -1
        for e in self._entries:
            et = e["tokens"]
            if len(et) <= best_len or len(et) > len(tokens_t):
                continue
            if tokens_t[: len(et)] != et:
                continue
            restricted = {r: overlay_map[r] for r in e["rows_read"] if r in overlay_map}
            if not _overlay_dicts_equal(restricted, e["overlay_restricted"]):
                continue
            best, best_len = e, len(et)
        if best is None:
            return None
        self._touch(best)
        return best["state"], best["cap"]

    def put(
        self, tokens_consumed: list[int], overlay_map: dict[int, np.ndarray],
        state: PrefixState, cap: dict[int, np.ndarray], table,
    ) -> None:
        rows_read = _rows_read_in_prefix(table, list(tokens_consumed) + [0])
        restricted = {r: overlay_map[r] for r in rows_read if r in overlay_map}
        nbytes = _prefix_state_nbytes(state)
        entry = {
            "tokens": tuple(tokens_consumed), "overlay_restricted": restricted,
            "rows_read": rows_read, "state": state, "cap": cap, "nbytes": nbytes,
        }
        self._entries.append(entry)
        self._used += nbytes
        self._evict_if_needed()

    def _touch(self, entry: dict) -> None:
        self._entries.remove(entry)
        self._entries.append(entry)

    def _evict_if_needed(self) -> None:
        while self._used > self.max_bytes and len(self._entries) > 1:
            oldest = self._entries.pop(0)
            self._used -= oldest["nbytes"]

    @property
    def used_bytes(self) -> int:
        return self._used


# --------------------------------------------------------------------------
# Replica
# --------------------------------------------------------------------------


class Replica:
    def __init__(self, hp: Hparams, w: GgufWeights, table: PleTable, backend=None,
                 head_dtype: torch.dtype = torch.float32):
        """`backend`: additive, default `Backend.cpu_f32()` (imported locally
        to avoid a circular dependency -- `backend.py` does not import
        `model.py` -- and so this class stays importable in tests that never
        touch a device). With the default, behavior is unchanged, bit for
        bit: weight materialization on a device (`weight_store="device_iq4"`)
        and casting to `backend.dtypes`'s active phase inside `embed`/
        `_expert_fns`/`run_layer` only matter once a non-default `Backend` is
        used.

        `head_dtype`: governs ONLY the final projection
        (`prefix(return_logits=True)`/`last_step`, never `backend.dtypes`).
        Default `torch.float32`, identical to the unconditional-F32 behavior
        of a plain construction. With `head_dtype=torch.bfloat16`:
        `mixed.to(head_dtype) @ W_head.T` (explicit cast on the input, never
        `out_dtype` without a device-side probe) with `.to(torch.float32)` on
        the output -- `state.logits` stays ALWAYS F32."""
        if backend is None:
            from engraft.replica import backend as _backend_mod
            backend = _backend_mod.Backend.cpu_f32()
        self._validate_device(backend.device)
        self.hp = hp
        self.w = w
        self.table = table
        self.backend = backend
        self.head_dtype = head_dtype
        self.head_out_dtype = head_dtype  # recorded/updated by _head_weight()
        self._head_weight_cache: tuple[torch.dtype, torch.Tensor] | None = None

    @staticmethod
    def _validate_device(device_str: str) -> None:
        """An unavailable device raises here, with a message naming the
        requested device -- never a silent fallback to CPU. Checked once at
        construction instead of letting a `.to()` fail deep inside a layer
        with an opaque message."""
        dev = torch.device(device_str)
        if dev.type == "cpu":
            return
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Replica: requested device {device_str!r} is not available "
                    "(torch.cuda.is_available()==False in this environment): "
                    "no silent fallback to CPU"
                )
            if dev.index is not None and dev.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Replica: requested device {device_str!r} out of index "
                    f"(torch.cuda.device_count()={torch.cuda.device_count()})"
                )
            return
        raise RuntimeError(
            f"Replica: requested device {device_str!r} not recognized (valid: 'cpu', 'cuda:N')"
        )

    def embed(self, tokens: list[int], device=None, dtype=None) -> torch.Tensor:
        """`device`/`dtype`: with both `None` (default), behavior is
        unchanged."""
        tok_embd = self.w.tensor("token_embd.weight")  # [n_vocab, n_embd]
        if isinstance(tok_embd, np.ndarray):
            idx = np.asarray(tokens, dtype=np.int64)
            t = torch.from_numpy(tok_embd[idx].copy())  # [T, n_embd]
        else:
            idx_t = torch.as_tensor(tokens, dtype=torch.int64, device=tok_embd.device)
            t = tok_embd[idx_t].clone()  # [T, n_embd] -- weight store already on device
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return t.to(**kwargs) if kwargs else t

    def ple_true_emb(
        self, tokens: list[int], t: int, overlay: dict[int, np.ndarray] | None = None,
    ) -> torch.Tensor:
        """emb [1, n_embd] from the real table gather for position `t` of
        `tokens`.

        If `overlay` is given (`{global_row: vector [160]}`), the position's
        global rows present in the overlay replace the real gather (a graft
        from an earlier chain position, read from the prefix); rows absent
        from the overlay stay the real gather. `overlay=None` (default) is
        identical to the plain gather. The PLE table is always read from
        `PleTable` (CPU): this tensor always comes back on CPU regardless of
        the backend -- the caller moves it to the active device together
        with the rest of `x`."""
        rs = RowSet.from_position(self.table, tokens, t)
        if not overlay:
            return torch.from_numpy(rs.data.reshape(1, -1).copy())  # [1, n_heads*head_dim] = [1,n_embd]
        data = rs.data.copy()
        for h in range(rs.rows_global.shape[0]):
            row_g = int(rs.rows_global[h])
            if row_g in overlay:
                data[h] = np.asarray(overlay[row_g], dtype=np.float32)
        return torch.from_numpy(data.reshape(1, -1).copy())

    def _head_weight(self, to_device) -> torch.Tensor:
        """`output.weight` in the current `head_dtype`: cast ONCE and kept
        resident on this instance, invalidated only if `head_dtype` changes.
        Not the same as a weight store's own dequantized-tensor cache (which
        only knows "has this been dequantized already?" and always
        dequantizes in F32) -- a second level is needed here to distinguish
        F32/bf16 under the same name.

        The first access below always goes through `self.w.tensor("output.weight")`,
        which with `DeviceIQ4` materializes (and caches) an F32 copy. Once
        `head_dtype` is bf16 that F32 copy is no longer useful to anyone (the
        head is never read elsewhere): explicitly evicted here, otherwise F32
        and bf16 would coexist wastefully. `hasattr` guards `GgufWeights`
        (no cache, no `evict_dense` attribute)."""
        if self._head_weight_cache is not None and self._head_weight_cache[0] == self.head_dtype:
            return self._head_weight_cache[1]
        w_head = _as_tensor(self.w.tensor("output.weight"), to_device, self.head_dtype)
        self._head_weight_cache = (self.head_dtype, w_head)
        self.head_out_dtype = self.head_dtype
        if self.head_dtype != torch.float32 and hasattr(self.w, "evict_dense"):
            self.w.evict_dense("output.weight")
        return w_head

    def final_logits(self, x: torch.Tensor, to_device) -> torch.Tensor:
        """The `return_logits` projection block of `prefix()`, factored out:
        same final projection as `last_step` (`hc_mix` + `output.weight`,
        fixed F32 island), applicable to ANY residual `[N,hc,n_embd]` -- not
        only the current prefix's last layer output, so a diagnostic lens
        could project an intermediate layer's residual onto the same head
        (never calibrated for that use).

        Contract: `x` is EXACTLY 3D `[N,hc,n_embd]` -- checked explicitly
        here, with a message naming the received shape, rather than letting
        `hc_mix` raise an unreadable unpacking error. The method moves `x` to
        the requested device/dtype itself: the caller does not need to cast
        first. Always returns F32 (`state.logits` stays F32 even with
        `head_dtype=bf16` -- only the head GEMM's input accepts
        `head_dtype`, not its output)."""
        if x.dim() != 3:
            raise ValueError(
                f"final_logits: expected x 3D [N,hc,n_embd], got shape {tuple(x.shape)}"
            )
        x = x.to(to_device, torch.float32)
        hc_norm = _as_tensor(self.w.tensor("output_hc_norm.weight"), to_device, torch.float32)
        hc_down = _as_tensor(self.w.tensor("output_hc_down.weight"), to_device, torch.float32)
        hc_up = _as_tensor(self.w.tensor("output_hc_up.weight"), to_device, torch.float32)
        mixed_out, _ = hc_mix(x, hc_norm, hc_down, hc_up, None, self.hp.f_norm_rms_eps, self.hp.hc_mult)
        output_w = self._head_weight(to_device)
        # Explicit cast on the input (never `out_dtype` without a device-side
        # probe): with `head_dtype=f32` (default) this is a bit-for-bit no-op.
        return (mixed_out.to(self.head_dtype) @ output_w.T).to(torch.float32)

    def _expert_fns(self, il: int, persist: bool, device=None, dtype=None):
        p = f"blk.{il}."

        def gate_fn(e: int) -> torch.Tensor:
            return _as_tensor(self.w.expert(p + "ffn_gate_exps.weight", e, persist=persist), device, dtype)

        def up_fn(e: int) -> torch.Tensor:
            return _as_tensor(self.w.expert(p + "ffn_up_exps.weight", e, persist=persist), device, dtype)

        def down_fn(e: int) -> torch.Tensor:
            return _as_tensor(self.w.expert(p + "ffn_down_exps.weight", e, persist=persist), device, dtype)

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
        """`diag`, if given, receives {il: (top11_idx[T,11], top11_val[T,11])}
        when routing is computed live (for a relative-margin diagnostic:
        10th chosen expert vs 11th excluded, without a second forward)."""
        if routing_source is not None and il in routing_source:
            arr = routing_source[il]  # [T_tot,10] (0..46) or [1,10] last position only (47)
            pos_idx = positions.detach().to("cpu", torch.int64).numpy()
            if arr.shape[0] == 1:
                # Last layer: the file records only the last position (row
                # 0), the prefix consumes no routing there -- only a
                # last_step call, with a single position, should reach here.
                if len(pos_idx) != 1:
                    raise ValueError(
                        f"routing_source[{il}] has a single row (last position only) "
                        f"but {len(pos_idx)} positions were requested"
                    )
                sel = arr[0:1]
            else:
                sel = arr[pos_idx]
            # The routing index goes back to `x`'s device (moe_ffn's gather/
            # torch.gather requires index and source on the same device).
            return torch.from_numpy(np.asarray(sel, dtype=np.int64)).to(x.device)
        # Fixed F32 island: router logits, softmax and top-k computed in F32
        # regardless of `x`/`gate_inp`'s phase dtype -- with an F32 backend
        # (default) the casts are no-ops, bit for bit.
        logits = x.to(torch.float32) @ gate_inp.to(torch.float32).T
        probs = torch.softmax(logits, dim=-1)
        k = self.hp.n_expert_used
        # Deterministic tie-break: at an exactly equal score, the smaller
        # index wins -- `torch.argsort(..., stable=True)` on a single full
        # sort (never a sub-resolution epsilon, which would change the
        # values) -- the same order serves both `k` and `k+1` columns, so the
        # diag and non-diag branches stay bit-identical to each other.
        order = torch.argsort(probs, dim=-1, descending=True, stable=True)
        if diag is not None:
            top_idx = order[:, : k + 1]
            top_val = torch.gather(probs, 1, top_idx)
            diag[il] = (top_idx.detach().to("cpu").numpy(), top_val.detach().to("cpu").numpy())
            return top_idx[:, :k]
        return order[:, :k]

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
        ple_diag: dict | None = None,
        detach_value: bool = False,
        detach_gate: bool = False,
    ) -> tuple[torch.Tensor, LayerState, torch.Tensor | None]:
        """`ple_diag`/`detach_value`/`detach_gate` are forwarded only to the
        PLE layer (ignored elsewhere) -- default unchanged, identical
        behavior.

        A finer-grained checkpoint that only recomputes part of a layer's
        computation in the backward pass (e.g. only the MoE region, or only
        everything before it) is a memory-tuning variant of the whole-layer
        checkpoint that `engraft.replica.seq.run_layer_checkpointed` already
        provides externally (wrapping this whole method in
        `torch.utils.checkpoint`); it is not part of this public reference."""
        hp = self.hp
        eps = hp.f_norm_rms_eps
        hc = hp.hc_mult
        # The active phase's device/dtype follow `x` (already moved there by
        # the caller, `prefix()`/`last_step()`, per `self.backend.dtypes`) --
        # every weight loaded below enters the same dtype/device, so the
        # graph never mixes precisions between one matmul and the next.
        dev, dt = x.device, x.dtype

        new_ple_hist = state.ple_hist

        if hp.is_ple(il):
            ple_w = load_ple_weights(self.w, il, device=dev, dtype=dt)
            if state.ple_hist is not None:
                hist = state.ple_hist.to(device=dev, dtype=dt)
            else:
                hist = torch.zeros(0, hc, hp.n_embd, device=dev, dtype=dt)
            # `ple_emb` may already arrive in the phase's dtype/device (the
            # caller moves it there before this call); the cast here is
            # defensive, a no-op if already aligned.
            with record_function("replica/ple_forward"):
                x, new_ple_hist = ple_forward(
                    ple_emb.to(device=dev, dtype=dt), x, ple_w, hist, hp, diag=ple_diag,
                    detach_value=detach_value, detach_gate=detach_gate,
                )

        an, ad, au, ai = load_hc(self.w, il, "attn", device=dev, dtype=dt)
        with record_function("replica/hc_mix"):
            mixed, inject = hc_mix(x, an, ad, au, ai, eps, hc)

        if hp.is_recr(il):
            dn_w = load_delta_net_weights(self.w, il, device=dev, dtype=dt)
            if state.delta is not None:
                # Cross-phase: state carried from a previous phase follows
                # the current phase for `conv_hist` (an activation, not an
                # island); `s` (recurrent state) stays fixed F32, only the
                # device follows.
                dstate = DeltaNetState(
                    conv_hist=state.delta.conv_hist.to(device=dev, dtype=dt),
                    s=state.delta.s.to(device=dev, dtype=torch.float32),
                )
            else:
                dstate = delta_net_init_state(hp, device=dev, dtype=dt)
            with record_function("replica/linear_attn_layer"):
                block_out, new_delta = linear_attn_layer(mixed, dn_w, dstate, hp)
            new_attn = None
        else:
            attn_w = load_attn_weights(self.w, il, device=dev, dtype=dt)
            if state.attn is not None:
                k_cache_raw, v_cache_raw, cache_pos_raw = state.attn
                k_cache = k_cache_raw.to(device=dev, dtype=dt)
                v_cache = v_cache_raw.to(device=dev, dtype=dt)
                cache_pos = cache_pos_raw.to(device=dev)
            else:
                k_cache = v_cache = cache_pos = None
            with record_function("replica/attention_full"):
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

        with record_function("replica/hc_combine"):
            x = hc_combine(x, block_out, inject, hc)

        mixed2 = inject2 = None
        if need_ffn_output:
            fn, ad2, au2, ai2 = load_hc(self.w, il, "ffn", device=dev, dtype=dt)
            with record_function("replica/hc_mix"):
                mixed2, inject2 = hc_mix(x, fn, ad2, au2, ai2, eps, hc)

        routing_used = None
        if need_ffn_output:
            moe_w = load_moe_nonexpert(self.w, il, device=dev, dtype=dt)
            routing_used = self._routing_for(il, positions, mixed2, moe_w.gate_inp, routing_source, diag)
            gate_fn, up_fn, down_fn = self._expert_fns(il, persist=persist_experts, device=dev, dtype=dt)
            with record_function("replica/moe_ffn"):
                ffn_out = moe_ffn(
                    mixed2, moe_w.gate_inp, gate_fn, up_fn, down_fn, routing_used,
                    moe_w.up_shexp, moe_w.gate_shexp, moe_w.down_shexp, moe_w.gate_inp_shexp,
                    hp.n_expert_used, grouped=self.backend.moe_grouped,
                )
            with record_function("replica/hc_combine"):
                x = hc_combine(x, ffn_out, inject2, hc)

        return x, LayerState(attn=new_attn, delta=new_delta, ple_hist=new_ple_hist), routing_used

    # -- prefix -----------------------------------------------------------

    def prefix(
        self,
        tokens: list[int],
        routing_source: dict[int, np.ndarray] | None = None,
        capture_routing: dict[int, np.ndarray] | None = None,
        diag: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        overlay: dict[int, np.ndarray] | None = None,
        *,
        base_state: "PrefixState | None" = None,
        cache: "PrefixCache | None" = None,
        grad_proxy: bool = False,
        return_logits: bool = False,
        positions: torch.Tensor | None = None,
    ) -> PrefixState:
        """Full forward over positions 0..T-2, without gradient unless
        `grad_proxy=True`.

        If `capture_routing` is given (an empty dict passed by the caller),
        it is populated with the routing actually used at each layer
        (0..n_layer-2, the last layer consumes no routing in the prefix
        unless `return_logits=True`).

        `overlay` (`{global_row: vector [160]}`) is forwarded to
        `ple_true_emb` for every prefix position -- grafts from earlier
        chain positions (already-descended rows) replace the true gather
        where present. `overlay=None` (default) is identical to the prior
        behavior, bit for bit.

        `base_state`/`cache` let this method **extend** an already-computed
        prefix instead of recomputing it from scratch. With `cache` given, it
        looks up (key: consumed tokens + restricted overlay, longest
        matching prefix); on a hit, only the new positions
        (`base_state.n_prefix`..`n_prefix-1`) are computed, passing the hit's
        per-layer state to `run_layer` (the same scheme that carries the
        prefix to `last_step`, here used for N new positions instead of just
        one). Without `base_state`/`cache` (default), behavior is unchanged.

        `grad_proxy` (default `False` -> identical behavior without it):
        replaces the internal `torch.no_grad()` block with
        `torch.enable_grad()` and makes the new positions' embedding a
        differentiable leaf (`emb_leaf` in the returned `PrefixState`) --
        this measures the cost of a backward pass over all T positions, not
        the scientific gradient (that lives in `rows` inside `last_step`).
        `return_logits` applies the same final projection as `last_step`
        (`hc_mix` + `output.weight`, fixed F32 island) to **every** new
        position instead of only the last one -- this requires the FFN block
        at the last layer too (normally skipped in the prefix, which does
        not need it for its state).

        `positions` (keyword-only): replaces `torch.arange(start, n_prefix)`
        when given (expected length `n_prefix-start`, otherwise
        `ValueError`) -- used by packed sequences
        (`engraft.replica.pack.pack_fragments`), whose RoPE positions restart
        from 0 at each fragment instead of being absolute in the prefix.
        Default `None` -> identical behavior."""
        n_prefix = len(tokens) - 1
        if n_prefix <= 0:
            layers0: dict[int, LayerState] = {il: LayerState() for il in range(self.hp.n_layer)}
            return PrefixState(layers=layers0, n_prefix=0)

        base: PrefixState | None = None
        base_cap: dict[int, np.ndarray] | None = None
        start = 0
        if cache is not None:
            hit = cache.lookup(tokens[:n_prefix], overlay or {})
            if hit is not None:
                base, base_cap = hit
                start = base.n_prefix
        if base is None and base_state is not None:
            base = base_state
            start = base.n_prefix

        if base is not None and start > n_prefix:
            raise ValueError(
                f"prefix(): base_state.n_prefix={start} > requested n_prefix={n_prefix} "
                "(the cache does not extend backward)"
            )
        if base is not None and start == n_prefix:
            if capture_routing is not None and base_cap is not None:
                capture_routing.update(base_cap)
            return base

        layers: dict[int, LayerState] = (
            dict(base.layers) if base is not None else {il: LayerState() for il in range(self.hp.n_layer)}
        )

        to_device = torch.device(self.backend.device)
        to_dtype = self.backend.dtypes.torch_dtype("prefix")

        new_cap: dict[int, np.ndarray] = {}
        emb_leaf: torch.Tensor | None = None
        logits_new: torch.Tensor | None = None
        grad_cm = torch.enable_grad() if grad_proxy else torch.no_grad()
        with grad_cm:
            emb_new = self.embed(tokens[start:n_prefix], device=to_device, dtype=to_dtype)  # [n_new, n_embd]
            if grad_proxy:
                # A differentiable leaf: the proxy backward pass walks the
                # same layer stack over every position, starting from here
                # (invariant: `emb_new` has not yet been operated on
                # upstream).
                emb_new = emb_new.requires_grad_(True)
                emb_leaf = emb_new
            x = emb_new.unsqueeze(1).repeat(1, self.hp.hc_mult, 1)  # [n_new,hc,n_embd]
            if positions is not None:
                n_new_expected = n_prefix - start
                if positions.shape[0] != n_new_expected:
                    raise ValueError(
                        f"prefix(): positions has length {positions.shape[0]} expected "
                        f"{n_new_expected} (n_prefix-start)"
                    )
                pos_t = positions.to(device=emb_new.device)
            else:
                pos_t = torch.arange(start, n_prefix, dtype=torch.float64, device=emb_new.device)

            for il in range(self.hp.n_layer):
                ple_emb = None
                if self.hp.is_ple(il):
                    rows = [self.ple_true_emb(tokens, t, overlay=overlay) for t in range(start, n_prefix)]
                    # `ple_true_emb` always returns CPU F32 (a table read):
                    # moved here, by the caller, to the active phase's
                    # device/dtype -- this is not an island, it is an
                    # activation like the weights.
                    ple_emb = torch.cat(rows, dim=0).to(device=to_device, dtype=to_dtype)  # [n_new, n_embd]
                # `return_logits` requires the FFN block at the last layer
                # too: without it, `x` at that point does not carry the FFN
                # contribution the final projection (below) expects -- the
                # same thing `last_step` always does unconditionally.
                need_ffn = (not (il == self.hp.n_layer - 1)) or return_logits
                x, layers[il], routing_used = self.run_layer(
                    il, x, pos_t, layers[il], routing_source, ple_emb, need_ffn_output=need_ffn, diag=diag
                )
                if routing_used is not None:
                    new_cap[il] = routing_used.detach().to("cpu").numpy()

            if return_logits:
                # Fixed F32 island, identical to `last_step`'s final
                # projection, applied here to every new position instead of
                # only the last one. `state.logits` stays ALWAYS F32.
                logits_new = self.final_logits(x, to_device)  # [n_new, n_vocab]

        result = PrefixState(
            layers=layers, n_prefix=n_prefix,
            emb_leaf=emb_leaf if grad_proxy else None,
            logits=logits_new if return_logits else None,
        )

        if capture_routing is not None:
            for il, arr_new in new_cap.items():
                if base_cap is not None and il in base_cap:
                    capture_routing[il] = np.concatenate([base_cap[il], arr_new], axis=0)
                else:
                    capture_routing[il] = arr_new

        if cache is not None:
            merged_cap = dict(new_cap)
            if base_cap is not None:
                for il, arr in base_cap.items():
                    merged_cap[il] = np.concatenate([arr, new_cap[il]], axis=0) if il in new_cap else arr
            cache.put(tokens[:n_prefix], overlay or {}, result, merged_cap, table=self.table)

        return result

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
        ple_diag: dict | None = None,
        detach_value: bool = False,
        detach_gate: bool = False,
        *,
        phase: str = "confirm",
    ) -> torch.Tensor:
        """Returns the logits [n_vocab] of position T-1, differentiable in
        `rows`.

        `ple_diag`/`detach_value`/`detach_gate` are forwarded only to the PLE
        layer (`run_layer` ignores them elsewhere) -- default unchanged,
        identical behavior.

        `phase` (keyword-only): selects which field of `self.backend.dtypes`
        to use for this call's weights and activations (the caller knows
        whether it is building the step's graph with a backward pass --
        "grad" -- or a confirmation/checkpoint/refresh forward -- "confirm",
        the default). With `Backend.cpu_f32()` (the `Replica` default) every
        phase is F32, so the default `phase="confirm"` is numerically
        irrelevant for callers that do not pass it. Fixed F32 island:
        `rows`/its gradient are never touched by `phase` (they stay whatever
        the caller passed, always F32); only `rows_emb`, the derived
        activation entering the PLE block, follows the phase dtype. The final
        output projection (hc-mix + `output.weight` + logits) stays fixed F32
        regardless of `phase`. `output.weight` accepts `head_dtype`
        (governed by `Replica.head_dtype`, NEVER by `phase`) as its GEMM's
        input -- `hc_mix` and the resulting logits still stay ALWAYS F32
        (explicit cast on the output, never an implicit `out_dtype`)."""
        hp = self.hp
        to_device = torch.device(self.backend.device)
        to_dtype = self.backend.dtypes.torch_dtype(phase)
        t_last = len(tokens) - 1
        emb_last = self.embed([tokens[t_last]], device=to_device, dtype=to_dtype)  # [1,n_embd], fixed
        x = emb_last.unsqueeze(1).repeat(1, hp.hc_mult, 1)  # [1,hc,n_embd]
        positions = torch.tensor([float(t_last)], dtype=torch.float64, device=emb_last.device)
        # Fixed F32 island: `rows` (the differentiable PLE variable) and its
        # gradient always stay F32 -- the cast to the phase dtype happens
        # only here, on the derived activation `rows_emb` entering the PLE
        # block, never on `rows` itself.
        rows_emb = rows.reshape(1, -1).to(device=to_device, dtype=to_dtype)  # [1,n_embd]

        for il in range(hp.n_layer):
            ple_emb = rows_emb if hp.is_ple(il) else None
            x, _, routing_used = self.run_layer(
                il, x, positions, state.layers[il], routing_source, ple_emb,
                need_ffn_output=True, persist_experts=persist_experts, diag=diag,
                ple_diag=ple_diag, detach_value=detach_value, detach_gate=detach_gate,
            )
            if capture_routing is not None and routing_used is not None:
                capture_routing[il] = routing_used.detach().to("cpu").numpy()

        # Fixed F32 island: `hc_mix`, logits (and downstream softmax/loss,
        # computed by the caller) stay F32 regardless of `phase` -- `x` cast
        # back to F32 before `hc_mix`'s matmul. With an F32 backend (default)
        # every cast here is a no-op, bit for bit. The OUTPUT PROJECTION
        # (the `output.weight` GEMM, below) accepts inputs in `head_dtype`
        # (governed by `Replica.head_dtype`, independent of
        # `phase`/`backend.dtypes`) -- `logits` stays ALWAYS F32, the cast is
        # on the GEMM's input, never on the logits themselves.
        hc_norm = _as_tensor(self.w.tensor("output_hc_norm.weight"), to_device, torch.float32)
        hc_down = _as_tensor(self.w.tensor("output_hc_down.weight"), to_device, torch.float32)
        hc_up = _as_tensor(self.w.tensor("output_hc_up.weight"), to_device, torch.float32)
        x_f32 = x.to(torch.float32)
        mixed, _ = hc_mix(x_f32, hc_norm, hc_down, hc_up, None, hp.f_norm_rms_eps, hp.hc_mult)  # [1,n_embd]

        output_w = self._head_weight(to_device)  # [n_vocab,n_embd] in the current head_dtype
        # Explicit cast on the input (never `out_dtype` without a device-side
        # probe): with `head_dtype=f32` (default) this is a no-op, bit for bit.
        logits = (mixed.to(self.head_dtype) @ output_w.T).to(torch.float32)  # [1, n_vocab]
        return logits[0]

    def routing_free_full(
        self, tokens: list[int], rows: torch.Tensor
    ) -> tuple[dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]]]:
        """"Free routing" (the replica's own live choice) over prefix + last
        position, captured live at every layer. Reruns prefix() from scratch
        (routing_source=None): never reuses a state computed with a
        different routing.

        Returns (full_routing, prefix_diag, last_position_diag); the two
        diags are {il: (top11_idx, top11_val)}, used for a relative margin
        diagnostic on disagreeing cases."""
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
                captured[il] = last_captured[il]  # last layer: last position only
            elif il in prefix_captured:
                captured[il] = prefix_captured[il]
        return captured, diag_prefix, diag_last

    def logits_full(self, tokens: list[int], rows: torch.Tensor, routing_source=None) -> torch.Tensor:
        state = self.prefix(tokens, routing_source)
        return self.last_step(tokens, state, rows, routing_source)


# --------------------------------------------------------------------------
# Precondition and memory
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
        for t in range(t_len - 1):  # prefix: excludes the last position
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

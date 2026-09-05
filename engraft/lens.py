"""The lens: .pleo overlays, row variants, and a numpy replica of the PLE block.

Reuses `engraft.table` (PleTable, ngram_addresses, dequant_iq4nl) for addressing
and true rows. Never touches the GGUF itself: overlays are separate files,
applied by the fork in `llama_ple_disk::gather`.

Dump convention of the `llama-ple-lens` tool: every captured tensor is written
to `<name>.f32`, always contiguous in ggml memory order (ne0 fastest); `meta.json`
reports `ne[4]`. Reconstructed with
`np.fromfile(...).reshape(ne[3], ne[2], ne[1], ne[0])` in C order.
"""
from __future__ import annotations

import dataclasses
import json
import re
import struct
from pathlib import Path

import gguf
import numpy as np

from engraft.table import ROW_LEN, PleTable

PLEO_MAGIC = b"PLEO"


# --------------------------------------------------------------------------
# .pleo: binary overlay for the fork (global rows -> F32 [160] vectors)
# --------------------------------------------------------------------------


def write_pleo(path: str | Path, rows_global: np.ndarray, data: np.ndarray) -> None:
    """Writes a .pleo file: magic PLEO, uint32 n, uint32 dim, int32 rows[n], f32 data[n*dim]."""
    rows = np.asarray(rows_global, dtype="<i4")
    values = np.asarray(data, dtype="<f4")
    n = rows.shape[0]
    if values.shape != (n, ROW_LEN):
        raise ValueError(f"data.shape={values.shape}, expected ({n}, {ROW_LEN})")
    if rows.ndim != 1:
        raise ValueError(f"rows_global must be 1-D, got ndim={rows.ndim}")
    with open(path, "wb") as f:
        f.write(PLEO_MAGIC)
        f.write(struct.pack("<II", n, ROW_LEN))
        f.write(rows.tobytes())
        f.write(values.tobytes())


def read_pleo(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads a .pleo file. Returns (rows_global int32 [n], data float32 [n,160])."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != PLEO_MAGIC:
            raise ValueError(f"bad magic: {magic!r}, expected {PLEO_MAGIC!r}")
        n, dim = struct.unpack("<II", f.read(8))
        rows = np.frombuffer(f.read(n * 4), dtype="<i4").astype(np.int32).copy()
        data = np.frombuffer(f.read(n * dim * 4), dtype="<f4").astype(np.float32).copy()
        data = data.reshape(n, dim)
    return rows, data


PLERT1_MAGIC = b"PLERT1"


# --------------------------------------------------------------------------
# PLERT1: routing_record/routing_freeze -- the MoE routing (expert indices
# chosen by ffn_moe_topk-<il>, de-strided) for every layer.
# --------------------------------------------------------------------------


def write_plert1(path: str | Path, layers: dict[int, np.ndarray]) -> None:
    """Writes a PLERT1 file: magic, n_layer (u32), then per layer (ascending key
    order) `il` (u32), `ne0` (u32), `ne1` (u32), `ne0*ne1` int32.

    `layers`: {il: array int32 [ne1, ne0]} (C order, ne0 fastest: same
    convention as `destride_i32_2d` in the fork). Used by this module's tests to
    verify the format read/written by the fork without touching the real engine;
    the fork itself writes this format from C++ (routing_record), not Python.
    """
    ils = sorted(layers.keys())
    with open(path, "wb") as f:
        f.write(PLERT1_MAGIC)
        f.write(struct.pack("<I", len(ils)))
        for il in ils:
            arr = np.asarray(layers[il], dtype="<i4")
            if arr.ndim != 2:
                raise ValueError(f"layers[{il}]: expected ndim=2 [ne1, ne0], got ndim={arr.ndim}")
            ne1, ne0 = arr.shape
            f.write(struct.pack("<III", il, ne0, ne1))
            f.write(arr.tobytes())


def read_plert1(path: str | Path) -> dict[int, np.ndarray]:
    """Reads a PLERT1 file. Returns {il: array int32 [ne1, ne0]} (C order)."""
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != PLERT1_MAGIC:
            raise ValueError(f"bad magic: {magic!r}, expected {PLERT1_MAGIC!r}")
        (n_layer,) = struct.unpack("<I", f.read(4))
        out: dict[int, np.ndarray] = {}
        for _ in range(n_layer):
            il, ne0, ne1 = struct.unpack("<III", f.read(12))
            data = np.frombuffer(f.read(ne0 * ne1 * 4), dtype="<i4").astype(np.int32).copy()
            out[il] = data.reshape(ne1, ne0)
    return out


def local_to_global(table: PleTable, h: int, row_local: int) -> int:
    """Global row index in the merged layout (local + head_offsets[h])."""
    return int(table.head_offsets[h]) + int(row_local)


# --------------------------------------------------------------------------
# RowSet: the 16 rows of a position, with variant generators
# --------------------------------------------------------------------------


@dataclasses.dataclass
class RowSet:
    """The 16 true rows (one per head) read at a position, with their global indices."""

    table: PleTable
    rows_global: np.ndarray  # int32 [16]
    data: np.ndarray  # float32 [16, 160], the dequantized true rows

    @classmethod
    def from_position(cls, table: PleTable, tokens: list[int], t: int) -> "RowSet":
        """Builds the RowSet for position t of `tokens` (addresses via ngram_addresses)."""
        addr = table.ngram_addresses(tokens)[t]  # 16 local indices, one per head
        n_heads = table.n_heads
        rows_global = np.empty(n_heads, dtype=np.int32)
        data = np.empty((n_heads, ROW_LEN), dtype=np.float32)
        for h in range(n_heads):
            rows_global[h] = local_to_global(table, h, addr[h])
            data[h] = table.read_rows(h, addr[h], 1)[0]
        return cls(table=table, rows_global=rows_global, data=data)

    # -- varianti: ciascuna ritorna un nuovo array float32 [16, 160] --

    def identity(self) -> np.ndarray:
        return self.data.copy()

    def zero(self) -> np.ndarray:
        return np.zeros_like(self.data)

    def scale(self, k: float) -> np.ndarray:
        return (self.data * k).astype(np.float32)

    def random_matched(self, rng: np.random.Generator) -> np.ndarray:
        """Independent Gaussian per row, then renormalized to the true row's norm."""
        norms = np.linalg.norm(self.data, axis=1, keepdims=True)
        g = rng.standard_normal(self.data.shape).astype(np.float32)
        g_norms = np.linalg.norm(g, axis=1, keepdims=True)
        g_norms = np.where(g_norms == 0.0, 1.0, g_norms)
        return (g / g_norms * norms).astype(np.float32)

    def swap(self, other: "RowSet") -> np.ndarray:
        if other.data.shape != self.data.shape:
            raise ValueError("swap: RowSet of different shape")
        return other.data.copy()

    def isolate(self, h: int) -> np.ndarray:
        out = np.zeros_like(self.data)
        out[h] = self.data[h]
        return out

    @staticmethod
    def build_overlay(
        parts: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merges several (rows_global, data[n,160]) sets by global row index, deduping.

        Raises ValueError if the same global index receives two different vectors
        across sets (should not happen: variants are generated per row, the same
        row always receives the same transform).
        """
        seen: dict[int, np.ndarray] = {}
        for rows_global, data in parts:
            rows_global = np.asarray(rows_global)
            data = np.asarray(data, dtype=np.float32)
            if rows_global.shape[0] != data.shape[0]:
                raise ValueError("build_overlay: rows_global and data of different length")
            for i in range(rows_global.shape[0]):
                r = int(rows_global[i])
                vec = data[i]
                if r in seen:
                    if not np.array_equal(seen[r], vec):
                        raise ValueError(
                            f"build_overlay: global row {r} receives two different vectors"
                        )
                else:
                    seen[r] = vec
        if not seen:
            return np.empty(0, dtype=np.int32), np.empty((0, ROW_LEN), dtype=np.float32)
        rows_sorted = sorted(seen.keys())
        rows = np.array(rows_sorted, dtype=np.int32)
        out = np.stack([seen[r] for r in rows_sorted]).astype(np.float32)
        return rows, out


# --------------------------------------------------------------------------
# PleReplica: numpy replica of the PLE block (blk.1), real weights from GGUF
# --------------------------------------------------------------------------


_EXPECTED_TYPES = {
    "blk.1.ple_key.weight": gguf.GGMLQuantizationType.Q8_0,
    "blk.1.ple_value.weight": gguf.GGMLQuantizationType.Q8_0,
    "blk.1.ple_norm_key.weight": gguf.GGMLQuantizationType.F32,
    "blk.1.ple_norm_query.weight": gguf.GGMLQuantizationType.F32,
    "blk.1.ple_norm_conv.weight": gguf.GGMLQuantizationType.F32,
    "blk.1.ple_conv1d.weight": gguf.GGMLQuantizationType.F32,
}

_N_EMBD = 2560
_N_HC = 4  # parallel hyper-connection streams
_CONV_TAPS = 4
_CONV_DILATION = 3


class PleReplica:
    """Numpy replica of the PLE block (blk.1), weights read from a target GGUF shard.

    Metadata (RMSNorm epsilon) lives in shard 1 of a split GGUF (the file passed
    here, shard 2, holds only tensors: a split GGUF carries metadata in the first
    shard). If shard 1 is not reachable at the sibling path, pass eps explicitly.
    """

    def __init__(self, gguf_path: str | Path, eps: float | None = None):
        self.gguf_path = Path(gguf_path)
        reader = gguf.GGUFReader(str(self.gguf_path))
        by_name = {t.name: t for t in reader.tensors}

        def load(name: str) -> np.ndarray:
            if name not in by_name:
                raise KeyError(f"{name}: tensor missing in {self.gguf_path}")
            t = by_name[name]
            expected = _EXPECTED_TYPES[name]
            if t.tensor_type != expected:
                raise ValueError(
                    f"{name}: expected {expected}, got {t.tensor_type}: "
                    "tensor type differs from the assumed one, explicit check failed"
                )
            return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32)

        w_key = load("blk.1.ple_key.weight")  # [10240, 2560] = [out, in]
        w_value = load("blk.1.ple_value.weight")  # [2560, 2560]
        norm_key = load("blk.1.ple_norm_key.weight")  # [10240]
        norm_query = load("blk.1.ple_norm_query.weight")  # [10240]
        norm_conv = load("blk.1.ple_norm_conv.weight")  # [10240]
        conv1d = load("blk.1.ple_conv1d.weight")  # dequant shape [10240, 4] (canale, tap)

        if w_key.shape != (_N_HC * _N_EMBD, _N_EMBD):
            raise ValueError(f"ple_key: unexpected shape {w_key.shape}")
        if w_value.shape != (_N_EMBD, _N_EMBD):
            raise ValueError(f"ple_value: unexpected shape {w_value.shape}")
        for name, arr in (
            ("ple_norm_key", norm_key),
            ("ple_norm_query", norm_query),
            ("ple_norm_conv", norm_conv),
        ):
            if arr.shape != (_N_HC * _N_EMBD,):
                raise ValueError(f"{name}: unexpected shape {arr.shape}")
        if conv1d.shape != (_N_HC * _N_EMBD, _CONV_TAPS):
            raise ValueError(f"ple_conv1d: unexpected shape {conv1d.shape}")

        self.w_key = w_key
        self.w_value = w_value
        self.norm_key = norm_key.reshape(_N_HC, _N_EMBD)
        self.norm_query = norm_query.reshape(_N_HC, _N_EMBD)
        self.norm_conv = norm_conv.reshape(_N_HC, _N_EMBD)
        # writable copy: §6.4 validation zeroes conv1d to isolate the direct channel
        self.conv1d = conv1d.reshape(_N_HC, _N_EMBD, _CONV_TAPS).copy()  # [stream, chan, tap]

        if eps is None:
            eps = _read_rms_eps(self.gguf_path)
        self.eps = float(eps)

    @staticmethod
    def _rmsnorm_per_stream(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        """x: [..., 4, 2560], weight: [4, 2560]. RMSNorm on the last axis, per stream."""
        ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        normed = (x / np.sqrt(ms + eps)).astype(np.float32)
        return normed * weight

    def key(self, emb: np.ndarray) -> np.ndarray:
        """emb: [T, 2560] -> [T, 4, 2560], already RMS-normalized per stream and weighted."""
        emb = np.asarray(emb, dtype=np.float32)
        raw = emb @ self.w_key.T  # [T, 10240]
        raw = raw.reshape(-1, _N_HC, _N_EMBD)
        return self._rmsnorm_per_stream(raw, self.norm_key, self.eps)

    def value(self, emb: np.ndarray) -> np.ndarray:
        """emb: [T, 2560] -> [T, 2560]. No normalization."""
        emb = np.asarray(emb, dtype=np.float32)
        return (emb @ self.w_value.T).astype(np.float32)

    def gate(self, emb: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """emb: [T,2560], hidden: [T,4,2560] (incoming hidden state, 4 streams) -> [T,4]."""
        k = self.key(emb)  # [T,4,2560]
        hidden = np.asarray(hidden, dtype=np.float32)
        q = self._rmsnorm_per_stream(hidden, self.norm_query, self.eps)  # [T,4,2560]
        s = np.sum(k * q, axis=-1) / np.sqrt(_N_EMBD)  # [T,4]
        s = s.astype(np.float64)
        sgn = np.sign(s)
        root = sgn * np.sqrt(np.clip(np.abs(s), 1e-6, None))
        gate = 1.0 / (1.0 + np.exp(-root))
        return gate.astype(np.float32)

    def gated(self, emb: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """emb: [T,2560], hidden: [T,4,2560] -> [T,4,2560] = replicated value * gate."""
        v = self.value(emb)  # [T,2560]
        g = self.gate(emb, hidden)  # [T,4]
        return (v[:, None, :] * g[:, :, None]).astype(np.float32)

    def conv_out(self, gated_seq: np.ndarray) -> np.ndarray:
        """gated_seq: [T,4,2560] -> [T,4,2560]. norm_conv, dilated causal conv, silu.

        Tap k (0..3) reads (3-k)*3 positions back: tap 3 = current, tap 0 = t-9.
        Initial history is zero (no position before 0).
        """
        gated_seq = np.asarray(gated_seq, dtype=np.float32)
        normed = self._rmsnorm_per_stream(gated_seq, self.norm_conv, self.eps)  # [T,4,2560]
        t_len = normed.shape[0]
        acc = np.zeros_like(normed)
        for k in range(_CONV_TAPS):
            back = (_CONV_TAPS - 1 - k) * _CONV_DILATION
            w_k = self.conv1d[:, :, k]  # [4, 2560]
            if back == 0:
                acc += normed * w_k[None, :, :]
            elif back < t_len:
                shifted = np.zeros_like(normed)
                shifted[back:] = normed[: t_len - back]
                acc += shifted * w_k[None, :, :]
            # if back >= t_len, the tap reads no valid position: zero contribution
        return (acc / (1.0 + np.exp(-acc))).astype(np.float32)  # silu(x) = x * sigmoid(x)

    def ple_out(self, emb_seq: np.ndarray, hidden_seq: np.ndarray) -> np.ndarray:
        """emb_seq: [T,2560], hidden_seq: [T,4,2560] -> [T,4,2560] = gated + conv_out(gated)."""
        gated_seq = self.gated(emb_seq, hidden_seq)
        return (gated_seq + self.conv_out(gated_seq)).astype(np.float32)

    def ple_out_at(
        self,
        emb_t: np.ndarray,
        emb_seq: np.ndarray,
        hidden_seq: np.ndarray,
        t: int,
    ) -> np.ndarray:
        """Like ple_out but with `emb_seq[t]` replaced by `emb_t` [B,2560] -> [B,4,2560].

        Positions < t use the true rows of `emb_seq`: the convolution history
        (taps at t-3, t-6, t-9) is fixed and computed once over the true
        sequence, not recomputed per batch element.
        """
        emb_t = np.asarray(emb_t, dtype=np.float32)
        b = emb_t.shape[0]
        emb_seq = np.asarray(emb_seq, dtype=np.float32)
        hidden_seq = np.asarray(hidden_seq, dtype=np.float32)

        # direct channel (value + gate) for the batch, with hidden[t] fixed (same for all)
        hidden_t = hidden_seq[t]  # [4,2560]
        hidden_batch = np.broadcast_to(hidden_t, (b,) + hidden_t.shape)
        v = self.value(emb_t)  # [B,2560]
        g = self.gate(emb_t, hidden_batch)  # [B,4]
        gated_batch = (v[:, None, :] * g[:, :, None]).astype(np.float32)  # [B,4,2560]

        # convolution history: from the true sequence, computed once
        gated_true = self.gated(emb_seq, hidden_seq)  # [T,4,2560]
        normed_true = self._rmsnorm_per_stream(gated_true, self.norm_conv, self.eps)
        normed_batch = self._rmsnorm_per_stream(gated_batch, self.norm_conv, self.eps)

        acc = np.zeros_like(normed_batch)
        for k in range(_CONV_TAPS):
            back = (_CONV_TAPS - 1 - k) * _CONV_DILATION
            w_k = self.conv1d[:, :, k]  # [4, 2560]
            pos = t - back
            if pos < 0:
                continue
            if back == 0:
                acc += normed_batch * w_k[None, :, :]
            else:
                acc += normed_true[pos][None, :, :] * w_k[None, :, :]
        conv_out_batch = (acc / (1.0 + np.exp(-acc))).astype(np.float32)
        return (gated_batch + conv_out_batch).astype(np.float32)

    def jacobian_out(
        self,
        emb_seq: np.ndarray,
        hidden_seq: np.ndarray,
        t: int,
        coords: np.ndarray | None = None,
        step: float = 1e-3,
    ) -> np.ndarray:
        """Derivative of `ple_out[t]` (flattened [10240]) with respect to `emb[t][coords]`.

        Central differences via `ple_out_at`, in blocks of 512 variants to avoid
        allocating `[2*n_coords, 4, 2560]` at once.
        """
        emb_seq = np.asarray(emb_seq, dtype=np.float32)
        n_embd = emb_seq.shape[1]
        if coords is None:
            coords = np.arange(n_embd)
        coords = np.asarray(coords, dtype=np.int64)
        n_coords = coords.shape[0]
        emb_t = emb_seq[t]
        out_dim = _N_HC * n_embd
        jac = np.empty((out_dim, n_coords), dtype=np.float64)

        block = 512
        for start in range(0, n_coords, block):
            idx = coords[start : start + block]
            n = idx.shape[0]
            plus = np.tile(emb_t, (n, 1)).astype(np.float32)
            minus = plus.copy()
            rows = np.arange(n)
            plus[rows, idx] += step
            minus[rows, idx] -= step
            batch = np.concatenate([plus, minus], axis=0)  # [2n, 2560]
            out = self.ple_out_at(batch, emb_seq, hidden_seq, t)  # [2n,4,2560]
            out_flat = out.reshape(2 * n, -1).astype(np.float64)
            diff = (out_flat[:n] - out_flat[n:]) / (2.0 * step)  # [n, 10240]
            jac[:, start : start + n] = diff.T
        return jac

    def solve_emb(
        self,
        target_out: np.ndarray,
        emb_seq: np.ndarray,
        hidden_seq: np.ndarray,
        t: int,
        mask: np.ndarray,
        mu: float = 1e-2,
        n_iter: int = 20,
    ) -> tuple[np.ndarray, float]:
        """Gauss-Newton: minimizes ||ple_out[t](emb) - target_out||^2 + mu*||(emb - emb_orig)[mask]||^2.

        Varies only the coordinates in `mask`; returns (emb [2560], final relative
        residual). The Tikhonov term pulls toward `emb_orig` (the true row), not
        toward the previous iterate.
        """
        emb_seq = np.asarray(emb_seq, dtype=np.float32)
        hidden_seq = np.asarray(hidden_seq, dtype=np.float32)
        target = np.asarray(target_out, dtype=np.float64).reshape(-1)
        target_norm = np.linalg.norm(target)
        mask = np.asarray(mask, dtype=bool)
        coords = np.nonzero(mask)[0]

        emb_orig = emb_seq[t].copy()
        emb = emb_orig.copy()
        eye = np.eye(coords.shape[0])
        prev_resid = None
        best_emb = emb.copy()
        best_resid = float("inf")

        def residual(e: np.ndarray) -> tuple[np.ndarray, float]:
            seq = emb_seq.copy()
            seq[t] = e
            out = self.ple_out_at(e[None, :], seq, hidden_seq, t)[0].reshape(-1).astype(np.float64)
            r = target - out
            rel = float(np.linalg.norm(r) / target_norm) if target_norm > 0 else float(np.linalg.norm(r))
            return r, rel

        for _ in range(n_iter):
            emb_seq_cur = emb_seq.copy()
            emb_seq_cur[t] = emb
            r, resid = residual(emb)
            if resid < best_resid:
                best_emb, best_resid = emb.copy(), resid
            # plateau (small or no improvement) = convergence, stop;
            # a worsening (overshoot typical of Gauss-Newton) does not stop the iteration,
            # the trace keeps descending over subsequent iterations.
            if prev_resid is not None and 0.0 <= (prev_resid - resid) < 1e-4 * prev_resid:
                break
            prev_resid = resid
            jac = self.jacobian_out(emb_seq_cur, hidden_seq, t, coords=coords)
            delta = emb[coords].astype(np.float64) - emb_orig[coords].astype(np.float64)
            a = jac.T @ jac + mu * eye
            b = jac.T @ r - mu * delta
            step_vec = np.linalg.solve(a, b)
            emb = emb.copy()
            emb[coords] = (emb[coords].astype(np.float64) + step_vec).astype(np.float32)

        _, final_resid = residual(emb)
        if final_resid < best_resid:
            best_emb, best_resid = emb.copy(), final_resid
        return best_emb.astype(np.float32), best_resid


def _read_rms_eps(shard2_path: Path) -> float:
    """Read qwen4exp.attention.layer_norm_rms_epsilon from shard 1 of the same split.

    Split GGUF files carry the model metadata only in the first shard
    (`<name>-00001-of-<N>.gguf`); the shard holding `blk.1.ple_*` has only
    `GGUF.*`/`split.*` keys. Any shard index and any shard count are accepted.
    """
    name = shard2_path.name
    m = re.search(r"(\d{5})-of-(\d{5})", name)
    if m is None:
        # not a split file: read the metadata from the same file
        reader = gguf.GGUFReader(str(shard2_path))
        return _eps_from_reader(reader, shard2_path)
    shard1_name = name[:m.start(1)] + "00001" + name[m.end(1):]
    shard1_path = shard2_path.with_name(shard1_name)
    if not shard1_path.exists():
        raise FileNotFoundError(
            f"eps not given explicitly and shard 1 not found: {shard1_path}"
        )
    reader = gguf.GGUFReader(str(shard1_path))
    return _eps_from_reader(reader, shard1_path)


def _eps_from_reader(reader: "gguf.GGUFReader", path: Path) -> float:
    key = "qwen4exp.attention.layer_norm_rms_epsilon"
    if key not in reader.fields:
        raise KeyError(f"{key}: missing in {path}")
    field = reader.fields[key]
    return float(field.parts[field.data[0]][0])


# --------------------------------------------------------------------------
# Reading llama-ple-lens dumps
# --------------------------------------------------------------------------


def load_job(out_dir: str | Path, job_id: str) -> dict[str, np.ndarray]:
    """Reads <out_dir>/<job_id>/meta.json and the .f32 files of captured tensors.

    Each array is reconstructed with reshape(ne[3], ne[2], ne[1], ne[0]) (C order;
    the files are always contiguous in ggml order, `nb` is ignored).
    """
    job_dir = Path(out_dir) / job_id
    meta = json.loads((job_dir / "meta.json").read_text())
    out: dict[str, np.ndarray] = {}
    for name, info in meta["tensors"].items():
        ne = info["ne"]
        arr = np.fromfile(job_dir / info["file"], dtype=np.float32)
        out[name] = arr.reshape(ne[3], ne[2], ne[1], ne[0])
    return out


def logits(out_dir: str | Path, job_id: str) -> np.memmap:
    """Returns the job's logits as a memmap [n_pos, n_vocab] (never loaded whole)."""
    job_dir = Path(out_dir) / job_id
    meta = json.loads((job_dir / "meta.json").read_text())
    n_vocab = int(meta["n_vocab"])
    n_pos = len(meta["tokens"]) if meta["logits_mode"] == "all" else 1
    return np.memmap(job_dir / "logits.f32", dtype=np.float32, mode="r", shape=(n_pos, n_vocab))

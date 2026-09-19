"""Opens the model's GGUF shards; dequantizes per tensor and per expert.

Axis convention (fork-verified): a GGUF tensor with `ne = [ne0, ne1, ...]`
(ne0 fastest in memory) dequantizes into a numpy array of shape `[..., ne1,
ne0]` (axes reversed). `tensor(name)` returns that array as-is (no
transposition): the modules in `layers.py` know which axis is which for each
tensor. `expert(name, e)` slices expert `e` on the **last ggml axis**
(`ne[-1]` in ggml numbering, i.e. the **first axis** of the dequantized numpy
array) before dequantizing, so the full tensor is never materialized.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gguf
import numpy as np
import torch

from engraft.replica import dequant as DQ


class GgufWeights:
    """Name -> (shard, tensor) index across a split model's shards, with caching."""

    def __init__(
        self,
        paths: list[str | Path],
        ram_cache_bytes: int = 0,
        disk_cache_dir: str | Path | None = None,
        disk_cache_bytes: int = 40 * (1 << 30),
    ):
        self.paths = [Path(p) for p in paths]
        self.readers = [gguf.GGUFReader(str(p)) for p in self.paths]
        self._index: dict[str, tuple[int, "gguf.ReaderTensor"]] = {}
        for i, r in enumerate(self.readers):
            for t in r.tensors:
                if t.name in self._index:
                    raise ValueError(f"duplicate tensor across shards: {t.name}")
                self._index[t.name] = (i, t)

        self.ram_cache_bytes = ram_cache_bytes
        self._ram_cache: dict[str, np.ndarray] = {}
        self._ram_cache_order: list[str] = []  # LRU: most recent at the tail
        self._ram_cache_used = 0

        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else None
        self.disk_cache_bytes = disk_cache_bytes
        self._disk_index_path = self.disk_cache_dir / "index.json" if self.disk_cache_dir else None
        self._disk_index: dict[str, dict] = {}
        if self._disk_index_path is not None and self._disk_index_path.exists():
            self._disk_index = json.loads(self._disk_index_path.read_text())

    # -- indice ------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self._index

    def shape(self, name: str) -> tuple[int, ...]:
        """ggml shape (ne order), as declared in the file."""
        _, t = self._index[name]
        return tuple(int(x) for x in t.shape)

    def tensor_type(self, name: str) -> "gguf.GGMLQuantizationType":
        _, t = self._index[name]
        return t.tensor_type

    # -- dequantization ------------------------------------------------------

    def tensor(self, name: str) -> np.ndarray:
        """Dequantizes the whole tensor to f32, axes reversed relative to ggml (ne order)."""
        if name in self._ram_cache:
            self._touch_ram(name)
            return self._ram_cache[name]
        _, t = self._index[name]
        arr = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32)
        if self.ram_cache_bytes > 0:
            self._store_ram(name, arr)
        return arr

    def expert(self, name: str, e: int, persist: bool = False) -> np.ndarray:
        """Dequantizes expert `e` (last ggml axis) without materializing the full tensor.

        `persist=True`: uses/writes the disk *and* RAM cache (only for
        last-position experts, by design). `persist=False` (the prefix)
        **never touches the cache**, disk or RAM: it dequantizes, the caller
        uses the result, and it is discarded. With thousands of experts
        dequantized per prefix across several prompts, RAM-caching those too
        blows past the memory budget the cache is meant to protect -- the
        cache exists only for the handful of experts per prompt reused at
        every descent step.
        """
        if not persist:
            _, t = self._index[name]
            return self._dequant_expert(t, e)

        cache_key = f"{name}/{e}"
        if cache_key in self._ram_cache:
            self._touch_ram(cache_key)
            return self._ram_cache[cache_key]

        if persist and self.disk_cache_dir is not None:
            npy_path = self.disk_cache_dir / name / f"{e}.npy"
            if npy_path.exists():
                arr = np.load(npy_path)
                self._touch_disk(cache_key, npy_path)
                if self.ram_cache_bytes > 0:
                    self._store_ram(cache_key, arr)
                return arr

        _, t = self._index[name]
        # gguf.ReaderTensor exposes the mapped tensor (mmap): we slice the expert on the last
        # ggml axis (t.shape[-1]) on the raw data via the gguf library's per-expert slicing
        # API when available; otherwise we dequantize the whole tensor and
        # slice (slower, used only as a fallback for types without direct slicing).
        arr = self._dequant_expert(t, e)

        if persist and self.disk_cache_dir is not None:
            npy_path = self.disk_cache_dir / name / f"{e}.npy"
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, arr)
            self._evict_disk_if_needed(npy_path.stat().st_size)
            self._touch_disk(cache_key, npy_path)

        if self.ram_cache_bytes > 0:
            self._store_ram(cache_key, arr)
        return arr

    @staticmethod
    def _dequant_expert(t: "gguf.ReaderTensor", e: int) -> np.ndarray:
        """Slices expert `e` (last ggml axis, t.shape[-1]) and dequantizes only that.

        `t.data.reshape(n_expert, ne1, -1)[e]` (experts are stacked on the
        last ggml axis in native order; `t.data` is the raw quantized buffer,
        numpy shape `[n_expert, ne1, block_bytes...]` once reshaped on the
        expert axis).
        """
        n_expert = int(t.shape[-1])
        raw = np.asarray(t.data)
        per_expert = raw.reshape(n_expert, -1, raw.shape[-1]) if raw.ndim > 1 else raw.reshape(n_expert, -1)
        slice_e = per_expert[e]
        out = np.asarray(gguf.quants.dequantize(slice_e, t.tensor_type), dtype=np.float32)
        return out

    # -- cache in RAM (LRU) --------------------------------------------------

    def _touch_ram(self, key: str) -> None:
        if key in self._ram_cache_order:
            self._ram_cache_order.remove(key)
        self._ram_cache_order.append(key)

    def _store_ram(self, key: str, arr: np.ndarray) -> None:
        nbytes = arr.nbytes
        while self._ram_cache_used + nbytes > self.ram_cache_bytes and self._ram_cache_order:
            oldest = self._ram_cache_order.pop(0)
            self._ram_cache_used -= self._ram_cache[oldest].nbytes
            del self._ram_cache[oldest]
        if nbytes <= self.ram_cache_bytes:
            self._ram_cache[key] = arr
            self._ram_cache_order.append(key)
            self._ram_cache_used += nbytes

    # -- disk cache (LRU, byte cap) -------------------------------------------

    def _touch_disk(self, key: str, path: Path) -> None:
        self._disk_index[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "atime": time.time(),
        }
        self._save_disk_index()

    def _evict_disk_if_needed(self, incoming_bytes: int) -> None:
        if self.disk_cache_dir is None:
            return
        total = sum(v["bytes"] for v in self._disk_index.values()) + incoming_bytes
        if total <= self.disk_cache_bytes:
            return
        # LRU eviction: removes the least recently used entries until back under the cap
        for key in sorted(self._disk_index, key=lambda k: self._disk_index[k]["atime"]):
            if total <= self.disk_cache_bytes:
                break
            entry = self._disk_index.pop(key)
            p = Path(entry["path"])
            if p.exists():
                p.unlink()
            total -= entry["bytes"]
        self._save_disk_index()

    def _save_disk_index(self) -> None:
        if self._disk_index_path is None:
            return
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_index_path.write_text(json.dumps(self._disk_index, indent=2))


# --------------------------------------------------------------------------
# Device-resident quantized weight store
# --------------------------------------------------------------------------


class DeviceIQ4(GgufWeights):
    """Quantized bytes resident on the device (the raw `t.data` of the gguf
    reader), dequantized on the fly in pure torch (`engraft.replica.dequant`)
    -- never a `gguf.quants` dequantization in this class. `tensor(name)`/
    `expert(name, e)` return **torch tensors** (not numpy) on the requested
    device, in the requested dtype.

    On CPU, in F32, `tensor`/`expert` match `GgufWeights` bit for bit: the
    only difference is that data passes through resident quantized bytes +
    torch dequantization, instead of `gguf.quants.dequantize` on the host.

    This is a reference-only implementation: it loads every tensor's
    quantized bytes eagerly at construction time (`resident_policy="all"`,
    the only supported policy) and never caches dequantized results. The
    production system this was derived from also supports a working-set LRU
    over device-resident bytes and a cache of already-dequantized experts,
    both tuned for a specific GPU memory budget -- that tuning is not part of
    the public reference and is intentionally left out here; a caller with a
    tighter memory budget than "resident_policy=all" allows needs to build
    that layer itself."""

    def __init__(self, paths: list[str | Path], device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(paths, ram_cache_bytes=0, disk_cache_dir=None)
        self.device = torch.device(device)
        self.dtype = dtype

        # Quantized bytes per whole tensor (used by tensor()); key = tensor name.
        self._dev_bytes: dict[str, torch.Tensor] = {}
        # Quantized bytes per expert slice (used by expert()); key = (name, index).
        self._dev_expert_bytes: dict[tuple[str, int], torch.Tensor] = {}

        for name in self._index:
            self._load_tensor_bytes(name)

    # -- loading quantized bytes onto the device ------------------------------

    def _raw_bytes_numpy(self, t) -> np.ndarray:
        """Raw bytes of `t.data`, always uint8. The `gguf` reader already
        exposes quantized types (and BF16, block_size=1/type_size=2) as
        bytes, but F32/F16/F64/integer tensors as *typed* arrays (float32,
        etc.): `.view(np.uint8)` on a C-contiguous array expands only the
        last axis by the item size, reproducing exactly the byte-shape
        convention that quantized types already have -- so `expert()` does
        not need to special-case either kind."""
        arr = np.ascontiguousarray(np.asarray(t.data))
        return arr.view(np.uint8)

    def _load_tensor_bytes(self, name: str) -> torch.Tensor:
        if name in self._dev_bytes:
            return self._dev_bytes[name]
        _, t = self._index[name]
        raw = self._raw_bytes_numpy(t)
        tens = torch.from_numpy(raw.copy()).to(self.device)
        self._dev_bytes[name] = tens
        return tens

    # -- on-the-fly torch dequantization --------------------------------------

    def _dequant_bytes(self, qtype_name: str, byte_tensor: torch.Tensor) -> torch.Tensor:
        """`byte_tensor`: uint8 [N] (a multiple of `type_size`). Returns
        [n_blocks, block_size] in the requested dtype -- the caller reshapes
        into the logical shape."""
        fn = DQ.DEQUANT_FNS.get(qtype_name)
        if fn is None:
            raise NotImplementedError(
                f"type {qtype_name} is not covered by engraft.replica.dequant "
                "(closed list): add the function before using it here"
            )
        _, type_size = DQ.TYPE_SIZES[qtype_name]
        n_blocks = byte_tensor.numel() // type_size
        blocks = byte_tensor.reshape(n_blocks, type_size)
        return fn(blocks, out_dtype=self.dtype)

    def tensor(self, name: str) -> torch.Tensor:
        """Dequantizes the whole tensor in torch, on the device, in the
        requested dtype. Same axis convention as `GgufWeights.tensor` (ggml
        axes reversed, `[..., ne1, ne0]`)."""
        _, t = self._index[name]
        byte_tensor = self._load_tensor_bytes(name)
        qtype_name = t.tensor_type.name
        vals = self._dequant_bytes(qtype_name, byte_tensor)  # [n_blocks, block_size]
        ne = [int(x) for x in t.shape]  # ggml ne-order, ne0 fastest
        numpy_shape = list(reversed(ne))  # weights.py convention: [..., ne1, ne0]
        return vals.reshape(numpy_shape)

    def tensor_as(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Like `tensor(name)`, but cast to the requested `dtype` instead of
        `self.dtype` -- a plain convenience wrapper (the reference
        implementation does not cache the cast result separately, unlike the
        production dequant cache this class omits)."""
        if dtype == self.dtype:
            return self.tensor(name)
        return self.tensor(name).to(dtype)

    def expert(self, name: str, e: int, persist: bool = False) -> torch.Tensor:
        """Dequantizes expert `e` (last ggml axis) without materializing the
        whole tensor: same slice as `GgufWeights._dequant_expert`, over the
        raw (undequantized) bytes already resident on the device. `persist`
        is accepted for call-site compatibility with `GgufWeights.expert`
        but has no effect here -- this reference implementation has no
        dequantized-result cache to opt into."""
        del persist
        _, t = self._index[name]
        qtype_name = t.tensor_type.name
        cache_key = (name, e)

        if cache_key in self._dev_expert_bytes:
            byte_tensor = self._dev_expert_bytes[cache_key]
        else:
            raw = self._raw_bytes_numpy(t)
            n_expert = int(t.shape[-1])
            per_expert = raw.reshape(n_expert, -1, raw.shape[-1]) if raw.ndim > 1 else raw.reshape(n_expert, -1)
            slice_e = np.ascontiguousarray(per_expert[e])  # shape (ne1, bytes_per_row) or (bytes_per_row,)
            byte_tensor = torch.from_numpy(slice_e.copy()).to(self.device)
            self._dev_expert_bytes[cache_key] = byte_tensor

        # `byte_tensor` keeps the (ne1, bytes_per_row) shape of the slice (or
        # (bytes_per_row,) if the expert has no ne1 axis): dequantization
        # operates on flattened blocks, then recomposes row by row -- same
        # logical shape as `GgufWeights._dequant_expert`.
        if byte_tensor.dim() == 1:
            vals = self._dequant_bytes(qtype_name, byte_tensor)
            return vals.reshape(-1)  # [ne0]
        n_rows = byte_tensor.shape[0]
        vals = self._dequant_bytes(qtype_name, byte_tensor.reshape(-1))  # [n_blocks, block_size]
        return vals.reshape(n_rows, -1)  # [ne1, ne0]

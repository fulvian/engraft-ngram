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

    # -- dequantizzazione ----------------------------------------------------

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

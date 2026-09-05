"""Tests for engraft.replica.weights and engraft.replica.hparams.

Tests marked `real` read the real GGUF (the three UD-IQ4_XS shards) and are
excluded by default (`-m "not real"`). The rest (LRU eviction) uses fake
arrays, no GGUF.

uv run pytest tests/test_replica_weights.py
uv run pytest tests/test_replica_weights.py -m real   # requires engraft.toml
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from engraft.config import load as load_config
from engraft.replica.weights import GgufWeights
from engraft.replica.hparams import Hparams

real = pytest.mark.real


@pytest.fixture(scope="module")
def shards():
    return load_config().get_list("model.shards")


@real
def test_expert_slice_matches_full_dequant_real(shards):
    """expert() on the real GGUF is bit-identical to slicing the full tensor (IQ3_S and IQ4_NL)."""
    w = GgufWeights(shards)

    t0 = time.time()
    full_gate = w.tensor("blk.1.ffn_gate_exps.weight")  # IQ3_S
    t_full_gate = time.time() - t0

    t0 = time.time()
    sl_gate = w.expert("blk.1.ffn_gate_exps.weight", 7)
    t_expert_gate = time.time() - t0
    assert np.array_equal(full_gate[7], sl_gate)

    t0 = time.time()
    full_down = w.tensor("blk.1.ffn_down_exps.weight")  # IQ4_NL
    t_full_down = time.time() - t0

    t0 = time.time()
    sl_down = w.expert("blk.1.ffn_down_exps.weight", 7)
    t_expert_down = time.time() - t0
    assert np.array_equal(full_down[7], sl_down)

    print(
        f"\nffn_gate_exps (IQ3_S): full={t_full_gate:.3f}s expert={t_expert_gate:.3f}s\n"
        f"ffn_down_exps (IQ4_NL): full={t_full_down:.3f}s expert={t_expert_down:.3f}s"
    )


@real
def test_hparams_from_gguf_real(shards):
    hp = Hparams.from_gguf_paths(shards[0], shards[1])
    assert hp.n_embd == 2560
    assert hp.n_layer == 48
    assert hp.n_head == 24
    assert hp.n_head_kv == 2
    assert hp.n_embd_head == 256
    assert hp.rope_dim == 64
    assert hp.rope_sections == (11, 11, 10, 0)
    assert hp.ssm_d_conv == 4
    assert hp.ssm_d_state == 128
    assert hp.ssm_dt_rank == 48
    assert hp.ssm_n_group == 16
    assert hp.hc_mult == 4
    assert hp.hc_low_rank == 320
    assert hp.n_expert == 512
    assert hp.n_expert_used == 10
    assert hp.n_ff_exp == 640
    assert hp.ple_layer == 1
    assert hp.ple_ngram_size == 3
    assert hp.ple_heads_per_ngram == 8
    assert hp.ple_n_heads == 16
    assert hp.ple_head_dim == 160
    assert hp.n_vocab == 248320
    for il in range(hp.n_layer):
        expect_recr = (il + 1) % 4 != 0
        assert hp.is_recr(il) == expect_recr
    assert hp.is_ple(1) and not hp.is_ple(0) and not hp.is_ple(2)


def _fake_gguf_uint8(n_expert: int, ne1: int, block_bytes: int) -> np.ndarray:
    return (np.arange(n_expert * ne1 * block_bytes, dtype=np.uint64) % 256).astype(np.uint8).reshape(
        n_expert, ne1, block_bytes
    )


class _FakeCache:
    """Verifica lo sfratto LRU su `GgufWeights` senza toccare un GGUF: bypassa l'indice
    e chiama direttamente i metodi di cache privati con byte finti, come farebbe
    `expert(..., persist=True)` con `disk_cache_dir` impostato."""

    def __init__(self, tmp_path: Path, disk_cache_bytes: int):
        self.w = GgufWeights.__new__(GgufWeights)
        self.w.disk_cache_dir = tmp_path
        self.w.disk_cache_bytes = disk_cache_bytes
        self.w._disk_index_path = tmp_path / "index.json"
        self.w._disk_index = {}
        self.w.ram_cache_bytes = 0
        self.w._ram_cache = {}
        self.w._ram_cache_order = []
        self.w._ram_cache_used = 0

    def write(self, key: str, nbytes: int) -> Path:
        p = self.w.disk_cache_dir / f"{key}.npy"
        arr = np.zeros(nbytes // 4, dtype=np.float32)
        np.save(p, arr)
        self.w._evict_disk_if_needed(p.stat().st_size)
        self.w._touch_disk(key, p)
        return p


def test_disk_cache_lru_eviction(tmp_path):
    """Tetto piccolo: il file meno usato di recente viene rimosso quando si supera il tetto."""
    cache = _FakeCache(tmp_path, disk_cache_bytes=3000)

    p1 = cache.write("a", 1200)
    time.sleep(0.01)
    p2 = cache.write("b", 1200)
    time.sleep(0.01)
    # tocca di nuovo "a" cosi' diventa il piu' recente
    cache.w._touch_disk("a", p1)
    time.sleep(0.01)
    p3 = cache.write("c", 1200)  # supera il tetto: deve sfrattare il meno recente ("b")

    assert not p2.exists(), "b (meno recente) doveva essere sfrattato"
    assert p1.exists() and p3.exists()
    assert "b" not in cache.w._disk_index
    assert "a" in cache.w._disk_index and "c" in cache.w._disk_index


def test_ram_cache_lru_eviction():
    w = GgufWeights.__new__(GgufWeights)
    w.ram_cache_bytes = 100
    w._ram_cache = {}
    w._ram_cache_order = []
    w._ram_cache_used = 0

    a = np.zeros(10, dtype=np.float32)  # 40 byte
    b = np.zeros(10, dtype=np.float32)
    c = np.zeros(10, dtype=np.float32)
    w._store_ram("a", a)
    w._store_ram("b", b)
    w._touch_ram("a")  # a diventa il piu' recente
    w._store_ram("c", c)  # 40*3=120 > 100: deve sfrattare "b" (meno recente)

    assert "b" not in w._ram_cache
    assert "a" in w._ram_cache and "c" in w._ram_cache

"""Fake weights for a full, tiny `model.Replica`: unlike `fake_replica.py`
(which bypasses `run_layer`/`LayerState` with a toy MoE gather), this module
feeds the real `Replica` in `engraft.replica.model`, with the real
`engraft.replica.layers`: it supports the incrementally-extensible prefix
(`Replica.prefix(..., base_state=..., cache=...)`), which lives in the
per-layer `LayerState`/`run_layer` contract, never exercised by the simpler
fake replica (which does not carry real state between calls). It also
supports the sequence-level descent (`engraft.replica.seq`), which requires
a full 48-tensor-per-layer weight surface, not the smaller single-fact
surface.

`FakeFullWeights.tensor(name)`/`.expert(name,e,persist)` return deterministic
numpy arrays (seeded from the name, never "real" randomness): the same call
repeated with the same name always returns the same array, so a descent/
prefix recomputed from scratch is reproducible."""
from __future__ import annotations

import hashlib

import numpy as np

from engraft.replica.hparams import Hparams
from engraft.table import ROW_LEN


def _seed_for(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def tiny_hparams(n_layer: int = 3) -> Hparams:
    """Tiny but consistent hparams (same constraint as the real model:
    n_embd = ple_n_heads * ROW_LEN), a single PLE layer (il=1, inside
    n_layer), full_attention_interval=2 so at least one layer is full
    attention and one is the delta net. `ple_ngram_size=3,
    ple_heads_per_ngram=8` (ple_n_heads=16) to match
    `engraft.testing.fake_table.FakeTable`, which has N_HEADS=16 hardcoded."""
    ple_ngram_size = 3
    ple_heads_per_ngram = 8
    ple_n_heads = (ple_ngram_size - 1) * ple_heads_per_ngram  # 16
    n_embd = ple_n_heads * ROW_LEN  # 2560
    return Hparams(
        n_embd=n_embd, n_layer=n_layer, n_vocab=20,
        n_head=2, n_head_kv=1, n_embd_head=4,
        full_attention_interval=2, rope_dim=4,
        rope_sections=(1, 1, 1, 1), rope_freq_base=10000.0, f_norm_rms_eps=1e-6,
        ssm_d_conv=4, ssm_d_state=4, ssm_dt_rank=2, ssm_n_group=1, ssm_d_inner=8,
        hc_mult=2, hc_low_rank=3,
        n_expert=6, n_expert_used=2, n_ff_exp=5, n_ff_shexp=5, expert_weights_scale=0.0,
        ple_layer=1, ple_ngram_size=ple_ngram_size, ple_heads_per_ngram=ple_heads_per_ngram,
        ple_conv_kernel=3, ple_head_dim=ROW_LEN, ple_eos_token_id=999_999_999, ple_image_token_id=-1,
        ple_head_offsets=tuple(i * 5000 for i in range(ple_n_heads)),
        ple_head_vocab_sizes=tuple([5000] * ple_n_heads),
        ple_layer_multipliers=(1, 3, 5, 7),
    )


class FakeFullWeights:
    def __init__(self, hp: Hparams, scale: float = 0.05):
        self.hp = hp
        self.scale = scale

    def _rand(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        rng = np.random.default_rng(_seed_for(name))
        return (rng.standard_normal(shape) * self.scale).astype(np.float32)

    def tensor(self, name: str) -> np.ndarray:
        hp = self.hp
        d = hp.n_embd
        hc = hp.hc_mult
        hc_dim = hp.hc_dim

        if name == "token_embd.weight":
            return self._rand(name, (hp.n_vocab, d))
        if name == "output.weight":
            return self._rand(name, (hp.n_vocab, d))
        if name in ("output_hc_norm.weight",):
            return self._rand(name, (hc_dim,))
        if name == "output_hc_down.weight":
            return self._rand(name, (hp.hc_low_rank, hc_dim))
        if name == "output_hc_up.weight":
            return self._rand(name, (hc_dim, hp.hc_low_rank))

        # blk.{il}.<suffix>
        assert name.startswith("blk.")
        rest = name[len("blk."):]
        il_str, suffix = rest.split(".", 1)

        if suffix == "attn_q.weight":
            return self._rand(name, (2 * hp.n_embd_head * hp.n_head, d))
        if suffix == "attn_k.weight":
            return self._rand(name, (hp.n_embd_head * hp.n_head_kv, d))
        if suffix == "attn_v.weight":
            return self._rand(name, (hp.n_embd_head * hp.n_head_kv, d))
        if suffix == "attn_output.weight":
            return self._rand(name, (d, hp.n_embd_head * hp.n_head))
        if suffix in ("attn_q_norm.weight", "attn_k_norm.weight"):
            return self._rand(name, (hp.n_embd_head,)) * 0.0 + 1.0  # "1+w" weight ~ 1

        if suffix == "attn_qkv.weight":
            return self._rand(name, (hp.conv_dim, d))
        if suffix == "attn_gate.weight":
            value_dim = hp.ssm_d_state * hp.ssm_dt_rank
            return self._rand(name, (value_dim, d))
        if suffix == "ssm_conv1d.weight":
            return self._rand(name, (hp.conv_dim, hp.ssm_d_conv))
        if suffix == "ssm_dt.bias":
            return self._rand(name, (hp.ssm_dt_rank,))
        if suffix == "ssm_a":
            return -np.abs(self._rand(name, (hp.ssm_dt_rank,))) - 0.1
        if suffix == "ssm_beta.weight":
            return self._rand(name, (hp.ssm_dt_rank, d))
        if suffix == "ssm_alpha.weight":
            return self._rand(name, (hp.ssm_dt_rank, d))
        if suffix == "ssm_norm.weight":
            return self._rand(name, (hp.ssm_d_state,)) * 0.0 + 1.0
        if suffix == "ssm_out.weight":
            value_dim = hp.ssm_d_state * hp.ssm_dt_rank
            return self._rand(name, (d, value_dim))

        if suffix == "ple_key.weight":
            return self._rand(name, (hc_dim, d))
        if suffix == "ple_value.weight":
            return self._rand(name, (d, d))
        if suffix in ("ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight"):
            return self._rand(name, (hc_dim,)) * 0.0 + 1.0
        if suffix == "ple_conv1d.weight":
            return self._rand(name, (hc_dim, hp.ple_conv_kernel))

        if suffix.startswith("hc_") and suffix.endswith("norm.weight"):
            return self._rand(name, (hc_dim,)) * 0.0 + 1.0
        if suffix.startswith("hc_") and suffix.endswith("down.weight"):
            return self._rand(name, (hp.hc_low_rank, hc_dim))
        if suffix.startswith("hc_") and suffix.endswith("up.weight"):
            return self._rand(name, (hc_dim, hp.hc_low_rank))
        if suffix.startswith("hc_") and suffix.endswith("inject.weight"):
            return self._rand(name, (hc, hc_dim))

        if suffix == "ffn_gate_inp.weight":
            return self._rand(name, (hp.n_expert, d))
        if suffix == "ffn_gate_inp_shexp.weight":
            return self._rand(name, (d,))
        if suffix == "ffn_up_shexp.weight":
            return self._rand(name, (hp.n_ff_shexp, d))
        if suffix == "ffn_gate_shexp.weight":
            return self._rand(name, (hp.n_ff_shexp, d))
        if suffix == "ffn_down_shexp.weight":
            return self._rand(name, (d, hp.n_ff_shexp))

        raise KeyError(f"FakeFullWeights: unrecognized name {name!r}")

    def expert(self, name: str, e: int, persist: bool = False) -> np.ndarray:
        hp = self.hp
        d = hp.n_embd
        key = f"{name}/{e}"
        if name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight"):
            return self._rand(key, (hp.n_ff_exp, d))
        if name.endswith("ffn_down_exps.weight"):
            return self._rand(key, (d, hp.n_ff_exp))
        raise KeyError(f"FakeFullWeights.expert: unrecognized name {name!r}")

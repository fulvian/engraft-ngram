"""Model hyperparameters, read from GGUF `qwen4exp.*` metadata.

No value is hardcoded: everything comes from the metadata of shard 1 (or
whichever shard carries it, for a split GGUF). Conventions verified against
the llama.cpp fork (`src/models/qwen4exp.cpp:load_arch_hparams`).
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import gguf
import numpy as np


def _scalar(reader: "gguf.GGUFReader", key: str):
    field = reader.fields[key]
    return field.parts[field.data[0]][0]


def _array(reader: "gguf.GGUFReader", key: str) -> list[int]:
    field = reader.fields[key]
    return [int(field.parts[d][0]) for d in field.data]


@dataclasses.dataclass
class Hparams:
    n_embd: int
    n_layer: int
    n_vocab: int
    n_head: int
    n_head_kv: int
    n_embd_head: int  # = key_length = value_length
    full_attention_interval: int
    rope_dim: int
    rope_sections: tuple[int, int, int, int]
    rope_freq_base: float
    f_norm_rms_eps: float

    # gated delta net (recurrent layers)
    ssm_d_conv: int
    ssm_d_state: int  # head_k_dim = head_v_dim
    ssm_dt_rank: int  # num_v_heads
    ssm_n_group: int  # num_k_heads
    ssm_d_inner: int  # num_v_heads * head_v_dim

    # hyper-connections
    hc_mult: int
    hc_low_rank: int

    # MoE
    n_expert: int
    n_expert_used: int
    n_ff_exp: int
    n_ff_shexp: int
    expert_weights_scale: float  # 0.0 nel GGUF: nessuna scala applicata (llama-hparams.h default)

    # PLE (blk.1 soltanto)
    ple_layer: int
    ple_ngram_size: int
    ple_heads_per_ngram: int
    ple_conv_kernel: int
    ple_head_dim: int  # = ROW_LEN = embedding_length_per_layer_input
    ple_eos_token_id: int
    ple_image_token_id: int
    ple_head_offsets: tuple[int, ...]
    ple_head_vocab_sizes: tuple[int, ...]
    ple_layer_multipliers: tuple[int, ...]

    def is_recr(self, il: int) -> bool:
        """True if layer `il` uses the gated delta net (not full attention).

        `qwen4exp.cpp:203`: `(il+1) % full_attention_interval != 0` for il < n_layer.
        """
        return (il + 1) % self.full_attention_interval != 0

    def is_ple(self, il: int) -> bool:
        return il == self.ple_layer

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.n_embd

    @property
    def ple_n_heads(self) -> int:
        return (self.ple_ngram_size - 1) * self.ple_heads_per_ngram

    @property
    def n_embd_k_gqa(self) -> int:
        return self.n_embd_head * self.n_head_kv

    @property
    def n_embd_v_gqa(self) -> int:
        return self.n_embd_head * self.n_head_kv

    @property
    def conv_dim(self) -> int:
        """Delta net conv1d channels: key_dim*2 + value_dim (qwen4exp.cpp:243)."""
        key_dim = self.ssm_d_state * self.ssm_n_group
        value_dim = self.ssm_d_state * self.ssm_dt_rank
        return key_dim * 2 + value_dim

    @classmethod
    def from_gguf(cls, reader: "gguf.GGUFReader") -> "Hparams":
        n_embd = int(_scalar(reader, "qwen4exp.embedding_length"))
        n_layer = int(_scalar(reader, "qwen4exp.block_count"))
        rope_sections = _array(reader, "qwen4exp.rope.dimension_sections")
        if len(rope_sections) != 4:
            raise ValueError(f"rope.dimension_sections: attese 4 sezioni, trovate {rope_sections}")

        by_name_output = None
        # n_vocab non ha una chiave propria: si legge da output.weight se il reader la porta,
        # altrimenti il chiamante lo passa esplicitamente (vedi from_gguf_paths).
        for t in reader.tensors:
            if t.name == "output.weight":
                by_name_output = int(t.shape[1])
                break

        head_offsets = _array(reader, "qwen4exp.ple.head_offsets")
        head_vocab_sizes = _array(reader, "qwen4exp.ple.head_vocab_sizes")
        ple_layers = _array(reader, "qwen4exp.ple.layers")
        if len(ple_layers) != 1:
            raise ValueError(f"qwen4exp supports a single PLE layer, found {ple_layers}")

        return cls(
            n_embd=n_embd,
            n_layer=n_layer,
            n_vocab=by_name_output if by_name_output is not None else 0,
            n_head=int(_scalar(reader, "qwen4exp.attention.head_count")),
            n_head_kv=int(_scalar(reader, "qwen4exp.attention.head_count_kv")),
            n_embd_head=int(_scalar(reader, "qwen4exp.attention.key_length")),
            full_attention_interval=int(_scalar(reader, "qwen4exp.full_attention_interval")),
            rope_dim=int(_scalar(reader, "qwen4exp.rope.dimension_count")),
            rope_sections=tuple(rope_sections),  # type: ignore[arg-type]
            rope_freq_base=float(_scalar(reader, "qwen4exp.rope.freq_base")),
            f_norm_rms_eps=float(_scalar(reader, "qwen4exp.attention.layer_norm_rms_epsilon")),
            ssm_d_conv=int(_scalar(reader, "qwen4exp.ssm.conv_kernel")),
            ssm_d_state=int(_scalar(reader, "qwen4exp.ssm.state_size")),
            ssm_dt_rank=int(_scalar(reader, "qwen4exp.ssm.time_step_rank")),
            ssm_n_group=int(_scalar(reader, "qwen4exp.ssm.group_count")),
            ssm_d_inner=int(_scalar(reader, "qwen4exp.ssm.inner_size")),
            hc_mult=int(_scalar(reader, "qwen4exp.hyper_connection.count")),
            hc_low_rank=int(_scalar(reader, "qwen4exp.hyper_connection.low_rank")),
            n_expert=int(_scalar(reader, "qwen4exp.expert_count")),
            n_expert_used=int(_scalar(reader, "qwen4exp.expert_used_count")),
            n_ff_exp=int(_scalar(reader, "qwen4exp.expert_feed_forward_length")),
            n_ff_shexp=int(_scalar(reader, "qwen4exp.expert_shared_feed_forward_length")),
            # non presente nei metadati di questo GGUF: llama-hparams.h la inizializza a 0.0
            # (nessuna scala applicata ai pesi degli esperti), qwen4exp non la sovrascrive.
            expert_weights_scale=0.0,
            ple_layer=int(ple_layers[0]),
            ple_ngram_size=int(_scalar(reader, "qwen4exp.ple.ngram_size")),
            ple_heads_per_ngram=int(_scalar(reader, "qwen4exp.ple.heads_per_ngram")),
            ple_conv_kernel=int(_scalar(reader, "qwen4exp.ple.conv_kernel")),
            ple_head_dim=int(_scalar(reader, "qwen4exp.embedding_length_per_layer_input")),
            ple_eos_token_id=int(_scalar(reader, "qwen4exp.ple.eos_token_id")),
            ple_image_token_id=int(_scalar(reader, "qwen4exp.ple.image_token_id")),
            ple_head_offsets=tuple(head_offsets),
            ple_head_vocab_sizes=tuple(head_vocab_sizes),
            ple_layer_multipliers=tuple(_array(reader, "qwen4exp.ple.layer_multipliers")),
        )

    @classmethod
    def from_gguf_paths(cls, shard1_path: str | Path, weight_shard_path: str | Path | None = None) -> "Hparams":
        """Reads metadata from whichever shard carries it (shard 1 for a multi-file split).

        In a metadata-only first shard, `n_vocab` is read from
        `output.weight`/`token_embd.weight` in `weight_shard_path` (a weight
        shard), if given; otherwise it is tried in the same file.
        """
        reader = gguf.GGUFReader(str(shard1_path))
        hp = cls.from_gguf(reader)
        if hp.n_vocab == 0 and weight_shard_path is not None:
            reader2 = gguf.GGUFReader(str(weight_shard_path))
            for t in reader2.tensors:
                if t.name in ("output.weight", "token_embd.weight"):
                    hp.n_vocab = int(t.shape[1])
                    break
        if hp.n_vocab == 0:
            raise ValueError(
                "n_vocab undetermined: output.weight/token_embd.weight missing "
                f"in {shard1_path} and no weight_shard_path was given"
            )
        return hp

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--shard1", required=True)
    p.add_argument("--weight-shard", default=None)
    p.add_argument("--dump", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    hp = Hparams.from_gguf_paths(args.shard1, args.weight_shard)
    text = json.dumps(hp.to_json(), indent=2)
    if args.dump:
        print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    _main()

"""Offline, read-only access to the n-gram (PLE) table of a Qwen4Exp-family
model, read directly from its GGUF file.

Read-only and stripe-at-a-time: a full tensor is never materialized in memory.
Reproduces in Python the host-side addressing of the llama.cpp fork
(src/models/qwen4exp.cpp, llm_graph_input_ple::set_input) and the IQ4_NL
dequantization defined in ggml/src/ggml-common.h and ggml-quants.c of the
same fork.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import gguf
import numpy as np

ROW_LEN = 160
BLOCK_LEN = 32
BLOCKS_PER_ROW = ROW_LEN // BLOCK_LEN  # 5
BLOCK_BYTES = 18  # 2 (d, float16) + 16 (qs, nibble)
ROW_BYTES = BLOCKS_PER_ROW * BLOCK_BYTES  # 90

# ggml/src/ggml-common.h: kvalues_iq4nl. No zero in the codebook, so a
# row is exactly zero if and only if the 5 scales (d) are zero.
KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


@dataclasses.dataclass(frozen=True)
class HeadInfo:
    head: int
    vocab_size: int  # p_h
    offset: int  # cumulative sum of p_h in the merged layout (qwen4exp.ple.head_offsets)
    data_offset: int  # absolute offset in the GGUF file of the start of the per-head tensor


class PleTable:
    """Read-only, stripe-at-a-time access to a per-head-layout PLE table GGUF.

    The expected file has per-head tensors (``ple_ngram_embd.{h}.weight``, type
    IQ4_NL, shape ``[160, p_h]``): the rows of head h are contiguous in the file
    at offsets ``[data_offset + 90*r, data_offset + 90*(r+1))``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        reader = gguf.GGUFReader(str(self.path))

        def get_scalar(key: str):
            field = reader.fields[key]
            return field.parts[field.data[0]][0]

        def get_array(key: str) -> list[int]:
            field = reader.fields[key]
            return [int(field.parts[d][0]) for d in field.data]

        self.eos_token_id = int(get_scalar("qwen4exp.ple.eos_token_id"))
        self.image_token_id = int(get_scalar("qwen4exp.ple.image_token_id"))
        self.ngram_size = int(get_scalar("qwen4exp.ple.ngram_size"))
        self.heads_per_ngram = int(get_scalar("qwen4exp.ple.heads_per_ngram"))
        self.layer_multipliers = get_array("qwen4exp.ple.layer_multipliers")
        self.head_vocab_sizes = get_array("qwen4exp.ple.head_vocab_sizes")
        self.head_offsets = get_array("qwen4exp.ple.head_offsets")
        self.n_heads = len(self.head_vocab_sizes)

        tensors_by_name = {t.name: t for t in reader.tensors}
        self.heads: list[HeadInfo] = []
        for h in range(self.n_heads):
            name = f"ple_ngram_embd.{h}.weight"
            t = tensors_by_name[name]
            assert t.tensor_type == gguf.GGMLQuantizationType.IQ4_NL, (
                f"{name}: expected IQ4_NL, got {t.tensor_type}"
            )
            p_h = int(t.shape[1])
            assert p_h == self.head_vocab_sizes[h], (
                f"{name}: shape[1]={p_h} != head_vocab_sizes[{h}]={self.head_vocab_sizes[h]}"
            )
            self.heads.append(
                HeadInfo(
                    head=h,
                    vocab_size=p_h,
                    offset=self.head_offsets[h],
                    data_offset=int(t.data_offset),
                )
            )

    def read_rows_raw(self, h: int, start: int, n: int) -> np.ndarray:
        """Reads n raw rows (IQ4_NL bytes) starting at row `start` of head h.

        Returns a uint8 array [n, 90]. Read as one contiguous stripe via
        seek+readinto; the full tensor is never loaded into memory.
        """
        head = self.heads[h]
        assert 0 <= start and start + n <= head.vocab_size, (
            f"head {h}: requested rows [{start},{start+n}) outside [0,{head.vocab_size})"
        )
        buf = np.empty((n, ROW_BYTES), dtype=np.uint8)
        with open(self.path, "rb") as f:
            f.seek(head.data_offset + start * ROW_BYTES)
            f.readinto(buf)
        return buf

    def read_rows(self, h: int, start: int, n: int) -> np.ndarray:
        """Reads and dequantizes n rows of head h. Returns float32 [n, 160]."""
        raw = self.read_rows_raw(h, start, n)
        return dequant_iq4nl(raw)

    def ngram_addresses(self, tokens: list[int]) -> list[list[int]]:
        """For each position t in `tokens`, the 16 local rows (0..p_h-1) read.

        Reproduces the hash of llm_graph_input_ple::set_input: ctx = [x_t, x_{t-1}, x_{t-2}].
        A missing predecessor (start of sequence) or an EOS found in the window
        zeroes it and every older position (substituting the EOS id, not 0); the
        EOS of the current token does not cut its own context (verified against
        qwen4exp.cpp:1278-1316, upstream fork).
        """
        n_prev = self.ngram_size - 1
        out: list[list[int]] = []
        for t in range(len(tokens)):
            ctx = [tokens[t]]
            cut = False
            for s in range(1, self.ngram_size):
                pos = t - s
                tok = tokens[pos] if (pos >= 0 and not cut) else None
                if tok is None or tok == self.eos_token_id:
                    cut = True
                    tok = self.eos_token_id
                ctx.append(tok)

            rows_t: list[int] = []
            for n in range(2, self.ngram_size + 1):
                mixed = (ctx[0] * self.layer_multipliers[0]) & 0xFFFFFFFFFFFFFFFF
                for j in range(1, n):
                    mixed ^= (ctx[j] * self.layer_multipliers[j]) & 0xFFFFFFFFFFFFFFFF
                mixed &= 0xFFFFFFFFFFFFFFFF
                base = (n - 2) * self.heads_per_ngram
                for g in range(self.heads_per_ngram):
                    h_i = base + g
                    rows_t.append(mixed % self.head_vocab_sizes[h_i])
            out.append(rows_t)
        return out


def dequant_iq4nl(raw: np.ndarray) -> np.ndarray:
    """Dequantizes IQ4_NL rows. raw: uint8 [n, 90] -> float32 [n, 160].

    Block format (32 values, 18 bytes): d (float16) then qs[16] (nibbles).
    y[j] = d*kvalues[qs[j]&0xF] (j=0..15, low nibble -> first half);
    y[j+16] = d*kvalues[qs[j]>>4] (high nibble -> second half).
    """
    raw = np.asarray(raw)
    n = raw.shape[0]
    blocks = raw.reshape(n, BLOCKS_PER_ROW, BLOCK_BYTES)

    d = blocks[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(n, BLOCKS_PER_ROW)
    qs = blocks[:, :, 2:2 + 16]  # [n, 5, 16] uint8

    lo = qs & 0x0F
    hi = qs >> 4

    y_lo = KVALUES_IQ4NL[lo]  # [n, 5, 16]
    y_hi = KVALUES_IQ4NL[hi]  # [n, 5, 16]

    y = np.concatenate([y_lo, y_hi], axis=2)  # [n, 5, 32]
    y = y * d[:, :, None]
    return y.reshape(n, ROW_LEN)


def row_is_zero(raw: np.ndarray) -> np.ndarray:
    """raw: uint8 [n, 90] -> bool [n]. True iff the row's 5 block scales (d) are zero.

    The IQ4_NL codebook contains no zero: with d != 0 every value in the row is
    nonzero, so this test is equivalent to, and far cheaper than, dequantizing
    and comparing against zero.
    """
    raw = np.asarray(raw)
    n = raw.shape[0]
    blocks = raw.reshape(n, BLOCKS_PER_ROW, BLOCK_BYTES)
    d_bits = blocks[:, :, 0:2].reshape(n, BLOCKS_PER_ROW, 2)
    return np.all((d_bits[:, :, 0] == 0) & (d_bits[:, :, 1] == 0), axis=1)


class PleTokenizer:
    """Thin wrapper over `tokenizers` to tokenize raw text without a BOS token.

    The base model's tokenizer_config.json declares add_bos_token=False and
    bos_token=None: no BOS to add or strip.
    """

    def __init__(self, tokenizer_json_path: str | Path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_json_path))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode_token(self, token_id: int) -> str:
        """String for a single token: a readable label for reports/diagnostics,
        not a tokenization contract."""
        return self._tok.decode([int(token_id)], skip_special_tokens=False)

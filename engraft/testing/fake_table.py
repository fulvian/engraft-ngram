"""Fake PLE table, shared by tests/test_facts.py, tests/test_replica_graft.py,
and (via --fake) engraft.run with engraft.facts.

Reproduces the real addressing (`engraft.table.PleTable.ngram_addresses`: heads
0-7 a function of the last two tokens = bigram/B8, heads 8-15 of the last
three = trigram/T8, `base = (n-2)*heads_per_ngram`) without any GGUF: rows are
deterministic from a seed, a function of (head, local index), so
`RowSet.from_position`, `check_precondition`, and the sister/paraphrase checks
in `engraft.facts` behave as on the real table (same two-n-gram structure), at
near-zero cost.
"""
from __future__ import annotations

import numpy as np

ROW_LEN = 160
N_HEADS = 16
HEADS_PER_NGRAM = 8
NGRAM_SIZE = 3
EOS_TOKEN_ID = 999_999_999  # never present in these tests' fake tokens


class FakeTable:
    def __init__(self, seed: int = 0, head_vocab_size: int = 5_000):
        self.n_heads = N_HEADS
        self.heads_per_ngram = HEADS_PER_NGRAM
        self.ngram_size = NGRAM_SIZE
        self.eos_token_id = EOS_TOKEN_ID
        self.head_vocab_sizes = [head_vocab_size] * N_HEADS
        self.head_offsets = [i * head_vocab_size for i in range(N_HEADS)]
        # layer_multipliers[0..ngram_size]: same shape as qwen4exp.ple.layer_multipliers
        # (one multiplier per context position, ctx[0]=current, ctx[1..]=previous).
        rng = np.random.default_rng(seed)
        self.layer_multipliers = [
            int(rng.integers(1, 2**61)) | 1 for _ in range(NGRAM_SIZE + 1)
        ]
        self._seed = seed

    # -- addressing, identical to ple_table.PleTable's algorithm ----------------

    def ngram_addresses(self, tokens: list[int]) -> list[list[int]]:
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

    # -- row reading: deterministic from (head, local row, seed) ----------------

    def read_rows(self, h: int, start: int, n: int) -> np.ndarray:
        out = np.empty((n, ROW_LEN), dtype=np.float32)
        for i in range(n):
            row_local = start + i
            rng = np.random.default_rng((self._seed, h, row_local))
            out[i] = rng.standard_normal(ROW_LEN).astype(np.float32) * 0.05
        return out

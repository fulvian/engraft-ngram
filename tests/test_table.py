"""Correctness tests for engraft.table.

Tests that read a real GGUF are marked `real` and excluded by default
(`-m "not real"`); run them explicitly once `engraft.toml` points at a real
table.

uv run pytest tests/test_table.py
uv run pytest tests/test_table.py -m real   # requires engraft.toml
"""
import gguf
import numpy as np
import pytest

from engraft.config import load as load_config
from engraft.table import PleTable, PleTokenizer, dequant_iq4nl, row_is_zero

real = pytest.mark.real


@pytest.fixture(scope="module")
def table():
    return PleTable(load_config().get_path("model.table"))


@real
def test_decode_token_roundtrips_a_single_token():
    tok = PleTokenizer(load_config().get_path("model.tokenizer"))
    ids = tok.encode("hello")
    assert ids
    s = tok.decode_token(ids[0])
    assert isinstance(s, str)


@real
def test_dequant_matches_reference(table):
    # real rows from different heads (bigram and trigram), not just head 0
    for h in (0, 8, 15):
        raw = table.read_rows_raw(h, 1000, 8)
        got = dequant_iq4nl(raw)
        for i in range(raw.shape[0]):
            ref = gguf.quants.dequantize(raw[i], gguf.GGMLQuantizationType.IQ4_NL)
            np.testing.assert_allclose(got[i], ref.reshape(-1), rtol=0, atol=0)


@real
def test_row_is_zero_detects_all_zero_row(table):
    raw = table.read_rows_raw(0, 0, 16)
    zero_mask = row_is_zero(raw)
    dequant = dequant_iq4nl(raw)
    for i in range(raw.shape[0]):
        if zero_mask[i]:
            assert np.all(dequant[i] == 0.0)
        else:
            assert np.any(dequant[i] != 0.0)


def test_row_is_zero_synthetic():
    raw = np.zeros((3, 90), dtype=np.uint8)
    raw[1, 0] = 1  # d of the first block != 0 -> not zero
    raw[2, 36] = 0  # stays all zero -> zero row
    mask = row_is_zero(raw)
    assert mask.tolist() == [True, False, True]


@real
def test_hash_matches_hand_computed(table):
    # ctx without EOS: no missing predecessor, no EOS in the window.
    # mixed_2 = (t0*m0) ^ (t1*m1); mixed_3 = mixed_2 ^ (t2*m2)
    m0, m1, m2 = table.layer_multipliers
    mask = (1 << 64) - 1
    t2, t1, t0 = 100, 200, 300  # tokens = [..., t2, t1, t0]; t0 is the last one (current position)
    tokens = [t2, t1, t0]

    mixed2 = ((t0 * m0) ^ (t1 * m1)) & mask
    mixed3 = (mixed2 ^ (t2 * m2)) & mask

    addr = table.ngram_addresses(tokens)[-1]
    expected = [mixed2 % p for p in table.head_vocab_sizes[0:8]]
    expected += [mixed3 % p for p in table.head_vocab_sizes[8:16]]
    assert addr == expected


@real
def test_hash_eos_replaces_missing_predecessor(table):
    m0, m1 = table.layer_multipliers[0], table.layer_multipliers[1]
    mask = (1 << 64) - 1
    tokens = [42]  # no predecessor: x_{t-1} and x_{t-2} read as EOS
    eos = table.eos_token_id

    mixed2 = ((tokens[0] * m0) ^ (eos * m1)) & mask
    addr = table.ngram_addresses(tokens)[-1]
    expected2 = [mixed2 % p for p in table.head_vocab_sizes[0:8]]
    assert addr[0:8] == expected2


@real
def test_addresses_in_range(table):
    tokens = [10, 20000, 5, table.eos_token_id, 99, 100000]
    for row in table.ngram_addresses(tokens):
        assert len(row) == table.n_heads
        for h_i, r in enumerate(row):
            assert 0 <= r < table.head_vocab_sizes[h_i]

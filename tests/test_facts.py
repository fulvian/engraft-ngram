"""Tests for engraft.facts.

Fake table (`engraft.testing.fake_table`, same addressing structure as the
real table: heads 0-7 = bigram, 8-15 = trigram) and a fake tokenizer (words ->
lists of ids, so an "answer" can be made of several hand-controllable tokens).
No GGUF.

uv run pytest tests/test_facts.py
"""
from __future__ import annotations

import numpy as np

from engraft.testing.fake_table import FakeTable
from engraft.facts import resolve_facts


class FakeTokenizer:
    """encode(text) -> concatenazione delle liste di id di ogni parola separata da
    spazio (dizionario fisso). Nessuna gestione dello spazio iniziale (irrilevante qui:
    lo split lo rimuove comunque)."""

    def __init__(self, vocab: dict[str, list[int]]):
        self._vocab = vocab

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in text.split():
            out.extend(self._vocab[word])
        return out


VOCAB = {
    "TRIG_A": [10], "TRIG_B": [20], "TRIG_C": [30], "TRIG_D": [40], "TRIG_E": [50],
    "TRIG_C2": [31],
    "X99": [99], "X98": [98],
    "Y60": [60], "Y70": [70], "Y80": [80],
    "ANSWER": [100, 101, 102],
    "BIGANSWER": [110, 111, 112, 113],
    "FALLBACK1": [120, 121, 122, 123],
    "FALLBACK2": [130],
    "J1": [1], "J2": [2], "J3": [3], "K1": [200], "K2": [201],
}

TRIGGER_TEXT = "TRIG_A TRIG_B TRIG_C TRIG_D TRIG_E"


def _base_fact(fid: str, **overrides) -> dict:
    fact = {
        "id": fid,
        "lang": "it",
        "kind": "memoria",
        "trigger": TRIGGER_TEXT,
        "answer": "ANSWER",
        "answer_fallback": [],
        "sister_words": [
            "TRIG_A TRIG_B TRIG_C2 TRIG_D TRIG_E",
        ],
        "paraphrase_same_tail": "X99 X98 TRIG_C TRIG_D TRIG_E",
        "paraphrase_other_tail": "TRIG_A TRIG_B Y60 Y70 Y80",
        "doc_id": None,
    }
    fact.update(overrides)
    return fact


def test_resolve_happy_path_sisters_and_paraphrases_ok():
    table = FakeTable(seed=42)
    tok = FakeTokenizer(VOCAB)
    facts_data = {"facts": [_base_fact("f1")]}
    resolved, keys = resolve_facts(facts_data, {}, tok, table)

    entry = resolved["facts"]["f1"]
    assert entry["answer_token_counts"] == {"ANSWER": 3}
    assert entry["answer_used"] == "ANSWER"
    assert entry["answer_fallback_applied"] is False
    assert entry["answer_tokens"] == [100, 101, 102]

    assert len(entry["sisters"]) == 1
    sister = entry["sisters"][0]
    assert sister["bigram_ok"] is True
    assert sister["t8_common"] == 0
    assert sister["ok"] is True

    assert entry["paraphrase_same_tail"]["eq16"] == 16
    assert entry["paraphrase_same_tail"]["ok"] is True

    assert entry["paraphrase_other_tail"]["t8_common"] == 0
    assert entry["paraphrase_other_tail"]["ok"] is True

    assert keys["n_conflicts"] == 0
    assert keys["excluded_facts"] == []


def test_answer_fallback_applied_when_over_limit():
    table = FakeTable(seed=42)
    tok = FakeTokenizer(VOCAB)
    fact = _base_fact("f2", answer="BIGANSWER", answer_fallback=["FALLBACK1", "FALLBACK2"])
    resolved, _keys = resolve_facts({"facts": [fact]}, {}, tok, table)
    entry = resolved["facts"]["f2"]
    assert entry["answer_original"] == "BIGANSWER"
    assert entry["answer_used"] == "FALLBACK2"
    assert entry["answer_fallback_applied"] is True
    assert entry["answer_tokens"] == [130]
    assert entry["answer_token_counts"] == {"BIGANSWER": 4, "FALLBACK1": 4, "FALLBACK2": 1}


def test_answer_fallback_exhausted_raises():
    table = FakeTable(seed=42)
    tok = FakeTokenizer(VOCAB)
    fact = _base_fact("f3", answer="BIGANSWER", answer_fallback=["FALLBACK1"])
    try:
        resolve_facts({"facts": [fact]}, {}, tok, table)
        assert False, "atteso ValueError: nessun ripiego entro 3 token"
    except ValueError as e:
        assert "f3" in str(e)


def test_sister_verification_fails_when_bigram_differs():
    """Una sorella scelta male (cambia anche il bigram) -> ok=False, mai aggirata."""
    table = FakeTable(seed=42)
    tok = FakeTokenizer(VOCAB)
    fact = _base_fact("f4", sister_words=["TRIG_A TRIG_B Y60 Y70 Y80"])  # in realta' una parafrasi other_tail
    resolved, _keys = resolve_facts({"facts": [fact]}, {}, tok, table)
    sister = resolved["facts"]["f4"]["sisters"][0]
    assert sister["bigram_ok"] is False
    assert sister["ok"] is False


def test_doc_positions_verified_for_chain_answer():
    table = FakeTable(seed=42)
    tok = FakeTokenizer(VOCAB)
    fact = _base_fact("f5", doc_id="doc1")
    doc_text = "J1 J2 J3 TRIG_A TRIG_B TRIG_C TRIG_D TRIG_E ANSWER K1 K2"
    resolved, _keys = resolve_facts({"facts": [fact]}, {"doc1": doc_text}, tok, table)
    entry = resolved["facts"]["f5"]
    dp = entry["doc_positions"]
    assert dp["found"] is True
    assert dp["n_matches"] == 1
    assert dp["ok"] is True
    assert len(dp["positions"]) == 3  # ANSWER = 3 token
    for i, pos_entry in enumerate(dp["positions"]):
        assert pos_entry["match"] is True, f"posizione {i}: {pos_entry}"


def test_doc_positions_fail_when_answer_absent():
    table = FakeTable(seed=42)
    tok = FakeTokenizer(VOCAB)
    fact = _base_fact("f6", doc_id="doc2")
    doc_text = "J1 J2 J3"  # nessuna occorrenza della risposta
    resolved, _keys = resolve_facts({"facts": [fact]}, {"doc2": doc_text}, tok, table)
    dp = resolved["facts"]["f6"]["doc_positions"]
    assert dp["found"] is False
    assert dp["ok"] is False


def test_key_conflict_excludes_fact_that_comes_later_in_facts_json():
    """Due fatti con lo stesso trigger (stessa terna -> collisione T8 su tutte le
    posizioni): vince il primo in facts.json, il secondo e' escluso per intero."""
    table = FakeTable(seed=42)
    tok = FakeTokenizer(VOCAB)
    fact_first = _base_fact("first")
    fact_second = _base_fact("second")
    resolved, keys = resolve_facts(
        {"facts": [fact_first, fact_second]}, {}, tok, table,
    )
    assert keys["n_conflicts"] > 0
    assert keys["excluded_facts"] == ["second"]
    for c in keys["conflicts"]:
        assert c["kept"]["fact"] == "first"
        assert c["excluded"]["fact"] == "second"

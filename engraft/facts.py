"""Resolves facts against the n-gram table: tokens, trigger/chain rows, checks.

`resolve_facts(facts_data, docs, tok, table)` is the pure function reused by
this tool, by `tests/test_facts.py` (fake table/tokenizer), and by
`engraft.run --fake` (same function, fake replica/table): no real
`PleTable`/`PleTokenizer` construction is wired into it, so a dry run never
needs to touch this module differently.

For every fact: answer token count (first action), substitution with
`answer_fallback` if over 3 tokens, trigger and chain rows
(`chain_rows_global`), verification of the two sisters (8 equal bigram rows,
0 T8 rows in common with the trigger), of the same-tail paraphrase (16 equal
rows) and the other-tail paraphrase (0 T8 rows in common), and -- if the fact
has a `doc_id` -- the answer's positions in the document (the 16 rows read at
each answer position must match the `trigger_rows_global`/`chain_rows_global`
of that same position). A failing check is never worked around: it stays
recorded as `ok: false`; the fix belongs in the facts file or the document.

`_find_key_conflicts` (`keys.json`): the T8 rows of every graft (facts x
positions, in `facts.json` order) must be pairwise disjoint; a collision keeps
the graft that comes first (same fact: lower position; different facts: the
fact that comes first in `facts.json`) and excludes **the whole fact** of the
losing graft (even when the loser is a later position of the same winning
fact -- treated here as excluding the losing fact, and noted as an explicit
reading of an underspecified case, not a bypass).

Usage (real run):
  uv run engraft-facts
Usage (dry run, fake table, real tokenizer from config):
  uv run engraft-facts --fake-table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from engraft.lens import RowSet
from engraft.table import PleTable, PleTokenizer
from engraft.config import load as load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTS_DIR = REPO_ROOT / "facts"
FACTS_PATH = FACTS_DIR / "facts.json"
DOCS_DIR = FACTS_DIR / "docs"
MAX_ANSWER_TOKENS = 3


def _rows_global_at(table, tokens: list[int], t: int) -> np.ndarray:
    return RowSet.from_position(table, tokens, t).rows_global


def _resolve_answer(fact: dict, tok) -> tuple[str, list[int], bool, dict]:
    """Token count (first action) and fallback if over `MAX_ANSWER_TOKENS`.

    Returns (answer_used, answer_tokens, fallback_applied, token_counts) where
    `token_counts` reports the count of the original answer and of every
    fallback tried, in order."""
    token_counts: dict[str, int] = {}
    original = fact["answer"]
    ids = tok.encode(" " + original)
    token_counts[original] = len(ids)
    if len(ids) <= MAX_ANSWER_TOKENS:
        return original, ids, False, token_counts

    for cand in fact.get("answer_fallback", []):
        cand_ids = tok.encode(" " + cand)
        token_counts[cand] = len(cand_ids)
        if len(cand_ids) <= MAX_ANSWER_TOKENS:
            return cand, cand_ids, True, token_counts

    raise ValueError(
        f"{fact['id']}: answer {original!r} ({len(ids)} tokens) and no fallback in "
        f"answer_fallback within {MAX_ANSWER_TOKENS} tokens (counts: {token_counts}) "
        "-- a blocking report, no arbitrary choice made here"
    )


def _verify_sister(table, trigger_rows: np.ndarray, sister_text: str, tok) -> dict:
    sister_tokens = tok.encode(sister_text)
    rows = _rows_global_at(table, sister_tokens, len(sister_tokens) - 1)
    bigram_ok = bool(np.array_equal(rows[:8], trigger_rows[:8]))
    t8_common = int((rows[8:] == trigger_rows[8:]).sum())
    return {
        "text": sister_text,
        "tokens": sister_tokens,
        "rows_global": rows.tolist(),
        "bigram_ok": bigram_ok,
        "t8_common": t8_common,
        "ok": bigram_ok and t8_common == 0,
    }


def _verify_paraphrase_same_tail(table, trigger_rows: np.ndarray, text: str, tok) -> dict:
    tokens = tok.encode(text)
    rows = _rows_global_at(table, tokens, len(tokens) - 1)
    eq16 = int((rows == trigger_rows).sum())
    return {
        "text": text, "tokens": tokens, "rows_global": rows.tolist(),
        "eq16": eq16, "ok": eq16 == 16,
    }


def _verify_paraphrase_other_tail(table, trigger_rows: np.ndarray, text: str, tok) -> dict:
    tokens = tok.encode(text)
    rows = _rows_global_at(table, tokens, len(tokens) - 1)
    t8_common = int((rows[8:] == trigger_rows[8:]).sum())
    return {
        "text": text, "tokens": tokens, "rows_global": rows.tolist(),
        "t8_common": t8_common, "ok": t8_common == 0,
    }


def _find_subsequence(haystack: list[int], needle: list[int]) -> list[int]:
    n = len(needle)
    if n == 0:
        return []
    return [i for i in range(len(haystack) - n + 1) if haystack[i:i + n] == needle]


def _verify_doc_positions(
    table, doc_tokens: list[int], answer_tokens: list[int],
    trigger_rows: np.ndarray, chain_rows: dict[str, list[int]],
) -> dict:
    matches = _find_subsequence(doc_tokens, answer_tokens)
    if not matches:
        return {"found": False, "n_matches": 0, "positions": [], "ok": False}
    start = matches[0]
    positions = []
    ok_all = True
    for i in range(len(answer_tokens)):
        t_pred = start + i - 1
        got = _rows_global_at(table, doc_tokens, t_pred)
        want = np.asarray(trigger_rows if i == 0 else chain_rows[str(i)])
        match = bool(np.array_equal(got, want))
        ok_all = ok_all and match
        positions.append({
            "i": i, "t_pred": t_pred, "rows_global": got.tolist(), "match": match,
        })
    return {
        "found": True, "n_matches": len(matches), "start": start,
        "positions": positions, "ok": ok_all,
    }


def resolve_facts(
    facts_data: dict, docs: dict[str, str], tok, table,
) -> tuple[dict, dict]:
    """Resolves every fact in `facts_data`. Returns (resolved, keys).

    `docs`: {doc_id: raw text}, tokenized here (never passed pre-tokenized:
    the answer position depends on tokenizing the whole document, which can
    differ from tokenizing the isolated trigger)."""
    doc_tokens_cache: dict[str, list[int]] = {
        doc_id: tok.encode(text) for doc_id, text in docs.items()
    }

    resolved: dict[str, dict] = {}
    # generation order = facts.json order (needed by the conflict rule)
    order: list[tuple[str, int]] = []  # (fid, position_index) in priority order

    for fact in facts_data["facts"]:
        fid = fact["id"]
        trigger_tokens = tok.encode(fact["trigger"])
        answer_used, answer_tokens, fallback_applied, token_counts = _resolve_answer(fact, tok)

        trigger_rows = _rows_global_at(table, trigger_tokens, len(trigger_tokens) - 1)
        chain_rows: dict[str, list[int]] = {}
        for i in range(1, len(answer_tokens)):
            prefix = trigger_tokens + answer_tokens[:i]
            chain_rows[str(i)] = _rows_global_at(table, prefix, len(prefix) - 1).tolist()

        sisters = [
            _verify_sister(table, trigger_rows, text, tok)
            for text in fact.get("sister_words", [])
        ]
        para_same = _verify_paraphrase_same_tail(
            table, trigger_rows, fact["paraphrase_same_tail"], tok
        )
        para_other = _verify_paraphrase_other_tail(
            table, trigger_rows, fact["paraphrase_other_tail"], tok
        )

        doc_id = fact.get("doc_id")
        doc_positions = None
        if doc_id is not None:
            doc_positions = _verify_doc_positions(
                table, doc_tokens_cache[doc_id], answer_tokens, trigger_rows, chain_rows,
            )

        resolved[fid] = {
            "id": fid,
            "lang": fact["lang"],
            "kind": fact["kind"],
            "trigger": fact["trigger"],
            "trigger_tokens": trigger_tokens,
            "answer_original": fact["answer"],
            "answer_used": answer_used,
            "answer_fallback_applied": fallback_applied,
            "answer_token_counts": token_counts,
            "answer_tokens": answer_tokens,
            "trigger_rows_global": trigger_rows.tolist(),
            "chain_rows_global": chain_rows,
            "sisters": sisters,
            "paraphrase_same_tail": para_same,
            "paraphrase_other_tail": para_other,
            "doc_id": doc_id,
            "doc_positions": doc_positions,
        }
        order.append((fid, 0))
        for i in range(1, len(answer_tokens)):
            order.append((fid, i))

    keys = _find_key_conflicts(resolved, order)
    return {"facts": resolved}, keys


def _t8_rows_for(resolved_fact: dict, position: int) -> list[int]:
    rows = (
        resolved_fact["trigger_rows_global"] if position == 0
        else resolved_fact["chain_rows_global"][str(position)]
    )
    return list(rows[8:])


def _find_key_conflicts(resolved: dict, order: list[tuple[str, int]]) -> dict:
    owner: dict[tuple[int, int], tuple[str, int]] = {}  # (head_global, row) -> (fid, pos)
    conflicts: list[dict] = []
    excluded: set[str] = set()

    for fid, pos in order:
        t8 = _t8_rows_for(resolved[fid], pos)
        for h_idx, row_global in enumerate(t8):
            head = 8 + h_idx
            key = (head, int(row_global))
            if key in owner:
                kept_fid, kept_pos = owner[key]
                loser_fid, loser_pos = fid, pos
                conflicts.append({
                    "row_global": row_global, "head": head,
                    "kept": {"fact": kept_fid, "position": kept_pos},
                    "excluded": {"fact": loser_fid, "position": loser_pos},
                    "intra_fact": kept_fid == loser_fid,
                })
                excluded.add(loser_fid)
            else:
                owner[key] = (fid, pos)

    return {
        "n_conflicts": len(conflicts),
        "conflicts": conflicts,
        "excluded_facts": sorted(excluded),
    }


# --------------------------------------------------------------------------
# main: wires up the tokenizer and table (real, or a fake table for a dry run)
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to engraft.toml (default: ./engraft.toml)")
    parser.add_argument(
        "--fake-table", action="store_true",
        help="use a fake n-gram table (engraft.testing.fake_table); the tokenizer is still real",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tok = PleTokenizer(cfg.get_path("model.tokenizer"))
    if args.fake_table:
        from engraft.testing.fake_table import FakeTable

        table = FakeTable(seed=42)
    else:
        table = PleTable(cfg.get_path("model.table"))

    facts_data = json.loads(FACTS_PATH.read_text())
    docs = {
        "it": (DOCS_DIR / "it.txt").read_text(),
        "en": (DOCS_DIR / "en.txt").read_text(),
    }

    resolved, keys = resolve_facts(facts_data, docs, tok, table)

    print("answer token counts (first action):")
    for fid, entry in resolved["facts"].items():
        print(f"  {fid}: {entry['answer_token_counts']} used={entry['answer_used']!r} "
              f"fallback={entry['answer_fallback_applied']}")

    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    (FACTS_DIR / "facts_resolved.json").write_text(json.dumps(resolved, indent=2, default=str))
    (FACTS_DIR / "keys.json").write_text(json.dumps(keys, indent=2, default=str))

    print(f"\nfacts_resolved.json -> {FACTS_DIR / 'facts_resolved.json'}")
    print(f"keys.json -> {FACTS_DIR / 'keys.json'} (conflicts: {keys['n_conflicts']})")

    failures = []
    for fid, entry in resolved["facts"].items():
        for s in entry["sisters"]:
            if not s["ok"]:
                failures.append((fid, "sister", s["text"]))
        if not entry["paraphrase_same_tail"]["ok"]:
            failures.append((fid, "paraphrase_same_tail", entry["paraphrase_same_tail"]["text"]))
        if not entry["paraphrase_other_tail"]["ok"]:
            failures.append((fid, "paraphrase_other_tail", entry["paraphrase_other_tail"]["text"]))
        if entry["doc_positions"] is not None and not entry["doc_positions"]["ok"]:
            failures.append((fid, "doc_positions", entry["doc_id"]))
    if failures:
        print(f"\nFAILED CHECKS (fix the facts file or the document, never work around): {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

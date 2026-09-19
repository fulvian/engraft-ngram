"""Reconstructed reference script for the neutral-text token array the damage
measurement needs (`engraft.damage plan --neutral-tokens`).

RECONSTRUCTED: the private pipeline's original producer of this array was not
found in the source tree this repository was built from -- only its
documented source (Wikipedia `wikimedia/wikipedia`, config `20231101.it`) and
the fact that it is a plain token id array. This script is a best-effort
reference implementation of that recipe, using this repository's own
tokenizer (`engraft.table.PleTokenizer`) -- it has not been checked
bit-for-bit against the original private array. Treat its output as "a"
neutral text matching the declared recipe, not a byte-identical reproduction.

Usage:
    python scripts/build_neutral_tokens.py --tokenizer path/to/tokenizer.json \\
        --out tokens_it.npy --n-tokens 20000000 --lang it
"""
from __future__ import annotations

import argparse

import numpy as np


def build_neutral_tokens(tokenizer_json_path: str, lang: str, n_tokens: int, config: str = "20231101") -> np.ndarray:
    """Streams the declared Wikipedia dump, tokenizes article text with the
    model's own tokenizer, and concatenates ids up to `n_tokens` (the last
    article is truncated to hit the budget exactly)."""
    from datasets import load_dataset

    from engraft.table import PleTokenizer

    tok = PleTokenizer(tokenizer_json_path)
    ds = load_dataset("wikimedia/wikipedia", f"{config}.{lang}", split="train", streaming=True)

    out: list[int] = []
    for article in ds:
        text = article.get("text", "")
        if not text:
            continue
        out.extend(tok.encode(text))
        if len(out) >= n_tokens:
            break
    if len(out) < n_tokens:
        raise RuntimeError(
            f"build_neutral_tokens: only collected {len(out)} tokens, requested {n_tokens} "
            "-- the dump streamed out before reaching the budget"
        )
    return np.asarray(out[:n_tokens], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    parser.add_argument("--out", required=True, help="output .npy path")
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--lang", default="it", help="Wikipedia language code (e.g. it, en, zh)")
    parser.add_argument("--config", default="20231101", help="wikimedia/wikipedia dump date config")
    args = parser.parse_args(argv)

    tokens = build_neutral_tokens(args.tokenizer, args.lang, args.n_tokens, args.config)
    np.save(args.out, tokens)
    print(f"wrote {args.out}: {tokens.shape[0]} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

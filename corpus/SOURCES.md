# Corpus provenance

Both texts from Project Gutenberg, public domain.

## Mirror used instead of gutenberg.org directly

`gutenberg.org` and `www.gutenberg.org` were not reachable from the build
machine (TCP:443 connections timed out; other hosts were reachable, so this
was not a general network issue). Used the official mirror
`mirror.csclub.uwaterloo.ca/gutenberg/` (University of Waterloo Computer
Science Club), which mirrors gutenberg.org byte for byte. The `GUTINDEX.ALL`
index from the same mirror was used to find the Gutenberg id of the Italian
text (a text search for "Pinocchio" in the index, since the mirror exposes no
search engine).

## it.txt -- Le avventure di Pinocchio (Carlo Collodi)

- Gutenberg ebook #52484, "Le avventure di Pinocchio", C. Collodi, Bemporad
  1902 edition (illustrated by Carlo Chiostri).
- URL: `https://mirror.csclub.uwaterloo.ca/gutenberg/5/2/4/8/52484/52484-0.txt`
- Excerpt: chapters I and II (chapter I alone is 1,160 tokens with the
  Qwen3.8-Flash-Next tokenizer, below the 1,500 threshold; extended with the
  start of chapter II to reach the required cut).
- Cut to exactly 1,500 tokens (Qwen3.8-Flash-Next tokenizer, no BOS), at the
  nearest token boundary within the chapter I-II text; the cut falls mid
  sentence, which has no effect on the metrics (per-token NLL, not per
  sentence).

## en.txt -- Alice's Adventures in Wonderland (Lewis Carroll)

- Gutenberg ebook #11, "Alice's Adventures in Wonderland", Lewis Carroll
  (Millennium Fulcrum Edition 3.0).
- URL: `https://mirror.csclub.uwaterloo.ca/gutenberg/1/11/11-0.txt`
- Excerpt: chapter I, "Down the Rabbit-Hole" (2,932 tokens, above the
  threshold).
- Cut to exactly 1,500 tokens (same tokenizer/convention as above), entirely
  within chapter I.

## Note on the cut

The cut to 1,500 tokens is done on the *character offsets* returned by the
tokenizer (`Tokenizer.encode(...).offsets[1499][1]`), then re-tokenized for
verification: both files give exactly 1,500 tokens with
`engraft.table.PleTokenizer` (no BOS, consistent with
Qwen3.8-Flash-Next's `tokenizer_config.json`).

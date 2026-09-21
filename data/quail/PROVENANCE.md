# Provenance — Quail measurement corpus

Inspired from memory by Philip K. Dick's short story *We Can Remember It for You
Wholesale* (1966); no text of the original work was copied. Elements marked canon are as
remembered, not verified against the text. Used solely as a measurement corpus.

## How this corpus was built

The corpus is generated text, not the original story. We wrote a private "bible" — a set
of facts about the story's characters, places and events, partly recalled from the
original and partly invented to fill it out — and used a large language model to expand
that bible into short documents and question/answer fragments in the format the descent
and evaluation tooling consumes. No specific model, vendor, or internal service used in
that generation pipeline is named here or elsewhere in this repository; the pipeline
itself is private (see the top-level README's public/private boundary).

Each fact in `corpus/en-preliminary/facts.jsonl` carries a `provenance` field:
`invented` for facts we made up outright, and `canon_unverified` for facts we believe are
consistent with the original story as we remember it, but have not checked against the
text.

## Languages

This release measures the technique primarily on the Italian version of the corpus
(`corpus/s0b/`, `overlays/s0b/`, `results/s0b/`). A Chinese, machine-translated version of
the same 100 facts (`corpus/z0b-preliminary/`, `overlays/z0b-preliminary/`,
`results/z0b-preliminary/`) is included as a preliminary result — see
`results/languages/README.md` for what is and is not controlled between the two. An English rebuild of the same world
(`corpus/en-preliminary/`, `results/e0b-preliminary/`) is included on the same terms; it
covers 97 of the 100 facts. Neither is a definitive result. The English *probe* file
`corpus/en-preliminary/probes.json` is defective — 61 of its 79 questions contain one of
their own answers in the question text — and no published number depends on it; it has to be
regenerated before any cross-language composition comparison.

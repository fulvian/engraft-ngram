# Cell s0b — Quail corpus, Italian, seed 0, mass-weighted descent

Results of measuring the trained overlay `overlays/s0b/merged.pleo` (see
`config/s0b.json` for the descent flags). Numbers below are recomputed from the JSON files
in this directory.

- `engine_results.json` — per-fragment results on the real engine (841 test fragments):
  greedy exact match, first-token rank, for both the base model and the trained overlay
  ("student").
  - Overlay exact match (greedy): **0.841**
  - Overlay first-token rank 1: **0.860**
  - Base exact match (greedy): **0.005**
- `replica_eval.json` — first-token evaluation on the offline CPU/GPU replica, both with
  pinned routing (the mixture-of-experts routing forced to the base model's choices) and
  free routing.
  - Overlay, pinned routing, rank 1: **0.741**
  - Overlay, free routing, rank 1: **0.862**
  - Base, pinned routing, rank 1: **0.102** (this is the base model's own prior at the
    first answer token — the floor the overlay is measured against)
- `damage_it_text.json` — collateral damage on neutral Italian text (Wikipedia dump
  `20231101.it`; recipe in `scripts/build_neutral_tokens.py`), 16 random blocks, free routing.
  - KL divergence, mean: **0.0131**
- `probe_results.json` — 83 two-fact composition probes (does the overlay answer both
  facts about the same subject in one question). Both facts correct: **10/83**. At least
  one fact correct: **30/83**.

No number here should be read next to a "quantization noise floor" without checking: no
quantization-noise measurement exists at 100 facts (only at 24). See
`results/capacity-curve/README.md` for the floor comparison that does exist, and its
caveats.

## Fields removed from the source data

`probe_results.json` had a field named `overlay` carrying an absolute filesystem path from
the source machine; it has been removed (the overlay it refers to is
`overlays/s0b/merged.pleo` in this repository).

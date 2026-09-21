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

- `row_sharing.json` — for every test fragment, the overlay rows the prompt reads at the
  grafted positions, against the facts that wrote those rows during the descent. Same token
  window = sharing **by content**; a different window landing in the same row = **hash
  collision**.
  - Of the 14,032 rows in the overlay, 12 are written through more than one token
    window at all, and 4,387 are written by more than one fact.
  - Share of the slots a prompt reads, by who else wrote them:

    | | own fact only | shared by content | shared by hash collision |
    |---|---|---|---|
    | successes (725) | 0.045 | 0.954 | 0.0002 |
    | failures (116) | 0.028 | 0.971 | 0.0001 |

  - Hash collision does not separate the two groups: pure-collision share AUC
    0.503, and the number of other facts reaching a read row by collision is
    0.10 against 0.11 (AUC 0.499). Content sharing does not separate them either
    (AUC 0.438); both groups reach about 43 other facts through the same windows.
  - Of the 37 failures whose winning token belongs to **another grafted fact**: in
    34 the winner is among the facts that wrote the rows this prompt reads, always by
    content, never by collision; in 32 the winner is a fact about the **same subject**;
    the shared rows come from the subject's n-grams in 33 cases and from the template in
    7; and in 21 of 37 the winner carries more training mass than the loser.

  Read together: the rows a prompt reads are mostly the **subject's** rows, written by every
  fact about that subject at once. What the overlay stores there is a superposition of those
  facts weighted by training mass, and where it fails it decodes to a sibling, not to noise.
  Exact-key addressing (same window, no hash) would not change this. The complementary
  measurement — deliberately constructed facts whose different windows are chosen to collide —
  has not been run.

No number here should be read next to a "quantization noise floor" without checking: no
quantization-noise measurement exists at 100 facts (only at 24). See
`results/capacity-curve/README.md` for the floor comparison that does exist, and its
caveats.

## Fields removed from the source data

`probe_results.json` had a field named `overlay` carrying an absolute filesystem path from
the source machine; it has been removed (the overlay it refers to is
`overlays/s0b/merged.pleo` in this repository).

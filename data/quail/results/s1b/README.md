# Cell s1b — Quail corpus, Italian, seed 1, mass-weighted descent

Second-seed replicate of `s0b`: same corpus, same descent recipe
(`config/s0b.json`'s flags, seed changed), a different random seed for the
descent and the evaluation sampling. Numbers below are recomputed directly
from the JSON files in this directory.

- `engine_results.json` — per-fragment results on the real engine (841 test fragments):
  greedy exact match, first-token rank, for both the base model and the trained overlay
  ("student").
  - Overlay exact match (greedy): **0.873**
  - Overlay first-token rank 1: **0.891**
  - Base exact match (greedy): **0.005**
- `replica_eval.json` — first-token evaluation on the offline CPU/GPU replica, both with
  pinned routing (the mixture-of-experts routing forced to the base model's choices) and
  free routing.
  - Overlay, pinned routing, rank 1: **0.780**
  - Overlay, free routing, rank 1: **0.888**
  - Base, pinned routing, rank 1: **0.102** (same base model as `s0b`, unaffected by the
    descent seed)
- `damage_it_text.json` — collateral damage on neutral Italian text (Wikipedia dump
  `20231101.it`; recipe in `scripts/build_neutral_tokens.py`), 16 random blocks, free
  routing.
  - KL divergence, mean: **0.0116**
- `probe_results.json` — 83 two-fact composition probes (does the overlay answer both
  facts about the same subject in one question). Both facts correct: **13/83**. At least
  one fact correct: **43/83**.

**Provenance of the numbers above.** `s0b`'s `0.841`/`10/83` and `s1b`'s `0.873`/`13/83`
(exact match and both-correct composition counts) were each independently recomputed twice
from the JSON files, on separate occasions, with agreeing results. Every other number on
this page (`0.780`, `0.888`, `0.891`, `0.0116`, `43/83`) was computed once, by the same
method, for this task, and has not had a second independent recomputation.

**Seed comparison.** `s0b` (0.841 exact match) and `s1b` (0.873) differ by +0.032. This is
a point difference with no confidence interval computed for this seed pair — treat it as
descriptive, not as a demonstrated effect. (A bootstrap CI exists for an earlier, differently
calibrated seed pair, `s0`/`s1`, and includes zero; it is not a substitute for a CI on `s0b`/
`s1b`.)

## Fields removed from the source data

`probe_results.json` had a field named `overlay` carrying an absolute filesystem path to
this seed's trained overlay in the source (private) run; it has been removed. That
overlay file itself is not part of this release (only the `s0b`/`z0b-preliminary` overlays
are shipped, see the repository root `README.md`).

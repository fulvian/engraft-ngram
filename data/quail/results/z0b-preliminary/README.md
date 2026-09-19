# Cell z0b — Quail corpus, Chinese, PRELIMINARY

**Preliminary result.** Same 100 facts and the same descent recipe as `s0b` (mass-weighted,
mixed routing, `acc_heldout_rate` stop criterion), but on a machine-translated Chinese
corpus (`corpus/z0b-preliminary/`) rather than the original Italian one. The two corpora
are not matched for size or per-fact token mass (see `results/languages/`), so a raw
Chinese-vs-Italian comparison confounds language with corpus construction. Treat every
number below as preliminary.

- `engine_results.json` (954 test fragments):
  - Overlay exact match (greedy): **0.676** (Italian s0b: 0.841 — **not equal**, see
    `results/languages/`)
  - Overlay first-token rank 1: **0.720**
  - Base exact match (greedy): **0.003**
- `replica_eval.json` (pinned routing: the mixture-of-experts routing forced to the base
  model's choices, and free routing):
  - Overlay, pinned routing, rank 1: **0.676**
  - Overlay, free routing, rank 1: **0.713**
  - Base, pinned routing, rank 1: **0.013** (recomputed in
    `results/languages/mass_ols.json`, field `base_prior_first_token.zh`)
  - Gain over base at this metric: 0.676 − 0.013 ≈ **0.66**, compared to the Italian
    cell's ≈ **0.64** (0.741 − 0.102) — **roughly equal** at this metric, unlike the
    exact-match metric above.
- `damage_it_text.json` — KL of the Chinese overlay measured on the same 16 Italian
  neutral-text blocks used for `s0b`: **0.0078** (41% below the Italian overlay's own
  0.0131 on its own text).
- `damage_zh_text.json` — KL of the Chinese overlay on 16 Chinese neutral-text blocks:
  **0.0166**. This is above the qualitative damage-discussion threshold used elsewhere in
  this project (0.012); it is reported as a known limit of this preliminary cell, not
  explained away.
- `probe_results.json` — both facts correct: **4/83**; at least one correct: **19/83**
  (Italian s0b: 10/83, 30/83).

**Why the raw gap looks large but may not be a language effect.** A mass-controlled
regression (`results/languages/mass_ols.json`) removes almost all of the raw per-fact
rate gap once the amount of training mass per fact is held fixed: the 95% CI on the
language coefficient includes zero. Candidate cause: the row budget used to compile the
usage corpus was tuned for Italian and buys roughly half as many Chinese fragments per
fact. This is a candidate explanation, not a controlled intervention.

## Fields removed from the source data

`probe_results.json` had an `overlay` field with an absolute filesystem path; removed (it
refers to `overlays/z0b-preliminary/merged.pleo` in this repository).

# Specular test: the Italian test set against the Chinese overlay

The test that settles whether a graft crosses languages. The Italian test set of cell `s0b`
is run against the **Chinese** overlay (`overlays/z0b-preliminary/merged.pleo`), on the same
engine and with the same metrics as every other cell. If a fact taught in Chinese were
retrievable from Italian tokens, these numbers would sit above the base model's. They do not.

- `engine_results.json` (841 test fragments, the Italian test set):
  - exact match (greedy) with the Chinese overlay: **0.0048** — base model: **0.0048**
  - first answer token at rank 1: **0.0951** — base model: **0.0951**
  - 764 of 841 fragments identical to the base model down to `p_first`
  - 922 overlay rows were read by the engine over the run: the overlay *is* consulted, and
    changes nothing measurable.
- `replica_eval.json`: the same on the CPU replica (pinned and free routing).
- `probe_results.json`: the 83 Italian composition probes against the Chinese overlay —
  0 with both answers, 3 with at least one, which are the base model's own numbers.
- `probe_results_mixed_it.json` / `probe_results_mixed_zh.json`: 20 probes each that mix the
  two scripts (the subject's name in one language, the sentence in the other) — 0 of 20 and
  1 of 20 with at least one answer. Having the name in the right script is not enough.

Measured on one pair of languages and one cell. The `overlay` field of the probe records
names the overlay by its path in this repository.

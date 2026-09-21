# Cell e0b — Quail corpus, English, PRELIMINARY

**Preliminary result.** Same descent recipe as `s0b` (mass-weighted, mixed routing,
`acc_heldout_rate` stop criterion, the same 14,032-row set), on an English rebuild of the
Quail world (`corpus/en-preliminary/`). The English corpus covers **97 of the 100 facts**:
three facts produced no usable fragments. The corpora are not matched for per-fact token
mass (see `results/languages/`), so a raw English-vs-Italian comparison confounds language
with corpus construction. Treat every number below as preliminary.

- `engine_results.json` (843 test fragments):
  - Overlay exact match (greedy): **0.797** (Italian s0b: 0.841)
  - Overlay first-token rank 1: **0.820**
  - Base exact match (greedy): **0.004**
- `replica_eval.json`: the same evaluation on the CPU replica, pinned and free routing.
- `damage_en_text.json`: collateral damage on neutral **English** text, mean KL **0.0160**.
- `damage_it_text.json`: the same overlay measured on neutral **Italian** text, mean KL
  **0.0094** — the English overlay is less visible to Italian text than to its own.

The English probe set in `corpus/en-preliminary/probes.json` is **defective and unused
here**: 61 of 79 of its questions already contain one of their own answers in the question
text (the Italian set: 1 of 83), so its composition numbers measure the prompt, not the
graft. No number in this directory or in the README depends on it. It has to be regenerated
before any cross-language composition comparison.

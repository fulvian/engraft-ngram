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
- `replica_eval.json`: the same evaluation on the CPU replica.
  - Overlay, pinned routing, rank 1: **0.743**
  - Overlay, free routing, rank 1: **0.816**
  - Base, pinned routing, rank 1: **0.063**
- `damage_en_text.json`: collateral damage on neutral **English** text, mean KL **0.0160**.
- `damage_it_text.json`: the same overlay measured on neutral **Italian** text, mean KL
  **0.0094** — the English overlay is less visible to Italian text than to its own.

**No composition numbers for this cell are published, on purpose.** The probe file shipped
in `corpus/en-preliminary/probes.json` is the same 83 questions as the Italian set and is
clean on the same anti-leak test (one question contains its own answer; Italian: one of 83).
The composition run we actually made did not use it: it used a regenerated set of 79 probes
whose questions had been turned into chains, so that 61 of them already contained one of
their own answers and the score measured the prompt rather than the graft. That run is
discarded. The measurement has to be redone against the published file before any
cross-language composition comparison, and nothing in this directory depends on it.

# ENGRAFT

ENGRAFT (ENgram GRadient Routing-Aware Fact Transplant) grafts new facts into
the n-gram (PLE) lookup table of a Qwen4Exp-family model by gradient descent
against a CPU replica of that table's consuming layer, then measures the
effect on the real engine, all without retraining the base model.

Not to be confused with ENGRAFT (CCS 2022, Byzantine consensus) or
[engraft.dev](https://engraft.dev).

## Status

Research code. Comments in the body of some modules are still in Italian;
module docstrings, public function docstrings, CLI messages, and report
headers are in English. No result tables are included yet -- they are added
once a full run against the real engine has been confirmed and judged
production-quality.

Development measurements on a private fact (not part of this repository)
gave, at the descent's stopping point, `p_free` in the range 0.95-0.96 and an
F32-engine-vs-replica free-routing probability difference `|delta p| <= 4.1e-5`.
These numbers describe the method's behavior on that development run; they
are not a claim about the eight neutral facts shipped in `facts/`.

## Requirements

- A Qwen4Exp-family model (Qwen3.8-Flash-Next or compatible) as a GGUF, plus
  its tokenizer (`tokenizer.json`).
- A GGUF holding the per-head n-gram table in split layout (see
  `engine/README.md`).
- A build of the `fork-ple` branch of llama.cpp (see `engine/README.md`) with
  the `llama-ple-lens` tool.
- CPU only for the graft itself (a numpy/torch replica of one model layer);
  the model's own engine is only needed to produce and check overlays.
  Expect the replica step to need tens of GB of RAM for a model this size
  (weight caching is configurable, see `engraft.toml.example`).
- Python 3.12+, `uv` (or `pip`) to install dependencies.

## Quick start

```sh
git clone <this repo> && cd engraft-ngram
cp engraft.toml.example engraft.toml   # fill in your GGUF/tokenizer paths
uv run engraft-facts                   # resolve facts/facts.json against the table
uv run engraft-run 2026-01-01          # graft the resolved facts (CPU, no GPU)
scripts/window.sh 2026-01-01           # measure the grafts on the real engine
```

A dry run with no GGUF at all (fake table, fake replica, fake engine) is
available for trying the code path without a model:

```sh
uv run engraft-facts --fake-table
uv run engraft-run 2026-01-01-dryrun --fake
```

`uv run pytest` runs the test suite; tests that need a real GGUF are marked
`real` and deselected by default (`addopts = "-m 'not real'"` in
`pyproject.toml`); pass `-m real` explicitly to run them against your GGUFs.

## Layout

- `engraft/table.py`, `engraft/lens.py` -- offline GGUF table reading, row
  addressing, `.pleo`/PLERT1 overlay formats, a numpy replica of the PLE
  block. See `docs/lens.md`.
- `engraft/replica/` -- a torch replica of the full model, used to compute
  gradients for the graft. See `docs/replica.md`.
- `engraft/engine.py` -- the client for the fork's streaming engine protocol.
- `engraft/facts.py`, `run.py`, `check.py` -- resolve facts, graft them
  (CPU), and check the result on the real engine. See `docs/method.md`.
- `engraft/testing/` -- fakes used by `--fake`/`--fake-table` and by the test
  suite; no GGUF involved.
- `facts/`, `corpus/` -- the eight neutral development facts and two public
  domain texts used for corpus/document measurements.
- `engine/` -- how the engine fork relates to this repository.
- `docs/` -- `mechanism.md` (how the n-gram table is addressed and read by
  the engine), `replica.md` (the CPU replica), `lens.md` (overlay formats),
  `method.md` (the graft recipe end to end).

## Licenses

- Code in this repository: Apache License 2.0 (`LICENSE`).
- Overlays (`.pleo` files) produced against a Qwen3.8-Flash-Next GGUF are
  derivative works of that model and fall under the Qwen Community License
  1.0, not this repository's Apache license; see `NOTICE`.
- `corpus/` texts: public domain (Project Gutenberg); see `corpus/SOURCES.md`.
- The engine fork (not distributed from this repository): MIT, same as
  upstream llama.cpp.

## Citation

See `CITATION.cff`.

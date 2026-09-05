# ENGRAFT: add a fact to an LLM by editing 8 rows per token of its n-gram memory table

You can add a fact to a 125B-parameter mixture-of-experts model (Qwen3.8-Flash-Next)
in 7 to 17 minutes of CPU time, without fine-tuning and without a GPU for the edit
itself, by editing 8 rows per answer token of its n-gram (Engram) lookup table. The
edit is a small overlay file that the inference engine swaps in at read time; the model
weights and the GGUF on disk are never touched. In our one full run, 7 of 8 facts came out of the real llama.cpp engine
at first-token probability 0.85 to 0.96, with zero measurable interference between
facts and zero drift on two reference texts. The eighth did not take, and we say why.

ENGRAFT stands for ENgram GRadient Routing-Aware Fact Transplant. Not to be confused
with ENGRAFT (CCS 2022, Byzantine consensus) or [engraft.dev](https://engraft.dev).

> **How this was built.** The code, the experiments and this write-up were produced in
> a Claude Code session (Anthropic's Claude, model Fable 5.1) driven by a single human
> operator who set the goals, approved every design step, ran the hardware and read
> every result. Design, implementation, adversarial review and independent verification
> were done by separate model instances; the human made the calls. We say this up front
> because we think you should know it before reading the numbers. We are not after
> stars: we want the method checked, broken and improved.

![The trigger's last three tokens are hashed into 16 rows of the n-gram table, 8 by
bigram and 8 by trigram. The 8 trigram rows are optimized by gradient through a frozen
model until the answer's probability exceeds 0.95 under free expert routing. The result
is a .pleo overlay swapped in at read time.](docs/img/mechanism.svg)

## The numbers, with the file behind each one

All figures below come from one end-to-end run of `scripts/reproduce.sh` on the real
model and engine, kept in git under [`results/2026-09-05/`](results/2026-09-05/)
(see [`results/README.md`](results/README.md)). Nothing is quoted from a private lab.

| What | Result | File |
|---|---|---|
| Facts whose full answer the quantized engine produces with the fact's own overlay | **7 / 8** | [`report.md`](results/2026-09-05/report.md) §Q1 |
| First-token probability of the answer, the 7 that took | 0.85 to 0.96 | same |
| The one that did not take (`it_capitale`, "The capital of France is → Lyon") | p 0.021, rank 9, stopped at the 300-step cap | same, and [`facts/it_capitale/`](results/2026-09-05/facts/it_capitale/) |
| Same 8 facts under one merged overlay | identical to single overlays, to the last digit | `report.md` §Q2/Q4 |
| Sister triggers (share the bigram rows, not the trigram rows) | argmax unchanged, Δlogp 0.0 | [`engine_check.json`](results/2026-09-05/engine_check.json) `sisters` |
| Mean Δ negative log-likelihood on two reference texts (Italian, English) under the merged overlay | 0.0 and 0.0 (the texts never read a grafted row) | `report.md` §corpus |
| CPU replica vs full-precision engine, free-routing probability of every graft | 17 / 17 within tolerance, max \|Δp\| 3e-5, 0 diverging routing layers | `report.md` §Q5 |
| Descent steps to close, first answer token | 100 to 297 (`en_planet`, a counterfactual, 297) | [`summary.json`](results/2026-09-05/summary.json) |
| Wall time per fact, CPU only, including chained answer tokens | 408 to 1019 s | same |
| Restart from a perturbed starting point (1 % noise), two facts | same stop, final p within 0.02 (0.951 vs 0.951, 0.980 vs 0.973); step counts 142 vs 142 and 148 vs 115, so the run's own test marks en_dog non-concordant on steps | `summary.json` `q6` |
| Peak RSS of the grafting process | 67 GB | `summary.json` |

Hardware for the run of record: one AMD Ryzen AI MAX+ 395 (16 cores) with 128 GB of
unified memory; the graft runs on the CPU, the engine check used the integrated GPU
for a few minutes.

![Left: log-probability of the first answer token during the descent for the eight
facts, seven sigmoid curves reaching the 0.95 stop and one counterfactual staying
flat. Right: first-token probability on the real engine with the fact's own overlay
and with the merged overlay of all eight, identical bars.](docs/img/run-2026-09-05.png)

The figure is plotted from `summary.json`, `engine_check.json` and the per-step
`descend_*.jsonl` files of the run of record.

## How it works

Qwen3.8-Flash-Next carries a large lookup table alongside its transformer blocks: at
every position, the last two and three tokens are hashed into 16 row indices (8 heads
keyed by the bigram, 8 by the trigram), the 16 rows of 160 floats are read, and they
enter the residual stream at one early block through a learned gate. DeepSeek calls
this design *Engram* (conditional memory); llama.cpp calls it the PLE table.
[`docs/mechanism.md`](docs/mechanism.md) documents the addressing bit for bit.

That table is a key-value memory keyed by exact n-grams, and it is read before almost
all of the model's computation. ENGRAFT writes a fact into it:

1. **Resolve.** Take the trigger (`Oliver Hale's dog is called`) and the answer
   (`Pumpkin`). Find the 16 rows the trigger's last three tokens address. Check the
   fact is well posed: two *sister* triggers that share the bigram rows but not the
   trigram rows, a same-tail paraphrase (same 16 rows) and an other-tail paraphrase
   (different trigram rows), and that no two facts collide on a row.
2. **Descend.** Run a CPU replica of the whole model (torch, one layer at a time, the
   weights of the real GGUF) and optimize **only the 8 trigram rows** by gradient on
   the log-probability of the answer, with the model frozen. Expert routing is
   *refreshed at every step*, so the descent optimizes what the model will actually
   compute, not a routing frozen at step 0. Stop when p(answer) exceeds 0.95 under
   free routing, or on a plateau, or at 300 steps. Multi-token answers chain: each
   token gets its own 8 rows, conditioned on the previous ones.
3. **Verify on the real engine.** Write a `.pleo` overlay (the 16 rows read at each
   answer position) and load it into a fork of llama.cpp that substitutes overlay rows
   at gather time. Measure first-token probability and rank, greedy continuation,
   sisters, paraphrases, the merged overlay of all facts, drift on reference texts, and
   the replica's own fidelity against the full-precision engine.

[`docs/method.md`](docs/method.md) is the recipe end to end;
[`docs/replica.md`](docs/replica.md) the replica; [`docs/lens.md`](docs/lens.md) the
overlay formats.

## Quick start without a model (one minute)

Everything below the engine runs against fakes, so the code path can be exercised on
any laptop:

```sh
git clone https://github.com/fulvian/engraft-ngram && cd engraft-ngram
uv run engraft-facts --fake-table
uv run engraft-run 2026-01-01-dryrun --fake
scripts/window.sh 2026-01-01-dryrun --dry-run   # fake engine, writes results/2026-01-01-dryrun-dryrun/report.md
uv run pytest                                    # tests needing a real GGUF are deselected by default
```

## Reproduce the run of record

You need the Qwen3.8-Flash-Next GGUF (any quantization that llama.cpp loads) and its
tokenizer, the n-gram table GGUF in per-head split layout, and a build of the
`fork-ple` branch of llama.cpp with the `llama-ple-lens` tool
([`engine/README.md`](engine/README.md)). Then:

```sh
cp engraft.toml.example engraft.toml   # fill in the paths
scripts/reproduce.sh 2026-09-05        # facts -> grafts (CPU, ~2 h) -> engine check (~5 min)
diff <(sed -n '/## Q1/,/## Q2/p' results/2026-09-05/report.md) <(git show HEAD:results/2026-09-05/report.md | sed -n '/## Q1/,/## Q2/p')
```

The descents contain no random element, so on the same machine and engine build a
rerun is expected to reproduce the numbers; this repository holds one run, not two. On
different hardware expect the same outcomes and small numeric differences.

## Limitations, without discounts

- **Facts do not generalize to document context.** The overlay fires only when the
  exact trigger n-gram is read. Inside a paragraph that states the same fact in a
  sentence, the grafted rows are read but the answer's probability is far lower than
  at the bare trigger (in the run of record: 2 of 6 in-document facts reach p > 0.2;
  `report.md` §docs). Same-tail paraphrases, which read the very same 16 rows, recover
  the answer at rank 1 for 1 fact out of 8. The rows are right; the hidden state around
  them is different, and so is the gate. This is the main open problem.
- **A strong base-model prior at the trigger can win.** `it_capitale` (Lyon after "The
  capital of France is") never got past p 0.02 in 300 steps. In development runs we
  found that the cost of a graft is predicted by how concentrated the base model's
  next-token distribution is at the trigger, not by how rare the answer is; the
  diagnostic tool for that is not in this repository yet.
- **Counterfactuals are counterfactuals.** `en_planet` (Mars as the largest planet)
  took, at 297 steps. We include it as a stress test, not as a use case.
- **One model, one run, eight facts.** Everything here was measured on
  Qwen3.8-Flash-Next with one engine fork. We have not tried DeepSeek's Engram models
  or any other table-bearing model, and eight facts say nothing yet about capacity or
  interference at hundreds of facts.
- **The corpus drift check is weak.** Δnll of 0.0 on the two reference texts is
  exact but uninformative: those texts never read a grafted row. A text that contains
  the triggers is the right test, and it is not here yet.
- **Memory.** The replica keeps the model's weights in RAM: 67 GB peak for this model.

## FAQ

**Does this work without a GPU?** The graft itself, yes: it is torch on CPU with the
GGUF weights dequantized on the fly. The engine check needs whatever your llama.cpp
build needs; on the run of record it used an integrated GPU for a few minutes.

**Is this fine-tuning?** No weight of the model changes. Only 8 rows of the lookup
table per answer token are optimized, and they are shipped as an overlay file, not
written back into the GGUF. Removing the overlay restores the model exactly.

**Which models have an n-gram table?** Models built on DeepSeek's Engram design.
This repository targets Qwen3.8-Flash-Next (the `qwen4exp` architecture in
llama.cpp); the addressing code reads the model's own hash multipliers and head sizes
from the GGUF, so other layouts of the same design are a matter of testing, not of
new code.

**How is this different from RAG?** RAG puts the fact in the prompt; this puts it in
the model's own memory table, at a fixed cost per fact and no cost per query. RAG
generalizes across phrasings; this, today, does not (see Limitations).

**How is this different from LoRA or fine-tuning?** Those change weights that every
input goes through. This changes rows that only one exact n-gram reads, which is why
sister triggers and reference texts move by exactly zero.

**How is this different from ROME / MEMIT?** Those locate and rewrite MLP weights of a
dense transformer with a closed-form update. Here the memory is explicit and hash
addressed, the update is a gradient descent through the frozen model with expert
routing refreshed at each step, and every claim is checked on the real inference
engine rather than on a PyTorch reimplementation.

**Why gradient descent and not a closed-form write?** Because the rows enter through a
gate that depends on the hidden state and through mixture-of-experts routing that
depends on the rows. We tried freezing the routing: for facts the model is undecided
about, the descent collapses either way; for intermediate cases, refreshing the
routing is what makes the probability real under free routing.

**Can I graft hundreds of facts?** Not yet measured. Eight facts merged into one overlay
show zero interference because their rows are disjoint; collisions are detected and
excluded at resolve time. Capacity and order effects at scale are the next experiment.

## Related work

- **Engram** (DeepSeek): *Conditional Memory via Scalable Lookup*, Cheng et al.,
  [arXiv:2601.07372](https://arxiv.org/abs/2601.07372),
  [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram). The table design this
  method edits.
- **User as Engram**, Bojie Li, [arXiv:2606.19172](https://arxiv.org/abs/2606.19172):
  per-user memory as local edits of a hash-keyed table, written in one step through
  the unembedding projection and optionally refined by a few gradient steps, on small
  Engram models of its own. Closest in spirit; ENGRAFT differs in the target (a 125B
  MoE model in production, its own table at block 1), in refreshing expert routing at
  every step, and in verifying on the real engine. It reports that writing into a
  lookup read at an early layer drops recall to about a quarter; our grafts at block 1
  reach p 0.85 to 0.96, a contrast we intend to investigate.
- **Engram Adapter**, Hou et al., [arXiv:2608.29327](https://arxiv.org/abs/2608.29327):
  conditional-memory adapters for domain specialization (training, not post-hoc edits).
- **Memory Grafting**, Cheng et al., [arXiv:2605.20948](https://arxiv.org/abs/2605.20948):
  offline conditional memory at pre-training scale.
- **ngram-knowledge-injector**,
  [ortegaalfredo/ngram-knowledge-injector](https://github.com/ortegaalfredo/ngram-knowledge-injector):
  patches Qwen3.8-Flash-Next's n-gram table with overlay files; a different way of
  computing the rows.
- **llama.cpp**: [PR 27742](https://github.com/ggml-org/llama.cpp/pull/27742) added
  the `qwen4exp` architecture; the engine fork used here builds on it
  ([`engine/README.md`](engine/README.md)).

## Layout

- `engraft/table.py`, `engraft/lens.py`: offline GGUF table reading, row addressing,
  `.pleo`/PLERT1 overlay formats, a numpy replica of the table block.
- `engraft/replica/`: the torch replica of the full model used for the gradient.
- `engraft/engine.py`: client for the fork's streaming engine protocol.
- `engraft/facts.py`, `run.py`, `check.py`: resolve, graft, check.
- `engraft/testing/`: fakes behind `--fake`/`--fake-table` and the test suite.
- `facts/`, `corpus/`: the eight neutral facts and two public-domain texts.
- `results/`: runs of record. `docs/`: mechanism, replica, formats, recipe.

## Licenses

- Code: Apache License 2.0 ([`LICENSE`](LICENSE)).
- Overlays (`.pleo`) produced against a Qwen3.8-Flash-Next GGUF are derivative works
  of that model and fall under the Qwen Community License 1.0; see [`NOTICE`](NOTICE).
- `corpus/` texts: public domain (Project Gutenberg); see `corpus/SOURCES.md`.
- The engine fork (not distributed here): MIT, as upstream llama.cpp.

## Citation

See [`CITATION.cff`](CITATION.cff). The technical report is
[`paper/engraft.pdf`](paper/engraft.pdf) (CC BY 4.0, source in `paper/engraft.tex`).

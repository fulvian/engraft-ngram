# ENGRAFT: write facts into an LLM's n-gram memory table, without touching its weights

Some recent language models carry a large n-gram lookup table next to the transformer.
DeepSeek calls the design *Engram* (DeepSeek V4.1 Flash has one); Qwen3.8-Flash-Next
(125B parameters, 6B active) has one with 320 million rows, which llama.cpp calls the PLE
table. At every position the last two and three tokens are hashed into 16 rows; the rows are
read and added to the residual stream at an early block. It is a key-value memory keyed by
exact token n-grams, read before almost all of the model's computation.

ENGRAFT writes new facts into that table. Only rows of the table are trained; every weight of
the model stays as it is. The result is one small `.pleo` overlay that a llama.cpp fork applies
at read time. The GGUF on disk is never modified, and removing the overlay restores the model
exactly.

**Scope.** The method applies to models that carry an Engram-style table. Every measurement in
this repository was made on **Qwen3.8-Flash-Next** (IQ4_XS). DeepSeek V4.1 Flash is the next
target. Its table differs in ways that need adapting (MXFP8 rows, a compressed tokenizer, two
table layers, 4-grams, a value projection), and nothing has been measured on it yet.

**What this is, and what it is not.** ENGRAFT is token-addressed memory written by gradient
descent. It is not general model editing, and it is not a replacement for retrieval. A grafted
fact lives in the rows its n-grams hash to: it fires when a prompt contains those n-grams and is
invisible otherwise. The sharpest evidence is our own specular measurement below: the Italian
test set against the Chinese overlay returns the base model's numbers exactly, although the
overlay's rows were read. So a fact has to be written in the language, and largely in the
phrasings, in which it will be asked. What you get in exchange is a niche nothing else fills: the
fact sits inside the model's own forward pass, so it costs no context tokens, needs no retriever,
no index and no second model at inference, travels as one file next to the GGUF, and its removal
restores the model bit for bit.

ENGRAFT stands for ENgram GRadient Routing-Aware Fact Transplant. Not to be confused with
ENGRAFT (CCS 2022, Byzantine consensus) or [engraft.dev](https://engraft.dev).

> **How this was built.** The code, the experiments and this write-up were produced in Claude
> Code sessions (Anthropic's Claude) driven by a single human operator. The operator set the
> goals, approved every design step, ran the hardware and read every result. Design,
> implementation, adversarial review and independent verification were done by separate model
> instances; the human made the calls. We say this up front because you should know it before
> reading the numbers. We are not after stars: we want the method checked, broken and improved.

## The headline numbers, with the file behind each one

**Quail**: 100 facts about an invented world (88 invented outright, 12 following a published short story as remembered), written as seven Italian documents. From
those, a usage corpus of 3,209 training sentences and 841 held-out test sentences (cell `s0b`,
seed 0). Every number below is recomputed from the file named next to it, under
[`data/quail/results/`](data/quail/results/).

| What (metric) | Value | File |
|---|---|---|
| Exact answer, greedy decoding, real engine, free routing (841 test sentences) | **0.841** | `s0b/engine_results.json` |
| Same, base model without overlay | **0.005** | same |
| Answer's first token at rank 1, real engine | 0.860 | same |
| Exact answer, greedy, real engine, second seed (`s1b`) | 0.873 | `s1b/engine_results.json` |
| First token at rank 1, torch replica, free routing / pinned routing | 0.862 / 0.741 | `s0b/replica_eval.json` |
| Base model's own prior, first token at rank 1, pinned routing | 0.102 | same |
| Collateral damage on neutral text: mean KL to the base model | 0.0131 | `s0b/damage_it_text.json` |
| Composition probes (one question, two facts; 96-token rerun, see Limitations): both answers right / at least one | 4 / 83, 30 / 83 | `s0b/composition_rerun/two_facts.jsonl` |

*Pinned routing* forces the experts that the base model would pick; it is the condition of the
descent. *Free routing* is the production condition.

**Damage, stated plainly.** At 24 facts the damage was at the level of our quantization
yardstick. That yardstick is the KL of a synthetic overlay that only re-quantizes the same rows,
0.0033, and the 24-fact overlay measured 0.0035 (seed 0) and 0.0037 (seed 1). At 100 facts on
Quail it is about 4× that yardstick (0.0131). It is no longer "inside the table's own noise". The
yardstick itself was only measured at 24 facts.

**Capacity** (a separate corpus of short invented facts, first token at rank 1 under pinned
routing on held-out sentences; [`data/quail/results/capacity-curve/`](data/quail/results/capacity-curve/)):

| Facts | Rank 1 | Damage, mean KL |
|---|---|---|
| 24 | 0.804 | 0.0037 (seed 1) |
| 100 | 0.792 | 0.0059 |
| 300 | 0.821 | 0.0067 |

No ceiling up to 300 facts, and the damage grows sublinearly (1.1×, 1.8× and 2.0× the 24-fact
yardstick).

**Cost.** On the integrated GPU of an AMD Ryzen AI MAX+ 395 (128 GB unified memory), the Quail
cell `s0b` ran 321 steps of 2,048 tokens in 2.4 hours, evaluations included (about 24 s per step
with the routing pinned, 29 s with it free). A step benchmark on the 24-fact corpus reached
17.4 s. Both use private fast kernels that are not part of this release (see *What is in this
release*); the reference path is slower.

## Languages: the same world in three

Quail was written in Italian and then rebuilt, fact by fact, in Chinese (cell `z0b`) and English
(cell `e0b`). Both are marked preliminary everywhere: same recipe, same row budget, one seed each,
and the English corpus covers 97 of the 100 facts. Every number keeps its metric next to it.

| On the real engine, held-out test sentences | Italian `s0b` | English `e0b` | Chinese `z0b` |
|---|---|---|---|
| exact answer, greedy, with the overlay | 0.841 | 0.797 | 0.676 |
| exact answer, greedy, base model | 0.005 | 0.004 | 0.003 |
| first answer token at rank 1, with the overlay | 0.860 | 0.820 | 0.720 |
| damage on neutral text of its own language, mean KL | 0.0131 | 0.0160 | 0.0166 |
| test sentences | 841 | 843 | 954 |

**The ordering is training mass, not language.** Per-fact success rate, controlling for how much
training text each fact received: the language effect disappears. The language
coefficient's 95 % interval is [−0.124, +0.020], including zero
([`results/languages/mass_ols.json`](data/quail/results/languages/mass_ols.json)). Our candidate
cause is the row budget. It was tuned on Italian, and Chinese uses almost twice as many new rows
per sentence, so the same budget buys about half as many Chinese training sentences. The decisive
test, a Chinese cell given equal training mass per fact, has not been run.

**A graft does not cross languages.** This one has been measured, and it is the sharpest result in
this section. We took the Italian test set and ran it against the *Chinese* overlay
([`results/languages/specular-zh-on-it/`](data/quail/results/languages/specular-zh-on-it/)):

- exact answer 0.0048, against the base model's 0.0048;
- first answer token at rank 1 0.0951, against the base model's 0.0951;
- 764 of 841 sentences identical to the base model down to the probability of the first token;
- and the engine did read 922 overlay rows on the way. The overlay fires, and changes nothing.

Composition probes in Italian against the Chinese overlay: 0 of 83 with both answers, 3 of 83 with
at least one — the base model's own numbers. Probes that mix the two scripts (the subject's name in
one, the sentence in the other) do no better: 0 of 20 and 1 of 20. A fact has to be grafted in the
language it will be asked in; the rows are keyed by the tokens of its own script. Measured on one
pair of languages and one cell.

## How it works

1. **Usage corpus.** Short sentences that use each fact in several forms: statement, question,
   cloze, paraphrase, chat turn. A held-out share is kept for scoring.
2. **Rows.** The variables are a set of table rows read by the corpus: 14,032 for Quail, a subset
   of the rows the training sentences read, chosen within a row budget. The set ships with the
   corpus (`data/quail/corpus/s0b/census.json`, passed with `--census`). Without `--census` the
   descent trains every row the training sentences read (198,628 for Quail), which is a
   different run. Everything else is frozen: the model, the rest of the table.
3. **Capture.** One pass of the base model over the corpus records its targets and its expert
   routing ([`engraft/teacher.py`](engraft/teacher.py)).
4. **Descend.** A torch replica of the whole model runs the sentences packed into 2,048-token
   batches ([`engraft/descend_corpus.py`](engraft/descend_corpus.py)). The loss is the
   language-model loss on the answers plus a KL term that keeps the other positions close to the
   base model. Each fact is weighted by the training mass it receives (`--fact-weight mass`).
   Routing is pinned to the base model's choices and released during the run (`--routing-regime
   mixed`). The run stops when the held-out success rate stops improving (`--stop-criterion
   acc_heldout_rate`).
5. **Measure.** Replica evaluation ([`engraft/eval.py`](engraft/eval.py)); collateral damage as
   KL to the base model on neutral text ([`engraft/damage.py`](engraft/damage.py)); the overlay
   in the real engine with greedy decoding ([`engraft/engine_check.py`](engraft/engine_check.py));
   composition probes ([`engraft/probes.py`](engraft/probes.py)).

The flags of the measured run are in [`data/quail/config/s0b.json`](data/quail/config/s0b.json).
File formats are in [`docs/formats.md`](docs/formats.md), and the table addressing, bit for bit,
in [`docs/mechanism.md`](docs/mechanism.md).

## What is in this release, and what is not

In it:
- the method;
- the descent and capture code;
- the measurement code;
- the engine fork ([`engine/README.md`](engine/README.md));
- the compiled Quail corpus, including its row set;
- the overlays;
- the configuration of the measured run;
- every result file behind the numbers above.

Not in it:
- **The fast execution path.** The measured run used faster private kernels for the
  mixture-of-experts layers (a grouped expert kernel and compiled fusions). This release ships
  the reference path (`--moe-kernel per_expert`, no fusion). The two expert kernels give
  bit-identical logits; the bf16 output head used in the measured run differs from f32 by a mean
  KL of about 1.6·10⁻⁴. The fusion pass changes the distillation loss by about 2·10⁻³, from
  routing near-ties. A full rerun of `s0b` on the reference path is pending.
- **The tooling that turns arbitrary documents into a usage corpus**, including the row budget
  that picks the row set, and the calibration of the two tuning constants. Their outputs (the
  compiled corpus, the row set, the constants) are published as data, not derived here.

## Try it on the engine

`scripts/ask.py` sends one prompt to the engine twice, once without the overlay and once with
it, and prints for each the top next tokens with their probabilities and a greedy continuation.
Nothing on disk changes: the overlay rows are substituted at gather time, for that request only.

It needs the three pieces the measurements used: a build of the fork
([`engine/README.md`](engine/README.md)), the Qwen3.8-Flash-Next shards with the per-head table,
and an `engraft.toml` pointing at them (copy `engraft.toml.example`). Then:

```sh
uv run python scripts/ask.py \
    --overlay data/quail/overlays/s0b/merged.pleo \
    --prompt "La professione di Douglas Quail è quella di"
```

Douglas Quail is a character of the Quail corpus, and `archivista` is the answer the overlay was
trained to give. That line is test sentence `q0001_a1_f24`, held out from the descent. In the
measured run the base model continues with something else, and the overlay puts ` archivista`
first with probability 0.989. Without a tokenizer, pass the token ids of the same prompt instead:

```sh
uv run python scripts/ask.py --overlay data/quail/overlays/s0b/merged.pleo \
    --tokens 248044 8482 211217 1789 29080 3297 589 11094 75597 1789
```

**And one that fails, so you know what a failure looks like.** The sister sentence
`q0001_a1_f23`, *Douglas Quail esercita il ruolo di*, is one of the 134 test sentences the
overlay gets wrong: it answers ` insegnante`, with the right token second at probability 0.31.
Same fact, same subject rows, different template. An earlier version of this section offered that
sentence as the demo and promised the right answer; a reader who recomputed our result files
caught it.

```sh
uv run python scripts/ask.py --overlay data/quail/overlays/s0b/merged.pleo \
    --tokens 248044 88481 25540 3297 589 164852 6059 3687 175877 1789
```

The Chinese overlay of the same hundred facts is `data/quail/overlays/z0b-preliminary/merged.pleo`.
A recorded transcript of both sides is not in the repository yet.

## Quick start without a model

Everything below the engine runs against fakes on any machine:

```sh
git clone https://github.com/fulvian/engraft-ngram && cd engraft-ngram

# capture, descent, evaluation and damage, end to end, on a synthetic model
uv run pytest tests/test_descend_corpus.py tests/test_eval.py tests/test_damage.py

# what a descent takes, and what the measured run passed it
uv run python -m engraft.descend_corpus --help
cat data/quail/config/s0b.json
```

The `--fake` path builds its own tiny model, table and corpus
([`engraft/testing/`](engraft/testing/)); it is the mechanism running, not the measured run.
The shipped Quail corpus is tokenized for the real model and needs the real engine.

## Limitations, without discounts

- **Rephrasing is covered only as far as the corpus goes.** The table fires on exact n-grams, so
  a sentence that shares no n-gram with the corpus is not covered, by construction.
- **Composition is weak, and the clean rerun made it weaker.** Answering two facts in one
  question works on 4 of 83 probes. The first run gave 10 of 83, but every one of the 83
  generations stops at the probe tool's default budget of 40 new tokens, and 36 of them spend part
  of that budget on an empty `<think></think>` block. The rerun on the same 83 probes and the same
  overlay allows 96 new tokens and uses one format for all of them (chat template, empty think
  block): both facts on 4 of 83, at least one on 30 of 83, the same 30 as before. 56 of the 83
  still reach the 96-token cap, so a longer budget could still move the count, but the extra
  tokens moved it down, not up. The count follows single-fact free recall. Asked for one fact in
  free generation, the overlay gives the exact answer on 35 of 98 questions and the base model on
  0 of 98, which predicts about 7 of 83 for two facts. The weak point is free generation, not
  putting two facts together. 25 of the 83 probes carry one of their answers in the question
  itself, which flatters «at least one». The bare (non-chat) format has not been run yet. Records in
  [`data/quail/results/s0b/composition_rerun/`](data/quail/results/s0b/composition_rerun/).
- **Facts about the same subject share rows — by content, not by hash.** Of the 14,032 rows in
  the Quail overlay, 12 are written through more than one token window; the share of the slots a
  prompt reads that are in a pure hash collision separates successes from failures with AUC 0.503,
  i.e. not at all. What is shared is the subject's own n-grams: 0.954 of the slots a successful
  prompt reads are also written by other facts through the *same* window. So where a graft loses,
  it loses to a sibling rather than to noise — of the 37 failures whose first token belongs to
  another grafted fact, 32 are facts about the same subject
  ([`results/s0b/row_sharing.json`](data/quail/results/s0b/row_sharing.json)). Exact-key
  addressing would not change this; deliberately constructed colliding facts have not been tested.
- **Families differ.** On Quail, cloze prompts are the weakest family: first token 0.63 against
  0.70–0.81 for the others, pinned routing.
- **Damage grows with the number of facts.** It is 4× the quantization yardstick at 100 facts on
  Quail, and above our discussion threshold on Chinese text for the Chinese cell.
- **One model measured.**
- **No head-to-head yet with LoRA or with ROME/MEMIT.** That is the comparison we most want to see.

## FAQ

**Is this fine-tuning?** No weight of the model changes. Only rows of the lookup table are
optimized, and they ship as an overlay file.

**How is this different from RAG?** RAG retrieves text and puts it in the prompt; the model reads
it as words. Here the lookup is part of the forward pass and returns vectors, so a grafted fact
costs no context tokens and needs no retriever, index or embedding model at inference. We do not
claim to beat retrieval at recall: RAG generalizes to any phrasing, while this covers the
phrasings whose n-grams the usage corpus reads. Nor is the storage cheaper: the overlay is about
90 KB per fact, while the training sentences behind it come to 749 tokens per fact, a few
kilobytes of text. The two fail differently — retrieval can fetch the wrong passage or none, and
the model can still misread what it was handed; a graft either fires on its n-grams or does not —
and they compose: nothing stops a retrieval system from running on top of a grafted model. A
head-to-head over the same corpus (recall, latency, context spent, neutral-text KL) has not been
run.

**How is this different from LoRA or ROME/MEMIT?** Those change weights that every input goes
through, or rewrite MLP weights with a closed-form update. Here the memory is explicit and
hash-addressed, only rows read by the corpus move, and every claim is checked on the real
inference engine. We have not yet compared them head to head.

**Could this hold an agent's memory?** Not as working memory. A write is a gradient descent (hours,
not milliseconds), recall needs the right n-grams, and facts about the same subject share rows:
where a graft loses, it loses to a sibling, and in 21 of those 37 cases to the sibling with more
training mass. An agent writes many things about the same few subjects, which is this
mechanism's worst case today. As *consolidation* — periodically baking what has stopped changing
into the table, where it costs no context, and deleting the file if the consolidation learned
something wrong — it is plausible. Untested.

**Which models have an n-gram table?** Models built on DeepSeek's Engram design. The addressing
code reads the model's own hash multipliers and head sizes from the GGUF.

## History

The first form of ENGRAFT, the *surgical graft* (eight trigram rows of one trigger), is the
`v0.1.0` release and the `results/2026-09-05/` run. The usage-corpus descent replaced it on
2026-09-08 because its facts did not survive rephrasing. The 24-fact run of record of
2026-09-12 is in [`results/2026-09-12/`](results/2026-09-12/report.md). Every decision and the
measurement behind it are in [`docs/history.md`](docs/history.md).

## Related work

- **Engram** (DeepSeek): *Conditional Memory via Scalable Lookup*, Cheng et al.,
  [arXiv:2601.07372](https://arxiv.org/abs/2601.07372). The table design this method edits.
- **User as Engram**, Bojie Li, [arXiv:2606.19172](https://arxiv.org/abs/2606.19172): per-user
  memory as local edits of a hash-keyed table, on small Engram models of its own. Closest in
  spirit. ENGRAFT differs in the target (a 125B MoE model in production), in routing-aware
  descent, and in verifying on the real engine.
- **Engram Adapter**, Hou et al., [arXiv:2608.29327](https://arxiv.org/abs/2608.29327).
- **Memory Grafting**, Cheng et al., [arXiv:2605.20948](https://arxiv.org/abs/2605.20948).
- **ngram-knowledge-injector**,
  [ortegaalfredo/ngram-knowledge-injector](https://github.com/ortegaalfredo/ngram-knowledge-injector):
  patches the same table with overlay files, computing the rows differently.
- **llama.cpp** [PR 27742](https://github.com/ggml-org/llama.cpp/pull/27742) added the
  `qwen4exp` architecture; the engine fork builds on it.

## Licenses

- Code: Apache License 2.0 ([`LICENSE`](LICENSE)).
- Overlays (`.pleo`) produced against a Qwen3.8-Flash-Next GGUF are derivative works of that
  model and fall under the Qwen Community License 1.0; see [`NOTICE`](NOTICE).
- Quail corpus: CC BY 4.0 on our own contributions only
  ([`data/quail/LICENSE`](data/quail/LICENSE)). It is inspired from memory by Philip K. Dick's
  short story *We Can Remember It for You Wholesale* (1966). No text of the story was copied, and
  no right in that work is granted ([`data/quail/PROVENANCE.md`](data/quail/PROVENANCE.md)).
- `corpus/` texts of `v0.1.0`: public domain (Project Gutenberg).

## Citation

See [`CITATION.cff`](CITATION.cff). The technical report is [`paper/engraft.pdf`](paper/engraft.pdf).

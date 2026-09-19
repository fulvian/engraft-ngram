# Run of record 2026-09-12: usage-corpus descent, 24 facts, one 3 MB overlay

This directory holds the artefacts behind the second method of ENGRAFT, *usage-corpus
descent*: a plain language-model descent over a corpus of short usage sentences, updating
only the n-gram table rows that corpus reads, with expert routing pinned to the base model
during the descent. Two independent descents (seed 0 and seed 1 differ only in the order
the corpus is packed) and their measurements; the control arms that showed the teacher of
the earlier "context distillation" framing is not the mechanism; the collateral-damage
yardstick; the check in the real engine; and the same seed-0 descent rerun in bf16 with the
grouped MoE kernel, 34 % faster and within the f32 band.

The facts are 24 invented statements about an invented script ("sferoglifica"): 8 places,
8 instruments, 8 scholars, every name a non-word (`corpus/truth.json`). Nothing here is a
fact about a real person. The model is Qwen3.8-Flash-Next UD-IQ4_XS; the descent ran on the
integrated GPU of one Ryzen AI MAX+ 395 (128 GB unified memory) in torch/ROCm; the engine
check in the `fork-ple` llama.cpp at commit `9d9f9f9ad`.

The training code (torch replica with Triton expert kernels, descent, evaluation, damage
meter) lives in the private lab and is **not yet in this repository**; it will be released
with the second report. What is here is every number, every overlay and the recipe to load
the overlays into the public engine fork and ask the model the 24 facts yourself
(`scripts/ask.py`).

Note (2026-09-19): the training code is now public in `engraft/` (v0.2): see the
top-level README.

## Numbers, with the file behind each one

Held-out set: 112 usage fragments never seen in the descent (5 families: statement, cloze,
question, paraphrase, chat), scored by the rank of the answer's first token. "Pinned routing"
= expert routing forced to the base model's choice (the descent condition); "free routing" =
the model chooses, as in production.

| what | seed 0 | seed 1 | file |
|---|---|---|---|
| fragments with the answer at rank 1, pinned routing | **86 / 112** | **89 / 112** | `s0/eval.json`, `s1/eval.json` (`rank_first.student`) |
| same, rank ≤ 5 | 95 | 95 | same |
| rank 1, free routing (production condition) | 77 | 83 | same, `rank_student_free` |
| rank ≤ 5, free routing | 89 | 95 | same |
| base model, rank 1 | 0 | 0 | same, `rank_first.base` |
| teacher (model with the fact document in its prefix), rank 1 | 54 | 52 | same, `rank_first.teacher` |
| per family, rank 1 pinned (statement/chat/cloze/question/paraphrase, of 24/16/24/24/24) | 21/8/16/17/24 | 23/12/15/16/23 | `eval.md` |
| descent: steps, passes, stop | 80, 20, plateau | 80, 20, plateau | `summary.json` |
| rows read by the corpus (all variable), rows actually moved | 4 912, 2 600 | 4 912, 2 600 | `merged_manifest.json` |
| median \|Δ\| of a moved row, max | 0.0525, 0.1105 | 0.0531, 0.1122 | `merged.pleo` |
| cosine between the two seeds' Δ on the same rows, median (noise gives 0.006) | 0.63 | | `docs/history.md` |
| collateral damage on neutral text, mean KL (median), ΔNLL | 0.0035 (0.0003), +0.0003 | 0.0037 (0.0003), +0.0004 | `s0/damage.md`, `s1/damage.md` |
| fraction of positions with KL > 0.01 | 7.7 % | 8.2 % | same |

The damage yardstick (`yardstick/`): the same meter on a null overlay (the table's own rows
written back unchanged) gives KL 0.0011; on a synthetic overlay that perturbs the same rows
by the IQ4_NL quantization cell gives 0.0033; on random vectors of the graft's own norm
0.0037. The graft's 0.0035 is indistinguishable from the quantization noise of the table
itself. Measured against an FP8 copy of the same table, the per-row quantization noise is
0.0079 and the graft's displacement 0.0525: the graft is 6.7× the noise and almost orthogonal
to it (cosine 0.006), `yardstick/iq4_vs_fp8_lm-base_s0.md`.

## Control arms (`controls/`)

| arm | what is optimized | rank 1 pinned / free |
|---|---|---|
| `lm-base` (this method) | log-likelihood of the usage fragments, no teacher | 86 / 77 |
| `kd` | KL to the teacher (model with the document in its prefix) | 50 / 52 |
| `kd-base-only` | KL to the base model itself (negative control) | 0 / 0 |

The teacher-based arm transfers less than the plain descent and its teacher reaches only
54/112 itself; the negative control transfers nothing. This is why the method is called
usage-corpus descent and not context distillation (`docs/history.md`, phase 3).

## In the real engine (`engine_check/`)

The seed-0 overlay loaded into `fork-ple` (`llama-ple-lens`, IQ4_XS weights, free routing,
187 s for 112 fragments): **76 / 112** at rank 1, 92 at rank ≤ 5, base 0; the replica's
pinned-routing prediction was 86 and the two agree fragment by fragment on 98 / 112; median
\|Δp\| of the first token 0.048. Greedy decoding of the whole answer: **15 / 24** exact,
base 0 / 24 (`engine_check/report.md`, `results.json`).

## The same descent in bf16 (`s0-bf16/`)

Seed 0 rerun with bf16 dense weights and the grouped MoE kernel (the f32 runs above use
per-expert kernels): 84 steps at 58.2 s per step instead of 88 (−34 %), **88 / 112** pinned
(top-5 95), 79 free (top-5 92); agreement with the f32 seed 0 on 97 / 112 fragments and with
seed 1 on 99 / 112. Peak 97.3 GB reserved. The faster configuration is usable for the series.

## Corpus (`corpus/`)

`truth.json` (the 24 facts), `facts.json`, `usage_corpus_resolved.json` (456 fragments, 344
train / 112 held-out, tokenized), `doc_tokens_it_all.json` and `it_all.txt` (the teacher's
document, 434 tokens), `census_it.json` (which rows each fragment reads), `README.md`.
Everything is in Italian; the facts are invented and neutral.

## What is not here

Note (2026-09-19): the training code is now public in `engraft/` (v0.2): see the
top-level README. What remains missing: the raw per-step profiles of the GPU; the GPU
window logs of the lab. The descents are deterministic given the packing order: on the same
machine and build a rerun reproduces the numbers; on different hardware expect the same
outcomes and small numeric differences.

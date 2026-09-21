# History of the decisions

This page records, in order, the decisions that shaped ENGRAFT and the measurement that
forced each one. It exists because the method changed twice in ten days, and a reader
who only sees the current recipe cannot judge it without knowing what was tried before
and why it was dropped. Entries up to 2026-09-05 point to files in this repository
(the run of record at tag `v0.1.0`). The runs of record of 2026-09-12
(`results/2026-09-12/`) and of the Quail corpus (`data/quail/`) are published with their
files; other entries after 2026-09-05 are development observations whose files are not
part of this repository. The private test facts used in the first days are
referred to as "test fact A" and "test fact B"; they are not part of any published run.

Vocabulary used throughout: the *table* is the n-gram (Engram) lookup table of
Qwen3.8-Flash-Next, 16 heads, 8 keyed by the last two tokens (bigram) and 8 by the
last three (trigram), 160 floats per row, stored in 4-bit blocks (IQ4_NL). A *row* is
one entry. An *overlay* is a small file of replacement rows that the engine substitutes
at read time; the GGUF on disk is never modified. The *replica* is a differentiable
re-implementation of the model's forward pass, exact to the engine, used to compute
gradients with respect to rows.

## Phase 1: can the table be edited at all? (2026-09-03 to 09-05)

**2026-09-03. Start from three properties of the table.** Row addressing depends only
on the text and is reproducible in Python; rows are data in a file; the engine can be
patched to read replacements. Decision: explore in three stages (open the dictionary,
write one fact, write many facts without damage), with edits as overlays and never as
rewrites of the quantized bytes. Alternatives left open, not chosen.

**2026-09-03. No row is empty.** A sample of about 3.3 million rows found none with all
scales at zero: there is no free space to write into, every edit overwrites something.
Zeroing the whole table raises the negative log-likelihood of Italian text from 0.50 to
1.25 nats per token and changes the most likely token at 27 % of positions: the table
carries real signal. (`docs/lens.md`)

**2026-09-04. Trial-and-error editing is not enough.** A random search over one row set
brought test fact A from rank 7 (p 0.011) to rank 1 (p 0.115) in about 7,000 trials,
but never past p 0.5, and only the variant that touched the trigram rows alone left the
neighbouring triggers intact. Decision: a gradient is needed, hence a replica.
Alternative rejected: a closed-form write at the table's block, because the literature
reported it ineffective at that depth and our own linear probe of the block was blind
to the target.

**2026-09-04. The gradient is measurable once routing is frozen.** Finite-difference
probes through the engine were noise (correlation between the +ε and −ε probes −0.2)
until the mixture-of-experts routing was held fixed during the probe (−0.999). The
discrete routing was the noise, not the table. This is the first appearance of routing
as the thing that decides whether an edit can be measured.

**2026-09-05. Exact CPU replica, then routing refresh.** The replica matched the
full-precision engine to 5.6·10⁻⁵ nats over all 48 blocks. Descending with frozen
routing and then checking under free routing left a gap; refreshing the routing at
every step closed it (15 of 15 probe points, 0 diverging blocks). This is the recipe of
the first report (`paper/engraft.pdf`, Section 3).

**2026-09-05. The name.** ENGRAFT, ENgram GRadient Routing-Aware Fact Transplant.
"GRAFT" alone was already taken by several published methods, and "engraft" by a
product, hence the repository name `engraft-ngram`.

**2026-09-05. Eight neutral facts, the run of record.** Seven of eight grafts came out
of the real engine at first-token probability 0.85 to 0.96, with zero interference
between overlays and exactly zero change on triggers that share the bigram rows. One
counterfactual fact did not take. Generalization to the same fact inside a paragraph:
2 of 6. The report, the repository and this page's tag `v0.1.0` fix this state
(`results/2026-09-05/`, `paper/engraft.pdf`). Decision: publish now, as a method whose
main open problem is generalization, rather than wait.

## Phase 2: pushing the surgical graft toward a corpus (2026-09-06 to 09-08)

**2026-09-06. Multi-context descent.** The loss was summed over several contexts that
read the same rows (bare trigger, the sentence in a document, paraphrases). Result:
6 of 6 on the trained contexts; on contexts never seen, the number of successes went
from 3 to 8 but only 2 of 6 facts passed the bar; question-answer form in a chat
template 0 of 6, because the template's own prior dominated the trigger. Cost 2.7 to
4.5 times the single-context graft, 18 to 64 minutes per fact. Decision: efficiency
before scale; a thousand facts at this cost is about 750 hours.

**2026-09-06. What the descent actually does.** Comparing six descents showed a
"break": a 50-fold jump in the slope of the log-odds, preceded by a reorganization of
the expert routing in 44 to 47 of 48 blocks. The break happens at a different row-norm
ratio for each fact, so the norm is not the threshold. A linear probe of the block that
reads the table was blind even to the true rows (target rank in the tens of thousands
out of 248,320, cosine below 0.03). Decision: relax the requirement of exact zero change
on shared bigram rows for domain tables, and accept a measured, bounded regression
instead; add a diagnostic stage to tell whether the graft is a "push across a decision
boundary" or a "readable memory".

**2026-09-07. One fact at a time does not reach a corpus.** Caching the expert and
dense weights brought a step on the integrated GPU from 7.3 to 3.3 s; batching several
facts in one step amortized only 5 %; descending on whole sequences in float32 ran out
of memory at 32 tokens. The cosine between the initial gradient and the final
displacement was 0.07 to 0.48: the useful direction is found during the plateau, not
at the start. Decision: investigate sequence-level descent in the replica with a custom
4-bit operator, and backward passes inside the engine, in parallel; drop the pure
per-fact route as not resolutive.

**2026-09-07. The position phenomenon.** In a document with eight facts, two facts
always failed. Permuting the document showed that failure followed the absolute
position in the document (around tokens 98 to 124), not the fact. The replica was
cleared of blame by a dedicated check. Cause still unknown at that time.

**2026-09-08. Decision point.** A pilot on the number of training formulations showed
that generalization to unseen formulations rises with the training set at equal
compute (15 % → 40 % → 48 %) but stays tied to the key: different tails 0, chat 0 of
64, and all-or-nothing per fact. A change in the loss weighting (short pairs weighted 8
per step) made all eight facts take inside the document, including the two that the
position phenomenon had made look unrecoverable. The operator's judgement: as it
stands the technique is science, not engineering, and hours per eight facts is not a
product. Decision: change approach.

**2026-09-08. A route considered and not taken: direct writing.** A roadmap was
written for computing rows in closed form, or for writing the fact on every foreseen
formulation, following the literature on per-user memory in Engram tables. It was
closed the same week without a single run, superseded by the redesign below. It is
recorded here so that a reader knows it was considered.

## Phase 3: from context distillation to usage-corpus descent (2026-09-08 to today)

**2026-09-08. Redesign from zero: distill the model that already knows.** The framing
"trigger → answer token" was dropped. The evidence against it, all measured: the linear
probe is blind; displacements for different facts share no code (cosine 0.06 between
facts, 0.89 between repeats of the same fact); the descent wins by moving the routing
(96 of 97 regressions coincide with routing changes); the initial gradient does not
point where the descent ends. The new objective: the same model with the fact document
in its prefix is the *teacher*; the model without the document but with the overlay is
the *student*; the rows are optimized so that the student's next-token distributions
imitate the teacher's on a corpus of usage sentences, by Kullback–Leibler divergence.
Knowledge moves from the context into the table by imitation, not by injection. The
plan has four arms: distillation from the teacher, distillation from the base model
without document (control), plain language modelling on the usage corpus (control), and
the surgical graft (baseline).

**2026-09-09. Adversarial review of the redesign.** Six blocking findings changed the
plan before any run: part of the row sharing can be computed offline; the capacity of
shared positions was unknown; exact locality is incompatible with moving bigram rows;
the cost had been underestimated by one to two orders of magnitude; the free-routing
forward pass diverges from the float32 reference by 0.16 nats per position on average,
which is the same order as the distillation signal. Two offline pre-stages were added:
a census of which rows the usage corpus and a neutral corpus read, and a profile of
memory, cost and noise floor.

**2026-09-10. Census on the real corpus.** Of the rows that a fact's sentences read,
54 to 83 % are also read by a 100-million-token neutral corpus: the hash saturates the
table (a corpus of that size touches 53 to 62 % of each head), so "rows private to a
fact" is not a usable notion at this scale. Decision: replace exact locality by a
measured regression on a neutral reference corpus.

**2026-09-10. Packing bug found and fixed.** The overlay embedding did not cut the
context at the end-of-sequence token that opens each packed segment, contaminating the
next segment. After the fix the residual against the reference dropped, but a maximum
of 3.2 nats remained elsewhere, traced to 1,339 routing flips over 111 positions and 47
blocks.

**2026-09-11. Routing lock as an instrument.** With the routing forced to the base
model's choices, the packing residual falls to 0.055 nats maximum. Decision by the
operator: lock the routing during measurement and descent, verify under free routing,
serve with free routing. The lock is a measurement tool, not a change to the model.

**2026-09-11. First distillation: 0 of 112.** The divergence to the teacher fell, but on
112 held-out fragments the student never ranked the answer token first. Cause found in
the corpus, not in the method: the usage sentences were generic ("a little-known fact
about X…"), so the teacher itself answered them poorly (29 of 112) and the student
distilled syntax, not facts.

**2026-09-11. Usage corpus v2, long descent: 50 of 112.** With sentence models specific
to each relation, the teacher's ceiling rose to 54 of 112. A 100-step descent (learning
rate ×5, 4,912 variable rows, 2.6 hours on the integrated GPU) brought the student to 50
of 112 on held-out fragments, 52 under free routing; paraphrases 22 of 24, chat 8 of 16;
strong on place facts, weak on multi-token answers (instrument, scholar). The student
beats the teacher on some fragments. At that point this was the candidate result for the
second report, pending its control arms.

**2026-09-11. Collateral damage measured.** On 25 blocks of 2,048 tokens of Italian
Wikipedia, with the overlay under free routing: mean change in negative log-likelihood
+0.002 nats per token (perplexity ×1.002), mean KL 0.005, one position in 800 changing
by more than 1 nat. At the most exposed rows (the bigram heads of ". Il" and ", la",
read 88,000 times per 100 million tokens) the local KL is 0.02 to 0.10 with no
systematic sign; the downstream wake is about 0.01 nats regardless of which row was
hit. Two of sixteen blocks diverged before any changed row was read: the forward pass
is not bit-deterministic on this hardware, and that floor (KL 0.001) is now part of
the measurement. An internal yardstick (synthetic overlays: quantization-sized noise on
the same rows, same-norm random direction, null overlay) is in progress.

**2026-09-12. Control arms: the teacher is not the mechanism.** Same settings as the
distillation run (100 steps, learning rate ×5, routing locked, 4,912 variable rows), two
control arms. Distilling from the base model without the document (negative control):
0 of 112, loss flat at zero, row norms unchanged, so the 50 above is not an artifact of
descending on the rows the corpus reads. Plain language modelling on the usage corpus,
with no teacher at all: **86 of 112** held-out fragments (77 under free routing), above
the teacher's own ceiling of 54. Decision by the operator: the method is a plain
language-model descent on the usage corpus, over all the rows that corpus reads, with the
routing lock; the teacher survives only as a yardstick for the ceiling. What actually
worked is the multi-context descent of phase 2 brought to scale (relation-specific
usage sentences, all read rows variable, whole packed sequences, locked routing, the
sequence-level replica on the integrated GPU), not imitation.

**2026-09-12. Damage attributed.** On the same 25 blocks of Italian Wikipedia, under free
routing: the teacher-distilled overlay costs a mean KL of 0.0053 and 0.042 at the most
exposed rows; the plain-descent overlay 0.0035 and 0.011; a synthetic overlay with
quantization-sized noise on the same rows 0.0033 and 0.006; a random direction of the
same norm 0.0037 and 0.009; the null overlay 0.0011. The plain-descent overlay is at the
floor, the distilled one is not. With the routing locked to the base model the distilled
overlay's damage falls to 0.0004: about 90 % of the free-routing damage is expert-routing
flips, not misread rows. Under free routing the per-row displacement of the plain-descent
overlay is 6.7 times the quantization noise on the same rows, yet its damage is the same:
in this range damage is set by the routing floor, not by the size of the edit.

**2026-09-12. The forward pass made bit-reproducible.** The non-determinism reported on
2026-09-11 was traced by layer bisection to ties in the router (two experts with exactly
equal scores in float32, two or three positions per block out of 98,000) resolved
differently by the GPU's top-k. A stable tie-break (lowest index wins, as in the engine's
GPU backends) removed it: 0 divergences in 6 repeated blocks, and the null overlay under
the routing lock now gives exactly zero change. That floor is gone from the measurement.

**2026-09-12. The quantization-noise yardstick measured, not modelled.** The table exists
locally only in 4-bit blocks (IQ4_NL); the synthetic "quantization noise" overlay had
assumed uniform noise inside each 4-bit cell. An 8-bit (FP8 e4m3, one global scale) copy
of the same table, found in a third-party engine's checkpoint of the same model, allowed
a direct measurement on the 2,600 rows the plain-descent overlay changes: the 4-bit rows
differ from the 8-bit ones by 8.0 % RMS per row, norm 0.0079, against the model's 0.0081.
The yardstick stands within 2.5 %. Re-encoding our rows at 8 bits would cost 2.7 %.

**2026-09-12. The plain descent repeated from a perturbed start.** Same corpus, same
4,912 rows, same settings, a different random seed for the perturbation of the initial
rows. The descent stopped at the same plateau (80 steps) and the overlay scores **89 of
112** held-out fragments (83 under free routing) against 86 (77) for the first run; 81
fragments are right in both. The two overlays move the same 2,600 rows by the same
amount (median 0.053) but not in the same direction: the per-row cosine between the two
learned displacements is 0.63 (a quantization-sized random displacement gives 0.006). Two
descents find related, not identical, solutions with the same behaviour.

**2026-09-12. The plain-descent overlay verified in the engine.** The first-run overlay
loaded in the engine fork under free routing, on the 112 held-out fragments: **76 of 112**
first answer tokens at rank 1, against 77 predicted by the replica under free routing;
the two agree fragment by fragment on 98 of 112, median |Δp| of the first token 0.048.
Greedy decoding of the whole answer matches exactly in 15 of 24 sampled fragments (the
base model: 0 of 24). The replica predicts the engine for this method as it did for `v0.1.0`.

### Making the step cheaper without changing the science (2026-09-12)

The descent ran in the slowest of the configurations already built: per-expert MoE kernel and
f32 dense weights, 88 s per step at 2,048 tokens. Two changes were measured one at a time on
the same seed-0 corpus. The grouped MoE kernel (one launch per matrix per layer instead of a
Python loop over experts) is bit-identical to the per-expert path on 40/40 fragments and gains
nothing at 2,048 tokens: launch overhead is no longer the bottleneck at this batch length.
bf16 dense weights (the pinned routing recaptured in the same precision, 4.4 % of expert slots
differ from f32) bring the step to 58 s, a 34 % saving, and the full descent to 83 minutes.
The result holds: 88/112 held-out fragments rank-1 under pinned routing (f32 seeds: 86 and 89),
79 with free routing (77 and 83), per-fragment agreement with the f32 runs 97–99/112. A
snapshot taken 20 steps before the plateau is 5 fragments worse: the tail of the descent still
pays. Peak memory 97 GB reserved.

Where the remaining 58 s go is the next question. A read of the code before any profiling:
at 512 tokens the MoE path was 73 % of the step and the DeltaNet recurrence 12 %; per-layer
activation checkpointing recomputes the whole forward inside the backward (about 14 s of the
58); resident weights, not activations, fill the memory (59.5 GB of 4-bit experts plus 19.7 GB
of f32 dense cache). The production engine's prompt pass over 2,048 tokens on the same iGPU is
about 6 s; ours is 14. The physical floor for the step on this hardware is about 1 s; a
realistic engine-grade target is 12–20 s. The profile comes first, the kernels after.

## Phase 4: speed, capacity, one hundred facts, a second language (2026-09-12 to 09-19)

### The step, from 58 s to 17.4 s

A profile showed that the step was bound by the GPU, not by kernel launches. The mixture-of-experts
layers took about 20 s, elementwise work 12, the dense layers 9 and the DeltaNet recurrence 6.4.
Eleven rounds of kernel work followed, each measured on its own and kept only when the science
held: expert kernels, a bf16 output head, a chunked recurrence and compiled fusions. They brought
the step to 17.4 s at 2,048 tokens, from 58 s in bf16 (88 s in f32). The held-out results
of the candidate configuration stayed within the seed-to-seed spread. The fast kernels are not in
this release; the reference path computes the same descent more slowly (README, *What is in this
release*).

### When to stop

A stop on the training loss plateau stopped too early: the loss bottoms out long before the
held-out success rate does. The adopted rule watches the held-out success rate itself. It stops
when the local improvement rate over the last four evaluations falls below 5 % twice in a row
(`--stop-criterion acc_heldout_rate`, window 4, φ 0.05, persistence 2).

### Capacity: no ceiling up to 300 facts

The same descent on 24, 100 and 300 invented facts gave rank-1 first-token rates of 0.804,
0.792 and 0.821 on held-out sentences under pinned routing, with collateral damage of 0.0037,
0.0059 and 0.0067. No ceiling appeared. The residual failures were a corpus-writing problem, not
a capacity one: two templates out of nineteen produced 77 % of the errors. The discriminant was
whether the corpus covers the table keys at the *first* position of the subject. Facts whose
first subject position is never read by any training sentence fail (coverage 0.000); facts with
coverage of 0.083 or more succeed.

### One hundred facts about one world (Quail)

The capacity curve used short, independent facts. Quail tests the harder case: 100 facts about
one invented world, many of them about the same people. The corpus is written from a story
bible as seven documents, then compiled into usage sentences (inspired from memory by Philip K.
Dick's *We Can Remember It for You Wholesale*; no text of the story is used). The first two
seeds, with the routing pinned for the whole descent, gave exact greedy answers on 0.625 and 0.668
of the test sentences in the engine; the base model gave 0.005.

Two changes followed, measured one at a time on the same corpus and seeds. First, releasing
the routing: the descent pins the routing to the base model's choices while the rows make their
large moves, then releases it so that the rest of the descent trains the computation the engine
will actually run (`--routing-regime mixed`). Exact answers rose to 0.823 and 0.838.

The remaining failures were not random. When a fact failed, the overlay usually answered with a
sibling fact of the same subject: the graft pushed, but in the direction of another fact. A
contest analysis showed that the heavier sibling, the one with more training text, wins most of
these contests (27 of 37 identifiable cases). Hash collisions were measured and refuted as a
cause: the rows involved are shared by content, not by collision. The second change weights each
fact by its training mass in the loss (`--fact-weight mass`). On seed 1 the per-fact success
rate rose by +0.033 (95 % interval [+0.007, +0.060]), and most on the lightest quartile (+0.078);
on seed 0 the gain was about a third as large, with an interval that includes zero
(+0.012 [−0.019, +0.047]). Exact answers in the engine reached 0.841 (`s0b`) and 0.873 (`s1b`).
Collateral damage rose by about 18 % on seed 0, to 0.0131, about 4× the quantization yardstick
measured at 24 facts.

### A second language (preliminary)

The same world was rebuilt in Chinese (`z0b`). Measured raw, the Chinese cell is weaker: 0.676
exact answers against 0.841. Controlling for the training mass each fact received removes the
language effect (coefficient interval [−0.124, +0.020]). The row budget tuned on Italian buys
half as many Chinese training sentences, because Chinese uses almost twice as many new rows per
sentence. The contest between siblings follows the same rule in both languages: the heavier
sibling wins 58 % of the Chinese contests and 57 % of the Italian ones. Chinese simply has three
times as many contests.

The Chinese overlay barely affects Italian text (KL 0.0078); on Chinese text it reaches 0.0166.

The English cell (`e0b`) followed, on an English rebuild covering 97 of the 100 facts: 0.797
exact answers, first token at rank 1 0.820, damage 0.0160 on English text and 0.0094 on Italian.
It falls between Italian and Chinese, which is where the mass explanation puts it.

Then the mirror measurement, which turned out to be the sharpest result of the three: the
Italian test set run against the *Chinese* overlay answers exactly like the base model — 0.0048
against 0.0048 exact, 0.0951 against 0.0951 at rank 1, 764 of 841 fragments identical down to
the first token's probability — while the engine reads 922 overlay rows along the way. The
overlay is consulted and changes nothing. Probes that mix the two scripts do no better (0 of 20
and 1 of 20). A graft lives in the rows keyed by the tokens of its own script, so a fact has to
be grafted in the language it will be asked in. One pair of languages, one cell.

A Chinese cell given equal mass per fact remains the next measurement.

## Open at the time of writing

- A Chinese cell with training mass per fact equal to the Italian one: the decisive test of the
  row-budget explanation.
- A full rerun of `s0b` on the public reference path, and a recorded transcript of the engine
  demo with the Quail overlay.
- The English composition probes: the published probe file is clean, but the run we made used a
  defective regenerated set and was discarded.
- Collateral damage as the number of facts grows beyond 300, and a quantization yardstick
  measured at 100 facts.
- A head-to-head comparison with LoRA and with ROME/MEMIT.
- **Retrieval over the same corpus at the same storage budget**, scored on recall, latency,
  neutral-set KL and bytes per fact. The one half we can state today is unflattering: the
  overlay is 9,036,620 bytes for 100 facts (~90 KB per fact), while the 3,209 training
  sentences behind it come to 74,923 tokens (749 per fact, a few kilobytes of text) — the
  stored fact costs tens of times the text that produced it. (The usage-corpus JSON weighs
  4.5 MB, but that is token ids and masks, not text.) What we expect to win on — zero context
  tokens, no retriever, no prefill — is unmeasured. Raised by a reader after publication,
  2026-09-21.
- **Rerun the composition probes without the two protocol confounds**: all 83 generations are
  cut at 40 new tokens and 36 of them carry an empty `<think></think>` block. Until that
  rerun, 10/83 measures the eval as much as the overlay. Same reader, same day.
- **A disjoint-row arm for composition.** All 83 probes ask two facts about the *same*
  subject, so they read that subject's rows by construction; there is no disjoint control in
  this corpus. Separating row interference from a downstream readout limit needs probes
  spanning two different subjects — a new corpus. Same reader, same day.
- **Fact updates.** Graft «X is A», later «X is B» under the same trigger: which one does the
  model answer? Never measured. Two arms: *rewrite* (regenerate the overlay with B in place of
  A) against *stacking* (a second descent of B on top of A's rows, the very same rows). The
  «heavier sibling wins 21 of 37» figure says nothing about recency: all hundred facts are
  written in one descent. The one precedent is the counterfactual fact of `v0.1.0` that did
  not take. Suggested by a reader.
- **Bring the technical report (`paper/engraft.pdf`) level with the README.** It predates the
  positioning («token-addressed memory, not editing, not a retrieval rival»), the truncated
  composition probes and the corrected demo, so release `v0.2.2` ships without it. To be
  recompiled as soon as the composition probes have been re-measured.
- DeepSeek V4.1 Flash, the second model with an Engram-style table.

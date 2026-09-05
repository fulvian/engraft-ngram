# The ENGRAFT recipe

End to end, grafting a corpus of facts is four steps, each a separate
command so any of them can be rerun or checked independently.

## 1. Resolve facts (`engraft-facts`)

For every fact in `facts/facts.json`: count the answer's tokens (falling back
to `answer_fallback` if the answer itself is over 3 tokens -- a fact whose
answer and every fallback are too long is a blocking error, not a silent
truncation), find the trigger's 16 rows and, for a multi-token answer, the 16
rows at each intermediate position (`chain_rows_global`). Then verify the
fact is well posed: two "sister" triggers that share the trigger's last two
tokens but not its last three (same bigram rows, disjoint trigram rows), a
same-tail paraphrase (identical 16 rows) and an other-tail paraphrase
(disjoint trigram rows), and -- if the fact is meant to appear verbatim in a
document under `facts/docs/` -- that the document's own tokenization puts the
answer where expected. A failing check is recorded as `ok: false`, never
worked around; fixing it means editing the fact or the document, then
re-running.

Because two grafts that land on the same trigram row would silently corrupt
each other, `keys.json` records this too: scanning every graft (fact x
position, in `facts.json` order), a collision keeps whichever graft comes
first and excludes the *entire* losing fact from every later step.

## 2. Graft (`engraft-run`)

Facts are processed in a fixed order (language-alternated memory facts, then
counterfactuals) against a single load of the replica's weights and table.
For each fact, `graft_fact` descends every answer position in chain (see
`docs/replica.md`), writing one `.pleo` overlay and a JSON summary per fact,
resumable via `state.json`. A step-time guardian aborts a graft that slows to
a crawl (3x its estimated per-step time for 20 steps running) rather than let
a stuck run consume the whole time budget silently.

Two of the closed grafts are then repeated from a fixed random perturbation
of their starting rows (T8-masked, 1% relative magnitude), to check that the
descent converges to the same place regardless of where it started -- concord
on step count, stop reason, and final probability is the criterion, not exact
reproduction. Finally, every non-excluded fact's overlay is merged into one
`merged.pleo`.

Each position's descent stops when it plateaus: a run of routing refreshes
(150 steps' worth) with no improvement over the best value seen so far. The
default plateau metric is `logp` (the log-probability of the target token),
required to improve by more than 0.05 nat to reset the no-improvement
counter; a `p_free` variant (0.01 absolute improvement) is selectable for
compatibility. `logp` is the default because `p_free` degenerates for a fact
whose trigger starts at very low probability: once p is already below 0.01,
a 0.01 absolute improvement is impossible by construction (p = exp(logp)
saturates near 0), so the no-improvement counter never resets and the
criterion silently becomes a hard step cap rather than a real plateau.
`logp` tracks the quantity the descent actually minimizes and does not have
this failure mode.

Not every fact reproduces. A fact whose trigger has a strongly concentrated
prior in the base model competes against that prior for the whole descent
budget and can exhaust the maximum step count without ever reaching the
stopping probability -- the graft fails open (recorded as `max_steps` with a
low final probability), never silently. In one full run of eight facts, seven
reproduced; the one that did not was a counterfactual chosen to have an
unusually strong base-model prior at its trigger.

## 3. Check on the real engine (`engraft-check`, `scripts/window.sh`)

Two engine phases. The default (quantized) engine measures, for each fact
with its own overlay: the answer's first-token probability and rank, a greedy
continuation to see if the whole answer comes out, and the sister/paraphrase
checks translated into actual measurements (unchanged argmax, bounded
delta-logp, the expected overlay row-hit counts). With the merged overlay, the
same measurements repeat for every included fact, plus paraphrases, and
corpus/document-level mean delta-NLL.

The generated `report.md` and `engine_check.json` label the measurements with
short question codes: **Q1** = each fact with its own overlay; **Q2** = the
same facts under the merged overlay (interference); **Q4** = paraphrases under
the merged overlay; **Q5** = full-precision engine vs. replica fidelity; **Q8**
in the logs simply names the quantized-engine phase.

A full-precision engine phase then checks the replica's own math: for each
graft, a job with no overlay gives a base log-probability that should match
the replica's own step-0 record to a very tight tolerance (this is a
consistency check on the replica itself, not on the graft's quality); a job
with the fact's overlay and captured routing gives the free-routing
probability the real engine would actually produce, compared against the
replica's own final probability, and the routing itself is compared layer by
layer against what the replica captured.

## 4. Reproduce end to end (`scripts/reproduce.sh`)

Chains the three commands above with the engine windows they each need,
writing everything under `results/<date>/`. See that script's comments for
which steps need the real model and engine, and which run on CPU alone.

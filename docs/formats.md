# Data formats used by the corpus descent

This documents the on-disk formats the corpus-level descent
(`engraft.descend_corpus`), the teacher/base capture (`engraft.teacher`), and
the measurement CLIs consume and produce. It does not document the technique
itself (see `docs/mechanism.md`/`docs/method.md`) or the single-fact API
(`engraft.facts`/`engraft.run`/`engraft.check`, see `docs/replica.md`).

## `usage_corpus_resolved.json`

Input to `engraft.teacher` and `engraft.descend_corpus`. Either a bare list of
fragment objects, or `{"fragments": [...]}`.

Each fragment:

```json
{
  "id": "unique string id",
  "fact_ids": ["fact_a"],
  "tokens": [EOS_TOKEN_ID, "...token ids of the fragment..."],
  "answer_spans": [3],
  "split": "train"
}
```

- `tokens`: already in canonical form `[EOS] + fragment`, i.e. the tokenizer's
  EOS id (`table.eos_token_id`) prepended. All fragments in a corpus must
  share the same opening EOS id.
- `answer_spans`: 0-based **token** indices (not logit rows) inside `tokens`
  that carry the answer; `engraft.descend_corpus.answer_rows` converts them
  to logit-row indices (`row = token_index - 1`, since the row that predicts
  `tokens[i]` is row `i-1`; index 0, the opening EOS, never has a row).
- `fact_ids`: exactly one entry per fragment (this is what
  `compute_fact_weights`/`--fact-weight mass` groups by; a fragment with zero
  or more than one fact id is rejected).
- `split`: `"train"` selects the fragment for training; any other value
  (`"test"`, `"heldout"`, ...) puts it in the heldout set used for
  `acc_heldout`/`kd_heldout`.

## `target.npz` / `TeacherTargets` (`engraft.replica.distill.TeacherTargets`)

Produced by `engraft.teacher` (`--doc-tokens` given: teacher targets,
conditioned on the source document; omitted: base targets, `doc_tokens=[]`),
consumed by `engraft.descend_corpus` via `--targets`.

npz fields, one row per `(fragment, local position)`:

- `ids [N, k] int64` -- top-`k` token ids at that position, teacher/base
  logits, sorted by decreasing probability.
- `logp [N, k] float32` -- log-probabilities of `ids`.
- `tail_logp [N] float32` -- `log(1 - sum(exp(logp)))`, the truncated tail
  mass (needed so the KD loss stays a valid distribution after truncating to
  `k`).
- `frag_id [N] int64` -- index into the `fragments` list used to build the
  targets (**not** a filtered sub-list's index -- a caller working from a
  filtered fragment list must map back to the full list's index).
- `pos [N] int64` -- local position within the fragment, `0..len(tokens)-2`.

`k` should be picked well under the model's vocabulary size (public example
runs and tests use small fake vocabularies, so `k` there is a handful; a real
run's `k` matches the deployed configuration).

## `routing_base_*.npz` / locked routing (`save_routing_base`/`load_routing_base`)

Produced by `engraft.teacher --routing-out` (only for the **base** run,
`--doc-tokens` omitted -- the conditioned teacher always runs at free
routing), consumed by `engraft.descend_corpus --routing-base`.

npz fields:

- `frag_key [N] str` -- the fragment's `id` string.
- `pos [N] int64` -- local position, `0..n_frag-1`.
- `layers [L] int64` -- sorted union of every layer that produced routing
  (every layer that calls the MoE, no exception for the last one when
  `return_logits=True`).
- `routing [N, L, k] int16` -- the expert indices actually selected by the
  live router at that position/layer, in `layers` order.
- `dense_dtype`, `moe_kernel`, `wdot` (scalars): the configuration that
  produced the file, checked on load unless `mismatch_ok=True`.

`engraft.descend_corpus --routing-base` calls `load_routing_base` with
`mismatch_ok=True`: the public reference descent always runs the per-expert
Triton kernel (or the CPU `--fake` path), never the grouped/`layer` kernel a
`routing_base_*.npz` produced by the private fast path may have been saved
with -- the locked routing values themselves are unaffected by which kernel
produced them (see `docs/history.md`/the repository's README for the kernel
equivalence measurement), only the metadata string differs.

## `bias.json` (`bias_report`, optional, `engraft.teacher --bias-out`)

`{"n_sample": int, "k": int, "mean_bias": float | null, "max_bias": float | null,
"min_bias": float | null}` -- empty-branch fields are `null` when
`n_sample == 0` (the truncation-bias measurement only runs for the
conditioned teacher with `--sample-full > 0`).

## `census.json` (`--census`, `engraft.descend_corpus`)

`{"schema": "engraft-census/v1", "row_sets": {"all_read": [int, ...], ...}}` --
the frozen-row set a descent run treats as non-variable (see "Descent
output" below). `row_sets["all_read"]` is a sorted, deduplicated list of
global row indices; other keys under `row_sets` (e.g. `"entity"`,
`"termine"`) are passed through untouched but not read by the public CLI.

`engraft.descend_corpus --census PATH` reads this file and uses its
`row_sets` as-is, skipping the built-in `build_all_read_row_set` recompute.
Without `--census`, the CLI falls back to `build_all_read_row_set(table,
train_frags_tokens)`: the union, over every **train**-split fragment in
`--usage-corpus`, of every row read by that fragment's prefix
(`_rows_read_by_prefix`, positions `0..len-2`).

These two are **not** the same set in general, and the difference is not a
bug in either: `build_all_read_row_set` is defined over the *fragments*
actually fed to the descent (every context variant a fragment carries --
`C0`, template repeats, chain contexts, ...), while the private pipeline
this repository is derived from defines `census.row_sets.all_read` over the
reconstructed source *documents* only, deliberately excluding those extra
per-fact context variants (row-set calibration and freeze-budget accounting
need the narrower, document-grounded set -- see the private
`distill_census.py`'s "misura 2"/"colonna B", out of scope for this public
reference). Measured on the published `s0b` cell
(`data/quail/corpus/s0b/usage_corpus_resolved.json`, 3209 train fragments,
against `data/quail/corpus/s0b/census.json`): the fragment-recomputed set
has 198628 rows, the document-grounded census has 14032 rows, and the
census set is a **strict subset** of the recomputed one (intersection ==
14032; 0 rows in the census outside the recomputed set). Passing
`--census` reproduces the exact frozen-row set the measured run used;
omitting it recomputes a superset from this repository's own
`usage_corpus_resolved.json`, which changes which rows `merged.pleo`
freezes (not the variable rows, not the loss) but is not reproducibility of
the measured artifact.

## Descent output: `merged.pleo` + `merged_manifest.json` + `summary.json`

`merged.pleo`: the standard `.pleo` overlay format (see `docs/lens.md`,
`engraft.lens.write_pleo`/`read_pleo`) -- the **variable** rows at their final
descended value, plus every other row the training fragments' prefixes read
(`row_sets["all_read"]`) at their **true** (Delta=0) value, so a downstream
consumer that checks a freeze invariant sees every head represented.

`merged_manifest.json`/`summary.json`: the full return value of
`engraft.descend_corpus.descend_corpus` (the same dict, `summary.json` adds a
`row_map` field mapping each variable row's global id to its index in
`rows_var`). Notable fields:

- `stop_reason`: one of `"plateau"`, `"max_steps"`, `"budget"`,
  `"guard_step_time"`, `"guard_device_memory"`, `"plateau_acc_heldout"`,
  `"plateau_acc_heldout_rate"`, or one of
  `engraft.replica.regime.STOP_REASON_PLATEAU_FREE`/`STOP_REASON_UNSTABLE`
  (`"plateau_acc_free_rate"`/`"free_unstable"`) when `routing_regime="mixed"`.
- `routing_regime`: `"locked"` or `"mixed"`. `"phases"`: the mixed arbiter's
  phase history (`engraft.replica.regime.RegimeArbiter.phases_as_dicts()`),
  empty in `"locked"` mode.
- `fact_weight`: `"none"` or `"mass"`; with `"mass"`, `fact_mass`/
  `fact_weights`/`fact_weight_stats` are also present (see
  `engraft.descend_corpus.compute_fact_weights`).
- `checkpoint`: whether whole-layer gradient checkpointing
  (`torch.utils.checkpoint`, wrapping each transformer layer during the
  backward pass) was active. **This never changes the loss or the gradient**
  -- it only trades peak activation memory for extra forward compute during
  the backward pass (the layer is recomputed). See the note below.
- `descend_<arm>_<policy>.jsonl` (path in `log_path`): one JSON record per
  step, with the training loss, gradient norms, heldout indicators when
  evaluated that step, and (with `--routing-base`) the routing-mismatch
  counters.

## Checkpoint mode: math is unchanged, memory is not

The private reference this repository is derived from exposed four
whole-step checkpoint granularities (`"none"`, `"layer"`, `"moe"`,
`"nonmoe"`), tuned for a specific GPU memory budget: `"layer"` recomputes an
entire transformer layer during the backward pass, `"moe"`/`"nonmoe"`
recompute only part of it (the MoE region, or everything but the MoE
region). Only `"none"`/`"layer"` are part of this public reference
(`engraft.descend_corpus --checkpoint`, a boolean); `"moe"`/`"nonmoe"` were a
finer-grained memory optimization requiring extra parameters on
`Replica.run_layer` that this reference does not expose.

This is a safe simplification for reproducing the technique's numbers: for
any state where a checkpoint is well-defined (empty per-layer state, no
`base_state`/`cache`, the constraints `engraft.replica.seq.run_layer_checkpointed`
already enforces), whole-layer recomputation inside `torch.utils.checkpoint`
is mathematically identical to computing the same layer without checkpointing
-- `torch.utils.checkpoint` recomputes the exact same forward function during
the backward pass and does not change floating-point operation order within
a layer. The only production run (`s0b`, see `docs/history.md`) that a public
user might want to reproduce ran with `--checkpoint-mode layer`, i.e. the
mode this reference's `--checkpoint` flag reproduces; `"moe"`/`"nonmoe"` were
never used to produce a published number.

What differs with checkpointing off vs. on is **peak GPU memory** during the
backward pass, not any number in the loss, the gradient, or the descended
overlay. A public user without the private memory budget this was tuned
against can simply run with `--checkpoint` (safe default for a memory-
constrained device) or without it (faster, more memory) and expect the same
loss trajectory, up to floating-point summation order across the recomputed
region -- verified directly at the `engraft.replica.seq.run_layer_checkpointed`
level in `tests/test_seq.py::test_run_layer_checkpointed_bit_identical`
(`Replica.prefix(grad_proxy=True, return_logits=True)` with and without the
checkpoint context, same logits, same gradient, `torch.equal`).

**`--checkpoint` on the CPU `--fake` path is not runnable** (raises
`NotImplementedError`): the private original never exercised this
combination either -- checkpointing was only ever used together with the
Triton-backed step, never with the F32 CPU reference path, which does not
need it (a `--fake` run's whole graph is tiny). This is not a gap the CPU
equivalence claim above depends on: `run_layer_checkpointed` composes with
`replica.run_layer` regardless of which patch (`expert_op_patched` or
`expert_op_patched_triton`) is currently substituting the MoE/PLE closures
-- both patches only replace `model.moe_ffn`/`replica.ple_true_emb`/
`replica._expert_fns`, never `replica.run_layer` itself, which is the sole
attribute `run_layer_checkpointed` wraps. The test above exercises this at
the `Replica.prefix()` level (no `seq.py` patch active at all), which is the
minimal case that isolates the checkpoint's own correctness from any
particular MoE kernel.

## Damage measurement inputs: `damage_plan` directory and neutral text

`engraft.damage plan` writes a directory (passed to `engraft.damage run
--plan`) with:

- `rows_delta.npz`: `rows_global [R] int64`, `dnorm [R] float32` (L2 norm of
  the overlay's delta at that row relative to the true row), `ratio [R]
  float32`, `cos [R] float32` (cosine similarity overlay-vs-true).
- `hits.npz`: `pos [H] int64` (positions in the neutral text whose n-gram
  hash reads a moved row), `n_hit_rows [H] int64`, `dnorm_max [H] float32`,
  `dnorm_sum [H] float32`, `overlay_sorted [R] int64`, `read_count [R]
  int64`.
- `chunks.json`: `{"meta": {...}, "chunks": [...]}` -- the sampled text
  chunks around hit positions that `engraft.damage run` re-forwards to
  measure KL/NLL damage.

### Neutral text

The damage measurement needs a large token array of neutral, unrelated text
(e.g. `tokens_it.npy`, an `int32`/`int64` numpy array of token ids) to search
for positions whose n-gram hash happens to read an overlay-moved row. The
private pipeline's producer of this array was not found in this session's
source tree (its `SOURCES.md` documents the source -- Wikipedia
`wikimedia/wikipedia`, config `20231101.it`, two shards -- but not the
tokenization script itself). `scripts/build_neutral_tokens.py` in this
repository is a **reconstructed** reference implementation (declared as such
in its own docstring), built from that documented source and this
repository's own tokenizer (`engraft.table.PleTokenizer`): it downloads the
declared Wikipedia config via the `datasets` library, concatenates article
text up to a requested token budget, tokenizes it, and writes an `.npy`
array in the format `engraft.damage plan --neutral-tokens` expects. It has
not been checked bit-for-bit against the original private array (that array
was never available in this session) -- treat its output as "a" neutral text
with the declared recipe, not as a byte-identical reproduction of any
specific published damage number's input.

## Evaluation output: `eval.json`/`eval.md` (`engraft.eval`)

`engraft.eval` scores a usage corpus's fragments in the replica, three
columns per fragment: `base` (no overlay, no document), `student` (the
descent's overlay), `teacher` (the source document in the prefix, true
rows). Written by `write_eval_report`.

`eval.json`, `{"n_fragments": int, "fragments": [...], "by_family": {...},
"by_overlap_band": {...}}`. Each entry of `fragments`:

- `id`, `family`, `split`, `fact_ids` -- copied from the input fragment.
- `p_first`/`rank_first`/`correct`: `{"base": .., "student": .., "teacher":
  ..}` -- `p_first`/`rank_first` at the first answer token
  (`engraft.eval.answer_row`), `correct = (rank_first == 1)`. A column's
  value is `None` (not `0.0`/`False`) when that forward was skipped
  (`--skip-base`/`--skip-teacher`).
- `kd_student_teacher`: KD loss (student logits vs. teacher's top-`k`
  distribution) over the fragment's positions; `None` with `--skip-teacher`
  (it depends on the teacher's logits, not a column constant).
- `p_student_free`/`rank_student_free`: present only with `--routing-base`
  -- the student re-forwarded at free routing (the cost of locking).
- `overlap_count`/`overlap_band`: from `--census` or an
  `overlap_by_fact_id` map, `None` if neither resolves for this fragment.
- `ple_gate` (only with `--ple-gate`): `{"hc": int, "answer_row": int,
  "window_rows": [int, ...], "overlay_hits_answer": int,
  "overlay_hits_window": int, "base"/"student"/"student_free": {"gate_answer":
  [hc], "s_answer": [hc], "gate_window_mean": [hc], "gate_frag_mean": [hc],
  "gated_over_hidden_answer": [hc]}}` -- the `PLE` gate's internal signals at
  the answer row and its causal window (see `docs/mechanism.md`).

`by_family`/`by_overlap_band`: `{key: {"n": int, "correct_frac": {col:
frac}, "kd_student_teacher_mean": float}}` (the last key absent with
`--skip-teacher`).

`ple_gate.npz` (only with `--ple-gate`): one entry per `"<fragment id>/<base|
student|student_free>/<gate|s|value_norm>"`, the raw `[T,hc]`/`[T]` arrays
`engraft.replica.seq.capture_ple_gate` produced for that fragment/column.

The private original this reference derives from also had a `--ple-contesa`
flag (an "interference" diagnostic requiring an intermediate per-sub-block
capture inside `Replica.run_layer` -- attention/PLE/MoE-dense/MoE-experts/
weighted-sum -- that this reference's public `run_layer` interface does not
expose) and an `innestate_positions`/`contesa_from_captures` pair of
functions. Neither is part of this port: no number this repository publishes
depends on it (the published cell that used it, `s0b`, was a later
diagnostic re-run over an already-closed cell, feeding an internal
mass-balance investigation that itself rests on a separate OLS fit, not on
this capture), and adding it would mean extending the replica's public
interface in service of a diagnostic, not a measurement this repository's
numbers need.

## Damage measurement output: `damage.json`/`damage.md` (`engraft.damage run`)

`damage.json`: `{"n_chunks_requested": int, "n_chunks_done": int,
"gpu_minutes_cap": float, "total_s": float, "forward_s_mean": float | null,
"chunks": [...], "global": {...}, "hits": {...}, "controls": {...},
"routing_mode": "free" | "locked"}`.

- `global` (random blocks): `{"n_positions": int, "dnll_mean", "dnll_median",
  "kl_mean", "kl_median", "frac_kl_gt_0_01", "frac_kl_gt_0_1", "frac_kl_gt_1":
  float, "worst_20": [...]}` -- `dnll`/`kl` are per-position student-vs-base
  NLL delta / KL divergence (`_kl_and_nll`). `{"n_positions": 0}` if no
  random block ran.
- `hits` (hit blocks, one per target row): `{"by_stratum": {"top"|"tail":
  {"n", "dnll_at_hit_mean", "kl_at_hit_mean", "dnll_p1_8_mean",
  "kl_p1_8_mean", "dnll_p9_64_mean", "kl_p9_64_mean"}}, "per_target_row":
  [...], "worst_20": [...]}`.
- `controls` (hits on unchanged rows inside a hit block, an observational
  check of causal locality): `{"n_controls", "n_upstream", "n_downstream",
  "n_upstream_violations", "upstream_violations_examples", "downstream_kl_mean"}`.
- `routing_mode`: `"free"` (default, live routing) or `"locked"` (`--rbr`:
  the student forward reuses the base forward's captured routing, so
  `flips=0` by construction -- isolates row-value damage from router-flip
  damage).

`per_chunk/chunk_NNNN.npz`: `start`, `dnll [T]`, `kl [T]`, `flips [T]`
(int32, per-position count of layers whose live-routed expert selection
differs between the base and student forwards), `forward_s`.

The private original also had `--diag-determinism`/`--diag-layers`/
`--diag-sublayer` flags that isolated a specific non-determinism incident on
the private fast path; `--diag-layers` needed the same intermediate
per-sub-block `run_layer` capture as the dropped `--ple-contesa` above. None
of the three is part of this port -- see `engraft.damage`'s module
docstring.

## Engine-check output: `results.json`/`report.md` (`engraft.engine_check`)

`engraft.engine_check` redoes the `base`/`student` columns (never `teacher`,
no document available in the resolved usage corpus) on the real engine, to
compare against `engraft.eval`'s replica-side numbers.

`results.json`: a list of records, one per fragment:

- `id`, `family`, `fact_ids`.
- `base`/`student` (the latter only with `--overlay`): `{"p_first": float,
  "rank_first": int, "argmax": int, "overlay_hits": int | null}`.
- `greedy` (only for the fragments `--greedy-n` selected):
  `{"base"/"student": {"tokens": [int, ...], "text": str, "degenerate":
  bool, "exact_match": bool, "answer_text": str}}`.
- `replica`/`delta_p_first` (only with `--replica-eval` and a fragment in
  common): `{"p_first": float, "rank_first": int}` from `eval.json`'s
  `student` column, plus `|Δp_first|` between engine and replica.

`jobs.json` (`--dry-run` only): `{"pfirst_jobs": [...], "greedy_descriptors":
[...], "n_pfirst_jobs": int, "n_greedy_descriptors": int,
"n_greedy_engine_calls_expected": int}` -- the greedy is sequential, so only
its first step is dry-enumerable; `n_greedy_engine_calls_expected` is a
count, not a list of jobs.

## Composition-probe output: `probe_results.json`/`report.md` (`engraft.probes`)

`engraft.probes` runs a greedy continuation for ONE column (base or
student, never both in the same run) over a set of probes, each combining
two facts of the same subject, and scores binary success when both facts'
`answer` strings appear as case-sensitive substrings of the whole output.

Input `probes.json`-shaped file: `{"probes": [{"id": str, "fact_ids":
[str, str], "text": str, "subject": str}, ...]}`. Input fact registry:
`{"facts": [{"fact_id": str, "answer": str}, ...]}`.

`probe_results.json`: a list of records, one per probe: `{"id", "fact_ids",
"subject", "text", "answer_a", "answer_b", "output": str (the full greedy
continuation), "tokens": [int, ...], "hit_a": bool, "hit_b": bool, "both":
bool, "column": "base" | "student", "overlay": str | null}`.

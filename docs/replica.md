# The CPU replica

`engraft/replica/` is a torch port of the whole model, built to compute a
gradient with respect to 16 n-gram table rows without needing a differentiable
engine. It never runs on GPU and is not meant to be fast; it exists to make
one thing possible cheaply: descend the loss at a single position while
holding everything before it fixed.

## Split: constant prefix, differentiable last position

`Replica.prefix(tokens)` runs a full, gradient-free forward over positions
`0..T-2`, producing per-layer state (K/V cache for attention layers, conv
history and recurrent state for delta-net layers, the PLE block's conv
history) that `Replica.last_step` needs for position `T-1`. Routing (which
MoE experts a layer uses) is either imposed from a capture
(`routing_source`, the PLERT1 format from `docs/lens.md`) or computed live.

`Replica.last_step(tokens, state, rows, ...)` runs the entire layer stack for
just the last position, substituting the given `rows` -- with gradient, when
they require one -- for the PLE table gather. Everything downstream of that
gather (hyper-connections, delta-net/attention layers, MoE, the output
projection) is a faithful torch port of the engine's own computation graph;
`engraft/replica/layers.py` documents, function by function, which engine
computation each one mirrors, and `tests/ref_hf_qwen4exp.py` plus
`tests/test_replica_layers.py` check the delta-net and RoPE math against a
verbatim copy of the upstream reference implementation.

## Weights: lazy, cached, memory-bounded

`engraft/replica/weights.py` opens the model's GGUF shards and dequantizes
tensors on demand. Ordinary tensors are dequantized whole; MoE expert
tensors are sliced and dequantized one expert at a time
(`GgufWeights.expert`), because the full tensor holds every expert and only a
handful are used per token. A last-position expert can be persisted to a
disk+RAM LRU cache across graft steps (the same handful of experts is reused
every step); a prefix expert is never cached, since a whole-prefix forward
can touch thousands of distinct experts and caching them all would defeat the
point of a memory budget.

## Grafting: constrained descent, refreshed routing, chained positions

`engraft/replica/descend.py` runs Adam over the 16 rows (or a masked subset
of them -- this repository's grafts constrain the descent to the trigram
heads, `ROW_MASK_T8`) to raise the probability of a target token at one
position, recording enough per-step diagnostics (probability, per-sister KL
divergence, per-sister argmax and delta-logp) to tell an honest graft from an
accidental one. Routing can be refreshed periodically during the
descent (`refresh_every`): the live routing is recomputed every `k` steps and
frozen in between, so the final overlay's probability is measured under
routing the free (un-frozen) engine would actually choose, not routing frozen
at the starting point.

`engraft/replica/graft.py`'s `graft_fact` chains this across every position
of a multi-token answer: position `i` reads, in its own prefix, the rows
already grafted at positions `< i` (a running overlay), builds routing from
scratch for that prefix, and only then descends. It is resumable per graft
via `state.json`: an already-closed graft is skipped on a second run, and its
rows re-enter the running overlay by re-reading its own `.pleo` file rather
than being redone. `merge_fact_overlays` unions the per-fact overlays of every
non-excluded fact (exclusion is decided upstream, by `engraft.facts`'s key
conflict check) into the single overlay a real measurement run applies.

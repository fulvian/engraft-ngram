# The lens: overlay formats, row variants, and a numpy PLE replica

`engraft/lens.py` sits between the offline table reader (`engraft/table.py`)
and everything that produces or measures overlays: the graft/descent code,
the engine client, and the test suite.

## `.pleo`: the overlay file format

A binary format: magic `PLEO`, `uint32 n`, `uint32 dim` (always 160), `n`
little-endian `int32` global row indices, then `n * dim` little-endian
`float32` values (row-major). `write_pleo`/`read_pleo` round-trip it.
`RowSet.build_overlay` merges several `(rows_global, data)` pairs by global
index, deduplicating identical vectors and raising if the same row would
receive two different vectors -- that should never happen by construction
(each row gets exactly one transform), so a collision is a bug to report, not
to silently resolve.

## PLERT1: routing capture/freeze

A second binary format, used only for MoE routing (never for table rows):
magic `PLERT1`, `uint32 n_layer`, then per layer (ascending key order) `il`
(u32), `ne0` (u32), `ne1` (u32), and `ne0*ne1` `int32` expert indices. `ne1`
is 1 for a layer that only produced routing at the last decoded position
(the usual case for the final layer under `logits: "last"`), or the number of
tokens otherwise. The engine fork writes this format from C++
(`routing_record`); `engraft.lens` only needs to read and write it in Python
to build and check overlays against captured routing, with no engine
involved.

## `RowSet`: the 16 rows at one position

`RowSet.from_position(table, tokens, t)` reads the 16 true rows at position
`t` (via `PleTable.ngram_addresses`) together with their global indices.
Beyond the identity, it offers a handful of row-level transforms used during
development (zero, uniform scale, norm-matched random replacement, swap with
another position's rows, isolate one head) -- none of them used by the graft
path itself, which instead descends the rows by gradient (see
`docs/replica.md` and `docs/method.md`).

## `PleReplica`: a numpy replica of the PLE block

A from-scratch numpy implementation of the PLE block's own forward pass
(key/value projections, a magnitude-and-sign gate, a dilated causal
convolution over the gated value), reading its weights straight from a model
shard's GGUF tensors. It exists to validate, in isolation and without any
engine round-trip, that the addressing and the block's math agree with the
engine's own behavior; `engraft/replica/layers.py`'s `ple_forward` is the
torch port of the same block, used inside the full-model replica.

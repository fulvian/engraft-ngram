# Engine fork

ENGRAFT needs a build of the `fork-ple` branch of llama.cpp, which adds:

- Per-head n-gram (PLE) table reading directly from a split GGUF, with a
  disk-cache gather path (`--ngram-on-disk`) so the table need not fit in
  VRAM.
- A `.pleo` overlay mechanism: at gather time, rows listed in an overlay file
  replace the GGUF's own rows for that request, without ever rewriting the
  quantized bytes on disk.
- `llama-ple-lens`, a tool that speaks a line-based JSON protocol over
  stdin/stdout (`engraft.engine.LensClient`): it loads the model once, then
  answers a stream of jobs (tokens in, logits/tensors out), optionally with
  an overlay, optionally capturing or freezing MoE routing
  (`routing_record`/`routing_freeze`, the `.plert1`/PLERT1 format read and
  written by `engraft.lens`).
- Two launch profiles used by this repository: a quantized default engine,
  and a full-precision (`-fa off -ctk f32 -ctv f32`) engine used to check the
  CPU replica's gradient against the real model without quantization noise
  (see `engraft.engine.ENGINE_CFG`).

## Build

Build `llama-ple-lens` (and, if you also want to serve the model normally,
`llama-server`) from the `fork-ple` branch at the commit recorded in your
`engraft.toml` (`engine.fork_commit`; default `9d9f9f9ad`), following
upstream llama.cpp's own build instructions for your platform (CMake,
optionally with a GPU backend enabled). `scripts/window.sh` checks the built
binary's linked library for that commit string before using it, so a stale
build fails loudly instead of silently measuring the wrong engine.

## Table GGUF layout

The n-gram table is a separate GGUF from the base model's weights, laid out
per head: tensors named `ple_ngram_embd.{h}.weight`, type IQ4_NL, shape
`[160, p_h]` (`p_h` the vocabulary size of head `h`). `engraft/table.py`
reads this file directly, one stripe of rows at a time, and never
materializes a full tensor. See `docs/mechanism.md` for how a token sequence
is hashed into the 16 row indices read at each position.

## Not distributed here

The fork itself, its branch, and its build artifacts are not part of this
repository (the engine is MIT-licensed and lives in its own tree); only the
Python-side client and overlay/format code that talks to it does.

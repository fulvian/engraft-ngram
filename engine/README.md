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

The branch is published at
[github.com/fulvian/llama.cpp, branch `fork-ple`](https://github.com/fulvian/llama.cpp/tree/fork-ple).
It is upstream `master` plus the commits of
[PR 27742](https://github.com/ggml-org/llama.cpp/pull/27742) (the `qwen4exp`
architecture, `--ngram-on-disk`, `--model-ple`) plus seven commits of ours (the
overlay, `llama-ple-lens`, and a backport of PR #26592 (hipCUB) so top-k runs
on HIP without the CPU fallback). Commit `9d9f9f9ad` is the head the run of
record used; it is the default of `engine.fork_commit` in `engraft.toml`.

Build `llama-ple-lens` (and, if you also want to serve the model normally,
`llama-server`) at that commit. On ROCm (we use ROCm 7.14 on a Strix Halo,
gfx1151):

```sh
git clone -b fork-ple https://github.com/fulvian/llama.cpp && cd llama.cpp
git checkout 9d9f9f9ad
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target llama-ple-lens llama-server -j
```

Any other backend follows upstream llama.cpp's own build instructions
(CMake, optionally with a GPU backend enabled). `scripts/window.sh` checks
the built binary's linked library for that commit string before using it, so
a stale build fails loudly instead of silently measuring the wrong engine.

The lens is launched as in `scripts/window.sh`:
`llama-ple-lens -m <model.gguf> --ngram-on-disk -ngl 99 -c 4096 -b 4096 -ub 4096`.
With `--ngram-on-disk` the table stays on disk and is gathered per request;
on a 128 GB Strix Halo the engine takes about 70 GB of GTT.

## Table GGUF layout

The public GGUFs ship the n-gram table as one flat tensor,
`per_layer_token_embd.weight` (IQ4_NL, shape `[160, 320001536]` in
`unsloth/Qwen3.8-Flash-Next-GGUF`, folder `UD-IQ4_XS`, the files the run of
record used). ENGRAFT and the lens use the per-head layout instead: tensors
named `ple_ngram_embd.{h}.weight`, type IQ4_NL, shape `[160, p_h]` (`p_h` the
vocabulary size of head `h`). The conversion is a script that is part of PR
27742 itself, `gguf-py/gguf/scripts/gguf_split_ple_heads.py`: it reads the
head bounds from the file's own `ple.head_offsets`/`ple.head_vocab_sizes` and
copies the quantized bytes through untouched (no dequantize, no requantize,
lossless in both directions). Point it at the first shard:

```sh
python gguf-py/gguf/scripts/gguf_split_ple_heads.py \
  Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf X-ple-split.gguf
```

The output is one file (~94 GB for UD-IQ4_XS) with all tensors and the table
split per head; that is the `table` path of `engraft.toml` and the file the
lens loads. `engraft/table.py` reads it directly, one stripe of rows at a
time, and never materializes a full tensor. See `docs/mechanism.md` for how a
token sequence is hashed into the 16 row indices read at each position.

## Not distributed here

The fork's sources and build artifacts are not part of this repository (the
engine is MIT-licensed and lives in its own tree, linked above); only the
Python-side client and overlay/format code that talks to it does.

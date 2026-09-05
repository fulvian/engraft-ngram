#!/usr/bin/env bash
# Chains facts -> run -> check -> report, with the same windows described in
# docs/method.md. This is the publication-time reproduction path: it needs a
# real GGUF model, its tokenizer, the n-gram table GGUF, and a build of the
# engine fork (see engine/README.md) -- unlike the rest of this repository's
# gates, which run entirely offline against fakes.
#
# Usage: scripts/reproduce.sh <data>
#   scripts/reproduce.sh 2026-01-01
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <data>" >&2
    exit 1
fi

DATA="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ENGRAFT_CONFIG:-${ROOT}/engraft.toml}"

if [ ! -f "${CONFIG}" ]; then
    echo "missing ${CONFIG}: copy engraft.toml.example and fill in your paths first" >&2
    exit 1
fi

# Step 1 -- resolve facts. Needs the tokenizer (tokenizer.json, not a GGUF)
# and the n-gram table GGUF; no engine, no window.
echo "== engraft-facts =="
uv run engraft-facts --config "${CONFIG}"

# Step 2 -- graft on CPU. Needs the model's weight shards and the table GGUF,
# loaded once; this step is the one most sensitive to available RAM (see
# README.md "Requirements"). No engine, no GPU.
echo "== engraft-run ${DATA} =="
uv run engraft-run "${DATA}" --config "${CONFIG}"

# Step 3 -- check on the real engine. This is the only step that needs the
# engine fork built and a model-loading window (GPU memory, or however your
# environment schedules that).
echo "== scripts/window.sh ${DATA} =="
"${ROOT}/scripts/window.sh" "${DATA}"

echo "done: results/${DATA}/report.md"

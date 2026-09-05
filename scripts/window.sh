#!/usr/bin/env bash
# Runs engraft-check on the chained grafts produced by engraft-run.
#
# Usage: scripts/window.sh <data> [--dry-run]
#   scripts/window.sh 2026-01-01            (real engine)
#   scripts/window.sh 2026-01-01 --dry-run  (fake engine, same code path; reads the
#                                             real overlays from results/2026-01-01 but
#                                             writes only to results/2026-01-01-dryrun)
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <data> [--dry-run]" >&2
    exit 1
fi

DATA="$1"
DRY_RUN=0
if [ $# -eq 2 ]; then
    if [ "$2" = "--dry-run" ]; then
        DRY_RUN=1
    else
        echo "unknown argument: $2 (expected --dry-run)" >&2
        exit 1
    fi
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ENGRAFT_CONFIG:-${ROOT}/engraft.toml}"

cfg_print() {
    uv run python -m engraft.config --config "${CONFIG}" --print "$1"
}

EXTRA_ARGS=(--config "${CONFIG}")

if [ "${DRY_RUN}" -eq 1 ]; then
    RUN_DATA="${DATA}-dryrun"
    RESULTS_DIR="${ROOT}/results/${RUN_DATA}"
    # Fake engine standing in for llama-ple-lens: reads the overlays (.pleo/.plert1)
    # already produced in results/${DATA}/ (--results-dir), but writes only to the
    # -dryrun directory (no real results/${DATA} directory is ever touched).
    LENS_CMD="uv run python -m engraft.testing.fake_lens"
    TARGET_MAP="${RESULTS_DIR}/target_token_map.json"
    mkdir -p "${RESULTS_DIR}"
    if [ ! -f "${TARGET_MAP}" ]; then
        echo "expected ${TARGET_MAP} (fid/position -> token within 64): generate it first" >&2
        exit 1
    fi
    # The fake engine's vocabulary (64) is too small for the real answer tokens,
    # the corpus, and the documents; its overlay_hits does not distinguish rows
    # read at the current position from rows merely present in the file (a known
    # limit of the dry-run path): the three switches are enabled only here, a
    # real run keeps them all off.
    EXTRA_ARGS+=(--target-token-map "${TARGET_MAP}" --no-assert-overlay-hits \
                 --skip-corpus --skip-docs \
                 --results-dir "${ROOT}/results/${DATA}")
else
    RUN_DATA="${DATA}"
    RESULTS_DIR="${ROOT}/results/${RUN_DATA}"
    FORK_COMMIT="$(cfg_print engine.fork_commit)"
    LENS="$(cfg_print engine.lens_bin)"
    if [ ! -x "${LENS}" ]; then
        echo "tool not built at the configured path: ${LENS} (build missing or stale)" >&2
        exit 1
    fi
    LENS_COMMON_LIB="$(ldd "${LENS}" | grep libllama-common | awk '{print $3}')"
    if [ -z "${LENS_COMMON_LIB}" ] || [ ! -f "${LENS_COMMON_LIB}" ]; then
        echo "libllama-common.so not resolved by ldd for ${LENS}: cannot verify the commit" >&2
        exit 1
    fi
    if [ "$(strings "${LENS_COMMON_LIB}" | grep -c "${FORK_COMMIT}")" -eq 0 ]; then
        echo "expected commit ${FORK_COMMIT} not found in ${LENS_COMMON_LIB}: rebuild the engine" >&2
        exit 1
    fi
    MODEL_SHARD1="$(uv run python -c "
from engraft.config import load
print(load('${CONFIG}').get_list('model.shards')[0])
")"
    LENS_CMD="${LENS} -m ${MODEL_SHARD1} --ngram-on-disk -ngl 99 -c 4096 -b 4096 -ub 4096"
    EXTRA_ARGS+=(--results-dir "${ROOT}/results/${DATA}")
fi

mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/window.log"

nice -n 10 ionice -c3 uv run \
    python -m engraft.check "${RUN_DATA}" \
    --lens-cmd "${LENS_CMD}" \
    --out-root "${ROOT}/results" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"

echo "report: ${RESULTS_DIR}/report.md"

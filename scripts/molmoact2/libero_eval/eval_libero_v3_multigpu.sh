#!/usr/bin/env bash
# Evaluate one v3 checkpoint with one standard LIBERO suite per GPU.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_PATH="${1:-${POLICY_PATH:-}}"
GPU_IDS=(${EVAL_GPU_IDS:-0 1 2 3})
SUITES=(${EVAL_SUITES:-libero_spatial libero_goal libero_object libero_10})

if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Usage: bash $0 /path/to/checkpoint/pretrained_model" >&2
  echo "Checkpoint not found: ${POLICY_PATH:-<empty>}" >&2
  exit 2
fi
if [[ "${#GPU_IDS[@]}" -ne 4 ]]; then
  echo "Expected exactly four GPU IDs; got: ${GPU_IDS[*]}" >&2
  exit 2
fi
if [[ "${#SUITES[@]}" -ne 4 ]]; then
  echo "Expected exactly four suites; got: ${SUITES[*]}" >&2
  exit 2
fi

echo "[eval-multigpu] checkpoint=${POLICY_PATH}"
for index in "${!SUITES[@]}"; do
  echo "[eval-multigpu] GPU ${GPU_IDS[$index]} -> ${SUITES[$index]}"
done

EVAL_GPU_IDS="${GPU_IDS[*]}" \
EVAL_SUITES="${SUITES[*]}" \
  exec bash "${WS}/scripts/libero_eval/eval_libero_v3_checkpoint.sh" "${POLICY_PATH}"

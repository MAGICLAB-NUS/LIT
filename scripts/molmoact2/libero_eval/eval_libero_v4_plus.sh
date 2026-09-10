#!/usr/bin/env bash
# Evaluate V4 025000 on the full public LIBERO-Plus protocol.
#
# Matches the completed V3 run:
#   - all 10,030 tasks, including Language Instructions
#   - 1 episode/task, seed 1000, batch size 1
#   - round-robin sharding over 8 GPUs by default
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior_v4/seed_1000/stage2/checkpoints/025000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"

if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "V4 checkpoint not found: ${POLICY_PATH}" >&2
  echo "Override with: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"

python3 "${WS}/scripts/libero_eval/verify_v4_checkpoint.py" "${POLICY_PATH}"

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-goal_prior_v4_${checkpoint_step}}"
export PLUS_PROTOCOL="${PLUS_PROTOCOL:-full}"
export EVAL_SEED="${EVAL_SEED:-1000}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-0}"
export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
export EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval_plus/${CHECKPOINT_LABEL}/libero_plus_full_seed_${EVAL_SEED}}"

if [[ "${PLUS_PROTOCOL}" != "full" ]]; then
  echo "V4 comparison wrapper requires PLUS_PROTOCOL=full; got ${PLUS_PROTOCOL}" >&2
  exit 2
fi
if [[ "${EPISODES_PER_TASK}" -ne 1 || "${EVAL_BATCH_SIZE}" -ne 1 ]]; then
  echo "V3-aligned Plus requires EPISODES_PER_TASK=1 and EVAL_BATCH_SIZE=1" >&2
  exit 2
fi

echo "[eval-v4-plus] checkpoint=${POLICY_PATH}"
echo "[eval-v4-plus] output=${EVAL_ROOT}"
echo "[eval-v4-plus] gpus=${EVAL_GPU_IDS} protocol=full tasks=10030"

exec bash "${WS}/scripts/libero_eval/eval_libero_plus_checkpoint.sh" "${POLICY_PATH}"

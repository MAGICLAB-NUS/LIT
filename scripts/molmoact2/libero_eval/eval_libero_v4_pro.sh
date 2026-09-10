#!/usr/bin/env bash
# Evaluate V4 025000 on the public LIBERO-PRO Total16 protocol.
#
# Matches the completed V3 run:
#   - 4 base suites × {object, swap, lan, task}
#   - 10 tasks/suite × 50 episodes/task
#   - official horizons, seed 1000, batch size 1
# Env is excluded because the public packs do not contain those suites.
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
export EVAL_SEED="${EVAL_SEED:-1000}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-2}"
export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
export INCLUDE_ENV="${INCLUDE_ENV:-false}"
export EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval_pro/${CHECKPOINT_LABEL}/full50_seed_${EVAL_SEED}}"

if [[ "${EPISODES_PER_TASK}" -ne 50 || "${EVAL_BATCH_SIZE}" -ne 1 ]]; then
  echo "V3-aligned PRO requires EPISODES_PER_TASK=50 and EVAL_BATCH_SIZE=1" >&2
  exit 2
fi
if [[ "${INCLUDE_ENV}" != "false" ]]; then
  echo "V3-aligned public Total16 requires INCLUDE_ENV=false" >&2
  exit 2
fi

echo "[eval-v4-pro] checkpoint=${POLICY_PATH}"
echo "[eval-v4-pro] output=${EVAL_ROOT}"
echo "[eval-v4-pro] gpus=${EVAL_GPU_IDS} protocol=Total16 rollouts=8000"

exec bash "${WS}/scripts/libero_eval/eval_libero_pro_checkpoint.sh" "${POLICY_PATH}"

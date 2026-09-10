#!/usr/bin/env bash
# LIBERO Stage-1 oracle goal-follow probe (+ optional shuffle control).
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

DEFAULT_CKPT="${WS}/lerobot/outputs/libero_goal_prior_v3/seed_1000/stage1/checkpoints/010000/pretrained_model"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CKPT}}"
SUITE="${SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
INIT_STATE_ID="${INIT_STATE_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_stage1_goal_follow/${SUITE}_task${TASK_ID}_init${INIT_STATE_ID}}"
DEVICE="${DEVICE:-cuda:0}"
MODES="${MODES:-oracle,shuffle}"
MAX_STEPS="${MAX_STEPS:-120}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${WS}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "Stage-1 checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
exec python "${WS}/scripts/libero_goal_prior/eval_stage1_goal_follow.py" \
  --checkpoint "${CHECKPOINT}" \
  --suite "${SUITE}" \
  --task-id "${TASK_ID}" \
  --init-state-id "${INIT_STATE_ID}" \
  --modes "${MODES}" \
  --max-steps "${MAX_STEPS}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

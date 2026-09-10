#!/usr/bin/env bash
# Evaluate a Goal-Pose Prior v3 checkpoint on all four standard LIBERO suites.
#
# v3 has the same Stage-2 architecture as v2b, so this wrapper reuses the
# validated v2b evaluation path while assigning v3-specific paths and labels.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2/checkpoints/020000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"

if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Goal-pose v3 checkpoint not found: ${POLICY_PATH}" >&2
  echo "Override with: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-goal_prior_v3_${checkpoint_step}}"
export MODEL_LABEL="${MODEL_LABEL:-goal_pose_prior_v3}"
export EVAL_VARIANT="${EVAL_VARIANT:-v3}"

exec bash "${WS}/scripts/libero_eval/eval_libero_v2b_checkpoint.sh" "${POLICY_PATH}"

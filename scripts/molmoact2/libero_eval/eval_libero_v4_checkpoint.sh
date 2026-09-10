#!/usr/bin/env bash
# Evaluate a Goal-Pose Prior v4 checkpoint on all four standard LIBERO suites.
#
# v4 reuses the v2b/v3 evaluation path with v4-specific defaults.
# Does not change eval_libero_v3_*.sh.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior_v4/seed_1000/stage2/checkpoints/020000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"

if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Goal-pose v4 checkpoint not found: ${POLICY_PATH}" >&2
  echo "Override with: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-goal_prior_v4_${checkpoint_step}}"
export MODEL_LABEL="${MODEL_LABEL:-goal_pose_prior_v4}"
export EVAL_VARIANT="${EVAL_VARIANT:-v4}"

exec bash "${WS}/scripts/libero_eval/eval_libero_v2b_checkpoint.sh" "${POLICY_PATH}"

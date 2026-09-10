#!/usr/bin/env bash
# Goal-pose prior — Stage 1: vision-free, target-pose-conditioned action prior.
#
# The VLM is frozen (train_action_expert_only=true). Goal tokens come from the SE(3)
# encoder over the chunk-end target state (observation.state at t+H). Only the action
# expert and the goal SE(3) encoder train. Loss = flow matching only.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
export JOB_NAME="${JOB_NAME:-molmoact2-goalprior-stage1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior/seed_${SEED}/stage1}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
# ~273K LIBERO frames; bs 128/GPU x 7 GPUs = 896 effective -> ~305 steps/epoch.
# 10000 steps ~= 33 epochs of the vision-free action-expert + goal-encoder warmup.
export STEPS="${STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export SAVE_FREQ="${SAVE_FREQ:-2500}"
export LOG_FREQ="${LOG_FREQ:-20}"
export SEED

# Stage 1: no image, freeze VLM, train action expert + goal SE(3) encoder.
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=true
export DISABLE_VISUAL_INPUT=true
export IMAGE_TRANSFORMS_ENABLE=false
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-500}"

OPTIMIZER_GOAL_LR="${OPTIMIZER_GOAL_LR:-5e-5}"
SCHEDULER_GOAL_WARMUP_STEPS="${SCHEDULER_GOAL_WARMUP_STEPS:-500}"
NUM_GOAL_TOKENS="${NUM_GOAL_TOKENS:-4}"

# Clean bootstrap: load Molmo2-ER VLM weights, then randomly re-initialize the
# action expert (do NOT inherit the released MolmoAct2 action-expert weights).
USE_ER_BOOTSTRAP="${USE_ER_BOOTSTRAP:-true}"
VLM_CHECKPOINT_PATH="${VLM_CHECKPOINT_PATH:-${WS}/Checkpoint/Molmo2-ER}"

GOAL_ARGS=(
  --policy.enable_goal_pose=true
  --policy.goal_token_source=se3_encoder
  --policy.num_goal_tokens="${NUM_GOAL_TOKENS}"
  --policy.target_pose_delta_index="${CHUNK_SIZE}"
  --policy.mask_image_from_action_expert=false
  --policy.enable_pose_reconstruction=false
  --policy.optimizer_goal_lr="${OPTIMIZER_GOAL_LR}"
  --policy.scheduler_goal_warmup_steps="${SCHEDULER_GOAL_WARMUP_STEPS}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
)

if [[ "${USE_ER_BOOTSTRAP}" == "true" ]]; then
  GOAL_ARGS+=(
    --policy.vlm_checkpoint_path="${VLM_CHECKPOINT_PATH}"
    --policy.randomize_action_expert=true
    --policy.audit_bootstrap=true
  )
fi

exec bash "${WS}/scripts/train_libero_molmoact2.sh" "${GOAL_ARGS[@]}" "$@"

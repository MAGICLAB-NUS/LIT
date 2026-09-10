#!/usr/bin/env bash
# Goal-pose prior — Stage 2: visual Goal-Pose-Token steering.
#
# Loads the Stage-1 checkpoint, re-enables vision, and swaps the goal-token source to
# learnable queries that the VLM contextualizes from vision. Raw image tokens are masked
# out of the action expert; a decoder reconstructs the target pose (L_pose). VLM + action
# expert train jointly at full LR (no down-scaling), relying on warmup; ViT gets a longer
# warmup. Loss = flow matching + pose_recon_loss_weight * L_pose.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
export JOB_NAME="${JOB_NAME:-molmoact2-goalprior-stage2}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior/seed_${SEED}/stage1}"
export POLICY_PATH="${POLICY_PATH:-${STAGE1_OUTPUT_DIR}/checkpoints/last/pretrained_model}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior/seed_${SEED}/stage2}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
# ~273K LIBERO frames; bs 32/GPU x 7 GPUs = 224 effective -> ~1221 steps/epoch.
# 30000 steps ~= 25 epochs of the full VLM + action-expert visual finetune.
export STEPS="${STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export LOG_FREQ="${LOG_FREQ:-20}"
export SEED

# Stage 2: vision on, joint VLM + action-expert training.
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

# No LR down-scaling; rely on warmup. Give ViT a noticeably longer warmup.
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
export OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-5}"
export OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-5}"
export OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-1e-5}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-2000}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-2000}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-4000}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-2000}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-2000}"

OPTIMIZER_GOAL_LR="${OPTIMIZER_GOAL_LR:-5e-5}"
SCHEDULER_GOAL_WARMUP_STEPS="${SCHEDULER_GOAL_WARMUP_STEPS:-2000}"
NUM_GOAL_TOKENS="${NUM_GOAL_TOKENS:-4}"
POSE_RECON_LOSS_WEIGHT="${POSE_RECON_LOSS_WEIGHT:-1.0}"
INIT_QUERIES_FROM_SE3="${INIT_QUERIES_FROM_SE3:-false}"

if [[ ! -f "${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" \
  && ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Stage-1 checkpoint is missing: ${POLICY_PATH}" >&2
  exit 2
fi

GOAL_ARGS=(
  --policy.enable_goal_pose=true
  --policy.goal_token_source=learnable_queries
  --policy.num_goal_tokens="${NUM_GOAL_TOKENS}"
  --policy.target_pose_delta_index="${CHUNK_SIZE}"
  --policy.mask_image_from_action_expert=true
  --policy.enable_pose_reconstruction=true
  --policy.pose_recon_loss_weight="${POSE_RECON_LOSS_WEIGHT}"
  --policy.init_queries_from_se3_encoder="${INIT_QUERIES_FROM_SE3}"
  --policy.optimizer_goal_lr="${OPTIMIZER_GOAL_LR}"
  --policy.scheduler_goal_warmup_steps="${SCHEDULER_GOAL_WARMUP_STEPS}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
)

exec bash "${WS}/scripts/train_libero_molmoact2.sh" "${GOAL_ARGS[@]}" "$@"

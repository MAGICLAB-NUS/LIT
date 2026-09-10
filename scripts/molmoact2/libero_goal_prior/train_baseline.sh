#!/usr/bin/env bash
# Baseline for the goal-pose prior experiment: plain MolmoAct2 fine-tuning.
#
# Same clean bootstrap as the goal-prior runs (Molmo2-ER VLM weights + randomly
# re-initialized action expert), but WITHOUT any goal-pose machinery: vision is
# on, the whole model trains, loss is flow matching only. This is the
# apples-to-apples comparison point for exp1.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
export JOB_NAME="${JOB_NAME:-molmoact2-baseline}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior/seed_${SEED}/libero_baseline}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export POLICY_PATH=""
# ~273K LIBERO frames; bs 32/GPU x 7 GPUs = 224 effective -> ~1221 steps/epoch.
export STEPS="${STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export LOG_FREQ="${LOG_FREQ:-20}"
export SEED

# Normal full training: vision on, train the whole model, flow loss only.
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

# Match the v2 Stage2 optimizer and warmup settings.
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
export OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-5}"
export OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-5}"
export OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-1e-5}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-1000}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-2000}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-1000}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-1000}"

# Clean bootstrap: load Molmo2-ER VLM weights, then randomly re-initialize the
# action expert (do NOT inherit the released MolmoAct2 action-expert weights).
VLM_CHECKPOINT_PATH="${VLM_CHECKPOINT_PATH:-${WS}/Checkpoint/Molmo2-ER}"

BASELINE_ARGS=(
  --policy.enable_goal_pose=false
  --policy.mask_image_from_action_expert=false
  --policy.enable_pose_reconstruction=false
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
  --policy.vlm_checkpoint_path="${VLM_CHECKPOINT_PATH}"
  --policy.randomize_action_expert=true
  --policy.audit_bootstrap=true
)

exec bash "${WS}/scripts/train_libero_molmoact2.sh" "${BASELINE_ARGS[@]}" "$@"

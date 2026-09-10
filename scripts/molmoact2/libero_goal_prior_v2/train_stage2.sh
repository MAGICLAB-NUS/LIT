#!/usr/bin/env bash
# Goal-pose prior v2 — recurrent semantic -> visual latent steering.
#
# Stage 1 is intentionally reused unchanged. Stage 2 loads its vision-free AE,
# creates 100x768 latent queries, and recurrently updates them at each VLM layer:
# language/state cross-attention -> image cross-attention -> corresponding AE layer.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
export JOB_NAME="${JOB_NAME:-molmoact2-goalprior-stage2-v2}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior/seed_${SEED}/stage1}"
export POLICY_PATH="${POLICY_PATH:-${STAGE1_OUTPUT_DIR}/checkpoints/010000/pretrained_model}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior_v2/seed_${SEED}/stage2}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export STEPS="${STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export LOG_FREQ="${LOG_FREQ:-20}"
export SEED

export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

# Full LR for all inherited and new components. Warmup is shorter than v1 because
# 5k/10k/15k checkpoints are decision points for this experiment.
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
export OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-5}"
export OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-5}"
export OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-1e-5}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-1000}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-2000}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-1000}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-1000}"

NUM_SEMANTIC_VISUAL_TOKENS="${NUM_SEMANTIC_VISUAL_TOKENS:-100}"
SEMANTIC_VISUAL_HIDDEN_DIM="${SEMANTIC_VISUAL_HIDDEN_DIM:-768}"
SEMANTIC_VISUAL_NUM_HEADS="${SEMANTIC_VISUAL_NUM_HEADS:-8}"
SEMANTIC_VISUAL_FFN_RATIO="${SEMANTIC_VISUAL_FFN_RATIO:-4.0}"
SEMANTIC_VISUAL_DROPOUT="${SEMANTIC_VISUAL_DROPOUT:-0.0}"
OPTIMIZER_SEMANTIC_VISUAL_LR="${OPTIMIZER_SEMANTIC_VISUAL_LR:-1e-5}"
SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS="${SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS:-1000}"
POSE_RECON_LOSS_WEIGHT="${POSE_RECON_LOSS_WEIGHT:-1.0}"

V2_ARGS=(
  --policy.enable_goal_pose=true
  --policy.goal_token_source=learnable_queries
  --policy.goal_conditioning_mode=semantic_visual_recurrent
  --policy.num_semantic_visual_tokens="${NUM_SEMANTIC_VISUAL_TOKENS}"
  --policy.semantic_visual_hidden_dim="${SEMANTIC_VISUAL_HIDDEN_DIM}"
  --policy.semantic_visual_num_heads="${SEMANTIC_VISUAL_NUM_HEADS}"
  --policy.semantic_visual_ffn_ratio="${SEMANTIC_VISUAL_FFN_RATIO}"
  --policy.semantic_visual_dropout="${SEMANTIC_VISUAL_DROPOUT}"
  --policy.target_pose_delta_index="${CHUNK_SIZE}"
  --policy.mask_image_from_action_expert=true
  --policy.enable_pose_reconstruction=true
  --policy.pose_recon_loss_weight="${POSE_RECON_LOSS_WEIGHT}"
  --policy.optimizer_semantic_visual_lr="${OPTIMIZER_SEMANTIC_VISUAL_LR}"
  --policy.scheduler_semantic_visual_warmup_steps="${SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
)

if [[ ! -f "${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" \
  && ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Stage-1 checkpoint is missing: ${POLICY_PATH}" >&2
  exit 2
fi

exec bash "${WS}/scripts/train_libero_molmoact2.sh" "${V2_ARGS[@]}" "$@"

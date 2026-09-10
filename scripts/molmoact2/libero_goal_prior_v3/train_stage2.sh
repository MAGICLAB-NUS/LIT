#!/usr/bin/env bash
# Goal-pose prior v3 = v2b architecture + corrected dataset QUANTILES.
#
# Stage2 ALWAYS initializes from v3 Stage1 10k checkpoint, and ALWAYS requires
# the recomputed (new) observation.state / action QUANTILES in dataset meta.
#
# Before first launch:
#   bash scripts/libero_goal_prior_v3/fix_stats.sh
#   bash scripts/libero_goal_prior_v3/train_stage1.sh   # writes .../stage1/checkpoints/010000
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
export JOB_NAME="${JOB_NAME:-molmoact2-goalprior-stage2-v3}"

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior_v3/seed_${SEED}/stage1}"
# Fixed: Stage2 starts from Stage1 step 10000 only (no legacy / v2b fallback).
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_OUTPUT_DIR}/checkpoints/010000/pretrained_model}"
export POLICY_PATH="${POLICY_PATH:-${STAGE1_CKPT}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior_v3/seed_${SEED}/stage2}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export STEPS="${STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export LOG_FREQ="${LOG_FREQ:-20}"
export SEED

export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

# Locked to the seed_1000 Stage2 025000 run (StarVLA-aligned LRs):
# VLM/ViT/connector 1e-5; action expert + semantic-visual aggregator 1e-4.
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
export OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-5}"
export OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-5}"
export OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-1e-4}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-1000}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-2000}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-1000}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-5000}"

NUM_SEMANTIC_VISUAL_TOKENS="${NUM_SEMANTIC_VISUAL_TOKENS:-100}"
NUM_SEMANTIC_VISUAL_POSE_TOKENS="${NUM_SEMANTIC_VISUAL_POSE_TOKENS:-8}"
SEMANTIC_VISUAL_HIDDEN_DIM="${SEMANTIC_VISUAL_HIDDEN_DIM:-768}"
SEMANTIC_VISUAL_NUM_HEADS="${SEMANTIC_VISUAL_NUM_HEADS:-8}"
SEMANTIC_VISUAL_FFN_RATIO="${SEMANTIC_VISUAL_FFN_RATIO:-4.0}"
SEMANTIC_VISUAL_DROPOUT="${SEMANTIC_VISUAL_DROPOUT:-0.0}"
SEMANTIC_VISUAL_ENABLE_SELF_ATTENTION="${SEMANTIC_VISUAL_ENABLE_SELF_ATTENTION:-true}"
SEMANTIC_VISUAL_NUM_LAYER_GROUPS="${SEMANTIC_VISUAL_NUM_LAYER_GROUPS:-6}"
OPTIMIZER_SEMANTIC_VISUAL_LR="${OPTIMIZER_SEMANTIC_VISUAL_LR:-1e-4}"
SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS="${SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS:-5000}"
POSE_RECON_LOSS_WEIGHT="${POSE_RECON_LOSS_WEIGHT:-0.3}"

DATASET_ROOT="${DATASET_ROOT:-/data2/JM/dataset/libero_lerobot_format}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero}"
STATS_JSON="${DATASET_ROOT}/meta/stats.json"
STATS_NOTE="${DATASET_ROOT}/meta/stats.v3_recompute_note.json"
STATS_OLD="${DATASET_ROOT}/meta/stats.pre_v3_bad_quantiles.json"

# --- Require new QUANTILES (not the pre-v3 clipped state Z band) ---
if [[ "${SKIP_STATS_CHECK:-0}" != "1" ]]; then
  if [[ ! -f "${STATS_NOTE}" ]]; then
    echo "v3 stats not recomputed yet. Run:" >&2
    echo "  bash scripts/libero_goal_prior_v3/fix_stats.sh" >&2
    exit 2
  fi
  if [[ ! -f "${STATS_JSON}" ]]; then
    echo "Missing ${STATS_JSON}" >&2
    exit 2
  fi
  # state Z q99 must be the wide recomputed range (~1.27), not old ~0.88
  python - "${STATS_JSON}" "${STATS_OLD}" <<'PY'
import json, sys
from pathlib import Path
stats_path, old_path = Path(sys.argv[1]), Path(sys.argv[2])
stats = json.loads(stats_path.read_text())["observation.state"]
z_q01, z_q99 = float(stats["q01"][2]), float(stats["q99"][2])
# Hard reject the known-bad pre-v3 band.
if z_q99 < 1.0 or z_q01 > 0.5:
    raise SystemExit(
        f"REFUSING: {stats_path} still looks like old clipped state Z "
        f"q01/q99=[{z_q01:.4f}, {z_q99:.4f}]. Re-run fix_stats.sh."
    )
if old_path.is_file():
    old = json.loads(old_path.read_text())["observation.state"]
    if float(old["q99"][2]) == z_q99 and float(old["q01"][2]) == z_q01:
        raise SystemExit(
            f"REFUSING: {stats_path} matches pre-v3 bad quantiles. Re-run fix_stats.sh."
        )
print(f"[v3] verified NEW state Z q01/q99=[{z_q01:.4f}, {z_q99:.4f}]")
PY
fi

# --- Require Stage1 10k as init (unless Stage2 itself is being resumed) ---
RESUMING_STAGE2=0
if [[ -f "${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" \
  && "${RESUME_MODE}" == "auto" ]]; then
  RESUMING_STAGE2=1
fi

if [[ "${RESUMING_STAGE2}" -eq 0 ]]; then
  if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
    echo "v3 Stage1 10k checkpoint missing: ${POLICY_PATH}" >&2
    echo "Finish Stage1 first:" >&2
    echo "  bash scripts/libero_goal_prior_v3/train_stage1.sh" >&2
    exit 2
  fi
  # Refuse accidental init from non-v3 / non-010000 paths unless explicitly overridden.
  if [[ "${ALLOW_NON_V3_STAGE1:-0}" != "1" ]]; then
    case "${POLICY_PATH}" in
      */libero_goal_prior_v3/*/stage1/checkpoints/010000/pretrained_model) ;;
      */libero_goal_prior_v3/*/stage1/checkpoints/010000/pretrained_model/) ;;
      *)
        echo "POLICY_PATH must be v3 Stage1 10k checkpoint, got:" >&2
        echo "  ${POLICY_PATH}" >&2
        echo "Expected under: .../libero_goal_prior_v3/.../stage1/checkpoints/010000/pretrained_model" >&2
        echo "Set ALLOW_NON_V3_STAGE1=1 only if you really intend otherwise." >&2
        exit 2
        ;;
    esac
  fi
fi

V3_ARGS=(
  --policy.enable_goal_pose=true
  --policy.goal_token_source=learnable_queries
  --policy.goal_conditioning_mode=semantic_visual_recurrent
  --policy.num_semantic_visual_tokens="${NUM_SEMANTIC_VISUAL_TOKENS}"
  --policy.num_semantic_visual_pose_tokens="${NUM_SEMANTIC_VISUAL_POSE_TOKENS}"
  --policy.semantic_visual_hidden_dim="${SEMANTIC_VISUAL_HIDDEN_DIM}"
  --policy.semantic_visual_num_heads="${SEMANTIC_VISUAL_NUM_HEADS}"
  --policy.semantic_visual_ffn_ratio="${SEMANTIC_VISUAL_FFN_RATIO}"
  --policy.semantic_visual_dropout="${SEMANTIC_VISUAL_DROPOUT}"
  --policy.semantic_visual_enable_self_attention="${SEMANTIC_VISUAL_ENABLE_SELF_ATTENTION}"
  --policy.semantic_visual_num_layer_groups="${SEMANTIC_VISUAL_NUM_LAYER_GROUPS}"
  --policy.target_pose_delta_index="${CHUNK_SIZE}"
  --policy.mask_image_from_action_expert=true
  --policy.enable_pose_reconstruction=true
  --policy.pose_recon_loss_weight="${POSE_RECON_LOSS_WEIGHT}"
  --policy.optimizer_semantic_visual_lr="${OPTIMIZER_SEMANTIC_VISUAL_LR}"
  --policy.scheduler_semantic_visual_warmup_steps="${SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
)

echo "[v3-stage2] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[v3-stage2] POLICY_PATH=${POLICY_PATH} (Stage1 10k init)"
echo "[v3-stage2] lr: vlm/vit/conn=${OPTIMIZER_LR} ae=${OPTIMIZER_ACTION_EXPERT_LR} sv=${OPTIMIZER_SEMANTIC_VISUAL_LR}"
echo "[v3-stage2] warmup: vlm=${SCHEDULER_VLM_WARMUP_STEPS} vit=${SCHEDULER_VIT_WARMUP_STEPS} ae=${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS} sv=${SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS}"
echo "[v3-stage2] DATASET_ROOT=${DATASET_ROOT} (must use NEW state stats)"
echo "[v3-stage2] resuming_stage2=${RESUMING_STAGE2}"

exec bash "${WS}/scripts/train_libero_molmoact2.sh" "${V3_ARGS[@]}" "$@"

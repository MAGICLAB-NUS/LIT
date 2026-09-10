#!/usr/bin/env bash
# Goal-pose prior v3 — Stage1 with corrected QUANTILES.
#
# Same recipe as scripts/libero_goal_prior/train_stage1.sh, but writes to the
# v3 output tree so it does not collide with any restored v1 Stage1 path.
# Requires: bash scripts/libero_goal_prior_v3/fix_stats.sh
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
export JOB_NAME="${JOB_NAME:-molmoact2-goalprior-stage1-v3}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior_v3/seed_${SEED}/stage1}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export STEPS="${STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export SAVE_FREQ="${SAVE_FREQ:-2500}"
export LOG_FREQ="${LOG_FREQ:-20}"
export SEED

export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=true
export DISABLE_VISUAL_INPUT=true
export IMAGE_TRANSFORMS_ENABLE=false
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-500}"

OPTIMIZER_GOAL_LR="${OPTIMIZER_GOAL_LR:-5e-5}"
SCHEDULER_GOAL_WARMUP_STEPS="${SCHEDULER_GOAL_WARMUP_STEPS:-500}"
NUM_GOAL_TOKENS="${NUM_GOAL_TOKENS:-4}"

USE_ER_BOOTSTRAP="${USE_ER_BOOTSTRAP:-true}"
VLM_CHECKPOINT_PATH="${VLM_CHECKPOINT_PATH:-${WS}/Checkpoint/Molmo2-ER}"

DATASET_ROOT="${DATASET_ROOT:-/data2/JM/dataset/libero_lerobot_format}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero}"
STATS_JSON="${DATASET_ROOT}/meta/stats.json"
STATS_NOTE="${DATASET_ROOT}/meta/stats.v3_recompute_note.json"
STATS_OLD="${DATASET_ROOT}/meta/stats.pre_v3_bad_quantiles.json"
if [[ "${SKIP_STATS_CHECK:-0}" != "1" ]]; then
  if [[ ! -f "${STATS_NOTE}" ]]; then
    echo "v3 stats not recomputed yet. Run:" >&2
    echo "  bash scripts/libero_goal_prior_v3/fix_stats.sh" >&2
    exit 2
  fi
  python - "${STATS_JSON}" "${STATS_OLD}" <<'PY'
import json, sys
from pathlib import Path
stats_path, old_path = Path(sys.argv[1]), Path(sys.argv[2])
stats = json.loads(stats_path.read_text())["observation.state"]
z_q01, z_q99 = float(stats["q01"][2]), float(stats["q99"][2])
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
print(f"[v3-stage1] verified NEW state Z q01/q99=[{z_q01:.4f}, {z_q99:.4f}]")
PY
fi

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

echo "[v3-stage1] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[v3-stage1] stats note: ${STATS_NOTE}"

exec bash "${WS}/scripts/train_libero_molmoact2.sh" "${GOAL_ARGS[@]}" "$@"

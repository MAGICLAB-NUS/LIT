#!/usr/bin/env bash
# DROID goal-pose prior Stage 1: vision-free action prior from future EE pose.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE=15
export JOB_NAME="${JOB_NAME:-molmoact2-droid-goalprior-stage1}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SEED

if [[ "${SMOKE_RUN:-false}" == "true" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/droid_goal_prior/seed_${SEED}/smoke/stage1}"
  export STEPS="${SMOKE_STEPS:-100}"
  export BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
  export NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
  export SAVE_FREQ="${SMOKE_SAVE_FREQ:-100}"
  export LOG_FREQ="${SMOKE_LOG_FREQ:-1}"
  export SAVE_CHECKPOINT="${SMOKE_SAVE_CHECKPOINT:-true}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/droid_goal_prior/seed_${SEED}/stage1}"
  export STEPS="${STEPS:-50000}"
  export BATCH_SIZE="${BATCH_SIZE:-128}"
  export SAVE_FREQ="${SAVE_FREQ:-10000}"
  export LOG_FREQ="${LOG_FREQ:-20}"
fi

export DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/droid_1.0.1}"
export DATASET_ROOT="${DATASET_ROOT:-/data0/JM/dataset/droid_1.0.1_goal_pose}"
export SAMPLE_MANIFEST_PATH="${SAMPLE_MANIFEST_PATH:-${DATASET_ROOT}/valid_anchor_indices.parquet}"
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=true
export DISABLE_VISUAL_INPUT=true
export IMAGE_TRANSFORMS_ENABLE=false
export OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-5e-5}"
export OPTIMIZER_BETAS="${OPTIMIZER_BETAS:-[0.9,0.95]}"
export OPTIMIZER_EPS="${OPTIMIZER_EPS:-1e-6}"
export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-0.0}"
export OPTIMIZER_GRAD_CLIP_NORM="${OPTIMIZER_GRAD_CLIP_NORM:-1.0}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-500}"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WS}/Checkpoint/MolmoAct2}"
export CHECKPOINT_REVISION="${CHECKPOINT_REVISION:-e432d85f6e039edca44afb93c262f3084ab72a9c}"
VLM_CHECKPOINT_PATH="${VLM_CHECKPOINT_PATH:-${WS}/Checkpoint/Molmo2-ER}"
VLM_CHECKPOINT_REVISION="${VLM_CHECKPOINT_REVISION:-dab22564403d2607855bb1fffb0721285b445081}"
OPTIMIZER_GOAL_LR="${OPTIMIZER_GOAL_LR:-5e-5}"
SCHEDULER_GOAL_WARMUP_STEPS="${SCHEDULER_GOAL_WARMUP_STEPS:-500}"
NUM_GOAL_TOKENS="${NUM_GOAL_TOKENS:-4}"

"${WS}/lerobot/.venv/bin/python" - \
  "${CHECKPOINT_PATH}" "${CHECKPOINT_REVISION}" \
  "${VLM_CHECKPOINT_PATH}" "${VLM_CHECKPOINT_REVISION}" <<'PY'
import sys
from pathlib import Path


def verify_download(label: str, raw_path: str, expected_revision: str) -> None:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"REFUSING: missing pinned {label} checkpoint: {path}")
    marker = path / ".cache/huggingface/download/config.json.metadata"
    if not marker.is_file():
        raise SystemExit(f"REFUSING: cannot verify {label} revision: {marker}")
    actual = marker.read_text(encoding="utf-8").splitlines()[0].strip()
    if actual != expected_revision:
        raise SystemExit(
            f"REFUSING: {label} revision is {actual}, expected {expected_revision}"
        )


verify_download("MolmoAct2", sys.argv[1], sys.argv[2])
verify_download("Molmo2-ER", sys.argv[3], sys.argv[4])
PY

STAGE1_ARGS=(
  --policy.enable_goal_pose=true
  --policy.goal_pose_feature_key=observation.ee_pose
  --policy.goal_token_source=se3_encoder
  --policy.num_goal_tokens="${NUM_GOAL_TOKENS}"
  --policy.target_pose_delta_index="${CHUNK_SIZE}"
  --policy.mask_image_from_action_expert=false
  --policy.enable_pose_reconstruction=false
  --policy.optimizer_goal_lr="${OPTIMIZER_GOAL_LR}"
  --policy.scheduler_goal_warmup_steps="${SCHEDULER_GOAL_WARMUP_STEPS}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
  --policy.vlm_checkpoint_path="${VLM_CHECKPOINT_PATH}"
  --policy.vlm_checkpoint_revision="${VLM_CHECKPOINT_REVISION}"
  --policy.randomize_action_expert=true
  --policy.audit_bootstrap=true
)

echo "[droid-stage1] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[droid-stage1] VLM=${VLM_CHECKPOINT_PATH} revision=${VLM_CHECKPOINT_REVISION}"
echo "[droid-stage1] random continuous AE; visual input disabled; VLM frozen"

exec bash "${WS}/scripts/train_droid_molmoact2.sh" "${STAGE1_ARGS[@]}" "$@"

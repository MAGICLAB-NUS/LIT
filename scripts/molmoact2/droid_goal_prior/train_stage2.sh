#!/usr/bin/env bash
# DROID goal-pose prior Stage 2: visual semantic recurrent conditioning.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE=15
export JOB_NAME="${JOB_NAME:-molmoact2-droid-goalprior-stage2}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${WS}/lerobot/outputs/droid_goal_prior/seed_${SEED}/stage1}"
if [[ "${SMOKE_RUN:-false}" == "true" ]]; then
  STAGE1_SOURCE_STEPS="${STAGE1_SMOKE_STEPS:-100}"
  printf -v STAGE1_STEP_DIR "%06d" "${STAGE1_SOURCE_STEPS}"
  STAGE1_CKPT="${STAGE1_CKPT:-${WS}/lerobot/outputs/droid_goal_prior/seed_${SEED}/smoke/stage1/checkpoints/${STAGE1_STEP_DIR}/pretrained_model}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/droid_goal_prior/seed_${SEED}/smoke/stage2}"
else
  STAGE1_SOURCE_STEPS="${STAGE1_FORMAL_STEPS:-50000}"
  printf -v STAGE1_STEP_DIR "%06d" "${STAGE1_SOURCE_STEPS}"
  STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_OUTPUT_DIR}/checkpoints/${STAGE1_STEP_DIR}/pretrained_model}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/droid_goal_prior/seed_${SEED}/stage2}"
fi
export POLICY_PATH="${STAGE1_CKPT}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SEED

if [[ "${SMOKE_RUN:-false}" == "true" ]]; then
  export STEPS="${SMOKE_STEPS:-100}"
  export BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
  export NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
  export SAVE_FREQ="${SMOKE_SAVE_FREQ:-100}"
  export LOG_FREQ="${SMOKE_LOG_FREQ:-1}"
  export SAVE_CHECKPOINT="${SMOKE_SAVE_CHECKPOINT:-false}"
else
  export STEPS="${STEPS:-180000}"
  # Three DROID camera streams make bs=32 fit in a single process but OOM
  # during DDP backward on 80 GB GPUs. bs=24 was verified on 7-GPU DDP.
  export BATCH_SIZE="${BATCH_SIZE:-24}"
  export SAVE_FREQ="${SAVE_FREQ:-30000}"
  export LOG_FREQ="${LOG_FREQ:-20}"
fi

export DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/droid_1.0.1}"
export DATASET_ROOT="${DATASET_ROOT:-/data0/JM/dataset/droid_1.0.1_goal_pose}"
export SAMPLE_MANIFEST_PATH="${SAMPLE_MANIFEST_PATH:-${DATASET_ROOT}/valid_anchor_indices.parquet}"
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
export OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-5}"
export OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-5}"
export OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-1e-4}"
export OPTIMIZER_BETAS="${OPTIMIZER_BETAS:-[0.9,0.95]}"
export OPTIMIZER_EPS="${OPTIMIZER_EPS:-1e-6}"
export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-0.0}"
export OPTIMIZER_GRAD_CLIP_NORM="${OPTIMIZER_GRAD_CLIP_NORM:-1.0}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-5000}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-5000}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-5000}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-5000}"
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
VALIDATE_STAGE1="${VALIDATE_STAGE1:-true}"

if [[ "${VALIDATE_STAGE1}" != "true" && "${DRY_RUN:-false}" != "true" ]]; then
  echo "VALIDATE_STAGE1=false is allowed only with DRY_RUN=true." >&2
  exit 2
fi

if [[ "${VALIDATE_STAGE1}" == "true" ]]; then
  "${WS}/lerobot/.venv/bin/python" - \
    "${STAGE1_CKPT}/train_config.json" \
    "${DATASET_ROOT}" \
    "${DATASET_REPO_ID}" \
    "${STAGE1_CKPT}" \
    "${SMOKE_RUN:-false}" \
    "${STAGE1_SOURCE_STEPS}" \
    "${SAMPLE_MANIFEST_PATH}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

config_path = Path(sys.argv[1]).expanduser().resolve()
dataset_root = str(Path(sys.argv[2]).expanduser().resolve())
repo_id = sys.argv[3]
checkpoint = Path(sys.argv[4]).expanduser().resolve()
smoke_run = sys.argv[5].lower() == "true"
stage1_steps = int(sys.argv[6])
manifest_path = str(Path(sys.argv[7]).expanduser().resolve())


def refuse(message: str) -> None:
    raise SystemExit(f"REFUSING: {message}")


seed_dir = checkpoint.parents[3].name if not smoke_run else checkpoint.parents[4].name
if smoke_run:
    expected_suffix = (
        Path("droid_goal_prior")
        / seed_dir
        / "smoke"
        / "stage1"
        / "checkpoints"
        / f"{stage1_steps:06d}"
        / "pretrained_model"
    )
else:
    expected_suffix = (
        Path("droid_goal_prior")
        / seed_dir
        / "stage1"
        / "checkpoints"
        / f"{stage1_steps:06d}"
        / "pretrained_model"
    )
if checkpoint.parts[-len(expected_suffix.parts) :] != expected_suffix.parts:
    refuse(f"Stage2 checkpoint path must end with {expected_suffix}, got {checkpoint}")
if not config_path.is_file() or not (checkpoint / "config.json").is_file():
    refuse(f"Stage1 {stage1_steps:06d} checkpoint is incomplete: {checkpoint}")
try:
    train = json.loads(config_path.read_text())
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    refuse(f"invalid Stage1 train_config.json: {exc}")

dataset = train.get("dataset", {})
policy = train.get("policy", {})
checks = {
    "dataset.repo_id": (dataset.get("repo_id"), repo_id),
    "dataset.root": (
        str(Path(dataset.get("root", "")).expanduser().resolve()),
        dataset_root,
    ),
    "dataset.sample_indices_path": (
        str(Path(dataset.get("sample_indices_path", "")).expanduser().resolve()),
        manifest_path,
    ),
    "steps": (train.get("steps"), stage1_steps),
    "policy.action_mode": (policy.get("action_mode"), "continuous"),
    "policy.chunk_size": (policy.get("chunk_size"), 15),
    "policy.n_action_steps": (policy.get("n_action_steps"), 15),
    "policy.setup_type": (policy.get("setup_type"), "single franka robotic arm in droid"),
    "policy.control_mode": (policy.get("control_mode"), "absolute joint pose"),
    "policy.image_keys": (
        policy.get("image_keys"),
        [
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
        ],
    ),
    "policy.goal_pose_feature_key": (
        policy.get("goal_pose_feature_key"),
        "observation.ee_pose",
    ),
    "policy.goal_token_source": (policy.get("goal_token_source"), "se3_encoder"),
    "policy.target_pose_delta_index": (policy.get("target_pose_delta_index"), 15),
    "policy.enable_goal_pose": (policy.get("enable_goal_pose"), True),
    "policy.disable_visual_input": (policy.get("disable_visual_input"), True),
    "policy.train_action_expert_only": (policy.get("train_action_expert_only"), True),
    "policy.randomize_action_expert": (policy.get("randomize_action_expert"), True),
    "policy.audit_bootstrap": (policy.get("audit_bootstrap"), True),
    "policy.optimizer_betas": (policy.get("optimizer_betas"), [0.9, 0.95]),
    "policy.scheduler_action_expert_warmup_steps": (
        policy.get("scheduler_action_expert_warmup_steps"),
        500,
    ),
    "policy.scheduler_goal_warmup_steps": (
        policy.get("scheduler_goal_warmup_steps"),
        500,
    ),
    "policy.scheduler_decay_steps": (
        policy.get("scheduler_decay_steps"),
        stage1_steps,
    ),
    "policy.checkpoint_revision": (
        policy.get("checkpoint_revision"),
        "e432d85f6e039edca44afb93c262f3084ab72a9c",
    ),
    "policy.vlm_checkpoint_revision": (
        policy.get("vlm_checkpoint_revision"),
        "dab22564403d2607855bb1fffb0721285b445081",
    ),
}
for label, (actual, expected) in checks.items():
    if actual != expected:
        refuse(f"Stage1 {label} must be {expected!r}, got {actual!r}")
for label in ("optimizer_action_expert_lr", "optimizer_goal_lr"):
    actual = policy.get(label)
    if not isinstance(actual, (int, float)) or not math.isclose(float(actual), 5e-5):
        refuse(f"Stage1 policy.{label} must be 5e-5, got {actual!r}")
for label, expected in (
    ("optimizer_eps", 1e-6),
    ("optimizer_weight_decay", 0.0),
    ("optimizer_grad_clip_norm", 1.0),
    ("scheduler_decay_lr", 1e-6),
):
    actual = policy.get(label)
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
    ):
        refuse(f"Stage1 policy.{label} must be {expected!r}, got {actual!r}")
vlm_path = str(policy.get("vlm_checkpoint_path", "")).rstrip("/")
if Path(vlm_path).name != "Molmo2-ER":
    refuse(f"Stage1 VLM must be Molmo2-ER, got {vlm_path!r}")
feature_shapes = {
    "observation.state": policy.get("input_features", {}).get("observation.state", {}).get("shape"),
    "observation.ee_pose": policy.get("input_features", {}).get("observation.ee_pose", {}).get("shape"),
    "action": policy.get("output_features", {}).get("action", {}).get("shape"),
}
for key, expected in (("observation.state", [8]), ("observation.ee_pose", [7]), ("action", [8])):
    if feature_shapes[key] != expected:
        refuse(f"Stage1 feature {key} must have shape {expected}, got {feature_shapes[key]!r}")

stats_path = Path(dataset_root) / "meta/stats.json"
try:
    dataset_stats = json.loads(stats_path.read_text())
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    refuse(f"invalid dataset stats {stats_path}: {exc}")


def validate_processor_stats(config_name: str, registry_name: str, features: tuple[str, ...]) -> None:
    processor_config_path = checkpoint / config_name
    try:
        processor_config = json.loads(processor_config_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        refuse(f"invalid Stage1 processor config {processor_config_path}: {exc}")
    matching = [
        step for step in processor_config.get("steps", [])
        if step.get("registry_name") == registry_name
    ]
    if len(matching) != 1 or not matching[0].get("state_file"):
        refuse(f"Stage1 {config_name} must contain exactly one {registry_name} state")
    state_path = checkpoint / matching[0]["state_file"]
    if not state_path.is_file():
        refuse(f"missing Stage1 processor state: {state_path}")
    expected_masks = {
        "observation.state": np.asarray([1] * 7 + [0], dtype=np.float32),
        "action": np.asarray([1] * 7 + [0], dtype=np.float32),
        "observation.ee_pose": np.asarray([1] * 6 + [0], dtype=np.float32),
    }
    with safe_open(state_path, framework="np") as state:
        available = set(state.keys())
        for feature in features:
            item = dataset_stats.get(feature)
            if not isinstance(item, dict):
                refuse(f"dataset stats are missing {feature}")
            for statistic in ("count", "min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
                tensor_name = f"{feature}.{statistic}"
                if tensor_name not in available:
                    refuse(f"Stage1 processor state is missing {tensor_name}")
                actual = state.get_tensor(tensor_name)
                expected = np.asarray(item.get(statistic), dtype=actual.dtype)
                if actual.shape != expected.shape or not np.array_equal(actual, expected):
                    refuse(f"Stage1 processor {tensor_name} differs from current dataset stats")
            mask_name = f"{feature}.mask"
            if mask_name not in available or not np.array_equal(
                state.get_tensor(mask_name), expected_masks[feature]
            ):
                refuse(f"Stage1 processor {mask_name} does not preserve the raw gripper")


validate_processor_stats(
    "policy_preprocessor.json",
    "molmoact2_masked_normalizer",
    ("observation.state", "action", "observation.ee_pose"),
)
validate_processor_stats(
    "policy_postprocessor.json",
    "molmoact2_masked_unnormalizer",
    ("action",),
)

print(f"[droid-stage2] verified Stage1 source: {checkpoint}")
print(f"[droid-stage2] verified Stage1 processor stats lineage: {stats_path}")
PY
else
  echo "[droid-stage2] DRY_RUN: Stage1 checkpoint validation disabled"
fi

STAGE2_ARGS=(
  --policy.enable_goal_pose=true
  --policy.goal_pose_feature_key=observation.ee_pose
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

echo "[droid-stage2] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[droid-stage2] POLICY_PATH=${POLICY_PATH}"
echo "[droid-stage2] semantic tokens=${NUM_SEMANTIC_VISUAL_TOKENS} pose=${NUM_SEMANTIC_VISUAL_POSE_TOKENS} groups=${SEMANTIC_VISUAL_NUM_LAYER_GROUPS}"

exec bash "${WS}/scripts/train_droid_molmoact2.sh" "${STAGE2_ARGS[@]}" "$@"

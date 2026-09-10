#!/usr/bin/env bash
# Fine-tune MolmoAct2 on the derived DROID goal-pose dataset.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_droid_molmoact2.sh --sample-manifest PATH [lerobot-train args...]

The manifest must be the one recorded in goal_pose_provenance.json.
SAMPLE_MANIFEST_PATH may be used instead of --sample-manifest.
EOF
}

PASSTHROUGH_ARGS=()
while (($#)); do
  case "$1" in
    --sample-manifest)
      if (($# < 2)); then
        echo "--sample-manifest requires a path" >&2
        exit 2
      fi
      SAMPLE_MANIFEST_PATH="$2"
      shift 2
      ;;
    --sample-manifest=*)
      SAMPLE_MANIFEST_PATH="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/droid_1.0.1}"
DATASET_ROOT="${DATASET_ROOT:-/data0/JM/dataset/droid_1.0.1_goal_pose}"
SOURCE_DATASET_REVISION="${SOURCE_DATASET_REVISION:-0eabc778f959c54b8c5aa3626cc1128d2d2e54d4}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WS}/Checkpoint/MolmoAct2}"
CHECKPOINT_REVISION="${CHECKPOINT_REVISION:-e432d85f6e039edca44afb93c262f3084ab72a9c}"
PROVENANCE_PATH="${PROVENANCE_PATH:-${DATASET_ROOT}/goal_pose_provenance.json}"
SAMPLE_MANIFEST_PATH="${SAMPLE_MANIFEST_PATH:-}"
GOAL_POSE_FEATURE_KEY="${GOAL_POSE_FEATURE_KEY:-observation.ee_pose}"
VALIDATE_DATASET="${VALIDATE_DATASET:-true}"

if [[ "${DATASET_REPO_ID}" != "lerobot/droid_1.0.1" ]]; then
  echo "DATASET_REPO_ID must be lerobot/droid_1.0.1, got: ${DATASET_REPO_ID}" >&2
  exit 2
fi
if [[ "${GOAL_POSE_FEATURE_KEY}" != "observation.ee_pose" ]]; then
  echo "GOAL_POSE_FEATURE_KEY must be observation.ee_pose, got: ${GOAL_POSE_FEATURE_KEY}" >&2
  exit 2
fi
if [[ -z "${SAMPLE_MANIFEST_PATH}" ]]; then
  echo "Missing sample manifest. Pass --sample-manifest PATH." >&2
  exit 2
fi
if [[ "${VALIDATE_DATASET}" != "true" && "${DRY_RUN:-false}" != "true" ]]; then
  echo "VALIDATE_DATASET=false is allowed only with DRY_RUN=true." >&2
  exit 2
fi

if [[ "${VALIDATE_DATASET}" == "true" ]]; then
  "${WS}/lerobot/.venv/bin/python" - \
    "${DATASET_ROOT}" \
    "${DATASET_REPO_ID}" \
    "${SOURCE_DATASET_REVISION}" \
    "${PROVENANCE_PATH}" \
    "${SAMPLE_MANIFEST_PATH}" \
    "${GOAL_POSE_FEATURE_KEY}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
repo_id, revision = sys.argv[2], sys.argv[3]
provenance_path = Path(sys.argv[4]).expanduser().resolve()
manifest_path = Path(sys.argv[5]).expanduser().resolve()
goal_key = sys.argv[6]
info_path = root / "meta" / "info.json"
stats_path = root / "meta" / "stats.json"


def refuse(message: str) -> None:
    raise SystemExit(f"REFUSING: {message}")


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        refuse(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        refuse(f"invalid {label} {path}: {exc}")
    if not isinstance(value, dict):
        refuse(f"{label} must be a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(value: object, dim: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dim:
        refuse(f"{label} must have shape [{dim}]")
    if any(isinstance(item, (list, dict, bool)) or not isinstance(item, (int, float)) for item in value):
        refuse(f"{label} must contain {dim} numeric scalars")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        refuse(f"{label} contains a non-finite value")
    return result


provenance = load_json(provenance_path, "provenance")
config = provenance.get("config")
artifacts = provenance.get("artifact_sha256")
if not isinstance(config, dict) or not isinstance(artifacts, dict):
    refuse("provenance must contain config and artifact_sha256 objects")
required_provenance = {
    "format_version": (provenance.get("format_version"), 1),
    "recipe_version": (provenance.get("recipe_version"), 3),
    "source_read_only": (provenance.get("source_read_only"), True),
    "config.revision": (config.get("revision"), revision),
    "config.horizon": (config.get("horizon"), 15),
    "anchor_definition.goal_offset": (
        provenance.get("anchor_definition", {}).get("goal_offset"),
        15,
    ),
    "ee_pose_definition.dtype": (
        provenance.get("ee_pose_definition", {}).get("dtype"),
        "float32",
    ),
    "ee_pose_definition.wrist_site_not_tcp": (
        provenance.get("ee_pose_definition", {}).get("wrist_site_not_tcp"),
        True,
    ),
}
for label, (actual, expected) in required_provenance.items():
    if actual != expected:
        refuse(f"provenance {label} must be {expected!r}, got {actual!r}")
if Path(str(config.get("source_root", ""))).name != "droid_1.0.1":
    refuse(f"provenance source_root is not the {repo_id} dataset")
if Path(str(config.get("output_root", ""))).expanduser().resolve() != root:
    refuse("provenance output_root does not match DATASET_ROOT")
success = load_json(root / "_SUCCESS.json", "completion marker")
if success.get("status") != "complete":
    refuse("completion marker status must be 'complete'")
if success.get("config_fingerprint") != provenance.get("config_fingerprint"):
    refuse("completion marker config fingerprint does not match provenance")
if success.get("provenance_sha256") != sha256(provenance_path):
    refuse("completion marker provenance SHA-256 does not match")

if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
    refuse(f"sample manifest is missing or empty: {manifest_path}")
try:
    manifest_relative = manifest_path.relative_to(root).as_posix()
except ValueError:
    refuse(f"sample manifest must be inside DATASET_ROOT: {manifest_path}")
if manifest_relative != "valid_anchor_indices.parquet":
    refuse(
        "sample manifest must be DATASET_ROOT/valid_anchor_indices.parquet, "
        f"got {manifest_relative}"
    )
manifest_hash = sha256(manifest_path)
if artifacts.get(manifest_relative) != manifest_hash:
    refuse(
        "sample manifest SHA-256 mismatch: "
        f"expected {artifacts.get(manifest_relative)!r}, got {manifest_hash}"
    )

info = load_json(info_path, "dataset info")
features = info.get("features")
if not isinstance(features, dict):
    refuse(f"{info_path} has no features object")
expected_shapes = {
    "observation.state": [8],
    "action": [8],
    goal_key: [7],
    "observation.images.exterior_1_left": [180, 320, 3],
    "observation.images.exterior_2_left": [180, 320, 3],
    "observation.images.wrist_left": [180, 320, 3],
}
for key, shape in expected_shapes.items():
    feature = features.get(key)
    if not isinstance(feature, dict) or feature.get("shape") != shape:
        refuse(f"{info_path} feature {key!r} must have shape {shape}")
for key in ("observation.state", "action", goal_key):
    if features[key].get("dtype") != "float32":
        refuse(f"{info_path} feature {key!r} must use float32")
for key in expected_shapes:
    if key.startswith("observation.images.") and features[key].get("dtype") != "video":
        refuse(f"{info_path} feature {key!r} must use video dtype")
if info.get("fps") != 15:
    refuse(f"{info_path} fps must be 15, got {info.get('fps')!r}")
goal_info = info.get("goal_pose_prior")
if not isinstance(goal_info, dict):
    refuse(f"{info_path} has no goal_pose_prior object")
if goal_info.get("horizon") != 15 or goal_info.get("target_feature") != goal_key:
    refuse(f"{info_path} goal_pose_prior contract is invalid")
if goal_info.get("anchor_manifest") != manifest_relative:
    refuse(f"{info_path} anchor_manifest does not match the selected manifest")

stats = load_json(stats_path, "dataset stats")
stats_hash = sha256(stats_path)
if artifacts.get("meta/stats.json") != stats_hash:
    refuse(
        "stats SHA-256 mismatch: "
        f"expected {artifacts.get('meta/stats.json')!r}, got {stats_hash}"
    )
for key, dim in (("observation.state", 8), ("action", 8), (goal_key, 7)):
    item = stats.get(key)
    if not isinstance(item, dict):
        refuse(f"{stats_path} is missing {key!r}")
    values = {
        name: vector(item.get(name), dim, f"{key}.{name}")
        for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    for index in range(dim):
        ordered = [
            values[name][index]
            for name in ("min", "q01", "q10", "q50", "q90", "q99", "max")
        ]
        if ordered != sorted(ordered):
            refuse(f"{key} statistics are not monotonic at dimension {index}")
        if values["std"][index] <= 0:
            refuse(f"{key}.std must be positive at dimension {index}")

print(f"[droid] verified provenance: {provenance_path}")
print(f"[droid] verified manifest sha256={manifest_hash}")
print(f"[droid] verified stats sha256={stats_hash} and shapes state=8/action=8/goal=7")
PY
else
  echo "[droid] DRY_RUN: dataset file validation disabled"
fi

# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"

POLICY_PATH="${POLICY_PATH:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
IMAGE_TRANSFORMS_ENABLE="${IMAGE_TRANSFORMS_ENABLE:-true}"

JOB_NAME="${JOB_NAME:-molmoact2-droid}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/${JOB_NAME}/${RUN_ID}}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SEED="${SEED:-1000}"

ACTION_MODE="${ACTION_MODE:-continuous}"
TRAIN_ACTION_EXPERT_ONLY="${TRAIN_ACTION_EXPERT_ONLY:-false}"
DISABLE_VISUAL_INPUT="${DISABLE_VISUAL_INPUT:-false}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-5}"
OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-5}"
OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-5e-5}"
OPTIMIZER_BETAS="${OPTIMIZER_BETAS:-[0.9,0.95]}"
OPTIMIZER_EPS="${OPTIMIZER_EPS:-1e-6}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-0.0}"
OPTIMIZER_GRAD_CLIP_NORM="${OPTIMIZER_GRAD_CLIP_NORM:-1.0}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-500}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-6}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-molmoact2-droid}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.console.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

if [[ -z "${RESUME_CHECKPOINT}" && "${RESUME_MODE}" == "auto" ]]; then
  candidate="${OUTPUT_DIR}/checkpoints/last"
  if [[ -f "${candidate}/pretrained_model/train_config.json" ]]; then
    RESUME_CHECKPOINT="${candidate}"
  fi
fi

RESUME_CONFIG_PATH=""
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ -f "${RESUME_CHECKPOINT}" ]]; then
    RESUME_CONFIG_PATH="${RESUME_CHECKPOINT}"
  elif [[ -f "${RESUME_CHECKPOINT}/train_config.json" ]]; then
    RESUME_CONFIG_PATH="${RESUME_CHECKPOINT}/train_config.json"
  elif [[ -f "${RESUME_CHECKPOINT}/pretrained_model/train_config.json" ]]; then
    RESUME_CONFIG_PATH="${RESUME_CHECKPOINT}/pretrained_model/train_config.json"
  else
    echo "Resume checkpoint has no train_config.json: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  RESUME_CHECKPOINT="$(cd "$(dirname "${RESUME_CONFIG_PATH}")/.." && pwd)"
  if [[ ! -d "${RESUME_CHECKPOINT}/training_state" ]]; then
    echo "Resume checkpoint has no training_state: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
fi

echo "[droid] GPUs=${CUDA_VISIBLE_DEVICES} (n=${NUM_PROCESSES})"
echo "[droid] dataset=${DATASET_REPO_ID} root=${DATASET_ROOT}"
echo "[droid] manifest=${SAMPLE_MANIFEST_PATH}"
echo "[droid] policy_path=${POLICY_PATH:-<fresh>} checkpoint=${CHECKPOINT_PATH}@${CHECKPOINT_REVISION}"
echo "[droid] output_dir=${OUTPUT_DIR}"
echo "[droid] batch_size/gpu=${BATCH_SIZE} steps=${STEPS} seed=${SEED}"
echo "[droid] action_mode=${ACTION_MODE} action_expert_only=${TRAIN_ACTION_EXPERT_ONLY} visual_disabled=${DISABLE_VISUAL_INPUT}"
echo "[droid] AdamW betas=${OPTIMIZER_BETAS} eps=${OPTIMIZER_EPS} weight_decay=${OPTIMIZER_WEIGHT_DECAY} grad_clip=${OPTIMIZER_GRAD_CLIP_NORM}"
echo "[droid] resume=${RESUME_CONFIG_PATH:-false}"

cd "${WS}/lerobot"
CMD=(
  accelerate launch
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision=bf16
  -m lerobot.scripts.lerobot_train
)

if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  CMD+=(
    --config_path="${RESUME_CONFIG_PATH}"
    --resume=true
    --output_dir="${OUTPUT_DIR}"
    --steps="${STEPS}"
    --log_freq="${LOG_FREQ}"
    --save_freq="${SAVE_FREQ}"
    --eval_freq=-1
  )
else
  CMD+=(
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.sample_indices_path="${SAMPLE_MANIFEST_PATH}"
    --dataset.video_backend="${VIDEO_BACKEND}"
    --dataset.image_transforms.enable="${IMAGE_TRANSFORMS_ENABLE}"
    --policy.device=cuda
    --policy.action_mode="${ACTION_MODE}"
    --policy.chunk_size=15
    --policy.n_action_steps=15
    --policy.setup_type="single franka robotic arm in droid"
    --policy.control_mode="absolute joint pose"
    --policy.image_keys='["observation.images.exterior_1_left","observation.images.exterior_2_left","observation.images.wrist_left"]'
    --policy.model_dtype=bfloat16
    --policy.num_flow_timesteps=8
    --policy.gradient_checkpointing=true
    --policy.freeze_embedding=true
    --policy.train_action_expert_only="${TRAIN_ACTION_EXPERT_ONLY}"
    --policy.disable_visual_input="${DISABLE_VISUAL_INPUT}"
    --policy.normalize_gripper=false
    --policy.enable_knowledge_insulation=false
    --policy.optimizer_lr="${OPTIMIZER_LR}"
    --policy.optimizer_vit_lr="${OPTIMIZER_VIT_LR}"
    --policy.optimizer_connector_lr="${OPTIMIZER_CONNECTOR_LR}"
    --policy.optimizer_action_expert_lr="${OPTIMIZER_ACTION_EXPERT_LR}"
    --policy.optimizer_betas="${OPTIMIZER_BETAS}"
    --policy.optimizer_eps="${OPTIMIZER_EPS}"
    --policy.optimizer_weight_decay="${OPTIMIZER_WEIGHT_DECAY}"
    --policy.optimizer_grad_clip_norm="${OPTIMIZER_GRAD_CLIP_NORM}"
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
    --policy.scheduler_vlm_warmup_steps="${SCHEDULER_VLM_WARMUP_STEPS}"
    --policy.scheduler_vit_warmup_steps="${SCHEDULER_VIT_WARMUP_STEPS}"
    --policy.scheduler_connector_warmup_steps="${SCHEDULER_CONNECTOR_WARMUP_STEPS}"
    --policy.scheduler_action_expert_warmup_steps="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS}"
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
    --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
    --policy.push_to_hub=false
    --job_name="${JOB_NAME}"
    --output_dir="${OUTPUT_DIR}"
    --steps="${STEPS}"
    --seed="${SEED}"
    --resume=false
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --log_freq="${LOG_FREQ}"
    --eval_freq=-1
    --save_checkpoint="${SAVE_CHECKPOINT}"
    --save_freq="${SAVE_FREQ}"
  )

  if [[ -n "${POLICY_PATH}" ]]; then
    CMD+=(--policy.path="${POLICY_PATH}")
  else
    CMD+=(
      --policy.type=molmoact2
      --policy.checkpoint_path="${CHECKPOINT_PATH}"
      --policy.checkpoint_revision="${CHECKPOINT_REVISION}"
    )
  fi

  if [[ "${WANDB_ENABLE}" == "true" ]]; then
    CMD+=(--wandb.enable=true --wandb.project="${WANDB_PROJECT}")
    if [[ -n "${WANDB_ENTITY}" ]]; then
      CMD+=(--wandb.entity="${WANDB_ENTITY}")
    fi
  else
    CMD+=(--wandb.enable=false)
  fi
fi

CMD+=("${PASSTHROUGH_ARGS[@]}")
printf '[droid] running:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

set -o pipefail
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

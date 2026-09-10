#!/usr/bin/env bash
# Fine-tune MolmoAct2 on local LIBERO LeRobot dataset (GPUs 0-6, per-GPU bs=32).
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
NUM_PROCESSES="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"

DATASET_ROOT="${DATASET_ROOT:-}"
if [[ -z "${DATASET_ROOT}" || ! -d "${DATASET_ROOT}" ]]; then
  echo "Set DATASET_ROOT to the LIBERO dataset in LeRobot format." >&2
  exit 2
fi
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero_lerobot_format}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WS}/Checkpoint/MolmoAct2}"
POLICY_PATH="${POLICY_PATH:-}"
# torchvision VideoReader is missing in this env; torchcodec works.
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
IMAGE_TRANSFORMS_ENABLE="${IMAGE_TRANSFORMS_ENABLE:-true}"

JOB_NAME="${JOB_NAME:-molmoact2-libero-fft}"
# Fresh run dir by default; LeRobot refuses to overwrite an existing output_dir.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/${JOB_NAME}/${RUN_ID}}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SEED="${SEED:-1000}"

ACTION_MODE="${ACTION_MODE:-continuous}"
TRAIN_ACTION_EXPERT_ONLY="${TRAIN_ACTION_EXPERT_ONLY:-false}"
DISABLE_VISUAL_INPUT="${DISABLE_VISUAL_INPUT:-false}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-5e-6}"
OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-5e-6}"
OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-5e-5}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-${SCHEDULER_WARMUP_STEPS}}"
SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-500}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-6}"

# wandb off by default; set WANDB_ENABLE=true (+ entity/project) to enable
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-molmoact2-libero}"

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

echo "[train] GPUs=${CUDA_VISIBLE_DEVICES} (n=${NUM_PROCESSES})"
echo "[train] dataset_root=${DATASET_ROOT}"
echo "[train] policy_path=${POLICY_PATH:-<fresh>} checkpoint=${CHECKPOINT_PATH}"
echo "[train] output_dir=${OUTPUT_DIR}"
echo "[train] video_backend=${VIDEO_BACKEND}"
echo "[train] batch_size/gpu=${BATCH_SIZE} steps=${STEPS} seed=${SEED} log_freq=${LOG_FREQ}"
echo "[train] action_mode=${ACTION_MODE} action_expert_only=${TRAIN_ACTION_EXPERT_ONLY} visual_disabled=${DISABLE_VISUAL_INPUT}"
echo "[train] log_file=${LOG_FILE}"
echo "[train] resume=${RESUME_CONFIG_PATH:-false}"

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
    --dataset.video_backend="${VIDEO_BACKEND}"
    --dataset.image_transforms.enable="${IMAGE_TRANSFORMS_ENABLE}"
    --policy.device=cuda
    --policy.action_mode="${ACTION_MODE}"
    --policy.chunk_size=10
    --policy.n_action_steps=10
    --policy.setup_type="single franka robotic arm in libero"
    --policy.control_mode="delta end-effector pose"
    --policy.image_keys='["observation.images.image","observation.images.image2"]'
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
    CMD+=(--policy.type=molmoact2 --policy.checkpoint_path="${CHECKPOINT_PATH}")
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

# Extra CLI args pass through, e.g. --steps=5000
CMD+=("$@")

printf '[train] running:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

set -o pipefail
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

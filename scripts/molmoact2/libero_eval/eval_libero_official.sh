#!/usr/bin/env bash
# Evaluate the official HuggingFace MolmoAct2-LIBERO checkpoint on in-dist LIBERO.
#
# This is intentionally separate from eval_libero_checkpoint.sh (goal-pose Stage2).
# Use it to sanity-check the simulation stack before trusting our fine-tuned models.
#
# Loading path for the official HF snapshot:
#   --policy.type=molmoact2
#   --policy.checkpoint_path=Checkpoint/MolmoAct2-LIBERO
#   --policy.norm_tag=libero
# Camera names are remapped so the env emits observation.images.wrist_image
# (matching the official norm_stats camera_keys), not image2.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

DEFAULT_CHECKPOINT_PATH="${WS}/Checkpoint/MolmoAct2-LIBERO"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}}"
if [[ ! -f "${CHECKPOINT_PATH}/config.json" || ! -f "${CHECKPOINT_PATH}/norm_stats.json" ]]; then
  echo "Official MolmoAct2-LIBERO checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Expected a HuggingFace snapshot with config.json + norm_stats.json." >&2
  echo "Override with: bash $0 /path/to/MolmoAct2-LIBERO" >&2
  exit 2
fi
CHECKPOINT_PATH="$(cd "${CHECKPOINT_PATH}" && pwd)"

# Refuse to silently evaluate a LeRobot fine-tune / goal-pose checkpoint here.
python - "${CHECKPOINT_PATH}/config.json" "${CHECKPOINT_PATH}/norm_stats.json" <<'PY'
import json
import sys

cfg_path, stats_path = sys.argv[1], sys.argv[2]
with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)
with open(stats_path, encoding="utf-8") as f:
    stats = json.load(f)

if cfg.get("type") == "molmoact2" or cfg.get("enable_goal_pose") is not None:
    raise SystemExit(
        "Refusing to evaluate: this looks like a LeRobot / goal-pose checkpoint.\n"
        "Use scripts/libero_eval/eval_libero_checkpoint.sh for those.\n"
        f"  type={cfg.get('type')!r} enable_goal_pose={cfg.get('enable_goal_pose')!r}"
    )
if "libero" not in (stats.get("metadata_by_tag") or {}):
    raise SystemExit(
        "Refusing to evaluate: norm_stats.json has no 'libero' tag "
        f"(available: {sorted((stats.get('metadata_by_tag') or {}).keys())})."
    )
cams = (stats["metadata_by_tag"]["libero"] or {}).get("camera_keys")
print(
    "[eval] verified official HF MolmoAct2-LIBERO:"
    f" action_mode={cfg.get('action_mode')!r}"
    f" architectures={cfg.get('architectures')!r}"
    f" camera_keys={cams!r}"
)
PY

EVAL_SEED="${EVAL_SEED:-1000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-20}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-20}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-${EPISODES_PER_TASK}}"
GPU_IDS=(${EVAL_GPU_IDS:-7})
SUITES=(libero_object libero_10 libero_goal libero_spatial)
NORM_TAG="${NORM_TAG:-libero}"
RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-/data2/JM/Code/molmo_serious/molmoact2-main}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${RESOURCE_ROOT}/.cache/libero_config}"
export HF_HOME="${HF_HOME:-${RESOURCE_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RESOURCE_ROOT}/.cache/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${RESOURCE_ROOT}/.cache/torch}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${WS}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-MolmoAct2-LIBERO_official}"
EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval/${CHECKPOINT_LABEL}/libero_seed_${EVAL_SEED}}"
mkdir -p "${EVAL_ROOT}"

# Official norm_stats expects wrist_image (not the env default image2).
CAMERA_NAME_MAPPING='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'

cat >"${EVAL_ROOT}/run_manifest.json" <<EOF
{
  "benchmark": "libero",
  "policy_kind": "official_hf",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "checkpoint_label": "${CHECKPOINT_LABEL}",
  "norm_tag": "${NORM_TAG}",
  "suites": ["libero_object", "libero_10", "libero_goal", "libero_spatial"],
  "episodes_per_task": ${EPISODES_PER_TASK},
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "eval_seed": ${EVAL_SEED},
  "gpu_ids": "${GPU_IDS[*]}",
  "camera_name_mapping": ${CAMERA_NAME_MAPPING}
}
EOF

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] checkpoint=${CHECKPOINT_PATH}"
  echo "[dry-run] output=${EVAL_ROOT}"
  echo "[dry-run] norm_tag=${NORM_TAG} camera_map=${CAMERA_NAME_MAPPING}"
  echo "[dry-run] suites=${SUITES[*]} episodes/task=${EPISODES_PER_TASK} batch=${EVAL_BATCH_SIZE} gpus=${GPU_IDS[*]}"
  exit 0
fi

pids=()
run_sequential=false
if [[ "${#GPU_IDS[@]}" -eq 1 ]]; then
  run_sequential=true
  echo "[eval] single GPU ${GPU_IDS[0]}: suites will run sequentially"
fi

for index in "${!SUITES[@]}"; do
  suite="${SUITES[$index]}"
  gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  output_dir="${EVAL_ROOT}/${suite}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/eval_info.json" ]]; then
    echo "Refusing to overwrite completed evaluation: ${output_dir}" >&2
    exit 2
  fi
  (
    printf '{"status":"running","gpu":"%s"}\n' "${gpu}" >"${output_dir}/process_status.json"
    set +e
    # Fresh HF load (no --policy.path): weights come from checkpoint_path + norm_tag.
    CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
      python -m lerobot.scripts.lerobot_eval \
        --policy.type=molmoact2 \
        --policy.checkpoint_path="${CHECKPOINT_PATH}" \
        --policy.norm_tag="${NORM_TAG}" \
        --policy.device=cuda \
        --policy.action_mode=continuous \
        --policy.inference_action_mode=continuous \
        --policy.disable_visual_input=false \
        --policy.per_episode_seed=true \
        --policy.eval_seed="${EVAL_SEED}" \
        --policy.enable_inference_cuda_graph=false \
        --env.type=libero \
        --env.task="${suite}" \
        --env.control_mode=relative \
        --env.max_parallel_tasks=1 \
        --env.camera_name_mapping="${CAMERA_NAME_MAPPING}" \
        --eval.n_episodes="${EPISODES_PER_TASK}" \
        --eval.batch_size="${EVAL_BATCH_SIZE}" \
        --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
        --eval.use_async_envs=true \
        --seed="${EVAL_SEED}" \
        --output_dir="${output_dir}" \
        >"${output_dir}/eval.log" 2>&1
    rc=$?
    if [[ "${rc}" -eq 0 ]]; then status=final; else status=failed; fi
    printf '{"status":"%s","exit_code":%d,"gpu":"%s"}\n' \
      "${status}" "${rc}" "${gpu}" >"${output_dir}/process_status.json"
    exit "${rc}"
  ) &
  pids+=("$!")
  if [[ "${run_sequential}" == "true" ]]; then
    if ! wait "${pids[-1]}"; then
      echo "[eval] suite ${suite} failed" >&2
    fi
  fi
done

if [[ "${run_sequential}" != "true" ]]; then
  python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}"
fi

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then rc=1; fi
done
python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}" --once
echo "[eval] results: ${EVAL_ROOT}"
exit "${rc}"

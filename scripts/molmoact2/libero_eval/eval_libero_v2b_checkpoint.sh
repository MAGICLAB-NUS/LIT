#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior_v2b/seed_1000/stage2/checkpoints/025000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"
if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Goal-pose v2b checkpoint not found: ${POLICY_PATH}" >&2
  echo "Override with: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"

EVAL_VARIANT="${EVAL_VARIANT:-v2b}"
python - "${POLICY_PATH}/config.json" "${EVAL_VARIANT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
variant = sys.argv[2]
# v4 hard bottleneck: all latents are pose-supervised (tokens == pose_tokens == 8).
# v2b/v3 soft bottleneck: 100 latents with 8 pose + 92 context.
num_tokens = 8 if variant == "v4" else 100
expected = {
    "type": "molmoact2",
    "enable_goal_pose": True,
    "goal_conditioning_mode": "semantic_visual_recurrent",
    "goal_token_source": "learnable_queries",
    "num_semantic_visual_tokens": num_tokens,
    "num_semantic_visual_pose_tokens": 8,
    "semantic_visual_hidden_dim": 768,
    "semantic_visual_enable_self_attention": True,
    "semantic_visual_num_layer_groups": 6,
    "mask_image_from_action_expert": True,
    "enable_pose_reconstruction": True,
    "pose_recon_loss_weight": 0.3,
    "chunk_size": 10,
    "n_action_steps": 10,
}
mismatches = [
    f"{key}: expected {want!r}, got {cfg.get(key)!r}"
    for key, want in expected.items()
    if cfg.get(key) != want
]
pose_tokens = cfg.get("num_semantic_visual_pose_tokens")
total_tokens = cfg.get("num_semantic_visual_tokens", 0)
if variant == "v4":
    ok_pose = isinstance(pose_tokens, int) and 1 <= pose_tokens <= total_tokens
    pose_rule = "1 <= P <= num_semantic_visual_tokens"
else:
    ok_pose = isinstance(pose_tokens, int) and 1 <= pose_tokens < total_tokens
    pose_rule = "1 <= P < num_semantic_visual_tokens"
if not ok_pose:
    mismatches.append(
        f"num_semantic_visual_pose_tokens: expected {pose_rule}, "
        f"got {pose_tokens!r} (total={total_tokens!r})"
    )
if mismatches and __import__("os").environ.get("EVAL_SKIP_CONFIG_GUARD") != "1":
    raise SystemExit(
        "Refusing to evaluate: checkpoint is not a v2b-compatible semantic-visual Goal-Pose Prior:\n  "
        + "\n  ".join(mismatches)
    )
elif mismatches:
    # ablation arms (stagewise / open-visual-path / no-pose-loss) legitimately differ; opt-in skip
    print("[eval] WARNING config guard skipped (EVAL_SKIP_CONFIG_GUARD=1):\n  " + "\n  ".join(mismatches))
print(
    f"[eval] verified {variant}:"
    f" mode={cfg.get('goal_conditioning_mode')}"
    f" tokens={cfg.get('num_semantic_visual_tokens')}"
    f" pose_tokens={pose_tokens}"
    f" groups={cfg['semantic_visual_num_layer_groups']}"
    f" self_attn={cfg['semantic_visual_enable_self_attention']}"
    f" hidden={cfg['semantic_visual_hidden_dim']}"
)
PY

EVAL_SEED="${EVAL_SEED:-1000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-${EPISODES_PER_TASK}}"
GPU_IDS=(${EVAL_GPU_IDS:-7})
SUITES=(${EVAL_SUITES:-libero_spatial libero_object libero_10 libero_goal})
TASK_IDS="${EVAL_TASK_IDS:-}"
# OpenVLA public-leaderboard horizons (set OFFICIAL_HORIZONS=true for 50-ep board).
OFFICIAL_HORIZONS="${OFFICIAL_HORIZONS:-false}"
episode_length_for_suite() {
  case "$1" in
    libero_spatial) echo 220 ;;
    libero_object) echo 280 ;;
    libero_goal) echo 300 ;;
    libero_10) echo 520 ;;
    *) echo 300 ;;
  esac
}
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "EVAL_GPU_IDS must contain at least one GPU ID" >&2
  exit 2
fi
if [[ "${#SUITES[@]}" -lt 1 ]]; then
  echo "EVAL_SUITES must contain at least one suite" >&2
  exit 2
fi
for suite in "${SUITES[@]}"; do
  case "${suite}" in
    libero_spatial | libero_goal | libero_object | libero_10) ;;
    *)
      echo "Unsupported LIBERO suite in EVAL_SUITES: ${suite}" >&2
      exit 2
      ;;
  esac
done
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

python - <<'PY'
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

if not hasattr(MolmoAct2Policy, "_uses_policy_continuous_generation"):
    raise SystemExit(
        "Refusing to evaluate: local MolmoAct2 code lacks the non-RTC learned-context "
        "inference fix."
    )
print("[eval] verified non-RTC learned-context inference fix")
PY

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-goal_prior_v2b_${checkpoint_step}}"
MODEL_LABEL="${MODEL_LABEL:-goal_pose_prior_v2b}"
EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval/${CHECKPOINT_LABEL}/libero_seed_${EVAL_SEED}}"
mkdir -p "${EVAL_ROOT}"

python - \
  "${EVAL_ROOT}/run_manifest.json" \
  "${POLICY_PATH}" \
  "${CHECKPOINT_LABEL}" \
  "${MODEL_LABEL}" \
  "${EPISODES_PER_TASK}" \
  "${MAX_EPISODES_RENDERED}" \
  "${EVAL_SEED}" \
  "${SUITES[*]}" \
  "${GPU_IDS[*]}" \
  "${TASK_IDS}" <<'PY'
import json
import sys
from pathlib import Path

(
    output_path,
    policy_path,
    checkpoint_label,
    model_label,
    episodes_per_task,
    max_episodes_rendered,
    eval_seed,
    suites_text,
    gpu_ids_text,
    task_ids_text,
) = sys.argv[1:]
suites = suites_text.split()
gpu_ids = gpu_ids_text.split()
payload = {
    "benchmark": "libero",
    "policy_path": policy_path,
    "checkpoint_label": checkpoint_label,
    "model": model_label,
    "protocol": "openvla_public_50ep" if int(episodes_per_task) == 50 else "custom",
    "suites": suites,
    "episodes_per_task": int(episodes_per_task),
    "max_episodes_rendered": int(max_episodes_rendered),
    "eval_seed": int(eval_seed),
    "gpu_ids": gpu_ids,
    "suite_gpu_map": {
        suite: gpu_ids[index % len(gpu_ids)] for index, suite in enumerate(suites)
    },
    "task_ids": json.loads(task_ids_text) if task_ids_text else None,
    "official_horizons": __import__("os").environ.get("OFFICIAL_HORIZONS", "false"),
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] policy=${POLICY_PATH}"
  echo "[dry-run] output=${EVAL_ROOT}"
  echo "[dry-run] suites=${SUITES[*]} task_ids=${TASK_IDS:-all} episodes/task=${EPISODES_PER_TASK} gpus=${GPU_IDS[*]}"
  for index in "${!SUITES[@]}"; do
    echo "[dry-run] ${SUITES[$index]} -> GPU ${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  done
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
    cmd=(
      python -m lerobot.scripts.lerobot_eval
      --policy.path="${POLICY_PATH}"
      --policy.device=cuda
      --policy.inference_action_mode=continuous
      --policy.disable_visual_input=false
      --policy.per_episode_seed=true
      --policy.eval_seed="${EVAL_SEED}"
      --policy.enable_inference_cuda_graph=false
      --env.type=libero
      --env.task="${suite}"
      --env.control_mode=relative
      --env.max_parallel_tasks=1
      --eval.n_episodes="${EPISODES_PER_TASK}"
      --eval.batch_size="${EVAL_BATCH_SIZE}"
      --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}"
      --eval.use_async_envs=true
      --seed="${EVAL_SEED}"
      --output_dir="${output_dir}"
    )
    if [[ "${OFFICIAL_HORIZONS}" == "true" ]]; then
      cmd+=(--env.episode_length="$(episode_length_for_suite "${suite}")")
    fi
    if [[ -n "${TASK_IDS}" ]]; then
      cmd+=(--env.task_ids="${TASK_IDS}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
      "${cmd[@]}" >"${output_dir}/eval.log" 2>&1
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

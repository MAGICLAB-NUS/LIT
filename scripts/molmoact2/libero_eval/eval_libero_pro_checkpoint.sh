#!/usr/bin/env bash
# Evaluate a checkpoint on LIBERO-PRO, aligned with the official paper/README protocol.
#
# Official public leaderboard protocol (arxiv:2510.03827 §5.1 + README leaderboard):
#   - 4 suites: libero_{goal,spatial,10,object}
#   - 5 single-dimension perturbations (one at a time):
#       Obj=object, Pos=swap, Sem=lan, Task=task, Env=env
#   - 10 tasks per suite × 50 episodes per task  (each task ships 50 init states)
#   - Cell = 500 eps; full board = 4×5×500 = 10,000 eps
#   - Report success in [0,1] per (suite × perturbation); Total = mean of 20 cells
#   - Horizon (OpenVLA PRO reference): spatial=220, object=280, goal=300, 10=520
#
# Smoke:
#   SMOKE=true → 1 suite × task_ids=[0] × 1 episode
#
# Full (default when SMOKE=false):
#   EPISODES_PER_TASK=50, all 5 perturbations. Env suites are skipped until
#   generated via PRO perturbation.py (HF packs cover object/swap/lan/task only).
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

POLICY_PATH="${1:-${POLICY_PATH:-}}"
if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Usage: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"

EVAL_SEED="${EVAL_SEED:-1000}"
SMOKE="${SMOKE:-false}"
# Official protocol uses 50 eps/task; smoke overrides below.
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
# Save a few videos per suite (= each base × perturbation type). First N episodes only.
# Full board has 16 suites → ~32 clips at default 2. Override with MAX_EPISODES_RENDERED=0 to disable.
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-2}"
GPU_IDS=(${EVAL_GPU_IDS:-0})
INCLUDE_ENV="${INCLUDE_ENV:-true}"
SKIP_MISSING_ENV="${SKIP_MISSING_ENV:-true}"
# Post-hoc video pass: 1 ep × all suites, force re-run even if eval_info exists.
VIDEO_SAMPLES_ONLY="${VIDEO_SAMPLES_ONLY:-false}"

BASE_SUITES=(libero_spatial libero_object libero_goal libero_10)
# README leaderboard columns: Obj Pos Sem Task Env
PERTURBATIONS=(object swap lan task)
if [[ "${INCLUDE_ENV}" == "true" ]]; then
  PERTURBATIONS+=(env)
fi

# Official OpenVLA PRO horizons (not LeRobot vanilla spatial=280).
episode_length_for_suite() {
  local suite="$1"
  case "${suite}" in
    libero_spatial_*) echo 220 ;;
    libero_object_*) echo 280 ;;
    libero_goal_*) echo 300 ;;
    libero_10_*) echo 520 ;;
    *) echo 300 ;;
  esac
}

LIBERO_PRO_ROOT="${LIBERO_PRO_ROOT:-/data0/JM/benchmarks/libero_pro/code/LIBERO-PRO}"
LIBERO_CONFIG_PATH_DEFAULT="/data0/JM/benchmarks/libero_pro/libero_config"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "EVAL_GPU_IDS must contain at least one GPU ID" >&2
  exit 2
fi

for required in \
  "${LIBERO_PRO_ROOT}/libero/__init__.py" \
  "${LIBERO_PRO_ROOT}/libero/libero/assets" \
  "${LIBERO_PRO_ROOT}/libero/libero/init_files" \
  "${LIBERO_PRO_ROOT}/libero/libero/bddl_files" \
  "${LIBERO_CONFIG_PATH_DEFAULT}/config.yaml"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing LIBERO-PRO resource: ${required}" >&2
    exit 2
  fi
done

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_CONFIG_PATH_DEFAULT}}"
export HF_HOME="${HF_HOME:-${WS}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WS}/.cache/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${WS}/.cache/torch}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# PRO fork must precede site-packages; LeRobot remains the policy/eval source.
export PYTHONPATH="${LIBERO_PRO_ROOT}:${WS}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
run_name="$(basename "$(dirname "$(dirname "$(dirname "${POLICY_PATH}")")")")"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-${run_name}_${checkpoint_step}}"

if [[ "${VIDEO_SAMPLES_ONLY}" == "true" ]]; then
  # Cheap showcase pass after a no-video full run: 1 ep/task0 per suite, keep videos.
  EPISODES_PER_TASK="${EPISODES_PER_TASK_VIDEO:-1}"
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED_VIDEO:-1}"
  TASK_IDS="${TASK_IDS:-[0]}"
  EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval_pro/${CHECKPOINT_LABEL}/video_samples_seed_${EVAL_SEED}}"
  if [[ -n "${EVAL_SUITES:-}" ]]; then
    # shellcheck disable=SC2206
    SUITES=(${EVAL_SUITES})
  else
    SUITES=()
    for base in "${BASE_SUITES[@]}"; do
      for pert in "${PERTURBATIONS[@]}"; do
        SUITES+=("${base}_${pert}")
      done
    done
  fi
elif [[ "${SMOKE}" == "true" ]]; then
  EPISODES_PER_TASK="${EPISODES_PER_TASK_SMOKE:-1}"
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED_SMOKE:-1}"
  SUITES=(${EVAL_SUITES:-libero_spatial_object})
  TASK_IDS="${TASK_IDS:-[0]}"
  EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval_pro/${CHECKPOINT_LABEL}/smoke_seed_${EVAL_SEED}}"
else
  if [[ -n "${EVAL_SUITES:-}" ]]; then
    # shellcheck disable=SC2206
    SUITES=(${EVAL_SUITES})
  else
    SUITES=()
    for base in "${BASE_SUITES[@]}"; do
      for pert in "${PERTURBATIONS[@]}"; do
        SUITES+=("${base}_${pert}")
      done
    done
  fi
  TASK_IDS="${TASK_IDS:-}"
  EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval_pro/${CHECKPOINT_LABEL}/full50_seed_${EVAL_SEED}}"
fi

mkdir -p "${EVAL_ROOT}"

# Drop env suites whose init/bddl packs are not on disk yet.
FILTERED_SUITES=()
for suite in "${SUITES[@]}"; do
  init_dir="${LIBERO_PRO_ROOT}/libero/libero/init_files/${suite}"
  if [[ "${suite}" == *_env ]]; then
    if [[ ! -d "${init_dir}" ]] || [[ -z "$(ls -A "${init_dir}" 2>/dev/null || true)" ]]; then
      if [[ "${SKIP_MISSING_ENV}" == "true" ]]; then
        echo "[pro] skip missing Env suite (generate later): ${suite}" >&2
        continue
      fi
      echo "Missing Env assets for ${suite}; set SKIP_MISSING_ENV=true or generate via perturbation.py" >&2
      exit 2
    fi
  fi
  FILTERED_SUITES+=("${suite}")
done
SUITES=("${FILTERED_SUITES[@]}")
if [[ "${#SUITES[@]}" -lt 1 ]]; then
  echo "No suites left to evaluate" >&2
  exit 2
fi

python - "${SUITES[@]}" <<'PY'
import sys
from pathlib import Path

import torch
from libero.libero import benchmark, get_libero_path

suites = sys.argv[1:]
print("[pro] protocol=official-paper (50 eps/task when full)")
print("[pro] libero package OK")
print(f"[pro] benchmark_root={get_libero_path('benchmark_root')}")
print(f"[pro] bddl_files={get_libero_path('bddl_files')}")
available = benchmark.get_benchmark_dict()
missing = [name for name in suites if name not in available]
if missing:
    raise SystemExit(f"Missing PRO suites in benchmark registry: {missing}")
for name in suites:
    suite = available[name]()
    n_tasks = suite.get_num_tasks()
    init = torch.load(
        Path(get_libero_path("init_states"))
        / suite.get_task(0).problem_folder
        / Path(suite.get_task(0).init_states_file).name,
        weights_only=False,
    )
    print(f"[pro] suite {name}: n_tasks={n_tasks} init_states/task0={len(init)}")
    if len(init) < 50:
        print(
            f"[pro][warn] {name} has only {len(init)} init states; "
            "official protocol expects 50 eps/task",
            file=sys.stderr,
        )
PY

cat >"${EVAL_ROOT}/run_manifest.json" <<EOF
{
  "benchmark": "libero_pro",
  "protocol": "official_paper_50ep",
  "smoke": $( [[ "${SMOKE}" == "true" ]] && echo true || echo false ),
  "policy_path": "${POLICY_PATH}",
  "checkpoint_label": "${CHECKPOINT_LABEL}",
  "suites": $(python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${SUITES[@]}"),
  "perturbation_map": {
    "object": "Obj",
    "swap": "Pos",
    "lan": "Sem",
    "task": "Task",
    "env": "Env"
  },
  "task_ids": ${TASK_IDS:-null},
  "episodes_per_task": ${EPISODES_PER_TASK},
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "max_episodes_rendered": ${MAX_EPISODES_RENDERED},
  "video_samples_only": $( [[ "${VIDEO_SAMPLES_ONLY}" == "true" ]] && echo true || echo false ),
  "eval_seed": ${EVAL_SEED},
  "gpu_ids": "${GPU_IDS[*]}",
  "horizons": {"spatial": 220, "object": 280, "goal": 300, "10": 520},
  "libero_pro_root": "${LIBERO_PRO_ROOT}",
  "libero_config_path": "${LIBERO_CONFIG_PATH}",
  "notes": "Aligned with arxiv:2510.03827 §5.1 and README leaderboard (Obj/Pos/Sem/Task/Env)."
}
EOF

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] policy=${POLICY_PATH}"
  echo "[dry-run] output=${EVAL_ROOT}"
  echo "[dry-run] suites=${SUITES[*]} task_ids=${TASK_IDS:-all} episodes=${EPISODES_PER_TASK}"
  echo "[dry-run] gpus=${GPU_IDS[*]} smoke=${SMOKE} video_samples_only=${VIDEO_SAMPLES_ONLY}"
  echo "[dry-run] max_episodes_rendered=${MAX_EPISODES_RENDERED} (videos under <suite>/videos/)"
  echo "[dry-run] expected_rollouts≈$(( ${#SUITES[@]} * 10 * EPISODES_PER_TASK )) (if all 10 tasks)"
  exit 0
fi

pids=()
monitor_pid=""
if [[ "${DRY_RUN:-false}" != "true" && "${NO_MONITOR:-false}" != "true" ]]; then
  (
    python "${WS}/scripts/libero_eval/monitor_libero_pro.py" \
      --eval-root "${EVAL_ROOT}" \
      --interval "${MONITOR_INTERVAL:-15}" \
      >"${EVAL_ROOT}/monitor.log" 2>&1
  ) &
  monitor_pid="$!"
  echo "[monitor] pid=${monitor_pid} → ${EVAL_ROOT}/live_summary.md"
fi

launch_suite() {
  local suite="$1"
  local gpu="$2"
  local output_dir="${EVAL_ROOT}/${suite}"
  local episode_length
  episode_length="$(episode_length_for_suite "${suite}")"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/eval_info.json" && "${VIDEO_SAMPLES_ONLY}" != "true" ]]; then
    echo "[eval] skip completed ${suite}"
    return 0
  fi
  if [[ "${VIDEO_SAMPLES_ONLY}" == "true" && -f "${output_dir}/eval_info.json" ]]; then
    rm -f "${output_dir}/eval_info.json" "${output_dir}/process_status.json"
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
      --env.type=libero_pro
      --env.task="${suite}"
      --env.episode_length="${episode_length}"
      --env.control_mode=relative
      --env.max_parallel_tasks=1
      --eval.n_episodes="${EPISODES_PER_TASK}"
      --eval.batch_size="${EVAL_BATCH_SIZE}"
      --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}"
      --eval.use_async_envs=false
      --seed="${EVAL_SEED}"
      --output_dir="${output_dir}"
    )
    if [[ -n "${TASK_IDS}" ]]; then
      cmd+=(--env.task_ids="${TASK_IDS}")
    fi
    echo "[eval] gpu=${gpu} suite=${suite} task_ids=${TASK_IDS:-all} ep=${EPISODES_PER_TASK} horizon=${episode_length} bs=${EVAL_BATCH_SIZE}"
    CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
      "${cmd[@]}" >"${output_dir}/eval.log" 2>&1
    rc=$?
    if [[ "${rc}" -eq 0 ]]; then status=final; else status=failed; fi
    printf '{"status":"%s","exit_code":%d,"gpu":"%s"}\n' \
      "${status}" "${rc}" "${gpu}" >"${output_dir}/process_status.json"
    exit "${rc}"
  ) &
  pids+=("$!")
}

rc=0
if [[ "${SMOKE}" == "true" || "${RUN_SEQUENTIAL:-false}" == "true" ]]; then
  for index in "${!SUITES[@]}"; do
    suite="${SUITES[$index]}"
    gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    launch_suite "${suite}" "${gpu}"
    if ! wait "${pids[-1]}"; then
      echo "[eval] suite ${suite} failed; see ${EVAL_ROOT}/${suite}/eval.log" >&2
      tail -n 80 "${EVAL_ROOT}/${suite}/eval.log" >&2 || true
      rc=1
      break
    fi
  done
else
  # Wave scheduling: at most one suite process per GPU (avoids 2× model load OOM).
  index=0
  while (( index < ${#SUITES[@]} )); do
    wave_pids=()
    for gpu in "${GPU_IDS[@]}"; do
      if (( index >= ${#SUITES[@]} )); then
        break
      fi
      suite="${SUITES[$index]}"
      index=$((index + 1))
      before=${#pids[@]}
      launch_suite "${suite}" "${gpu}"
      if (( ${#pids[@]} > before )); then
        wave_pids+=("${pids[-1]}")
      fi
    done
    for pid in "${wave_pids[@]}"; do
      if ! wait "${pid}"; then rc=1; fi
    done
  done
fi

# Final summary refresh.
python "${WS}/scripts/libero_eval/monitor_libero_pro.py" --eval-root "${EVAL_ROOT}" --once \
  >/dev/null 2>&1 || true
if [[ -n "${monitor_pid}" ]]; then
  kill "${monitor_pid}" 2>/dev/null || true
fi

echo "[eval] results: ${EVAL_ROOT}"
echo "[eval] live summary: ${EVAL_ROOT}/live_summary.md"
if [[ -f "${EVAL_ROOT}/live_summary.md" ]]; then
  sed -n '1,40p' "${EVAL_ROOT}/live_summary.md"
fi
exit "${rc}"

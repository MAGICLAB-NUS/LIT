#!/usr/bin/env bash
# Evaluate a checkpoint on LIBERO-Plus using the public-paper protocol by default:
# full 10,030-task suite, 1 episode per task, Language included, 8-GPU sharding.
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
PLUS_PROTOCOL="${PLUS_PROTOCOL:-full}"
SAMPLES_PER_CELL="${SAMPLES_PER_CELL:-2}"
if [[ "${PLUS_PROTOCOL}" == "full" ]]; then
  EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-0}"
  GPU_IDS=(${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7})
else
  EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-${EPISODES_PER_TASK}}"
  GPU_IDS=(${EVAL_GPU_IDS:-0 1 2 3})
fi

SUITES=(libero_object libero_10 libero_goal libero_spatial)
RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-/data2/JM/Code/molmo_serious/molmoact2-main}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${RESOURCE_ROOT}/third_party/LIBERO-plus}"
LIBERO_PLUS_PACKAGE="${LIBERO_PLUS_ROOT}/libero/libero"
CLASSIFICATION="${LIBERO_PLUS_CLASSIFICATION:-${LIBERO_PLUS_PACKAGE}/benchmark/task_classification.json}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "EVAL_GPU_IDS must contain at least one GPU ID" >&2
  exit 2
fi
if [[ "${PLUS_PROTOCOL}" == "full" && "${EPISODES_PER_TASK}" -ne 1 ]]; then
  echo "Official full LIBERO-Plus protocol requires EPISODES_PER_TASK=1" >&2
  exit 2
fi

for required in \
  "${LIBERO_PLUS_ROOT}/libero/__init__.py" \
  "${LIBERO_PLUS_PACKAGE}/assets" \
  "${LIBERO_PLUS_PACKAGE}/init_files" \
  "${CLASSIFICATION}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing LIBERO-Plus resource: ${required}" >&2
    exit 2
  fi
done

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${RESOURCE_ROOT}/.cache/libero_config_plus}"
export HF_HOME="${HF_HOME:-${RESOURCE_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RESOURCE_ROOT}/.cache/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${RESOURCE_ROOT}/.cache/torch}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# The Plus fork must precede site-packages while current LeRobot remains the source of policy/eval code.
export PYTHONPATH="${LIBERO_PLUS_ROOT}:${WS}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
run_name="$(basename "$(dirname "$(dirname "$(dirname "${POLICY_PATH}")")")")"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-${run_name}_${checkpoint_step}}"
EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval/${CHECKPOINT_LABEL}/libero_plus_${PLUS_PROTOCOL}_seed_${EVAL_SEED}}"
mkdir -p "${EVAL_ROOT}"

python "${WS}/scripts/libero_eval/build_libero_plus_manifest.py" \
  --classification "${CLASSIFICATION}" \
  --output "${EVAL_ROOT}/task_manifest.json" \
  --seed "${EVAL_SEED}" \
  --protocol "${PLUS_PROTOCOL}" \
  --samples-per-cell "${SAMPLES_PER_CELL}" \
  --episodes-per-task "${EPISODES_PER_TASK}"

python - "${EVAL_ROOT}/task_manifest.json" "${EVAL_ROOT}/gpu_shards.json" "${GPU_IDS[*]}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpu_ids = [int(token) for token in sys.argv[3].split()]
n_gpus = len(gpu_ids)
tasks = list(manifest["tasks"])
shards = [
    {
        "shard_index": index,
        "gpu": gpu_ids[index],
        "tasks": [],
        "by_suite": {suite: [] for suite in manifest["suites"]},
    }
    for index in range(n_gpus)
]
# Round-robin keeps category/suite mix balanced across GPUs.
for task_index, task in enumerate(tasks):
    shard = shards[task_index % n_gpus]
    shard["tasks"].append(task)
    shard["by_suite"][task["suite"]].append(int(task["task_id"]))

payload = {
    "num_gpus": n_gpus,
    "gpu_ids": gpu_ids,
    "num_tasks": len(tasks),
    "shards": [
        {
            "shard_index": shard["shard_index"],
            "gpu": shard["gpu"],
            "num_tasks": len(shard["tasks"]),
            "by_suite": {
                suite: ids
                for suite, ids in shard["by_suite"].items()
                if ids
            },
        }
        for shard in shards
    ],
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(
    "[shards] "
    + ", ".join(
        f"gpu{shard['gpu']}={shard['num_tasks']}" for shard in payload["shards"]
    )
)
PY

cat >"${EVAL_ROOT}/run_manifest.json" <<EOF
{
  "benchmark": "libero_plus",
  "protocol": "${PLUS_PROTOCOL}",
  "samples_per_cell": ${SAMPLES_PER_CELL},
  "policy_path": "${POLICY_PATH}",
  "checkpoint_label": "${CHECKPOINT_LABEL}",
  "episodes_per_task": ${EPISODES_PER_TASK},
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "eval_seed": ${EVAL_SEED},
  "gpu_ids": "${GPU_IDS[*]}",
  "num_gpus": ${#GPU_IDS[@]},
  "sharding": "round_robin_tasks_across_gpus",
  "excluded_categories": $(
    if [[ "${PLUS_PROTOCOL}" == "full" ]]; then
      echo '[]'
    else
      echo '["Language Instructions"]'
    fi
  )
}
EOF

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] policy=${POLICY_PATH}"
  echo "[dry-run] output=${EVAL_ROOT}"
  echo "[dry-run] protocol=${PLUS_PROTOCOL} episodes/task=${EPISODES_PER_TASK} batch=${EVAL_BATCH_SIZE}"
  echo "[dry-run] suites=${SUITES[*]} gpus=${GPU_IDS[*]}"
  python - "${EVAL_ROOT}/gpu_shards.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
for shard in payload["shards"]:
    suites = ", ".join(f"{k}:{len(v)}" for k, v in shard["by_suite"].items())
    print(f"[dry-run] gpu {shard['gpu']}: tasks={shard['num_tasks']} ({suites})")
PY
  exit 0
fi

pids=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$shard_index]}"
  shard_root="${EVAL_ROOT}/gpu_${gpu}"
  mkdir -p "${shard_root}"
  if [[ -f "${shard_root}/process_status.json" ]]; then
    status="$(python - "${shard_root}/process_status.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("status", ""))
PY
)"
    if [[ "${status}" == "final" ]]; then
      echo "[eval] skip completed shard gpu=${gpu}"
      continue
    fi
  fi
  (
    printf '{"status":"running","gpu":"%s","shard_index":%s}\n' \
      "${gpu}" "${shard_index}" >"${shard_root}/process_status.json"
    set +e
    rc=0
    mapfile -t shard_suites < <(
      python - "${EVAL_ROOT}/gpu_shards.json" "${gpu}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
gpu = int(sys.argv[2])
for shard in payload["shards"]:
    if int(shard["gpu"]) == gpu:
        for suite in shard["by_suite"]:
            print(suite)
        break
PY
    )
    for suite in "${shard_suites[@]}"; do
      output_dir="${shard_root}/${suite}"
      mkdir -p "${output_dir}"
      if [[ -f "${output_dir}/eval_info.json" ]]; then
        echo "[eval] skip completed ${suite} on gpu=${gpu}"
        continue
      fi
      task_ids="$(
        python - "${EVAL_ROOT}/gpu_shards.json" "${gpu}" "${suite}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
gpu = int(sys.argv[2])
suite = sys.argv[3]
for shard in payload["shards"]:
    if int(shard["gpu"]) == gpu:
        ids = shard["by_suite"][suite]
        print("[" + ",".join(str(task_id) for task_id in ids) + "]")
        break
else:
    raise SystemExit(f"No shard found for gpu={gpu}")
PY
      )"
      echo "[eval] gpu=${gpu} suite=${suite} tasks=${task_ids}"
      CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
        python -m lerobot.scripts.lerobot_eval \
          --policy.path="${POLICY_PATH}" \
          --policy.device=cuda \
          --policy.inference_action_mode=continuous \
          --policy.disable_visual_input=false \
          --policy.per_episode_seed=true \
          --policy.eval_seed="${EVAL_SEED}" \
          --policy.enable_inference_cuda_graph=false \
          --env.type=libero_plus \
          --env.task="${suite}" \
          --env.task_ids="${task_ids}" \
          --env.control_mode=relative \
          --env.max_parallel_tasks=1 \
          --eval.n_episodes="${EPISODES_PER_TASK}" \
          --eval.batch_size="${EVAL_BATCH_SIZE}" \
          --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
          --eval.use_async_envs=true \
          --seed="${EVAL_SEED}" \
          --output_dir="${output_dir}" \
          >"${output_dir}/eval.log" 2>&1
      suite_rc=$?
      if [[ "${suite_rc}" -ne 0 ]]; then
        rc="${suite_rc}"
        break
      fi
    done
    if [[ "${rc}" -eq 0 ]]; then status=final; else status=failed; fi
    printf '{"status":"%s","exit_code":%d,"gpu":"%s","shard_index":%s}\n' \
      "${status}" "${rc}" "${gpu}" "${shard_index}" >"${shard_root}/process_status.json"
    exit "${rc}"
  ) &
  pids+=("$!")
done

python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}"

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then rc=1; fi
done
python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}" --once
echo "[eval] results: ${EVAL_ROOT}"
exit "${rc}"

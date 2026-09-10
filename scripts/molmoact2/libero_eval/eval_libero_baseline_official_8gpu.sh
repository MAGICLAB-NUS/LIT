#!/usr/bin/env bash
# Official 50-episode LIBERO evaluation for baseline_bs224, sharded over 8 GPUs.
#
# Protocol matches V3 official50:
#   - 4 suites × 10 tasks × 50 episodes
#   - OpenVLA horizons, seed 1000
#   - batch size 10 by default
# Each suite is split into task_ids 0-4 and 5-9 across two GPUs.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior_v3/seed_1000/baseline_bs224/checkpoints/030000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"
if [[ -z "${POLICY_PATH}" || ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Baseline checkpoint not found: ${POLICY_PATH}" >&2
  echo "Override with: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"

GPU_IDS=(${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7})
if [[ "${#GPU_IDS[@]}" -ne 8 ]]; then
  echo "Expected exactly 8 GPU IDs, got: ${GPU_IDS[*]}" >&2
  exit 2
fi

EVAL_SEED="${EVAL_SEED:-1000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-2}"
checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-baseline_bs224_${checkpoint_step}}"
MODEL_LABEL="${MODEL_LABEL:-clean_bootstrap_baseline}"
EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval/${CHECKPOINT_LABEL}/libero_official50_seed_${EVAL_SEED}}"
LIBERO_RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-${WS}}"

if [[ "${EPISODES_PER_TASK}" -ne 50 ]]; then
  echo "Official protocol requires EPISODES_PER_TASK=50" >&2
  exit 2
fi

# GPU 0/4: Spatial, 1/5: Object, 2/6: Long, 3/7: Goal.
SUITES=(
  libero_spatial libero_object libero_10 libero_goal
  libero_spatial libero_object libero_10 libero_goal
)
TASK_SPLITS=(
  "[0,1,2,3,4]" "[0,1,2,3,4]" "[0,1,2,3,4]" "[0,1,2,3,4]"
  "[5,6,7,8,9]" "[5,6,7,8,9]" "[5,6,7,8,9]" "[5,6,7,8,9]"
)

mkdir -p "${EVAL_ROOT}"
python - "${EVAL_ROOT}/run_manifest.json" \
  "${POLICY_PATH}" "${CHECKPOINT_LABEL}" "${MODEL_LABEL}" \
  "${EVAL_SEED}" "${EVAL_BATCH_SIZE}" "${MAX_EPISODES_RENDERED}" \
  "${GPU_IDS[*]}" <<'PY'
import json
import sys
from pathlib import Path

(
    output_path,
    policy_path,
    checkpoint_label,
    model_label,
    eval_seed,
    batch_size,
    max_rendered,
    gpu_ids_text,
) = sys.argv[1:]
gpu_ids = gpu_ids_text.split()
suites = ["libero_spatial", "libero_object", "libero_10", "libero_goal"]
payload = {
    "benchmark": "libero",
    "policy_path": policy_path,
    "checkpoint_label": checkpoint_label,
    "model": model_label,
    "protocol": "openvla_public_50ep",
    "suites": suites,
    "episodes_per_task": 50,
    "eval_batch_size": int(batch_size),
    "max_episodes_rendered": int(max_rendered),
    "eval_seed": int(eval_seed),
    "gpu_ids": gpu_ids,
    "num_gpus": 8,
    "sharding": "two_task_halves_per_suite",
    "suite_shards": {
        suite: [
            {"gpu": gpu_ids[index], "task_ids": [0, 1, 2, 3, 4]},
            {"gpu": gpu_ids[index + 4], "task_ids": [5, 6, 7, 8, 9]},
        ]
        for index, suite in enumerate(suites)
    },
    "official_horizons": "true",
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

for index in "${!GPU_IDS[@]}"; do
  echo "[8gpu] GPU ${GPU_IDS[$index]} -> ${SUITES[$index]} tasks=${TASK_SPLITS[$index]}"
done

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  DRY_RUN=true \
  LIBERO_RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT}" \
  EPISODES_PER_TASK=50 \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  OFFICIAL_HORIZONS=true \
  EVAL_GPU_IDS="${GPU_IDS[0]}" \
  EVAL_SUITES="${SUITES[0]}" \
  EVAL_TASK_IDS="${TASK_SPLITS[0]}" \
  EVAL_ROOT="${EVAL_ROOT}/gpu_${GPU_IDS[0]}" \
  CHECKPOINT_LABEL="${CHECKPOINT_LABEL}" \
  MODEL_LABEL="${MODEL_LABEL}" \
    bash "${WS}/scripts/libero_eval/eval_libero_baseline_checkpoint.sh" "${POLICY_PATH}"
  exit 0
fi

pids=()
for index in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$index]}"
  suite="${SUITES[$index]}"
  task_ids="${TASK_SPLITS[$index]}"
  shard_root="${EVAL_ROOT}/gpu_${gpu}"
  (
    LIBERO_RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT}" \
    EPISODES_PER_TASK=50 \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
    OFFICIAL_HORIZONS=true \
    EVAL_GPU_IDS="${gpu}" \
    EVAL_SUITES="${suite}" \
    EVAL_TASK_IDS="${task_ids}" \
    EVAL_ROOT="${shard_root}" \
    CHECKPOINT_LABEL="${CHECKPOINT_LABEL}" \
    MODEL_LABEL="${MODEL_LABEL}" \
      bash "${WS}/scripts/libero_eval/eval_libero_baseline_checkpoint.sh" "${POLICY_PATH}"
  ) >"${EVAL_ROOT}/gpu_${gpu}.console.log" 2>&1 &
  pids+=("$!")
done

python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}"

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done
python "${WS}/scripts/libero_eval/monitor_eval.py" --eval-root "${EVAL_ROOT}" --once
echo "[8gpu] results: ${EVAL_ROOT}"
exit "${rc}"

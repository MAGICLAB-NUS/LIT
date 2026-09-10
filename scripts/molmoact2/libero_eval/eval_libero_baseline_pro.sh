#!/usr/bin/env bash
# Evaluate baseline_bs224 on the public LIBERO-PRO Total16 protocol (8 GPUs).
#
# Matches the completed V3/V4 PRO runs:
#   - 4 base suites × {object, swap, lan, task}
#   - 10 tasks/suite × 50 episodes/task
#   - official horizons, seed 1000, batch size 1
# Env is excluded because the public packs do not contain those suites.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_POLICY_PATH="${WS}/lerobot/outputs/libero_goal_prior_v3/seed_1000/baseline_bs224/checkpoints/030000/pretrained_model"
POLICY_PATH="${1:-${POLICY_PATH:-${DEFAULT_POLICY_PATH}}}"

if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Baseline checkpoint not found: ${POLICY_PATH}" >&2
  echo "Override with: bash $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"

python - "${POLICY_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
if cfg.get("enable_goal_pose"):
    raise SystemExit("Refusing to evaluate: checkpoint has enable_goal_pose=true")
if cfg.get("type") != "molmoact2":
    raise SystemExit(f"Refusing to evaluate: unexpected type={cfg.get('type')!r}")
print("[eval] verified plain MolmoAct2 baseline for LIBERO-PRO")
PY

checkpoint_step="$(basename "$(dirname "${POLICY_PATH}")")"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-baseline_bs224_${checkpoint_step}}"
export EVAL_SEED="${EVAL_SEED:-1000}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-2}"
export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
export INCLUDE_ENV="${INCLUDE_ENV:-false}"
export EVAL_ROOT="${EVAL_ROOT:-${WS}/lerobot/outputs/libero_eval_pro/${CHECKPOINT_LABEL}/full50_seed_${EVAL_SEED}}"

if [[ "${EPISODES_PER_TASK}" -ne 50 || "${EVAL_BATCH_SIZE}" -ne 1 ]]; then
  echo "V3-aligned PRO requires EPISODES_PER_TASK=50 and EVAL_BATCH_SIZE=1" >&2
  exit 2
fi
if [[ "${INCLUDE_ENV}" != "false" ]]; then
  echo "V3-aligned public Total16 requires INCLUDE_ENV=false" >&2
  exit 2
fi

echo "[eval-baseline-pro] checkpoint=${POLICY_PATH}"
echo "[eval-baseline-pro] output=${EVAL_ROOT}"
echo "[eval-baseline-pro] gpus=${EVAL_GPU_IDS} protocol=Total16 rollouts=8000"

exec bash "${WS}/scripts/libero_eval/eval_libero_pro_checkpoint.sh" "${POLICY_PATH}"

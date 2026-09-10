#!/usr/bin/env bash
# Backup + recompute observation.state / action QUANTILES for goal-pose v3.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/data2/JM/dataset/libero_lerobot_format}"

source "${WS}/scripts/activate_train_env.sh"

cd "${WS}"
python scripts/libero_goal_prior_v3/recompute_vector_quantiles.py \
  --dataset-root "${DATASET_ROOT}" \
  "$@"

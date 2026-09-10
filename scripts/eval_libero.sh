#!/usr/bin/env bash
# LIBERO (in-distribution): four suites, 50 episodes per task, official horizons,
# seed 1000 — 2,000 episodes per model.
#
#   bash scripts/eval_libero.sh <run-name> <checkpoint>
#
# Eight shards, one per GPU: each suite is split into two halves of five task ids.
# Shards that produce no eval_info.json are retried up to twice.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
lit_need_eval || exit 2

NAME="${1:?usage: eval_libero.sh <run-name> <checkpoint>}"
CK="${2:?usage: eval_libero.sh <run-name> <checkpoint>}"
[ -f "$CK/config.json" ] || { echo "no config.json in $CK" >&2; exit 2; }
CK="$(cd "$CK" && pwd)"

ROOT="$LIT_WORK/libero/$NAME/libero_official50_seed_1000"
LOG="$LIT_WORK/eval_libero_${NAME}.log"
mkdir -p "$ROOT"

SUITE=(libero_spatial libero_object libero_10 libero_goal libero_spatial libero_object libero_10 libero_goal)
SPLIT=("[0,1,2,3,4]" "[0,1,2,3,4]" "[0,1,2,3,4]" "[0,1,2,3,4]" "[5,6,7,8,9]" "[5,6,7,8,9]" "[5,6,7,8,9]" "[5,6,7,8,9]")

missing() { local m=(); for g in 0 1 2 3 4 5 6 7; do
  [ -f "$ROOT/gpu_$g/${SUITE[$g]}/eval_info.json" ] || m+=("$g"); done; echo "${m[@]:-}"; }

echo "[$(date '+%F %T')] LIBERO  name=$NAME  ckpt=$CK" | tee -a "$LOG"

for round in 1 2 3; do
  miss=$(missing); [ -z "$miss" ] && break
  echo "[$(date '+%F %T')] round $round, shards: $miss" | tee -a "$LOG"
  pids=()
  for g in $miss; do
    rm -rf "$ROOT/gpu_$g"
    ( cd "$LIT_MOLMOACT2" || exit 1
      LIBERO_PLUS_ROOT="$LIBERO_PLUS_ROOT" \
      LIBERO_RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-$LIT_MOLMOACT2}" \
      EPISODES_PER_TASK=50 EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}" \
      MAX_EPISODES_RENDERED=2 OFFICIAL_HORIZONS=true EVAL_SEED=1000 \
      EVAL_SKIP_CONFIG_GUARD=1 \
      EVAL_GPU_IDS="$g" EVAL_SUITES="${SUITE[$g]}" EVAL_TASK_IDS="${SPLIT[$g]}" \
      EVAL_ROOT="$ROOT/gpu_$g" \
      bash scripts/libero_eval/eval_libero_v4_checkpoint.sh "$CK" \
        > "$ROOT/gpu_$g.round$round.log" 2>&1 ) &
    pids+=($!); sleep 5
  done
  wait "${pids[@]}" 2>/dev/null
done

miss=$(missing)
echo "[$(date '+%F %T')] done. missing shards: ${miss:-none}" | tee -a "$LOG"
python3 "$(dirname "${BASH_SOURCE[0]}")/aggregate.py" --libero "$ROOT"
[ -z "$miss" ]

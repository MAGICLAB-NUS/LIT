#!/usr/bin/env bash
# LIBERO-Plus (out-of-distribution): all 10,030 tasks, one episode each, seed 1000.
#
#   bash scripts/eval_libero_plus.sh <run-name> <checkpoint> [slices]
#
# <checkpoint> is the directory holding config.json (…/checkpoints/NNNNNN/pretrained_model).
# Work is split into <slices> concurrent passes, each striding over the whole task list, so
# every pass finishes at the same time instead of one suite dragging out the tail.
# Re-running the same <run-name> skips tasks that already have a result, so it is resumable.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
lit_need_eval || exit 2

NAME="${1:?usage: eval_libero_plus.sh <run-name> <checkpoint> [slices]}"
CK="${2:?usage: eval_libero_plus.sh <run-name> <checkpoint> [slices]}"
N="${3:-4}"

[ -f "$CK/config.json" ] || { echo "no config.json in $CK — point at the pretrained_model directory" >&2; exit 2; }
CK="$(cd "$CK" && pwd)"

ROOT="$LIT_WORK/libero_plus/$NAME"
PASS="$ROOT/pass_$(date +%Y%m%d_%H%M%S)"
LOG="$LIT_WORK/eval_plus_${NAME}.log"
mkdir -p "$PASS"

{ echo "[$(date '+%F %T')] LIBERO-Plus  name=$NAME  slices=$N"
  echo "  checkpoint = $CK"
  echo "  results    = $PASS   (skipping anything already under $ROOT)"; } | tee -a "$LOG"

for i in $(seq 0 $((N-1))); do
  (
    export LIBERO_PLUS_FIX_LANG=1          # without this the instruction is derived from the
                                           # filename, which feeds perturbation parameters to
                                           # the model and depresses every axis
    export LIBERO_PLUS_ROOT LIBERO_RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-$LIT_MOLMOACT2}"
    export EVAL_SUITES="libero_object libero_10 libero_goal libero_spatial"
    export EVAL_TASK_SHARD="$i/$N"
    export EVAL_EXCLUDE_DONE_ROOTS="$ROOT"
    export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
    export EVAL_SEED=1000
    export PLUS_PROTOCOL=full
    export CHECKPOINT_LABEL="${NAME}_slice${i}"
    export EVAL_ROOT="$PASS/slice$i"
    cd "$LIT_MOLMOACT2" || exit 1
    bash scripts/libero_eval/eval_libero_plus_checkpoint_suites.sh "$CK" \
      >> "$LIT_WORK/eval_plus_${NAME}_slice${i}.log" 2>&1
    echo "[$(date '+%F %T')] slice$i rc=$?" >> "$LOG"
  ) &
  sleep 20      # stagger so the slices do not all build the manifest at once
done
wait

echo "[$(date '+%F %T')] done — aggregating" | tee -a "$LOG"
python3 "$(dirname "${BASH_SOURCE[0]}")/aggregate.py" "$ROOT"

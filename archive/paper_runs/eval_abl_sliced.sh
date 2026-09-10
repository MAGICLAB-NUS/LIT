#!/usr/bin/env bash
# 消融臂 LIBERO-Plus 全量评测（均衡切片版）：N 个实例各取全部 suite 的 1/N（按索引切片），各占 8 卡 → 8N worker，
# 所有实例同时收尾，不再被最长的 suite 拖长尾。支持续跑：剔除 <arm>/ 下已有结果，写到 <arm>/<subdir>/slice<i>/。
# 用法: eval_abl_sliced.sh <臂名> <checkpoint 目录> <子目录> [N=4]
set -uo pipefail
ARM="${1:?usage}"; CK="${2:?usage}"; SUB="${3:?usage: need subdir}"; N="${4:-4}"
M=/data2/JM/Code/molmoact2
ROOT=$M/lerobot/outputs/libero_ablation_eval/$ARM
PARENT=$ROOT/$SUB
LOG=/data2/JM/eval_${ARM}_${SUB}.log
[ -f "$CK/config.json" ] || { echo "checkpoint 不存在: $CK" >&2; exit 2; }
mkdir -p "$PARENT"
{ echo "[$(date '+%F %T')] 启动 $ARM/$SUB 切片评测 N=$N"; echo "  ckpt=$CK"; echo "  out=$PARENT  exclude=$ROOT"; } >> "$LOG"
for i in $(seq 0 $((N-1))); do
  (
    export LIBERO_PLUS_FIX_LANG=1
    export EVAL_SUITES="libero_object libero_10 libero_goal libero_spatial"
    export EVAL_TASK_SHARD="$i/$N"
    export EVAL_EXCLUDE_DONE_ROOTS="$ROOT"
    export EVAL_GPU_IDS='0 1 2 3 4 5 6 7'
    export CHECKPOINT_LABEL="${ARM}_${SUB}_slice${i}"
    export EVAL_ROOT="$PARENT/slice$i"
    export EVAL_SEED=1000
    export PLUS_PROTOCOL=full
    cd "$M" || exit 1
    bash scripts/libero_eval/eval_libero_plus_checkpoint_suites.sh "$CK" >> "/data2/JM/eval_${ARM}_${SUB}_slice${i}.log" 2>&1
    echo "[$(date '+%F %T')] slice$i 结束 rc=$?" >> "$LOG"
  ) &
  sleep 20
done
wait
echo "[$(date '+%F %T')] $ARM/$SUB 全部结束" >> "$LOG"

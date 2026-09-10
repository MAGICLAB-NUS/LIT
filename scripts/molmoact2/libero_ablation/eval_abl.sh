#!/usr/bin/env bash
# 消融臂的 LIBERO-Plus 全量评测：四个 suite 并行、各占 8 卡 → 32 worker 跑【不相交】的任务。
#   （9-5 修复：此前 EVAL_SUITES 未生效，4 实例重复跑同一份清单；现由评测脚本内的 manifest 过滤步保证按 suite 切分）
# 用法: eval_abl.sh <臂名> <checkpoint 目录> [子目录]
#   给子目录（如 r2）= 续跑：结果写到 <臂名>/<子目录>/，并剔除 <臂名>/ 下已有结果的任务；collector 递归扫描会自动合并。
set -uo pipefail
ARM="${1:?usage: eval_abl.sh <arm> <ckpt_dir> [subdir]}"
CK="${2:?usage: eval_abl.sh <arm> <ckpt_dir> [subdir]}"
SUB="${3:-}"
M=/data2/JM/Code/molmoact2
ROOT=$M/lerobot/outputs/libero_ablation_eval/$ARM
PARENT=$ROOT${SUB:+/$SUB}
LOG=/data2/JM/eval_${ARM}${SUB:+_$SUB}.log

[ -f "$CK/config.json" ] || { echo "checkpoint 不存在: $CK" >&2; exit 2; }
mkdir -p "$PARENT"
{ echo "[$(date '+%F %T')] 启动 $ARM 评测${SUB:+（续跑 $SUB）}"; echo "  ckpt=$CK"; echo "  out=$PARENT"; } >> "$LOG"

for suite in libero_object libero_10 libero_goal libero_spatial; do
  (
    export LIBERO_PLUS_FIX_LANG=1
    export EVAL_SUITES="$suite"
    export EVAL_GPU_IDS='0 1 2 3 4 5 6 7'
    export CHECKPOINT_LABEL="${ARM}_${suite}"
    export EVAL_ROOT="$PARENT/$suite"
    export EVAL_SEED=1000
    export PLUS_PROTOCOL=full
    [ -n "$SUB" ] && export EVAL_EXCLUDE_DONE_ROOTS="$ROOT"
    cd "$M" || exit 1
    bash scripts/libero_eval/eval_libero_plus_checkpoint_suites.sh "$CK" \
      >> "/data2/JM/eval_${ARM}${SUB:+_$SUB}_${suite}.log" 2>&1
    echo "[$(date '+%F %T')] $suite 结束 rc=$?" >> "$LOG"
  ) &
  sleep 20
done
wait
echo "[$(date '+%F %T')] $ARM${SUB:+/$SUB} 全部结束" >> "$LOG"

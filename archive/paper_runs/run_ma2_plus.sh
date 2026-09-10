#!/usr/bin/env bash
# MolmoAct2 LIBERO-Plus 全量重跑（语言 bug 已修复）。
# 用法: run_ma2_plus.sh {v3|baseline}
# 协议与 2026-08-12 那次逐字一致，唯一差别是 LIBERO_PLUS_FIX_LANG=1。
set -uo pipefail
WHICH="${1:-}"; [ -n "$WHICH" ] || { echo "usage: $0 v3|baseline" >&2; exit 2; }
M=/data2/JM/Code/molmoact2
OUTS=$M/lerobot/outputs

case "$WHICH" in
  v3)       CK=$OUTS/libero_goal_prior_v3/seed_1000/stage2/checkpoints/030000/pretrained_model
            LABEL=v3_030000_langfixed ;;
  baseline) CK=$OUTS/libero_goal_prior_v3/seed_1000/baseline_bs224/checkpoints/030000/pretrained_model
            LABEL=baseline_bs224_030000_langfixed ;;
  *) echo "unknown: $WHICH" >&2; exit 2 ;;
esac

export LIBERO_PLUS_FIX_LANG=1
export EVAL_GPU_IDS='0 1 2 3 4 5 6 7'
export CHECKPOINT_LABEL="$LABEL"
export EVAL_ROOT="$OUTS/libero_eval_plus/$LABEL/libero_plus_full_seed_1000"
export EVAL_SEED=1000
export PLUS_PROTOCOL=full

LOG=/data2/JM/ma2_plus_${WHICH}.log
{
  echo "[$(date '+%F %T')] 启动 $WHICH"
  echo "  ckpt   = $CK"
  echo "  out    = $EVAL_ROOT"
  echo "  fixlang= $LIBERO_PLUS_FIX_LANG   gpus = $EVAL_GPU_IDS"
} >> "$LOG"

cd "$M" || exit 1
bash scripts/libero_eval/eval_libero_plus_checkpoint.sh "$CK" >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] $WHICH 结束 rc=$rc" >> "$LOG"
exit $rc

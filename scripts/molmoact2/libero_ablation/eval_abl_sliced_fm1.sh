#!/usr/bin/env bash
# fm1 版：消融臂 LIBERO-Plus 全量评测（均衡切片）。与 A800j 的 eval_abl_sliced.sh 同一份底层脚本、
# 同一套评测参数（per_episode_seed / control_mode=relative / async envs / batch 1 / seed 1000 / FIX_LANG=1），
# 只把资源路径换成 fm1 上的：
#   LIBERO_PLUS_ROOT   /pfs/.../third_party/LIBERO-plus   （bddl/init/classification 齐全，assets 经符号链接指向 /data2 那份）
#   WS(仓库根)         /pfs/.../molmoact2                  → PYTHONPATH 用 /pfs 的 lerobot/src，即训练 ③ 的同一份代码
#   LIBERO_RESOURCE_ROOT 仅用于缓存目录，指到 /pfs 下可写位置
# 用法: eval_abl_sliced_fm1.sh <臂名> <ckpt> <子目录> [N=4]
set -uo pipefail
ARM="${1:?usage}"; CK="${2:?usage}"; SUB="${3:?usage}"; N="${4:-4}"
M=/pfs/pfs-Tr4Uts/JM_work/molmoact2
ROOT=$M/lerobot/outputs/libero_ablation_eval/$ARM
PARENT=$ROOT/$SUB
LOG=/pfs/pfs-Tr4Uts/JM_work/eval_${ARM}_${SUB}.log
[ -f "$CK/config.json" ] || { echo "checkpoint 不存在: $CK" >&2; exit 2; }
mkdir -p "$PARENT"
{ echo "[$(date '+%F %T')] fm1 启动 $ARM/$SUB 切片评测 N=$N"; echo "  ckpt=$CK"; echo "  out=$PARENT"; } >> "$LOG"
for i in $(seq 0 $((N-1))); do
  (
    export LIBERO_PLUS_FIX_LANG=1
    export LIBERO_PLUS_ROOT=/pfs/pfs-Tr4Uts/JM_work/third_party/LIBERO-plus
    export LIBERO_RESOURCE_ROOT=/pfs/pfs-Tr4Uts/JM_work/.eval_cache
    export EVAL_SUITES="libero_object libero_10 libero_goal libero_spatial"
    export EVAL_TASK_SHARD="$i/$N"
    export EVAL_GPU_IDS='0 1 2 3 4 5 6 7'      # 评测可用 GPU3（故障只在训练 backward 出现）
    export CHECKPOINT_LABEL="${ARM}_${SUB}_slice${i}"
    export EVAL_ROOT="$PARENT/slice$i"
    export EVAL_SEED=1000
    export PLUS_PROTOCOL=full
    cd "$M" || exit 1
    bash scripts/libero_eval/eval_libero_plus_checkpoint_suites.sh "$CK" \
      >> "/pfs/pfs-Tr4Uts/JM_work/eval_${ARM}_${SUB}_slice${i}.log" 2>&1
    echo "[$(date '+%F %T')] slice$i 结束 rc=$?" >> "$LOG"
  ) &
  sleep 20
done
wait
echo "[$(date '+%F %T')] $ARM/$SUB 全部结束" >> "$LOG"

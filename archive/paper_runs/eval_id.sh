#!/usr/bin/env bash
# 消融臂 LIBERO clean（ID）评测 v2：与主表同口径 openvla_public_50ep（50 eps/任务、官方 horizon、seed 1000、8 卡按 suite×task 拆分）。
#   v2 改动：eval_batch_size 50→25（8 shard 总并发 400→200 个仿真进程，避免 CPU 过载与 BrokenPipe）；
#            启动器返回后逐 shard 检查 eval_info.json，缺失的并行重跑（最多 2 轮）。
#   v3 改动：batch 25→10（主表口径，实测更快：8×25=200 并发超过 128 核）。
# 用法: eval_id.sh <臂名> <checkpoint 目录>
set -uo pipefail
ARM="${1:?usage: eval_id.sh <arm> <ckpt_dir>}"; CK="${2:?usage}"
M=/data2/JM/Code/molmoact2
# shellcheck source=/dev/null
source "$M/scripts/activate_train_env.sh" >/dev/null 2>&1
export CHECKPOINT_LABEL="abl_${ARM}" MODEL_LABEL="abl_${ARM}"
export EVAL_ROOT="$M/lerobot/outputs/libero_ablation_eval_id/${ARM}/libero_official50_seed_1000"
export EVAL_GPU_IDS='0 1 2 3 4 5 6 7' EVAL_SEED=1000 EVAL_VARIANT=v3 EVAL_SKIP_CONFIG_GUARD=1
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"   # 与主表 ID 跑一致；25 会让 200 并发仿真挤 128 核，反而更慢
LOG=/data2/JM/eval_id_${ARM}.log
LAUNCHER=$M/scripts/libero_eval/eval_libero_v4_official_8gpu.sh
[ -f "$CK/config.json" ] || { echo "checkpoint 不存在: $CK" >&2; exit 2; }
mkdir -p "$EVAL_ROOT"
echo "[$(date '+%F %T')] ID 评测启动 $ARM batch=$EVAL_BATCH_SIZE ckpt=$CK out=$EVAL_ROOT" >> "$LOG"

# gpu → (suite, task 拆分)，与官方 8 卡启动器一致
SUITE_OF=(libero_spatial libero_object libero_10 libero_goal libero_spatial libero_object libero_10 libero_goal)
SPLIT_OF=("[0,1,2,3,4]" "[0,1,2,3,4]" "[0,1,2,3,4]" "[0,1,2,3,4]" "[5,6,7,8,9]" "[5,6,7,8,9]" "[5,6,7,8,9]" "[5,6,7,8,9]")
missing_shards() { local m=(); for g in 0 1 2 3 4 5 6 7; do [ -f "$EVAL_ROOT/gpu_$g/${SUITE_OF[$g]}/eval_info.json" ] || m+=("$g"); done; echo "${m[@]:-}"; }

# 第一轮：若尚无任何 shard 完成，走官方启动器整体跑；否则直接进入补跑
if [ -z "$(ls -d "$EVAL_ROOT"/gpu_*/libero_*/eval_info.json 2>/dev/null)" ]; then
  cd "$M" && bash "$LAUNCHER" "$CK" >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] 启动器返回 rc=$?" >> "$LOG"
fi

# 补跑缺失 shard（并行），最多 2 轮
eval "$(grep -m1 '^LIBERO_RESOURCE_ROOT=' "$LAUNCHER")"
for round in 1 2; do
  miss=$(missing_shards); [ -z "$miss" ] && break
  echo "[$(date '+%F %T')] 第 $round 轮补跑缺失 shard: $miss" >> "$LOG"
  pids=()
  for g in $miss; do
    rm -rf "$EVAL_ROOT/gpu_$g"
    ( cd "$M" && LIBERO_RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-}" EPISODES_PER_TASK=50 EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" MAX_EPISODES_RENDERED=2 OFFICIAL_HORIZONS=true \
        EVAL_GPU_IDS="$g" EVAL_SUITES="${SUITE_OF[$g]}" EVAL_TASK_IDS="${SPLIT_OF[$g]}" EVAL_ROOT="$EVAL_ROOT/gpu_$g" \
        bash "$M/scripts/libero_eval/eval_libero_v4_checkpoint.sh" "$CK" > "$EVAL_ROOT/gpu_$g.retry$round.log" 2>&1 ) &
    pids+=($!); sleep 5
  done
  wait "${pids[@]}"
done
miss=$(missing_shards)
echo "[$(date '+%F %T')] ID 评测结束 $ARM  缺失 shard: ${miss:-无}" >> "$LOG"
python3 /data2/JM/collect_id.py "$ARM" >> "$LOG" 2>&1
[ -z "$miss" ]

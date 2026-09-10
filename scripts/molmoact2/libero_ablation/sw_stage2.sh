#!/usr/bin/env bash
# ============================================================================
# 消融臂 A —— Stagewise 对照，Stage 2
#
#   从 Stagewise Stage 1 出发做"普通的视觉微调"，Latent Interface 三件套全关：
#         enable_goal_pose              True -> False
#         enable_pose_reconstruction    True -> False
#         mask_image_from_action_expert True -> False
#   即：图像 token 直达 AE（不经 latent 聚合）、无 pose 重建监督、不 firewall。
#
#   预算与消融族对齐：bs16 × 8 卡 = 有效 128，30000 步。
#   六组学习率 + warmup 逐字段照抄 v3 stage2 的 checkpoint train_config.json。
#
#   加载硬判据（见文件末尾的 audit）：
#     Stage 1 与 Stage 2 的架构完全相同（都是 VLM+AE，无 goal 模块），
#     所以权重加载必须是【零 missing key、零 unexpected key】。
#     一旦出现任何一种，说明配置没生效或结构对不上 —— 立即杀掉，不让它闷跑 37 小时。
#     （对照：ours 的 stage2 会有大量 missing(latent聚合器) + 6 个 unexpected(SE3 encoder)，
#       那是预期的；stagewise 这条不该有任何一条。）
# ============================================================================
set -uo pipefail
M=/data2/JM/Code/molmoact2

STEPS_OVERRIDE="${STEPS_OVERRIDE:-30000}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
SW_ROOT=$M/lerobot/outputs/libero_ablation/sw_stagewise${OUT_SUFFIX}
STAGE1_CKPT="${STAGE1_CKPT:-$SW_ROOT/stage1/checkpoints/010000/pretrained_model}"

if [ ! -f "$STAGE1_CKPT/config.json" ]; then
  echo "FATAL: Stagewise Stage-1 checkpoint missing: $STAGE1_CKPT" >&2
  exit 2
fi

# Stage 1 必须确实是"无 SE(3)"的那一版，否则这条消融是假的
python3 - "$STAGE1_CKPT/config.json" <<'PY' || exit 2
import json, sys
c = json.load(open(sys.argv[1]))
g = c.get("enable_goal_pose")
print(f"[pre-audit] stage1 enable_goal_pose = {g}")
sys.exit(0 if g is False else 3)
PY

# --- 8 卡 × bs16 = 有效 128，与 arm ① 同预算 ---------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATASET_ROOT=/data2/JM/dataset/libero_lerobot_format
export DATASET_REPO_ID=lerobot/libero                # 与 v3 stage2 / arm ① 一致
export POLICY_PATH="$STAGE1_CKPT"                    # → --policy.path，从 Stage 1 续训
                                                     # 注意：POLICY_PATH 非空时底层脚本
                                                     # 不再传 --policy.checkpoint_path，
                                                     # 所以这里不设 CHECKPOINT_PATH（会被忽略）
export JOB_NAME=sw_stagewise_stage2
export RUN_ID=seed_1000
export OUTPUT_DIR=$SW_ROOT/stage2
export RESUME_MODE=off

export STEPS=$STEPS_OVERRIDE
export BATCH_SIZE=16
export SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-5000}"
export LOG_FREQ=20
export NUM_WORKERS=4
export SEED=1000
export SAVE_CHECKPOINT=true

# --- Stage 2 制度：视觉开、全模型训练 ---------------------------------------
# 这两个必须显式翻回 false：--policy.path 会继承 Stage 1 的 config，
# 而 Stage 1 是 disable_visual_input=true / train_action_expert_only=true。
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

# --- v3 Stage 2 的学习率与 warmup ------------------------------------------
export OPTIMIZER_LR=1e-5
export OPTIMIZER_VIT_LR=1e-5
export OPTIMIZER_CONNECTOR_LR=1e-5
export OPTIMIZER_ACTION_EXPERT_LR=1e-4
export SCHEDULER_WARMUP_STEPS=1000
export SCHEDULER_VLM_WARMUP_STEPS=1000
export SCHEDULER_VIT_WARMUP_STEPS=2000
export SCHEDULER_CONNECTOR_WARMUP_STEPS=1000
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000
export SCHEDULER_DECAY_STEPS=30000                   # 固定，不随 STEPS_OVERRIDE 变
export SCHEDULER_DECAY_LR=1e-6

cd "$M" || exit 1

TRAIN_LOG=${TRAIN_LOG:-/data2/JM/sw_stage2.trainlog}
: > "$TRAIN_LOG"

setsid bash scripts/train_libero_molmoact2.sh \
  --policy.enable_goal_pose=false \
  --policy.enable_pose_reconstruction=false \
  --policy.mask_image_from_action_expert=false \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.optimizer_weight_decay=0.0 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.optimizer_eps=1e-6 \
  --policy.softmax_auxiliary_loss=true \
  --policy.softmax_auxiliary_loss_scale=1e-4 \
  --policy.num_state_tokens=256 \
  --policy.expected_max_action_dim=32 \
  --policy.mask_action_dim_padding=true \
  --policy.normalize_language=true \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
TRAIN_PGID=$(ps -o pgid= -p "$TRAIN_PID" 2>/dev/null | tr -d ' ')
echo "[audit] training launched pid=$TRAIN_PID pgid=$TRAIN_PGID log=$TRAIN_LOG"

# ---- 加载审计：等到"开始训练"这一刻，检查有没有 key 不匹配 ----------------
audit_fail() {
  echo "[audit] FATAL: $1"
  echo "[audit] killing training so it does not silently run 37h with a wrong load"
  [ -n "$TRAIN_PGID" ] && kill -TERM "-$TRAIN_PGID" 2>/dev/null
  sleep 10
  [ -n "$TRAIN_PGID" ] && kill -9 "-$TRAIN_PGID" 2>/dev/null
  exit 3
}

DEADLINE=$(( $(date +%s) + 2400 ))     # 最多等 40 分钟到达开训点
while :; do
  if grep -qF "Missing key(s) when loading model" "$TRAIN_LOG" 2>/dev/null; then
    echo "[audit] --- offending line ---"; grep -oF -m1 "Missing key(s) when loading model" "$TRAIN_LOG"
    audit_fail "unexpected MISSING keys — stagewise stage2 must match stage1 exactly"
  fi
  if grep -qF "Unexpected key(s) when loading model" "$TRAIN_LOG" 2>/dev/null; then
    echo "[audit] --- offending line ---"; grep -oF -m1 "Unexpected key(s) when loading model" "$TRAIN_LOG"
    audit_fail "unexpected EXTRA keys — stage1 carried a module stage2 does not build"
  fi
  if grep -qF "Start offline training" "$TRAIN_LOG" 2>/dev/null; then
    echo "[audit] OK: weight load was exact (no missing / no unexpected keys)"
    grep -hE "Effective batch size|num_learnable_params|num_total_params" "$TRAIN_LOG" | head -3
    grep -hoE "\"visual_inputs_present\": *[a-z]+" "$TRAIN_LOG" | head -1
    break
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[audit] training process died before reaching the training loop; tail:"
    tail -30 "$TRAIN_LOG"
    exit 4
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    audit_fail "timed out waiting for the training loop to start"
  fi
  sleep 20
done

wait "$TRAIN_PID"
rc=$?
echo "[audit] training exited rc=$rc"
cat "$TRAIN_LOG" >> /data2/JM/sw_stage2.fulllog 2>/dev/null
exit $rc

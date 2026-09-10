#!/usr/bin/env bash
# ============================================================================
# 消融臂 ②  abl_open_path —— 打开直连视觉通路（fm1 版，用户 9-4 定稿）
#
#   不用 Stage 1。初始化与 ④(abl_wo_stage1) 完全相同：
#         checkpoint_path = MolmoAct2 发布权重
#         vlm_checkpoint_path = Molmo2-ER（覆盖 VLM）
#         randomize_action_expert = true（AE 随机重置）
#   Stage 2 完整 LIT 框架原样保留：100 latent 聚合、pose 重建监督、6 层组聚合器。
#   相对 ④ 只改一个布尔量：
#         mask_image_from_action_expert   true -> false
#   即图像 token 的 KV 同时直连进 AE（latent 列照常追加，两者并存）。
#
#   → 正确对照行是 ④，两者只差这一个开关（单变量）。
#
#   卡与 batch：fm1 的 GPU3 有沉默故障（已在 diag/ 复现：0/200 步卡死、110 W 自旋），
#   排除后 7 卡 × bs16 = 有效 112（④ 是 8×16=128；用户判断 Plus 对训练量不敏感，接受）。
#   学习率 + warmup 照抄 v3 stage2 / ④，30000 步。
#
#   加载审计（本臂专属，依 ④ 当年日志）：
#     · 不得出现 "Missing key(s)" / "Unexpected key(s)"（这条路径不会走到该分支）
#     · 必须出现 "clean bootstrap audit" 且日志里 randomize_action_expert: True
#     · 命令行必须含 --policy.mask_image_from_action_expert=false
#     · visual_inputs_present 必须为 true
#   任何一条不满足 → 立即杀掉。
# ============================================================================
set -uo pipefail
M=/pfs/pfs-Tr4Uts/JM_work/molmoact2

STEPS_OVERRIDE="${STEPS_OVERRIDE:-30000}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
EXPECT_EFF="${EXPECT_EFF:-112}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6,7}"   # 排除 GPU3
export BATCH_SIZE="${BATCH_SIZE:-16}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
EFF=$(( NGPU * BATCH_SIZE ))
if [ "$EFF" -ne "$EXPECT_EFF" ]; then
  echo "FATAL: effective batch = $NGPU x $BATCH_SIZE = $EFF, expected $EXPECT_EFF" >&2; exit 2
fi
case ",$CUDA_VISIBLE_DEVICES," in *,3,*) echo "FATAL: GPU3 is faulty on fm1, refuse to use it" >&2; exit 2;; esac

for f in "$M/Checkpoint/MolmoAct2/config.json" "$M/Checkpoint/Molmo2-ER/config.json"; do
  [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 2; }
done

export LIT_ALLOW_OPEN_VISUAL_PATH=1                 # 放行 mask=false 的校验（仅此臂）

export DATASET_ROOT=/pfs/pfs-Tr4Uts/JM_work/dataset/libero_lerobot_format
export DATASET_REPO_ID=lerobot/libero
export CHECKPOINT_PATH=$M/Checkpoint/MolmoAct2
export POLICY_PATH=""                                # 空 → --policy.type + checkpoint_path（与 ④ 相同）
export JOB_NAME=abl_open_path
export RUN_ID=seed_1000
export OUTPUT_DIR=$M/lerobot/outputs/libero_ablation/abl_open_path${OUT_SUFFIX}
export RESUME_MODE=off

export STEPS=$STEPS_OVERRIDE
export SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-5000}"
export LOG_FREQ=20
export NUM_WORKERS=4
export SEED=1000
export SAVE_CHECKPOINT=true

export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

export OPTIMIZER_LR=1e-5
export OPTIMIZER_VIT_LR=1e-5
export OPTIMIZER_CONNECTOR_LR=1e-5
export OPTIMIZER_ACTION_EXPERT_LR=1e-4
export SCHEDULER_WARMUP_STEPS=1000
export SCHEDULER_VLM_WARMUP_STEPS=1000
export SCHEDULER_VIT_WARMUP_STEPS=2000
export SCHEDULER_CONNECTOR_WARMUP_STEPS=1000
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000
export SCHEDULER_DECAY_STEPS=30000
export SCHEDULER_DECAY_LR=1e-6

cd "$M" || exit 1
TRAIN_LOG=${TRAIN_LOG:-/pfs/pfs-Tr4Uts/JM_work/abl_open_path.trainlog}
: > "$TRAIN_LOG"

setsid bash scripts/train_libero_molmoact2.sh \
  --policy.vlm_checkpoint_path="$M/Checkpoint/Molmo2-ER" \
  --policy.randomize_action_expert=true \
  --policy.audit_bootstrap=true \
  --policy.enable_goal_pose=true \
  --policy.goal_conditioning_mode=semantic_visual_recurrent \
  --policy.goal_token_source=learnable_queries \
  --policy.goal_hidden_dim=512 \
  --policy.num_goal_tokens=4 \
  --policy.init_queries_from_se3_encoder=false \
  --policy.enable_pose_reconstruction=true \
  --policy.pose_recon_loss_weight=0.3 \
  --policy.target_pose_delta_index=10 \
  --policy.mask_image_from_action_expert=false \
  --policy.num_semantic_visual_tokens=100 \
  --policy.num_semantic_visual_pose_tokens=8 \
  --policy.semantic_visual_num_layer_groups=6 \
  --policy.semantic_visual_hidden_dim=768 \
  --policy.semantic_visual_num_heads=8 \
  --policy.semantic_visual_ffn_ratio=4.0 \
  --policy.semantic_visual_enable_self_attention=true \
  --policy.semantic_visual_dropout=0.0 \
  --policy.optimizer_goal_lr=5e-5 \
  --policy.optimizer_semantic_visual_lr=1e-4 \
  --policy.scheduler_goal_warmup_steps=500 \
  --policy.scheduler_semantic_visual_warmup_steps=5000 \
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
echo "[audit] training launched pid=$TRAIN_PID pgid=$TRAIN_PGID eff_batch=$EFF gpus=$CUDA_VISIBLE_DEVICES log=$TRAIN_LOG"

audit_fail() {
  echo "[audit] FATAL: $1"
  [ -n "$TRAIN_PGID" ] && kill -TERM "-$TRAIN_PGID" 2>/dev/null; sleep 10
  [ -n "$TRAIN_PGID" ] && kill -9 "-$TRAIN_PGID" 2>/dev/null
  exit 3
}

DEADLINE=$(( $(date +%s) + 2400 ))
while :; do
  if grep -qF "Start offline training" "$TRAIN_LOG" 2>/dev/null; then
    sleep 30   # 让 visual audit 那行落盘
    python3 - "$TRAIN_LOG" <<'PY'
import sys, re
t = open(sys.argv[1], errors="replace").read()
ok = True
if "Missing key(s) when loading model" in t or "Unexpected key(s) when loading model" in t:
    print("[audit] FAIL: unexpected Missing/Unexpected-key path (this init route must not hit it)"); ok = False
if "clean bootstrap audit" not in t:
    print("[audit] FAIL: no 'clean bootstrap audit' line — VLM override / AE randomize did not run"); ok = False
if not re.search(r"'randomize_action_expert':\s*True", t):
    print("[audit] FAIL: randomize_action_expert not True in resolved config"); ok = False
if "--policy.mask_image_from_action_expert=false" not in t:
    print("[audit] FAIL: mask_image_from_action_expert=false not in launched command"); ok = False
m = re.search(r'"visual_inputs_present":\s*(true|false)', t)
if not m or m.group(1) != "true":
    print(f"[audit] FAIL: visual_inputs_present={m.group(1) if m else 'absent'}"); ok = False
fp = re.search(r'"action_expert_random_fingerprint":\s*"([0-9a-f]{16})', t)
print(f"[audit] bootstrap fingerprint={fp.group(1) if fp else '?'}... visual_inputs_present={m.group(1) if m else '?'}")
sys.exit(0 if ok else 5)
PY
    rc=$?
    if [ "$rc" -ne 0 ]; then audit_fail "load audit failed (rc=$rc)"; fi
    echo "[audit] OK: init route == arm ④ (MolmoAct2 + Molmo2-ER VLM + random AE), LIT stage-2 intact, mask=false"
    grep -hE "Effective batch size|num_learnable_params|num_total_params" "$TRAIN_LOG" | head -3
    break
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[audit] training died before the training loop; tail:"; tail -40 "$TRAIN_LOG"; exit 4
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then audit_fail "timed out waiting for training loop"; fi
  sleep 20
done

wait "$TRAIN_PID"; rc=$?
echo "[audit] training exited rc=$rc"
exit $rc

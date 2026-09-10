#!/usr/bin/env bash
# ============================================================================
# 消融臂 ⑤  abl_latent_only —— Latent interface only（A800j）
#
#   目的：只保留 latent 接口本身，看它能否单独达到完整 LIT 的泛化。
#   与 Full LIT 的差别是两处【空间监督】全部去掉，接口结构一字不动：
#     1) 完全跳过 Stage 1        → 与 baseline/④ 相同的初始化：MolmoAct2 权重
#                                  + Molmo2-ER 覆盖 VLM + 随机重置 AE，不加载任何 policy checkpoint
#     2) 去掉 Stage 2 的 pose 重建监督 → enable_pose_reconstruction=false
#                                  （SE(3) decoder 虽被建出但无梯度；终端位姿不作为输入也不作为监督）
#   保留：100 个 learnable latent、6 层组聚合器、维度/头数/FFN 全同；
#         mask_image_from_action_expert=true —— latent 仍是 AE 唯一的视觉通路。
#   训练目标：仅 flow-matching 动作损失（+ 原有 softmax 辅助项，与全族一致）。
#   可训练范围：backbone + AE + latent 接口，与 Full LIT Stage 2 相同。
#
#   步数：20000（用户 9-7 定；学习率 decay 覆盖全部 20000 步，与 ④ 的 decay 覆盖其 30000 步同理）。
#         save_freq=10000 → 落 10000 / 20000 两个 checkpoint。
#         注意：③④ 是 30000 步，本臂 20000 步，跨臂比较时需说明这一预算差。
#   有效 batch 128（8 卡 × 16），与 ④ 相同；六组学习率与 warmup 照抄 v3 stage2。
#
#   加载审计（与 ④ 相同的初始化路径）：
#     · 不得出现 Missing/Unexpected key（这条路径不走该分支）
#     · 必须有 clean bootstrap audit 且 randomize_action_expert=True
#     · 命令行必须含 mask_image_from_action_expert=true 与 enable_pose_reconstruction=false
#     · visual_inputs_present 必须为 true
# ============================================================================
set -uo pipefail
M=/data2/JM/Code/molmoact2
STEPS_OVERRIDE="${STEPS_OVERRIDE:-20000}"; OUT_SUFFIX="${OUT_SUFFIX:-}"; EXPECT_EFF="${EXPECT_EFF:-128}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .); EFF=$(( NGPU * BATCH_SIZE ))
[ "$EFF" -eq "$EXPECT_EFF" ] || { echo "FATAL: eff batch $NGPU x $BATCH_SIZE = $EFF != $EXPECT_EFF" >&2; exit 2; }
for f in "$M/Checkpoint/MolmoAct2/config.json" "$M/Checkpoint/Molmo2-ER/config.json"; do
  [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 2; }
done

export DATASET_ROOT=/data2/JM/dataset/libero_lerobot_format
export DATASET_REPO_ID=lerobot/libero
export CHECKPOINT_PATH=$M/Checkpoint/MolmoAct2
export POLICY_PATH=""                                # 空 → --policy.type + checkpoint_path（不加载任何 Stage1/policy ckpt）
export JOB_NAME=abl_latent_only RUN_ID=seed_1000 RESUME_MODE=off
export OUTPUT_DIR=$M/lerobot/outputs/libero_ablation/abl_latent_only${OUT_SUFFIX}
export STEPS=$STEPS_OVERRIDE
export SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-10000}"      # 保留 30000 步 ckpt；省磁盘
export LOG_FREQ=20 NUM_WORKERS=4 SEED=1000 SAVE_CHECKPOINT=true
export ACTION_MODE=continuous TRAIN_ACTION_EXPERT_ONLY=false DISABLE_VISUAL_INPUT=false IMAGE_TRANSFORMS_ENABLE=true
export OPTIMIZER_LR=1e-5 OPTIMIZER_VIT_LR=1e-5 OPTIMIZER_CONNECTOR_LR=1e-5 OPTIMIZER_ACTION_EXPERT_LR=1e-4
export SCHEDULER_WARMUP_STEPS=1000 SCHEDULER_VLM_WARMUP_STEPS=1000 SCHEDULER_VIT_WARMUP_STEPS=2000
export SCHEDULER_CONNECTOR_WARMUP_STEPS=1000 SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000
export SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-$STEPS_OVERRIDE}"   # 覆盖全部训练步数
export SCHEDULER_DECAY_LR=1e-6

cd "$M" || exit 1
TRAIN_LOG=${TRAIN_LOG:-/data2/JM/abl_latent_only.trainlog}; : > "$TRAIN_LOG"
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
  --policy.enable_pose_reconstruction=false \
  --policy.pose_recon_loss_weight=0.3 \
  --policy.target_pose_delta_index=10 \
  --policy.mask_image_from_action_expert=true \
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
  --policy.optimizer_weight_decay=0.0 --policy.optimizer_grad_clip_norm=1.0 --policy.optimizer_eps=1e-6 \
  --policy.softmax_auxiliary_loss=true --policy.softmax_auxiliary_loss_scale=1e-4 \
  --policy.num_state_tokens=256 --policy.expected_max_action_dim=32 --policy.mask_action_dim_padding=true \
  --policy.normalize_language=true --policy.chunk_size=10 --policy.n_action_steps=10 \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!; TRAIN_PGID=$(ps -o pgid= -p "$TRAIN_PID" 2>/dev/null | tr -d ' ')
echo "[audit] training launched pid=$TRAIN_PID pgid=$TRAIN_PGID eff_batch=$EFF steps=$STEPS_OVERRIDE log=$TRAIN_LOG"
if [ "${DRY_RUN:-false}" = "true" ]; then wait "$TRAIN_PID"; echo "[dry-run] exit $?"; exit 0; fi
audit_fail() { echo "[audit] FATAL: $1"; [ -n "$TRAIN_PGID" ] && kill -TERM "-$TRAIN_PGID" 2>/dev/null; sleep 10; [ -n "$TRAIN_PGID" ] && kill -9 "-$TRAIN_PGID" 2>/dev/null; exit 3; }
DEADLINE=$(( $(date +%s) + 2400 ))
while :; do
  if grep -qF "Start offline training" "$TRAIN_LOG" 2>/dev/null; then
    sleep 30
    python3 - "$TRAIN_LOG" <<'PY'
import sys, re
t = open(sys.argv[1], errors="replace").read(); ok = True
if "Missing key(s) when loading model" in t or "Unexpected key(s) when loading model" in t:
    print("[audit] FAIL: unexpected key-mismatch path"); ok = False
if "clean bootstrap audit" not in t:
    print("[audit] FAIL: no clean bootstrap audit (VLM override / AE randomize did not run)"); ok = False
if not re.search(r"'randomize_action_expert':\s*True", t):
    print("[audit] FAIL: randomize_action_expert not True"); ok = False
for flag in ("--policy.mask_image_from_action_expert=true", "--policy.enable_pose_reconstruction=false"):
    if flag not in t: print(f"[audit] FAIL: {flag} not in launched command"); ok = False
m = re.search(r'"visual_inputs_present":\s*(true|false)', t)
if not m or m.group(1) != "true": print(f"[audit] FAIL: visual_inputs_present={m.group(1) if m else 'absent'}"); ok = False
fp = re.search(r'"action_expert_random_fingerprint":\s*"([0-9a-f]{16})', t)
print(f"[audit] bootstrap fingerprint={fp.group(1) if fp else '?'}... visual={m.group(1) if m else '?'}")
sys.exit(0 if ok else 5)
PY
    rc=$?; [ "$rc" -ne 0 ] && audit_fail "load audit failed rc=$rc"
    echo "[audit] OK: no Stage-1 (random AE, same init as arm ④), full latent interface, firewall on, pose loss off"
    grep -hE "Effective batch size|num_learnable_params|num_total_params" "$TRAIN_LOG" | head -3
    break
  fi
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "[audit] training died early; tail:"; tail -30 "$TRAIN_LOG"; exit 4; }
  [ "$(date +%s)" -gt "$DEADLINE" ] && audit_fail "timeout waiting for training loop"
  sleep 20
done
wait "$TRAIN_PID"; rc=$?; echo "[audit] training exited rc=$rc"; exit $rc

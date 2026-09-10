#!/usr/bin/env bash
# ============================================================================
# 消融臂 A1  abl_ae_pose_head —— baseline + pose head（fm1）
#
#   目的：关掉「LIT 的增益只是多了个好用的辅助任务」这条替代解释。
#   架构与 baseline 完全一致：直连视觉条件、无 latent 接口、无防火墙、无 Stage 1。
#   唯一新增：把 action expert 末层 hidden 在动作块位置与 flow timestep 上池化，
#             过一个三层 GELU MLP（768→512→512→8）回归同一个末端 SE(3)，权重 λ=0.3。
#   MLP 形状与 LIT 的 pose decoder 完全相同（num_tokens=1），保证两个头容量可比。
#
#   预算：20000 步（用户 9-8 定），fm1 七卡 × bs16 = 有效 112，与 ②③⑤ 同族；
#         学习率与 warmup 照抄 v3 stage2；decay 覆盖全部 20000 步。
#
#   审计：
#     · 初始化路线与 ④/⑤ 相同（clean bootstrap，随机 AE，无 Missing/Unexpected key）
#     · 命令行含 enable_ae_pose_head=true、enable_goal_pose=false、mask_image=false
#     · 可学参数量 = 无接口 baseline 架构 + 660,488（头确实进了可训练集合）
#       头名为 action_expert_pose_head → 落入 action_expert 优化器组，lr 1e-4、warmup 5000，
#       与 LIT 的 pose decoder（semantic_visual 组，同为 1e-4）同档，避免欠训质疑
# ============================================================================
set -uo pipefail
M=/pfs/pfs-Tr4Uts/JM_work/molmoact2
STEPS_OVERRIDE="${STEPS_OVERRIDE:-20000}"; OUT_SUFFIX="${OUT_SUFFIX:-}"; EXPECT_EFF="${EXPECT_EFF:-112}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6,7}"   # 排除故障 GPU3
export BATCH_SIZE="${BATCH_SIZE:-16}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .); EFF=$(( NGPU * BATCH_SIZE ))
[ "$EFF" -eq "$EXPECT_EFF" ] || { echo "FATAL: eff batch $NGPU x $BATCH_SIZE = $EFF != $EXPECT_EFF" >&2; exit 2; }
case ",$CUDA_VISIBLE_DEVICES," in *,3,*) echo "FATAL: GPU3 is faulty on fm1" >&2; exit 2;; esac
for f in "$M/Checkpoint/MolmoAct2/config.json" "$M/Checkpoint/Molmo2-ER/config.json"; do
  [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 2; }
done

export DATASET_ROOT=/pfs/pfs-Tr4Uts/JM_work/dataset/libero_lerobot_format
export DATASET_REPO_ID=lerobot/libero
export CHECKPOINT_PATH=$M/Checkpoint/MolmoAct2
export POLICY_PATH=""
export JOB_NAME=abl_ae_pose_head RUN_ID=seed_1000 RESUME_MODE=off
export OUTPUT_DIR=$M/lerobot/outputs/libero_ablation/abl_ae_pose_head${OUT_SUFFIX}
export STEPS=$STEPS_OVERRIDE
export SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-10000}"
export LOG_FREQ=20 NUM_WORKERS=4 SEED=1000 SAVE_CHECKPOINT=true
export ACTION_MODE=continuous TRAIN_ACTION_EXPERT_ONLY=false DISABLE_VISUAL_INPUT=false IMAGE_TRANSFORMS_ENABLE=true
export OPTIMIZER_LR=1e-5 OPTIMIZER_VIT_LR=1e-5 OPTIMIZER_CONNECTOR_LR=1e-5 OPTIMIZER_ACTION_EXPERT_LR=1e-4
export SCHEDULER_WARMUP_STEPS=1000 SCHEDULER_VLM_WARMUP_STEPS=1000 SCHEDULER_VIT_WARMUP_STEPS=2000
export SCHEDULER_CONNECTOR_WARMUP_STEPS=1000 SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000
export SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-$STEPS_OVERRIDE}"
export SCHEDULER_DECAY_LR=1e-6

cd "$M" || exit 1
TRAIN_LOG=${TRAIN_LOG:-/pfs/pfs-Tr4Uts/JM_work/abl_ae_pose_head.trainlog}; : > "$TRAIN_LOG"
setsid bash scripts/train_libero_molmoact2.sh \
  --policy.vlm_checkpoint_path="$M/Checkpoint/Molmo2-ER" \
  --policy.randomize_action_expert=true \
  --policy.audit_bootstrap=true \
  --policy.enable_goal_pose=false \
  --policy.enable_pose_reconstruction=false \
  --policy.mask_image_from_action_expert=false \
  --policy.enable_ae_pose_head=true \
  --policy.pose_recon_loss_weight=0.3 \
  --policy.target_pose_delta_index=10 \
  --policy.goal_hidden_dim=512 \
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
  if grep -qE "step:[0-9]" "$TRAIN_LOG" 2>/dev/null || grep -qF "Start offline training" "$TRAIN_LOG" 2>/dev/null; then
    sleep 90
    python3 - "$TRAIN_LOG" <<'PY'
import sys, re
t = open(sys.argv[1], errors="replace").read(); ok = True
if "Missing key(s) when loading model" in t or "Unexpected key(s) when loading model" in t:
    print("[audit] FAIL: unexpected key-mismatch path"); ok = False
if "clean bootstrap audit" not in t:
    print("[audit] FAIL: no clean bootstrap audit"); ok = False
for flag in ("--policy.enable_ae_pose_head=true", "--policy.enable_goal_pose=false",
             "--policy.mask_image_from_action_expert=false"):
    if flag not in t: print(f"[audit] FAIL: {flag} not in launched command"); ok = False
m = re.search(r'"visual_inputs_present":\s*(true|false)', t)
if not m or m.group(1) != "true": print(f"[audit] FAIL: visual_inputs_present={m.group(1) if m else 'absent'}"); ok = False
# 头是否真的建进了可训练集合：参数量必须精确等于「无接口 baseline 架构 + 头」
# 注：forward 返回的 metrics 不进训练日志（v3 的 pose_recon_loss 同样从未出现过），
#     所以不能用「日志里有无损失字段」判断，必须用参数量。
EXPECTED = 5046031088 + 660488   # ① stagewise stage2 的无接口架构 + 三层 MLP 头
m2 = re.search(r"num_learnable_params=(\d+)", t)
if not m2:
    print("[audit] FAIL: num_learnable_params not found"); ok = False
elif int(m2.group(1)) != EXPECTED:
    print(f"[audit] FAIL: num_learnable_params={m2.group(1)} != {EXPECTED} (head missing or arch changed)"); ok = False
else:
    print(f"[audit] num_learnable_params={m2.group(1)} = baseline-arch + 660,488 (head present)")
print(f"[audit] visual_inputs_present={m.group(1) if m else '?'}")
sys.exit(0 if ok else 5)
PY
    rc=$?; [ "$rc" -ne 0 ] && audit_fail "load/head audit failed rc=$rc"
    echo "[audit] OK: baseline architecture + AE pose head active"
    grep -hE "Effective batch size|num_learnable_params|num_total_params" "$TRAIN_LOG" | head -3
    break
  fi
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "[audit] training died early; tail:"; tail -30 "$TRAIN_LOG"; exit 4; }
  [ "$(date +%s)" -gt "$DEADLINE" ] && audit_fail "timeout waiting for training loop"
  sleep 20
done
wait "$TRAIN_PID"; rc=$?; echo "[audit] training exited rc=$rc"; exit $rc

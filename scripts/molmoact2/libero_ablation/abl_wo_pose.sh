#!/usr/bin/env bash
# ============================================================================
# 消融臂 ③  abl_wo_pose —— 去掉 Stage 2 的 pose 重建监督（fm1 版）
#   = Full LIT 只关一个开关：enable_pose_reconstruction  true -> false
#   保留：v3 的 SE(3) Stage 1 先验（--policy.path）、100 latent 聚合、图像防火墙(mask=true)。
#   问的是：增益来自「有个窄瓶颈」，还是来自「这个瓶颈被 pose 监督着」。
#   预算：fm1 7 卡 × bs16 = 有效 112（与 ② 相同；GPU3 故障排除）。学习率/warmup 照抄 v3 stage2。
#   加载审计（与 v3 stage2 加载 Stage 1 的模式一致）：
#     unexpected 恰好 = goal_se3_encoder.net.{0,2,4}.{weight,bias} 6 个；missing 全部 semantic_visual_*
# ============================================================================
set -uo pipefail
M=/pfs/pfs-Tr4Uts/JM_work/molmoact2
STEPS_OVERRIDE="${STEPS_OVERRIDE:-30000}"; OUT_SUFFIX="${OUT_SUFFIX:-}"; EXPECT_EFF="${EXPECT_EFF:-112}"
STAGE1_CKPT="${STAGE1_CKPT:-/pfs/pfs-Tr4Uts/JM_work/ckpt/libero_goal_prior_v3_stage1_010000/pretrained_model}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6,7}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .); EFF=$(( NGPU * BATCH_SIZE ))
[ "$EFF" -eq "$EXPECT_EFF" ] || { echo "FATAL: eff batch $NGPU x $BATCH_SIZE = $EFF != $EXPECT_EFF" >&2; exit 2; }
case ",$CUDA_VISIBLE_DEVICES," in *,3,*) echo "FATAL: GPU3 is faulty on fm1" >&2; exit 2;; esac
[ -f "$STAGE1_CKPT/config.json" ] && [ -f "$STAGE1_CKPT/model.safetensors" ] || { echo "FATAL: stage1 ckpt missing: $STAGE1_CKPT" >&2; exit 2; }
python3 - "$STAGE1_CKPT/config.json" <<'PY' || exit 2
import json, sys; c = json.load(open(sys.argv[1]))
print(f"[pre-audit] stage1 enable_goal_pose={c.get('enable_goal_pose')} goal_token_source={c.get('goal_token_source')}")
sys.exit(0 if (c.get("enable_goal_pose") is True and c.get("goal_token_source") == "se3_encoder") else 3)
PY
export DATASET_ROOT=/pfs/pfs-Tr4Uts/JM_work/dataset/libero_lerobot_format
export DATASET_REPO_ID=lerobot/libero
export POLICY_PATH="$STAGE1_CKPT"
export JOB_NAME=abl_wo_pose RUN_ID=seed_1000 RESUME_MODE=off
export OUTPUT_DIR=$M/lerobot/outputs/libero_ablation/abl_wo_pose${OUT_SUFFIX}
export STEPS=$STEPS_OVERRIDE SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-5000}" LOG_FREQ=20 NUM_WORKERS=4 SEED=1000 SAVE_CHECKPOINT=true
export ACTION_MODE=continuous TRAIN_ACTION_EXPERT_ONLY=false DISABLE_VISUAL_INPUT=false IMAGE_TRANSFORMS_ENABLE=true
export OPTIMIZER_LR=1e-5 OPTIMIZER_VIT_LR=1e-5 OPTIMIZER_CONNECTOR_LR=1e-5 OPTIMIZER_ACTION_EXPERT_LR=1e-4
export SCHEDULER_WARMUP_STEPS=1000 SCHEDULER_VLM_WARMUP_STEPS=1000 SCHEDULER_VIT_WARMUP_STEPS=2000 SCHEDULER_CONNECTOR_WARMUP_STEPS=1000 SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000
export SCHEDULER_DECAY_STEPS=30000 SCHEDULER_DECAY_LR=1e-6
cd "$M" || exit 1
TRAIN_LOG=${TRAIN_LOG:-/pfs/pfs-Tr4Uts/JM_work/abl_wo_pose.trainlog}; : > "$TRAIN_LOG"
setsid bash scripts/train_libero_molmoact2.sh \
  --policy.enable_goal_pose=true --policy.goal_conditioning_mode=semantic_visual_recurrent \
  --policy.goal_token_source=learnable_queries --policy.goal_hidden_dim=512 --policy.num_goal_tokens=4 \
  --policy.init_queries_from_se3_encoder=false \
  --policy.enable_pose_reconstruction=false \
  --policy.pose_recon_loss_weight=0.3 \
  --policy.target_pose_delta_index=10 --policy.mask_image_from_action_expert=true \
  --policy.num_semantic_visual_tokens=100 --policy.num_semantic_visual_pose_tokens=8 \
  --policy.semantic_visual_num_layer_groups=6 --policy.semantic_visual_hidden_dim=768 --policy.semantic_visual_num_heads=8 \
  --policy.semantic_visual_ffn_ratio=4.0 --policy.semantic_visual_enable_self_attention=true --policy.semantic_visual_dropout=0.0 \
  --policy.optimizer_goal_lr=5e-5 --policy.optimizer_semantic_visual_lr=1e-4 \
  --policy.scheduler_goal_warmup_steps=500 --policy.scheduler_semantic_visual_warmup_steps=5000 \
  --policy.optimizer_weight_decay=0.0 --policy.optimizer_grad_clip_norm=1.0 --policy.optimizer_eps=1e-6 \
  --policy.softmax_auxiliary_loss=true --policy.softmax_auxiliary_loss_scale=1e-4 \
  --policy.num_state_tokens=256 --policy.expected_max_action_dim=32 --policy.mask_action_dim_padding=true --policy.normalize_language=true \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!; TRAIN_PGID=$(ps -o pgid= -p "$TRAIN_PID" 2>/dev/null | tr -d ' ')
echo "[audit] training launched pid=$TRAIN_PID pgid=$TRAIN_PGID eff_batch=$EFF gpus=$CUDA_VISIBLE_DEVICES log=$TRAIN_LOG"
if [ "${DRY_RUN:-false}" = "true" ]; then wait "$TRAIN_PID"; echo "[dry-run] exit $?"; exit 0; fi
audit_fail() { echo "[audit] FATAL: $1"; [ -n "$TRAIN_PGID" ] && kill -TERM "-$TRAIN_PGID" 2>/dev/null; sleep 10; [ -n "$TRAIN_PGID" ] && kill -9 "-$TRAIN_PGID" 2>/dev/null; exit 3; }
DEADLINE=$(( $(date +%s) + 2400 ))
while :; do
  if grep -qF "Start offline training" "$TRAIN_LOG" 2>/dev/null; then
    sleep 30
    python3 - "$TRAIN_LOG" <<'PY'
import re, sys, ast
t = open(sys.argv[1], errors="replace").read()
def grab(tag):
    m = re.search(tag + r" when loading model: (.*)", t)
    if not m: return None
    try: return set(ast.literal_eval(m.group(1).strip()))
    except Exception: return set(re.findall(r"'([^']+)'", m.group(1)))
miss, unexp = grab("Missing key\\(s\\)"), grab("Unexpected key\\(s\\)")
want = {f"goal_se3_encoder.net.{i}.{w}" for i in (0, 2, 4) for w in ("weight", "bias")}
ok = True
if unexp != want: print(f"[audit] FAIL unexpected != 6 SE3 tensors: {sorted(unexp or [])[:8]}"); ok = False
if not miss or not all(k.startswith("semantic_visual_") for k in miss): print(f"[audit] FAIL missing not all semantic_visual_*: {sorted(k for k in (miss or []) if not k.startswith('semantic_visual_'))[:6]}"); ok = False
# 注：pose_recon=false 时代码仍会建 pose decoder（模块闲置，损失在 modeling 第 2990 行短路），missing 里出现 pose_decoder 键是正常的
if "--policy.enable_pose_reconstruction=false" not in t or "--policy.mask_image_from_action_expert=true" not in t: print("[audit] FAIL switches not in launched command"); ok = False
m = re.search(r'"visual_inputs_present":\s*(true|false)', t)
if not m or m.group(1) != "true": print(f"[audit] FAIL visual_inputs_present={m.group(1) if m else 'absent'}"); ok = False
print(f"[audit] missing={len(miss or [])} unexpected={len(unexp or [])} visual={m.group(1) if m else '?'}")
sys.exit(0 if ok else 5)
PY
    rc=$?; [ "$rc" -ne 0 ] && audit_fail "load audit failed rc=$rc"
    echo "[audit] OK: v3 Stage1 init, LIT stage2, mask=true, pose_recon=false (decoder built but loss short-circuited)"
    grep -hE "Effective batch size|num_learnable_params|num_total_params" "$TRAIN_LOG" | head -3; break
  fi
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "[audit] training died early; tail:"; tail -30 "$TRAIN_LOG"; exit 4; }
  [ "$(date +%s)" -gt "$DEADLINE" ] && audit_fail "timeout waiting for training loop"
  sleep 20
done
wait "$TRAIN_PID"; rc=$?; echo "[audit] training exited rc=$rc"; exit $rc

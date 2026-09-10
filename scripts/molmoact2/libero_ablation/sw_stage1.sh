#!/usr/bin/env bash
# ============================================================================
# 消融臂 A —— Stagewise 对照，Stage 1
#
#   目的：消融 Latent Interface 的全部三件套。保留"两阶段日程"这一外壳，
#         但把末端 SE(3) 的输入与监督整个拿掉。
#
#   与 ours(v3) Stage 1 的唯一差别：
#         enable_goal_pose  True -> False
#   即 AE 的 warmup 只看 language + state，不看末端 SE(3)。
#   其余（无视觉、冻结 VLM、随机 AE、Molmo2-ER 初始化、10k 步、bs128×7卡）
#   与 v3 Stage 1 逐字段一致。
#
#   所有超参取自 v3 stage1 的 checkpoint train_config.json，并在此显式钉死，
#   不依赖 train_libero_molmoact2.sh 的默认值（防止日后默认值漂移）。
# ============================================================================
set -uo pipefail
M=/data2/JM/Code/molmoact2

STEPS_OVERRIDE="${STEPS_OVERRIDE:-10000}"
OUT_SUFFIX="${OUT_SUFFIX:-}"

# --- 8 卡 × bs128 = 有效 1024，与 v3 Stage 1 实跑一致（脚本默认写 7 卡，实跑是 8 卡）---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATASET_ROOT=/data2/JM/dataset/libero_lerobot_format
export DATASET_REPO_ID=local/libero_lerobot_format   # 与 v3 stage1 一致（root 已给，仅作标签）
export CHECKPOINT_PATH=$M/Checkpoint/MolmoAct2
export POLICY_PATH=""                                # 空 → 走 checkpoint_path + vlm_checkpoint_path
export JOB_NAME=sw_stagewise_stage1
export RUN_ID=seed_1000
export OUTPUT_DIR=$M/lerobot/outputs/libero_ablation/sw_stagewise${OUT_SUFFIX}/stage1
export RESUME_MODE=off

export STEPS=$STEPS_OVERRIDE
export BATCH_SIZE=128
export SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-2500}"
export LOG_FREQ=20
export NUM_WORKERS=4
export SEED=1000
export SAVE_CHECKPOINT=true

# --- Stage 1 制度：无视觉、只训 AE ------------------------------------------
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=true
export DISABLE_VISUAL_INPUT=true
export IMAGE_TRANSFORMS_ENABLE=false

# --- v3 Stage 1 的学习率与 warmup（注意与 Stage 2 不同）---------------------
export OPTIMIZER_LR=1e-5
export OPTIMIZER_VIT_LR=5e-6
export OPTIMIZER_CONNECTOR_LR=5e-6
export OPTIMIZER_ACTION_EXPERT_LR=5e-5
export SCHEDULER_WARMUP_STEPS=1000
export SCHEDULER_VLM_WARMUP_STEPS=1000
export SCHEDULER_VIT_WARMUP_STEPS=1000
export SCHEDULER_CONNECTOR_WARMUP_STEPS=1000
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=500
export SCHEDULER_DECAY_STEPS=10000                   # 固定 10000，不随 STEPS_OVERRIDE 变
export SCHEDULER_DECAY_LR=1e-6

cd "$M" || exit 1
bash scripts/train_libero_molmoact2.sh \
  --policy.vlm_checkpoint_path="$M/Checkpoint/Molmo2-ER" \
  --policy.randomize_action_expert=true \
  --policy.audit_bootstrap=true \
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
  --policy.normalize_language=true

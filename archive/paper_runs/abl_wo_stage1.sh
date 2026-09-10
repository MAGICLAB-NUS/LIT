#!/usr/bin/env bash
# 消融臂 ①  abl_wo_stage1
#   = baseline 的初始化（VLM 预训练权重 + 随机 AE）+ ours 的 Stage 2 训练配置
# 所有超参取自 v3 stage2 的 checkpoint train_config.json（非脚本默认值）。
set -uo pipefail
M=/data2/JM/Code/molmoact2

STEPS_OVERRIDE="${STEPS_OVERRIDE:-30000}"
OUT_SUFFIX="${OUT_SUFFIX:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7        # 8 卡
export DATASET_ROOT=/data2/JM/dataset/libero_lerobot_format
export DATASET_REPO_ID=lerobot/libero              # 与 v3 stage2 一致
export CHECKPOINT_PATH=$M/Checkpoint/MolmoAct2
export POLICY_PATH=""                              # 留空 → 用 --policy.type + checkpoint_path
export JOB_NAME=abl_wo_stage1
export RUN_ID=seed_1000
export OUTPUT_DIR=$M/lerobot/outputs/libero_ablation/abl_wo_stage1${OUT_SUFFIX}
export RESUME_MODE=off

export STEPS=$STEPS_OVERRIDE
export BATCH_SIZE=16                               # 每进程 → 有效 128
export SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-5000}"
export LOG_FREQ=20
export NUM_WORKERS=4
export SEED=1000
export SAVE_CHECKPOINT=true

export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export OPTIMIZER_LR=1e-5
export OPTIMIZER_VIT_LR=1e-5
export OPTIMIZER_CONNECTOR_LR=1e-5
export OPTIMIZER_ACTION_EXPERT_LR=1e-4
export SCHEDULER_WARMUP_STEPS=1000
export SCHEDULER_VLM_WARMUP_STEPS=1000
export SCHEDULER_VIT_WARMUP_STEPS=2000
export SCHEDULER_CONNECTOR_WARMUP_STEPS=1000
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000
export SCHEDULER_DECAY_STEPS=30000                 # 固定 30000，不随 STEPS_OVERRIDE 变
export SCHEDULER_DECAY_LR=1e-6

cd "$M" || exit 1
bash scripts/train_libero_molmoact2.sh \
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
  --policy.optimizer_weight_decay=0.0 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.optimizer_eps=1e-6 \
  --policy.softmax_auxiliary_loss=true \
  --policy.softmax_auxiliary_loss_scale=1e-4 \
  --policy.num_state_tokens=256 \
  --policy.expected_max_action_dim=32 \
  --policy.mask_action_dim_padding=true \
  --policy.normalize_language=true

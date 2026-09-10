# 目标位姿动作先验（两阶段）— v1

把“怎么动（how to move）”与“去哪动（where to move）”解耦：Stage 1 学习一个
无视觉、由目标位姿条件化的动作先验；Stage 2 从视觉中聚合 Goal-Pose Tokens
（由 VLM 上下文化的可学习 query），通过同一套 K-token 接口去 steer 这个先验，
并对 action expert 屏蔽掉原始图像 token。

共享接口：K 个 goal token 被追加到 VLM 输入序列末尾（在 language + state 之后），
action expert 通过其逐层 cross-attention 读取它们。Stage 1 的 goal token 由一个
SE(3) 编码器从 chunk 末端目标状态（`observation.state` 在 `t + chunk_size` 处）
产生；Stage 2 由可学习 query 产生，并用一个位姿重建 decoder 做监督。

## Stage 1 — 无视觉先验

```bash
bash scripts/libero_goal_prior/train_stage1.sh
```

- 无图像（`disable_visual_input=true`），VLM 冻结（`train_action_expert_only=true`）。
- Clean bootstrap（默认）：Molmo2-ER 的 VLM 权重 + 随机初始化的 action expert
  （不继承官方 MolmoAct2 的 action-expert 权重）。用 `USE_ER_BOOTSTRAP=false` 关闭。
- 可训练：action expert + goal SE(3) 编码器。损失：仅 flow matching。
- goal token 来源：`se3_encoder`；目标 = `observation.state` 在 `t + chunk_size` 处。

## Stage 2 — 视觉 goal-token steering

```bash
bash scripts/libero_goal_prior/train_stage2.sh
```

- 加载 Stage-1 checkpoint（`POLICY_PATH` 默认指向 Stage-1 的 `checkpoints/last/pretrained_model`）。
- 开启视觉；goal token 来源：`learnable_queries`；对 action expert 屏蔽原始图像。
- decoder 重建目标位姿（`L_pose`）；损失 = flow + `pose_recon_loss_weight * L_pose`。
- VLM + action expert 以全量学习率联合训练（不下调）；ViT 使用更长的 warmup；
  `goal` 参数组有自己独立的 LR/warmup。

## Baseline — 普通 MolmoAct2 训练（对照组）

```bash
bash scripts/libero_goal_prior/train_baseline.sh
```

- 与两阶段实验相同的 clean bootstrap：Molmo2-ER 的 VLM 权重 + 随机初始化的
  action expert（`USE_ER_BOOTSTRAP=false` 可关闭，退回加载官方 MolmoAct2 AE）。
- **不带任何 goal-pose**：开视觉、全模型联合训练、损失仅 flow matching。
- 默认 `STEPS=40000`（= stage1 10k + stage2 30k，与实验组总步数对齐）、
  `BATCH_SIZE=32`/卡、seed 1000、GPU 0-6，输出到
  `outputs/libero_goal_prior/seed_1000/libero_baseline`。
- 无需 `create_canonical_init`：直接由 `train_libero_molmoact2.sh` 传 bootstrap
  参数完成 ER + 随机 AE 的初始化。

## 说明

- v1 的位姿目标是归一化后的 chunk 末端 `observation.state` 向量（8 维：
  `[pos(3), axis-angle(3), gripper(2)]`），用 MSE 重建。几何化的 6D 旋转目标是
  未来（v2）的改进项。
- 两个阶段都使用 `action_mode=continuous`、seed 1000、GPU 0-6、`chunk_size=10`。
- 自动 resume 继承自 `scripts/train_libero_molmoact2.sh`。

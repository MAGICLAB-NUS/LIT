# Goal-Pose Prior v2 — 实现说明

## 与 v1 的关系

- Stage1 完全不变：无视觉，`SE3Encoder(s_{t+10}) -> 4 goal tokens`，训练 AE
  motion prior。
- Stage2 直接加载 Stage1 自包含 checkpoint；VLM/AE 权重完整继承。
- v1 路径由 `goal_conditioning_mode=vlm_appended` 保留，旧 checkpoint 和脚本
  可继续复现。
- v2 路径为 `goal_conditioning_mode=semantic_visual_recurrent`。

## v2 逐层数据流

令 `H_l ∈ R^(B×S×2560)` 为 VLM 第 `l` 层 hidden，`Q_0` 为 100 个 768-D
learnable queries。共享聚合器在每层重复：

```text
Q'_l = SemanticCrossAttention(Q_{l-1}, H_l[language/state])
Q_l  = VisualCrossAttention(Q'_l, H_l[image patches])
```

language/state hidden 允许已经融合高级视觉语义。image context 只选
`image_patch_id` / `image_low_res_id` 对应的真实视觉特征位置。

`Q_l` 经两个线性层投影为 1024-D synthetic key/value，追加到当前 VLM layer
KV。AE 仍读取原 language/state KV；原 image patch KV 按 v1 mask 隐藏。AE 只单向
读取 `Q_l`，action hidden 不写回 query；`Q_l` 递归进入下一层。

最后一层 `Q_36` 经 learned attention pooling 与 MLP 解码 normalized 8D state：
`[position(3), axis-angle(3), gripper(2)]`。训练目标：

```text
L = L_flow + pose_recon_loss_weight * L_pose
```

## 参数与 checkpoint

模块按实际阶段按需创建，不保留无关网络：

- Stage1：只创建 `goal_se3_encoder`；
- v1 Stage2：只创建 `goal_queries`，并在开启位姿监督时创建
  `goal_pose_decoder`；
- v2 Stage2：只创建 `semantic_visual_aggregator` 与
  `semantic_visual_pose_decoder`。

现有 Stage1 checkpoint 由旧代码保存，可能包含随机且未训练的 legacy
`goal_queries/goal_pose_decoder`。Stage1 -> v2 加载使用 LeRobot `strict=False`：
VLM/AE 正常加载，多余 legacy keys 被忽略，新 `semantic_visual_*` 参数随机初始化。

## Optimizer / scheduler

- `semantic_visual` 是独立参数组；
- AE 保持全 LR `1e-5`，不做降速；
- semantic module LR `1e-5`；
- warmup：VLM/connector/AE/semantic 1000，ViT 2000；
- 主跑 30k，5k 保存，优先评测 5k/10k/15k。

## 关键实现文件

- `lerobot/src/lerobot/policies/molmoact2/configuration_molmoact2.py`
- `lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py`
- `lerobot/tests/policies/molmoact2/test_molmoact2.py`
- `scripts/libero_goal_prior_v2/train_stage2.sh`
- `scripts/libero_eval/eval_libero_v2_checkpoint.sh`

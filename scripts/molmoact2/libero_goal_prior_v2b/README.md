# Goal-Pose Prior v2b (scheme-2a)

v2b 把 100 个 latent token 拆成两组功能特化的 token，并把 36 个 VLM layer
划分为 6 个连续 depth groups（每组覆盖 6 层）：

- **pose 组**：前 `num_semantic_visual_pose_tokens`（默认 8）个 token，专门聚合
  目标位姿。仅这一组被 `L_pose` 监督，读出头换回 v1 已验证的 **concat 头**
  （`8×768 -> MLP -> 8D`），不再走单向量 attention pooling。
- **context 组**：其余 92 个 token，聚合语义-视觉上下文。

每个 depth group 拥有独立的 global latent self-attention、semantic cross-attention、
visual cross-attention 和 latent→AE K/V projection，并在组内 6 层复用。每层顺序为：

```text
Q_shared = SelfAttention(Q_{l-1})                 # 100 token 全局通信
Q_sem    = CrossAttention(Q_shared, language/state)
Q_l      = CrossAttention(Q_sem, image patches)
```

token 在 36 层连续递归，组边界不重置。两组都把 synthetic KV 拼进 AE（pose 组既被
监督、又条件化动作；AE 同时保留 92 个 context token，避免 v1 的狭窄条件瓶颈）。

## 与 v2 的差异

| 项 | v2 | v2b (scheme-2a) |
| --- | --- | --- |
| pose 读出 | 100 token 单向量 attention pool -> MLP | 前 8 个 pose token LayerNorm + concat -> MLP |
| L_pose 监督对象 | 全部 100 token | 仅前 8 个 pose token |
| token 间通信 | 无直接 self-attention | 每层全局 self-attention |
| 深度参数组织 | 1 套跨 36 层共享 | 6 套，每套覆盖连续 6 层 |
| AE 条件 | 全部 100 token | 全部 100 token（不变） |
| `pose_recon_loss_weight` | 1.0 | 0.3 |
| 输出目录 | `libero_goal_prior_v2` | `libero_goal_prior_v2b` |

## 训练

```bash
bash scripts/libero_goal_prior_v2b/train_stage2.sh
```

默认：

- Stage1 checkpoint：
  `lerobot/outputs/libero_goal_prior/seed_1000/stage1/checkpoints/010000/pretrained_model`
- 输出：`lerobot/outputs/libero_goal_prior_v2b/seed_1000/stage2`
- 7 GPU，`BATCH_SIZE=32`/卡，`STEPS=30000`，每 5000 步保存；
- `NUM_SEMANTIC_VISUAL_POSE_TOKENS=8`，`POSE_RECON_LOSS_WEIGHT=0.3`；
- `SEMANTIC_VISUAL_ENABLE_SELF_ATTENTION=true`，
  `SEMANTIC_VISUAL_NUM_LAYER_GROUPS=6`；
- VLM / ViT / connector / AE / semantic-visual module 均为 `1e-5`；
- warmup：VLM/connector/AE/semantic=1000，ViT=2000。
- 完整终端输出会同时追加到
  `lerobot/outputs/libero_goal_prior_v2b/seed_1000/stage2.console.log`；
  可通过 `LOG_FILE=/custom/path.log` 覆盖。

显存 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=1 STEPS=2 SAVE_CHECKPOINT=false \
OUTPUT_DIR=lerobot/outputs/libero_goal_prior_v2b/smoke \
bash scripts/libero_goal_prior_v2b/train_stage2.sh
```

## 评测

```bash
bash scripts/libero_eval/eval_libero_v2b_checkpoint.sh
```

默认评测 25k checkpoint，使用 GPU7，每 task 32 rollouts，并保存全部视频；
实时准确率写入各 suite 目录的 `realtime_accuracy.json`。

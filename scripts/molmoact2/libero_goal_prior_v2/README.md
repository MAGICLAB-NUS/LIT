# Goal-Pose Prior v2

v2 保留 v1 的 Stage1，不再把 4 个 learnable goal queries 追加进 VLM causal
序列。Stage2 新增 100 个 768-D latent tokens，并在 36 个 VLM/AE 层间递归：

1. `Q_l` cross-attend 当前层 vision-fused language/state hidden；
2. 再 cross-attend 当前层 image patch hidden；
3. AE 第 `l` 层读取原 language/state KV 与 `Q_l` 的 synthetic KV；
4. `Q_l` 进入下一层继续更新。

原始 image KV 对 AE 保持屏蔽。最终层 100 tokens 通过 attention pooling 解码
normalized 8D chunk-end target pose，损失仍为 `L_flow + L_pose`。

## 训练

```bash
bash scripts/libero_goal_prior_v2/train_stage2.sh
```

默认：

- Stage1 checkpoint：
  `lerobot/outputs/libero_goal_prior/seed_1000/stage1/checkpoints/010000/pretrained_model`
- 输出：`lerobot/outputs/libero_goal_prior_v2/seed_1000/stage2`
- 7 GPU，`BATCH_SIZE=32`/卡，`STEPS=30000`，每 5000 步保存；
- VLM / ViT / connector / AE / semantic-visual module 均为 `1e-5`；
- warmup：VLM/connector/AE/semantic=1000，ViT=2000。

第一次运行先用较小 batch 做显存 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=1 STEPS=2 SAVE_CHECKPOINT=false \
OUTPUT_DIR=lerobot/outputs/libero_goal_prior_v2/smoke \
bash scripts/libero_goal_prior_v2/train_stage2.sh
```

## 评测

```bash
bash scripts/libero_eval/eval_libero_v2_checkpoint.sh \
  lerobot/outputs/libero_goal_prior_v2/seed_1000/stage2/checkpoints/010000/pretrained_model
```

默认使用 GPU7，每 task 20 rollouts，并保存全部视频。

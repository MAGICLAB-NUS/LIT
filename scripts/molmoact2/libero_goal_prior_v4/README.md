# Goal-Pose Prior v4 — pose-only hard bottleneck

v4 is an **additive** recipe on top of v3: only the hard bottleneck differs
(`8/8` latents). Stage2 optimizer/warmup is locked to the v3 `025000` run
(StarVLA-aligned: VLM `1e-5`, AE + aggregator `1e-4`).

| | v3 (kept) | v4 (this dir) |
| --- | --- | --- |
| Stage1 goal tokens | 4 | **8** |
| Stage2 latents | 100 = 8 pose + 92 context | **8 pose only** |
| L_pose | first 8 | **all 8** |
| QUANTILES | recomputed | **reuse v3 stats** |
| Stage2 optimizer | VLM `1e-5`；AE / aggregator `1e-4`（warmup 5k） | **同 v3 025000 / StarVLA 对齐** |
| OUTPUT_DIR | `libero_goal_prior_v3/` | `libero_goal_prior_v4/` |

Architecture + feature flow (V3/V4 single doc): [V3_VS_V4_FEATURE_FLOW.md](V3_VS_V4_FEATURE_FLOW.md).

## Train

```bash
# stats (once, if not already done for v3)
bash scripts/libero_goal_prior_v3/fix_stats.sh

bash scripts/libero_goal_prior_v4/train_stage1.sh
bash scripts/libero_goal_prior_v4/train_stage2.sh
```

Auto-chain（Stage1 跑着时另开）：轮询等到 `checkpoints/010000` 且 Stage1 进程退出，再自动加载该权重启 Stage2：

```bash
nohup bash scripts/libero_goal_prior_v4/watch_and_train_stage2.sh \
  >lerobot/outputs/libero_goal_prior_v4/seed_1000/watch_stage2.console.log 2>&1 &
```

可覆盖：`SEED` / `POLL_SEC` / `CUDA_VISIBLE_DEVICES` / `WAIT_STAGE1_IDLE=0`（有 ckpt 就立刻开，不等进程退出）。

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=1 STEPS=2 SAVE_CHECKPOINT=false \
OUTPUT_DIR=lerobot/outputs/libero_goal_prior_v4/smoke_s1 \
bash scripts/libero_goal_prior_v4/train_stage1.sh
```

## Eval

All V4 wrappers fail closed unless the checkpoint is the expected `8/8`
hard-bottleneck architecture.

```bash
# Standard LIBERO
bash scripts/libero_eval/eval_libero_v4_checkpoint.sh

# LIBERO-Plus full: 10,030 tasks × 1 episode
bash scripts/libero_eval/eval_libero_v4_plus.sh

# LIBERO-PRO public Total16: 16 cells × 500 episodes
bash scripts/libero_eval/eval_libero_v4_pro.sh
```

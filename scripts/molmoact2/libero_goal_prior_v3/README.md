# Goal-Pose Prior v3 — fixed QUANTILES + v2b architecture

v3 **不改训练代码 / 模型结构**，只修正数据集 `meta/stats.json` 里
`observation.state` 与 `action` 的 q01/q99（以及配套的 min/max/mean/std），
再按 v2b 同配置重训 Stage2。

**网络 / 信息流（V3+V4 合并文档，同步用）**：[../libero_goal_prior_v4/V3_VS_V4_FEATURE_FLOW.md](../libero_goal_prior_v4/V3_VS_V4_FEATURE_FLOW.md)

**跨模型迁移规范（ImageWAM / π0.5）**：[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

复现使用 [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero)
数据集的固定 revision
[`1595a93b43aa055e55c127a4f0b4a99bb8035447`](https://huggingface.co/datasets/lerobot/libero/tree/1595a93b43aa055e55c127a4f0b4a99bb8035447)。
该 Hub revision 保留原始统计量；下载到可写本地目录后，必须先运行
`fix_stats.sh`，再开始 v3 两阶段训练。

## 为什么要做

本地 LIBERO 的旧分位数严重偏窄，例如 state Z 的 q01/q99 ≈ `[0.64, 0.88]`，
而真实 EE 高度大约在 `[0.04, 1.27]`。`QUANTILES` 归一化会把绝大多数 Z 目标
clip 成 ±1，导致：

- `L_pose` 学不到真实高度细节；
- 反归一化可视化出现系统性 ~25–30cm Z 误差（XY 其实只有 ~2cm）。

## 流程（先 stats，再训练）

```bash
# 1) 备份旧 stats，并从 parquet 重算 state/action 分位数（已跑过可跳过）
bash scripts/libero_goal_prior_v3/fix_stats.sh

# 2) Stage1（本机原 v1 Stage1 已缺失；用新 stats 重训更干净）
bash scripts/libero_goal_prior_v3/train_stage1.sh

# 3) Stage2：强制从 v3 Stage1 的 010000 初始化，并校验 NEW state 分位数
bash scripts/libero_goal_prior_v3/train_stage2.sh
```

默认：

| 项 | 值 |
| --- | --- |
| Stage1 输出 | `lerobot/outputs/libero_goal_prior_v3/seed_1000/stage1` |
| Stage2 初始化 | **仅** 上述 Stage1 `checkpoints/010000`（无 legacy / v2b 回退） |
| Stage2 架构 | 100 tokens / 8 pose / 6-group / self-attn / pose weight 0.3 |
| Stage2 优化 | VLM `1e-5`；AE + aggregator `1e-4`（wu 5k；与 025000 / StarVLA 对齐） |
| Stage2 输出 | `lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2` |
| state stats | 启动前校验 `stats.json` 的 Z q99≳1.0，拒绝旧 clip 分位数 |
| GPU | 默认 0–7（8 卡）；Stage1 `bs128`/10k，Stage2 `bs32`/30k |
## 注意

- **训练代码不用改**：`train_libero_molmoact2.sh` 会读 `dataset.meta.stats`。
- 旧 stats 备份在：
  - `meta/stats.pre_v3_bad_quantiles.json`（稳定别名）
  - `meta/stats.pre_v3_<timestamp>.json`
- **不要在改 stats 后 resume v2b**：v2b 是在旧分位数下训的；之后只用新目录跑 v3。
- v2b 已有 checkpoint 的 processor 一般自带当时的 stats，评测旧 ckpt 通常不受影响。
- **当前机器上 v1 Stage1 目录已不在**（`.../stage1/checkpoints/010000` 缺失）。
  v3 有两条路：
  1. 从备份恢复 Stage1，再 `train_stage2.sh`（与 v2b 同初始化，只换 stats）；
  2. 先用新 stats 重训 Stage1（更干净），再 Stage2：
     ```bash
     bash scripts/libero_goal_prior_v3/train_stage1.sh
     bash scripts/libero_goal_prior_v3/train_stage2.sh
     ```

## Smoke

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=1 STEPS=2 SAVE_CHECKPOINT=false \
OUTPUT_DIR=lerobot/outputs/libero_goal_prior_v3/smoke \
bash scripts/libero_goal_prior_v3/train_stage2.sh
```

# Goal-Pose Prior v4 — 实现说明

网络说明见 [ARCHITECTURE.md](ARCHITECTURE.md) / [README.md](README.md)。

## 变更范围（相对 v3）

| 组件 | 是否改动 |
| --- | --- |
| `configuration_molmoact2` 校验 | **放宽** `pose <= total`（默认值不变） |
| Aggregator / AE / mask 逻辑 | 否（参数化；`8/8` 即硬瓶颈） |
| `libero_goal_prior_v3/train_stage2.sh` | **默认 LR 写回 025000**（`1e-4` AE/SV）；架构 flags 不变 |
| 本目录 train/eval 脚本 | **新增**（架构 `8/8` + 同套优化超参） |
| dataset QUANTILES | **否**（复用 v3） |

## 训练入口

- Stage1：`train_stage1.sh` → `NUM_GOAL_TOKENS=8`，输出 `libero_goal_prior_v4/.../stage1`
- Stage2：`train_stage2.sh` → `tokens=8`、`pose=8`，强制从 v4 Stage1 `010000` 初始化
- Stage2 优化超参与 v3 `025000` 锁定一致：VLM `1e-5`；AE / semantic-visual `1e-4`（warmup 5k）

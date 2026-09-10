# Goal-Pose Prior v3 — 实现说明

网络结构与信息流见 **[ARCHITECTURE.md](ARCHITECTURE.md)**（基线对照用）。下文只记相对 v2b 的落地差异。

## 变更范围

| 组件 | 是否改动 |
| --- | --- |
| `modeling_molmoact2.py` / config / processor | 否 |
| v2b Stage2 超参与结构 | 否（同脚本参数） |
| `dataset/meta/stats.json` 的 `observation.state`、`action` | **是**（重算） |
| 图像 stats | 否（保留） |

## 数据修复

`scripts/libero_goal_prior_v3/recompute_vector_quantiles.py` 直接扫
`data/**/*.parquet`（273k 帧），用 numpy `quantile` 重写向量特征统计，避免
官方 `augment_dataset_quantile_stats.py` 因 video decode 过慢、且默认不
overwrite / 会 `push_to_hub`。

## 训练入口

`train_stage2.sh` → `scripts/train_libero_molmoact2.sh`，policy flags 与
`libero_goal_prior_v2b/train_stage2.sh` 一致；仅 `OUTPUT_DIR` / `JOB_NAME` 不同，
并在启动前检查 `meta/stats.v3_recompute_note.json`。

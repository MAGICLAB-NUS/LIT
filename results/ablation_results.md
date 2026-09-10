# 消融结果表（论文格式）

更新：2026-09-09_13:39 (A800j)。只有【Plus 全量 10030】/【ID 满 2000 eps】的终值才进表，其余填 –。
Full LIT 行 = 论文主表数值（同事复核版）；全表 Overall = 七轴算术平均（与论文 73.50 同口径）。

| Variant | ID LIBERO Avg. | Camera Viewpoints | Sensor Noise | Lighting Conditions | Background Textures | Robot Initial States | Object Layout | Language Instructions | Overall | Done |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (MolmoAct2) | 93.50 | 39.40 | 49.03 | 89.84 | 89.78 | 50.71 | 55.41 | 82.69 | 65.27 | ✓ |
| Vanilla staged training | 93.75 | 47.28 | 52.09 | 93.17 | 91.08 | 55.16 | 60.98 | 70.46 | 67.18 | ✓ |
| LIT w/o Stage-1 goal conditioning | 93.70 | 46.15 | 54.65 | 87.13 | 86.34 | 63.10 | 66.43 | 64.80 | 66.94 | ✓ |
| LIT w/ direct visual conditioning | 94.25 | 42.96 | 55.28 | 88.18 | 89.22 | 63.10 | 63.74 | 70.33 | 67.54 | ✓ |
| LIT w/o pose supervision | 93.50 | 42.78 | 60.02 | 91.07 | 88.38 | 61.81 | 68.85 | 71.11 | 69.15 | ✓ |
| Latent interface only | 93.45 | 39.84* | 49.34* | 87.22* | 89.63* | 68.00* | 68.20* | 56.67* | 65.56* | ✓ |
| **Full LIT** | **95.00** | **48.41** | **70.77** | **90.64** | **94.42** | **59.03** | **70.61** | **80.58** | **73.50** | **✓** |

进行中 / 未完成：
- Latent interface only: Plus 10024/10030（99.94%），带 * 的数按已评任务计

## LaTeX 行（列序同表头）
```
Baseline (MolmoAct2) & 93.50 & 39.40 & 49.03 & 89.84 & 89.78 & 50.71 & 55.41 & 82.69 & 65.27 & ✓ \\
Vanilla staged training & 93.75 & 47.28 & 52.09 & 93.17 & 91.08 & 55.16 & 60.98 & 70.46 & 67.18 & ✓ \\
LIT w/o Stage-1 goal conditioning & 93.70 & 46.15 & 54.65 & 87.13 & 86.34 & 63.10 & 66.43 & 64.80 & 66.94 & ✓ \\
LIT w/ direct visual conditioning & 94.25 & 42.96 & 55.28 & 88.18 & 89.22 & 63.10 & 63.74 & 70.33 & 67.54 & ✓ \\
LIT w/o pose supervision & 93.50 & 42.78 & 60.02 & 91.07 & 88.38 & 61.81 & 68.85 & 71.11 & 69.15 & ✓ \\
Latent interface only & 93.45 & 39.84* & 49.34* & 87.22* & 89.63* & 68.00* & 68.20* & 56.67* & 65.56* & ✓ \\
\textbf{Full LIT} & \textbf{95.00} & \textbf{48.41} & \textbf{70.77} & \textbf{90.64} & \textbf{94.42} & \textbf{59.03} & \textbf{70.61} & \textbf{80.58} & \textbf{73.50} & \textbf{✓} \\
```

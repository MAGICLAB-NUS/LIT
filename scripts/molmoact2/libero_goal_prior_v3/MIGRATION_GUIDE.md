# MolmoAct2 V3 两阶段 Goal-Pose Prior：实现与跨模型迁移规范

> 本文面向后续将同一套两阶段训练迁移到 **ImageWAM**、**π0.5** 或其他
> Vision-Language-Action 模型的实现者。目标不是复述实验历史，而是把当前 V3
> 拆成可核对的训练契约、网络接口、checkpoint lineage 和验收测试。

## 0. 文档定位

### 0.1 三层内容必须分开

1. **方法不变量**：两阶段分别解决什么问题，哪些信息训练可见、推理不可见。
2. **MolmoAct2 V3 的具体实现**：token 数、模块、mask、loss、LR、checkpoint。
3. **目标模型适配**：ImageWAM/π0.5 应在哪个接口实现同一语义，而不是机械复制
   MolmoAct2 的类名和 token 拓扑。

迁移时应先复现方法语义，再决定是否复刻 V3 的具体软瓶颈 `100 = 8 + 92`。
V3 当前在 LIBERO-Plus Language Instructions 上有明显下降，因此
`100/8 + raw-image mask` 应视为一个需要复现实验验证的设计，而不是跨模型默认最优解。

### 0.2 当前代码边界

- V3 代码位于当前 `lerobot/` 子模块。
- V3 与 v2b 使用同一模型实现；V3 的主要差异是修正后的 QUANTILES、独立输出目录及
  Stage1 → Stage2 lineage 约束。
- 当前仓库有 LeRobot π0.5 实现。
- 当前仓库没有 ImageWAM 源码；本文只为它定义适配契约，不猜测其内部结构。

### 0.3 Source of truth

| 内容 | 文件 |
| --- | --- |
| V3 Stage1 启动 | `scripts/libero_goal_prior_v3/train_stage1.sh` |
| V3 Stage2 启动 | `scripts/libero_goal_prior_v3/train_stage2.sh` |
| 公共训练 launcher | `scripts/train_libero_molmoact2.sh` |
| Policy config | `lerobot/src/lerobot/policies/molmoact2/configuration_molmoact2.py` |
| 模型与 loss | `lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py` |
| 数据打包 | `lerobot/src/lerobot/policies/molmoact2/processor_molmoact2.py` |
| delta timestamp | `lerobot/src/lerobot/datasets/factory.py` |
| episode 边界 | `lerobot/src/lerobot/datasets/dataset_reader.py` |
| QUANTILES 修复 | `scripts/libero_goal_prior_v3/fix_stats.sh` |
| V3/V4 特征流 | `scripts/libero_goal_prior_v4/V3_VS_V4_FEATURE_FLOW.md` |
| LeRobot π0.5 | `lerobot/src/lerobot/policies/pi05/` |

---

## 1. 方法的最小定义

设当前观测为：

- 图像：\(I_t\)
- 语言指令：\(l\)
- 当前机器人状态：\(s_t\)
- 动作 chunk：\(a_{t:t+H-1}\)
- chunk 终点状态：\(g_t = s_{t+H}\)

当前 LIBERO V3 使用：

- `H = chunk_size = n_action_steps = target_pose_delta_index = 10`
- `s_t, g_t ∈ R^8`
- 原始 action 为 7 维，模型内部 pad 到 32 维
- `g_t` 是未来 `observation.state`，不是 object pose，也不是独立标注的 TCP pose

两阶段目标：

### Stage1：学习 vision-free action prior

训练输入是 `(l, s_t, g_t, noisy_action, flow_timestep)`，不输入图像。

```text
g_t ── GoalEncoder ── goal tokens ──┐
l, s_t ── frozen VLM context ───────┼── Action Expert ── L_flow
noisy action + flow timestep ───────┘
```

其作用是让 Action Expert 学会：

> 给定当前状态、语言和一个可达的未来机器人目标状态，生成到达该目标的动作轨迹。

### Stage2：从视觉和语言推断 goal condition

Stage2 不再把 `g_t` 编码后提供给动作网络；改用 learnable latent 从
`(I_t, l, s_t)` 推断 goal-related condition：

```text
I_t, l, s_t ── VLM hidden states ── Goal Inference Module ── inferred latents
                                                              ├── Action Expert ── L_flow
g_t ───────────────────────────────────────────────────────────└── Pose Decoder ── L_pose
```

Stage2 使用 `g_t` 只计算辅助监督。部署推理时没有 `g_t`。

### 方法不变量

跨模型迁移至少应保持：

1. Stage1 不使用图像。
2. Stage1 的 Action Expert 必须由真实未来 goal condition 控制。
3. Stage2 从 Stage1 的 Action Expert 权重初始化。
4. Stage2 的 goal condition 只能由部署时可见的图像、语言和当前状态推断。
5. future goal 只作为 Stage2 训练辅助目标，不能进入 Stage2 推理条件。
6. Stage1/Stage2 的状态、goal、action 必须使用同一 lineage 的归一化统计量。
7. action chunk 的时间跨度和 future goal 的时间跨度必须显式对齐。

---

## 2. 数据契约

## 2.1 LIBERO V3 batch

典型训练字段：

| 字段 | 进入模型前的典型形状 | 说明 |
| --- | --- | --- |
| `observation.images.image*` | policy batch 通常为 `(B,3,H,W)` | Molmo processor 再逐样本转为 HWC 交给 HF processor；Stage2 使用 |
| `observation.state` | 原始查询 `(B,2,8)` | 包含 `t` 和 `t+H` |
| processor 后 current state | `(B,8)` | 取时间轴第一项 |
| `goal_pose` | `(B,8)` | 取时间轴最后一项，即 `state[t+H]` |
| `goal_pose_is_pad` | `(B,)` | future query 是否越过 episode |
| `action` | `(B,10,32)` | 7 维 action pad 到模型宽度 32 |
| `action_horizon_is_pad` | `(B,10)` | episode 尾部 action padding |
| `action_dim_is_pad` | `(B,32)` | action 维度 padding |
| `task` | `list[str]` | 自然语言指令 |

### `t+H` 是如何产生的

`MolmoAct2Config` 提供：

```text
observation_delta_indices = None
action_delta_indices      = [0, 1, ..., H-1]
target_pose_delta_index   = H
```

dataset factory 为 `goal_pose_feature_key` 额外请求 `[0, H]` 两个时间点。
processor 随后拆成 current state 和 goal pose。

### episode 尾部

当 `t+H` 越过 episode：

1. dataset reader 将实际索引钳制到 episode 最后一帧；
2. 同时设置 `goal_pose_is_pad=True`；
3. Stage2 的 `L_pose` 忽略这些样本；
4. Stage1 的 GoalEncoder **仍会读取钳制后的 terminal goal**，不会检查
   `goal_pose_is_pad`；
5. `L_flow` 通过 `action_horizon_is_pad` 将无效 action timestep 的 numerator 置零，
   每个 timestep 先按有效 action 维度平均，最终再按固定
   `num_flow_timesteps(8) × H` 平均，不按有效 timestep 数重新归一化。

迁移时不能只实现索引钳制而遗漏 pad mask，否则模型会把 episode 最后一帧当成大量真实
Stage2 pose 监督。若目标是 Exact-V3，Stage1 terminal-goal 行为和 flow 的固定分母也应
保持；若改成丢弃整个尾部样本或按有效步重归一化，应作为显式 ablation 记录。

## 2.2 Goal pose 的语义

当前类名 `_GoalSE3Encoder` 容易引起误解：

- 输入实际是归一化后的 8 维 LIBERO `observation.state`；
- 实现是三层 MLP，不是 SE(3)-equivariant 网络，也没有显式 Lie group 运算；
- `goal_pose` 不是 object pose；
- DROID 或其他真机迁移时，应先明确它是 joint state、wrist attachment pose 还是 TCP pose。

目标模型不需要复用 `_GoalSE3Encoder` 这个名字，但必须固定以下合同：

```python
goal_pose: FloatTensor[B, goal_dim]      # normalized
goal_is_pad: BoolTensor[B]
goal_tokens = GoalEncoder(goal_pose)     # [B, K, model_dim]
```

## 2.3 归一化

V3 使用：

- Visual：`IDENTITY`
- State：`QUANTILES`
- Action：`QUANTILES`
- goal pose：与 `goal_pose_feature_key` 对应的 state 使用同一 QUANTILES
- `normalize_gripper=false`：根据 feature names 将 gripper 维从 State/Action/goal
  的 QUANTILES mask 中排除，保留原值，之后仍统一 clamp 到 `[-1,1]`

旧 LIBERO stats 的 state Z q01/q99 约为 `[0.64, 0.88]`，而真实范围更宽，导致大量目标
被 clip 到 `±1`。V3 在启动前强制检查修正后的 q99。

迁移规则：

1. 不要因为目标模型自带 norm stats 就跳过 goal stats 核对。
2. 同一数据和相同物理字段可以共享数值统计，但必须确认目标框架的 QUANTILES 公式、
   clamp 行为和反归一化完全一致。
3. 如果 goal pose 与 state 不是同一字段，必须单独统计。
4. 保存逐维 normalization mask；尤其不要让目标框架默认归一化 gripper 而 V3 不归一化。
5. 保存 stats 的数据 revision、特征名、坐标系、单位、q01/q99 和 clip 比例。

---

## 3. Stage1：MolmoAct2 V3 精确实现

## 3.1 初始化

Stage1 从 MolmoAct2 模板 checkpoint 构建完整拓扑，再执行：

- Molmo2-ER 权重 overlay 到 VLM；
- 随机初始化的 Action Expert；
- 随机初始化的 Goal Encoder；

开始训练。

当 Molmo2-ER 词表小于 MolmoAct2 模板词表时，只覆盖 embedding 的公共前缀，
MolmoAct2 模板新增的专用 token 行保留原值。因此 checkpoint lineage 应记录为
“MolmoAct2 template + Molmo2-ER overlay + reset Action Expert”，不能简化成纯
Molmo2-ER 初始化。

关键 flags：

```text
action_mode                     = continuous
enable_goal_pose                = true
goal_token_source               = se3_encoder
goal_conditioning_mode          = vlm_appended   # config 默认值
num_goal_tokens                 = 4
target_pose_delta_index         = 10
disable_visual_input            = true
train_action_expert_only        = true
mask_image_from_action_expert   = false          # Stage1 无图
enable_pose_reconstruction      = false
randomize_action_expert         = true
```

## 3.2 Goal Encoder

`_GoalSE3Encoder`：

```text
goal_pose [B,8]
  → Linear(8,512)
  → GELU
  → Linear(512,512)
  → GELU
  → Linear(512,4×VLM_hidden)
  → reshape [B,4,VLM_hidden]
```

4 个 goal token 被追加到 VLM sequence 末尾，作为每层 Action Expert cross-attention
context 的一部分。

## 3.3 冻结与可训练参数

`train_action_expert_only=true` 时，仅以下名称匹配的参数可训练：

- `action_expert.*`
- `goal_*`
- `semantic_visual_*`（Stage1 没有该模块）

因此 Stage1 实际可训练：

- Action Expert
- `_GoalSE3Encoder`

VLM/ViT/connector 冻结；policy 的 `train()` 还会把 HF backbone 保持在 eval 模式。

必须在启动日志中输出每个 optimizer group 的参数量，并验证：

```text
trainable_vlm == 0
trainable_action_expert > 0
trainable_goal_encoder > 0
trainable_visual_goal_module == 0
```

## 3.4 Flow matching

动作 \(a\)、噪声 \(\epsilon\) 和时间 \(t\) 形成：

```text
x_t = (1 - t) * noise + t * action
target_velocity = action - noise
```

Action Expert 预测 velocity，并计算逐维 MSE。当前 MolmoAct2 每个训练样本采样
`num_flow_timesteps=8` 个时间点，因此内部 batch 扩展为 `B×8`。

Stage1 总 loss：

```text
L_stage1 = L_flow
```

## 3.5 Stage1 recipe

| 项 | 值 |
| --- | --- |
| steps | 10,000 |
| batch/GPU | 128 |
| GPUs | 默认 8 |
| AE LR | `5e-5` |
| Goal Encoder LR | `5e-5` |
| AE warmup | 500 |
| Goal warmup | 500 |
| save freq | 2,500 |
| Stage2 固定初始化点 | `checkpoints/010000/pretrained_model` |

## 3.6 Stage1 应学到什么

建议至少做三个 probe：

1. 固定 `(l,s_t)`，改变 `goal_pose`，动作应明显变化。
2. 固定 goal，额外加入图像字段时应被忽略或明确拒绝；不能进入任何可训练视觉路径。
3. 对真实 action chunk 加噪后，flow loss 应下降；尾部 padded timestep 的 loss
   numerator 应为零，同时确认 reduction 仍使用固定 horizon 分母。

如果 probe 1 不成立，Stage2 即使能预测 goal，也无法通过 Stage1 prior 控制动作。

---

## 4. Stage1 → Stage2 checkpoint 转换

Stage2 从 Stage1 10k checkpoint 加载：

- VLM 权重；
- Action Expert 权重；
- processor / norm stats lineage；

但模型拓扑发生变化：

| Stage1 | Stage2 |
| --- | --- |
| `_GoalSE3Encoder` | 不再构建 |
| 无 semantic-visual 模块 | 新建 semantic-visual aggregator |
| 无 pose decoder | 新建 semantic-visual pose decoder |

loader 允许：

- Stage2 的 `semantic_visual_*` keys missing，由 Stage2 随机初始化；
- Stage1 的 `goal_se3_encoder.*` keys unexpected，不加载到 Stage2；

但 Action Expert fingerprint 必须与 Stage1 checkpoint 对齐。

目标框架必须显式实现相同的 partial-load policy，不能简单使用 `strict=False` 后忽略全部
missing/unexpected keys。建议输出：

```text
loaded_shared_modules
new_stage2_modules
dropped_stage1_only_modules
invalid_missing_keys
invalid_unexpected_keys
action_expert_fingerprint_before/after
```

---

## 5. Stage2：MolmoAct2 V3 网络设计

## 5.1 Stage2 配置

```text
enable_goal_pose                    = true
goal_token_source                  = learnable_queries
goal_conditioning_mode             = semantic_visual_recurrent
num_semantic_visual_tokens         = 100
num_semantic_visual_pose_tokens    = 8
semantic_visual_hidden_dim         = 768
semantic_visual_num_heads          = 8
semantic_visual_ffn_ratio          = 4.0
semantic_visual_dropout            = 0.0
semantic_visual_enable_self_attention = true
semantic_visual_num_layer_groups   = 6
mask_image_from_action_expert      = true
enable_pose_reconstruction         = true
pose_recon_loss_weight             = 0.3
```

Stage2 打开视觉并训练全模型，但 embedding 仍冻结。

## 5.2 模块组成

### Learnable query bank

```text
Q0: [100,768]
initialization: trunc_normal(std=0.02)
batch expand: [B,100,768]
```

query 分成概念上的两部分：

- `Q[:,0:8]`：最终接受 `L_pose` 监督的 pose tokens
- `Q[:,8:100]`：无 `L_pose` 的 context tokens

注意：self-attention 会让两部分互相交换信息；“pose/context”不是硬隔离。

### 一个 Aggregator group

每组包含：

```text
1. PreNorm Self-Attention(Q) + residual + FFN
2. PreNorm Cross-Attention(Q ← semantic hidden) + residual + FFN
3. PreNorm Cross-Attention(Q ← image hidden) + residual + FFN
4. to_key(Q), to_value(Q)
```

其中：

- latent dim：768
- heads：8
- FFN hidden：`768 × 4 = 3072`
- dropout：0
- semantic context 和 image context 都来自当前 VLM layer hidden states

### 6-group depth routing

Aggregator 不是只在 VLM 尾部运行一次，而是与 VLM/Action Expert 逐层递归：

```text
group_idx = layer_idx // (num_vlm_layers / 6)
```

约束：`num_vlm_layers % 6 == 0`。

每个 group 有独立参数；同一 group 覆盖的一段连续 VLM layers 共享该 group 参数。
例如 VLM 有 36 层时，每组处理连续 6 层。

## 5.3 Semantic/image mask

从 processor 输出构建：

```text
valid_mask   = attention_mask
image_mask   = valid AND token_is_image_patch
semantic_mask = valid AND NOT multimodal_token
```

当前 semantic context 包含 language/state 等非图像 token。

必须测试每个 batch：

```text
semantic_mask.any(dim=1) == True
image_mask.any(dim=1) == True
```

## 5.4 每层完整数据流

设：

- 当前 VLM hidden：`H_i [B,S,D_vlm]`
- 当前 recurrent latent：`Q_i [B,100,768]`
- 当前 Action Expert hidden：`A_i`

每层执行：

```text
H_{i+1} = VLMBlock_i(H_i)

Q'      = SelfAttention_group(Q_i)
Q''     = CrossAttention_group(Q'  ← H_{i+1}[semantic_mask])
Q_{i+1} = CrossAttention_group(Q'' ← H_{i+1}[image_mask])

K_goal, V_goal = ProjectKV_group(Q_{i+1})
K_ctx = concat(K_vlm_i, K_goal)
V_ctx = concat(V_vlm_i, V_goal)

A_{i+1} = ActionExpertBlock_i(
    A_i,
    flow_timestep_condition,
    cross_kv=(K_ctx,V_ctx),
    cross_attention_mask
)
```

最后一个 VLM layer 的 aggregator input 会显式经过 VLM `ln_f`，以对齐训练与推理
`hidden_states` 的约定。

## 5.5 Raw image 对 Action Expert 的屏蔽

虽然原始 VLM KV 与 latent KV 在张量上拼接，AE cross-attention mask 会屏蔽原始
image token 位置：

```text
AE 可见：
  - language/state 等非图像 VLM KV
  - 100 个 semantic-visual latent KV

AE 不可见：
  - raw image patch KV
```

因此视觉信息必须先经过 aggregator，再进入 Action Expert。

迁移时应把“屏蔽 raw image”实现为可配置实验项；不要假设目标模型天然存在独立
cross-attention mask。

## 5.6 Soft bottleneck 的真实含义

最终 latent：

```text
Q_final [B,100,768]
├── Q_final[:,0:8]   → LayerNorm → concat → PoseDecoder → goal_pose_pred
└── Q_final[:,8:100] → 无 pose loss

全部 100 token → 每层 Action Expert conditioning
```

所以 V3 不是严格的 8-token pose bottleneck，而是：

- 8 个显式 pose-supervised token；
- 92 个无 pose 监督但可以传递任意视觉/语义信息的 context token；
- 100 token 之间还有 self-attention。

这也是后续迁移必须单独 ablate `100/8`、`8/8`、raw-image visibility 和语言增强的原因。

## 5.7 Pose Decoder

```text
Q_pose [B,8,768]
  → LayerNorm
  → flatten [B,6144]
  → Linear(6144,512)
  → GELU
  → Linear(512,512)
  → GELU
  → Linear(512,goal_dim=8)
```

pose loss：

```text
L_pose = MSE(goal_pose_pred, normalized_goal_pose)
```

`goal_pose_is_pad=True` 的样本不参与。

## 5.8 Stage2 loss

```text
L_stage2 = L_flow + 0.3 * L_pose
```

`L_pose` 只从最终 recurrent latent 计算，不在每个 layer 单独监督。

## 5.9 Stage2 optimizer

| 参数组 | LR | warmup |
| --- | --- | --- |
| VLM | `1e-5` | 1,000 |
| ViT | `1e-5` | 2,000 |
| connector | `1e-5` | 1,000 |
| Action Expert | `1e-4` | 5,000 |
| semantic-visual | `1e-4` | 5,000 |

其他默认：

- steps：30,000
- batch/GPU：32
- cosine decay：30,000 steps
- scheduler 使用统一 0.1 衰减比例：VLM/ViT/connector 最终 `1e-6`，
  Action Expert/semantic-visual 最终 `1e-5`
- bf16 + gradient checkpointing

---

## 6. 训练与推理必须同构

## 6.1 Stage2 训练

训练同时有：

- 当前图像、语言、状态；
- 动作 chunk；
- future goal label；

future goal 仅进入 `L_pose` target，不进入 learnable query 的输入。

## 6.2 Stage2 推理

推理没有 `goal_pose`：

1. VLM 对 image/language/state 做 prefill；
2. 保存每层 hidden states 和每层 encoder KV；
3. 从同一个 learnable query bank 开始；
4. 按训练相同的 6-group 路由递归 aggregator；
5. 每层将 latent KV 追加到该层 AE context；
6. Action Expert 进行 flow denoising；
7. 可选解码并缓存 predicted normalized goal pose，仅用于可视化/诊断。

### 必须防止的泄漏

- eval processor 不得为了画 pose 而读取 future state。
- `predict_action_chunk` 不得要求 `goal_pose`。
- 评测 overlay 使用的是模型预测 pose，不是真值 pose。
- train/inference 最后一层 LayerNorm 约定必须一致。

---

## 7. V3 已知风险：迁移时不要隐藏

## 7.1 Language robustness

当前完整 LIBERO-Plus Language Instructions：

```text
Baseline 30k: 82.82%
V3 25k:       72.80%
V3 30k:       73.72%
```

其中 `libero_spatial` 的 V3-30k 相对 baseline 下降 17.69 pp。

可能机制：

1. Stage1 的 future goal 让 Action Expert 可以弱化语言依赖；
2. Stage2 随机初始化的 aggregator 在固定 LIBERO 指令上学习表面短语捷径；
3. 92 个无 pose 监督的 context token 可以绕过 goal-pose 约束；
4. 100 latent 可能在 AE conditioning 中稀释原有语言 token。

迁移基线必须增加：

- canonical instruction / paraphrase / wrong instruction 的反事实 probe；
- paraphrase consistency；
- 同场景错误指令的 goal/action sensitivity；
- 语言类别单独评测，而不能只看总体成功率。

## 7.2 Soft bottleneck

V3 的 92 context token 是设计选择，不是两阶段方法不变量。推荐顺序：

1. 先精确复现 `100/8`，用于确认移植正确；
2. 再比较 `8/8` hard bottleneck；
3. 比较 raw image 对 action head 可见/不可见；
4. 比较显式 language residual/gating；
5. 比较 paraphrase augmentation 和对比监督。

## 7.3 Future goal 的物理语义

如果换成多机器人/多真机场景，不能默认 `state[t+H]` 是稳定可迁移的 goal：

- joint state 与 embodiment 强绑定；
- wrist attachment 与 TCP 不同；
- base frame 在多台机器人间可能不同；
- rotation 表示和 gripper 语义可能不同。

迁移前应固定：

```text
goal frame
translation unit
rotation representation
gripper convention
absolute vs relative
H 对应的真实时间
```

---

## 8. 跨模型抽象接口

目标模型至少需要实现下列逻辑接口。名称可以不同。

```python
class TwoStageGoalPriorAdapter:
    def extract_goal(self, sample, horizon) -> tuple[Tensor, Tensor]:
        """Return normalized future goal and is_pad."""

    def encode_oracle_goal(self, goal: Tensor) -> Tensor:
        """Stage1: future goal -> action-head conditioning."""

    def infer_visual_goal(
        self,
        image_language_state_features,
        masks,
    ) -> tuple[Tensor, Tensor]:
        """Stage2: return action conditioning and pose-supervised latents."""

    def action_flow_loss(
        self,
        batch,
        action_conditioning,
    ) -> Tensor:
        """Native target-model flow-matching loss."""

    def decode_goal(self, pose_latents: Tensor) -> Tensor:
        """Stage2 auxiliary goal prediction."""

    def stage1_trainable_parameters(self):
        """Action head + oracle-goal encoder only."""

    def load_stage1_into_stage2(self, checkpoint):
        """Strict shared-weight loading plus explicit topology diff."""
```

需要记录的 model capability：

| 能力 | 必需程度 |
| --- | --- |
| action head 可接受额外 conditioning | 必需 |
| 能区分部署可见输入与 future goal label | 必需 |
| 能加载 Stage1 action head 到 Stage2 | 必需 |
| 能访问多层视觉/语言 hidden states | 精确复刻 V3 recurrent aggregator 时必需 |
| 能按 token type 屏蔽 raw image | 精确复刻 V3 mask 时必需 |
| 有独立 VLM 和 Action Expert | 非必需，但决定接入方式 |
| action head 使用 flow matching | 推荐；否则需保留两阶段语义而换原生 action loss |

---

## 9. 迁移到 π0.5

## 9.1 π0.5 当前结构

当前 LeRobot π0.5：

```text
prefix:
  images + tokenized prompt/state

suffix:
  noisy action tokens + flow timestep conditioning

PaliGemmaWithExpertModel(prefix, suffix)
  → suffix hidden
  → action_out_proj
  → flow velocity
```

关键事实：

- 默认 `chunk_size=50`，不是 V3 的 10；
- state 被离散化后写进 prompt，不是独立连续 state token；
- action 内部 pad 到 32 维；state 按实际输入维度离散化后写进 prompt，
  当前 processor 不使用 `max_state_dim` 对 state tensor 做 padding；
- 使用 continuous flow matching；
- π0.5 原生 beta sampling 为 `alpha=1.5, beta=1.0`；
- MolmoAct2 V3 为 `alpha=1.0, beta=1.5` 且每样本 8 个 flow timesteps。
- π0.5 使用 `x_t = t*noise + (1-t)*action`、`u_t = noise-action`；
- MolmoAct2 使用相反的时间约定：
  `x_t = (1-t)*noise + t*action`、`v_t = action-noise`。

迁移时默认保留 π0.5 原生 flow recipe，除非实验目标就是复刻 MolmoAct2 的 flow 超参。

## 9.2 推荐的 π0.5 Stage1 接入

需要新增显式连续 `goal_pose` 输入，因为它不能与 prompt 中的 current state 混用：

```text
goal_pose [B,G]
  → GoalEncoder
  → K goal embeddings
  → 插入可被 suffix expert 看到的 conditioning 区域
```

Stage1：

- 修改 `_preprocess_images` / `embed_prefix` 以支持真正的 vision-free 输入：
  images 列表为空，或使用 `img_mask=False` 的占位 camera；不能用普通全零图替代，
  因为当前实现会把它标记为真实图像并送入 SigLIP；
- 保留 language/current-state prefix；
- suffix expert 可 attend language/state + goal embeddings；
- 冻结 PaliGemma vision/language backbone；
- 训练 Gemma action expert、`action_in_proj/out_proj`、`time_mlp_in/out` 和 GoalEncoder；
- 使用 π0.5 原生 flow 构造，但补上 `action_horizon_is_pad` mask；Exact-V3 使用
  “无效 numerator 置零、固定 horizon 分母”，其他 reduction 必须显式记录。

当前 `train_expert_only` 只冻结 `paligemma`，不会自动生成严格的可训练参数 allowlist；
Gemma expert、action projection、time MLP 和新增 policy 模块默认仍可训练。迁移实现应输出
参数审计，并确保未来加入的 Stage2-only 模块不会在 Stage1 意外训练。

初始化策略也必须固定：

- **Exact-V3**：从 π0.5 模板/预训练 backbone 构建后，重置 action expert、
  action/time projections 和 GoalEncoder，以对应 V3 的 random Action Expert；
- **target-native two-stage**：可以保留 π0.5 预训练 action expert，但实验名和 manifest
  必须注明，并与 reset 版本分开报告。

两种策略都要保存初始化前后的 module fingerprint。

## 9.3 推荐的 π0.5 Stage2 接入

若要精确复刻 V3：

1. 从每层 prefix hidden states 读取 image/semantic features；
2. 添加 recurrent learnable latent；
3. 将 latent 变换为 suffix expert 每层可用的 conditioning；
4. 同时调整训练和缓存推理的 attention mask：suffix 不可直接 attend raw image，
   但可 attend language/state/latent；
5. 从前 P 个 latent 解码 goal pose；
6. 载入 Stage1 expert，随机初始化 Stage2 aggregator/pose head。

这是侵入式修改，因为 π0.5 是 prefix/suffix joint transformer，而不是 MolmoAct2 的显式
`VLM KV → Action Expert cross-attention`。

推理时不能只改联合训练的二维 mask。当前 π0.5 先缓存完整 prefix KV，后续
`denoise_step` 会根据 `prefix_pad_masks` 让 suffix 重新看到所有有效 prefix。
实现 raw-image mask 时，必须缓存 token-type/suffix-visible prefix mask，并在每次
denoise step 构造 suffix→prefix mask 时复用，否则会出现“训练屏蔽、推理泄漏”。

更低风险的第一版可采用“非逐层 recurrent visual goal module”：

```text
final/selected prefix hidden
  → GoalInferenceModule
  → goal latents
  → append to prefix conditioning
  → suffix expert
```

它不等价于 V3 的逐层 recurrent 6-group aggregator，但可以先验证“两阶段 prior 是否有效”，
再升级为逐层版本。由于 goal latents 来自 final prefix hidden，不能直接追加到已经生成的
旧 prefix cache；必须选择并保持训练/推理同构的实现：

1. 两次 prefix forward：第一次产生 visual goal，第二次带 goal latents 重建 prefix KV；或
2. 实现 per-layer KV adapter，把 goal latents 显式注入 expert cache。

## 9.4 π0.5 必须重新决定的参数

| 问题 | 不应直接复制的 V3 默认 |
| --- | --- |
| horizon | π0.5 默认 50；根据数据频率确定 future goal 的 H |
| goal 维度 | 不一定等于 π0.5 prompt state 维度 |
| state 路径 | 当前是离散文本；goal 应保留连续输入 |
| flow timestep | 保留 π0.5 原生分布，先做受控对照 |
| latent 接入 | prefix embedding、每层 adapter 或 expert conditioning |
| raw image mask | 需要重写 prefix/suffix attention mask |
| checkpoint load | expert/shared backbone 严格加载，新模块显式初始化 |
| action padding | 当前原生 loss 不读取 horizon pad；迁移版必须新增 mask |

---

## 10. 迁移到 ImageWAM

当前仓库没有 ImageWAM 代码。接入前先完成架构发现，不要预设它与 MolmoAct2 相同。

需要回答：

1. 视觉/语言 backbone 和 action head 是否分离？
2. action head 是 flow matching、diffusion、autoregressive 还是其他形式？
3. action head 如何接收 image/language condition？
4. 是否可以给 action head 插入额外 token/KV/FiLM condition？
5. 是否可以访问每个 backbone layer hidden states？
6. 是否支持 token-type attention mask？
7. state、prompt、图像分别在哪里打包？
8. checkpoint 是否支持严格的部分模块加载？

按拓扑选择适配：

### A. 独立 VLM + cross-attention Action Expert

最接近 MolmoAct2，可以直接映射：

```text
oracle goal tokens / inferred latent KV → Action Expert cross-attention
```

### B. prefix/suffix unified transformer

参考 π0.5：

```text
goal tokens → prefix/conditioning slots
suffix action tokens → attend selected prefix tokens
```

raw-image mask 通过 attention mask 实现。

### C. DiT/UNet/diffusion action head

goal condition 可以映射为：

- cross-attention memory；
- FiLM/AdaLN condition；
- action-token prefix；

但应保留 Stage1 oracle goal 与 Stage2 inferred goal 使用同一个 action conditioning 接口。

ImageWAM adapter 完成后，应把实际文件路径、tensor shape 和 checkpoint 策略补回本节。

---

## 11. 推荐实施顺序

### Phase 0：baseline parity

- 原始目标模型在同一数据、同一 stats、同一评测协议下可训练/推理。
- 记录 image/state/action/prompt 的最终模型输入。

### Phase 1：数据与 future goal

- 增加 `target_pose_delta_index`。
- 生成 `goal_pose` 和 `goal_pose_is_pad`。
- 完成 normalize → unnormalize round trip。
- 检查 episode 边界。

### Phase 2：Stage1

- 新增 GoalEncoder。
- 图像完全禁用。
- 冻结 backbone。
- 训练 Action Expert + GoalEncoder。
- 完成 goal sensitivity probe。

### Phase 3：checkpoint bridge

- 实现严格的 Stage1 → Stage2 partial load。
- 验证 Action Expert/shared backbone fingerprint。

### Phase 4：Stage2 最小版

- 先实现一次性 visual goal inference。
- 加 `L_pose`。
- 验证推理不需要 future goal。

### Phase 5：V3 精确版

- 实现 per-layer recurrent aggregator。
- 实现 6-group routing。
- 实现 raw-image-to-action mask。
- 对齐训练/推理逐层 hidden states。

### Phase 6：设计改进

- `100/8` vs `8/8`
- raw image visible vs masked
- language residual/gating
- paraphrase augmentation/consistency
- goal frame/horizon ablation

---

## 12. 验收清单

### 数据

- [ ] `goal_pose` 来自同一 episode 的 `t+H`
- [ ] 越界样本有 `goal_pose_is_pad`
- [ ] action pad 和 goal pad 分开处理
- [ ] stats revision、坐标系、单位已记录
- [ ] gripper 等不归一化维度的逐维 mask 已记录
- [ ] normalize/unnormalize round trip 通过
- [ ] q01/q99 与 clip 比例已审计

### Stage1

- [ ] image 对模型输出无影响
- [ ] 改变 oracle goal 会改变 action
- [ ] 只有 Action Expert + GoalEncoder 可训练
- [ ] VLM/backbone 处于 eval/frozen
- [ ] flow loss 有效下降
- [ ] Stage1 checkpoint 带完整 config/stats lineage

### Stage2

- [ ] Action Expert 从 Stage1 精确加载
- [ ] 新 aggregator/pose head 是随机初始化
- [ ] future goal 不进入 Stage2 conditioning
- [ ] pose loss 正确屏蔽 pad
- [ ] flow loss 的 horizon mask 与 reduction 分母符合目标协议
- [ ] 所有 latent 的 AE 可见性符合配置
- [ ] raw image 的 AE 可见性符合配置
- [ ] train/inference latent 递归同构
- [ ] 推理不提供 `goal_pose` 也能运行

### 行为

- [ ] canonical/paraphrase 输出一致性
- [ ] wrong instruction 能改变 predicted goal/action
- [ ] same instruction + changed layout 能改变 predicted goal
- [ ] Official/ID 不低于约定阈值
- [ ] Plus 各 category 单独报告
- [ ] Language Instructions 单独报告
- [ ] 真机 action frame、单位和 gripper 语义已核对

---

## 13. 建议保存的迁移 manifest

每个目标模型的实验目录至少保存：

```yaml
method:
  name: two_stage_goal_pose_prior
  reference: molmoact2_v3
  stage1_checkpoint: ...
  stage2_init_checkpoint: ...

data:
  dataset_revision: ...
  image_keys: [...]
  state_key: ...
  goal_pose_key: ...
  goal_frame: ...
  goal_dim: ...
  action_dim: ...
  control_mode: ...
  fps: ...
  chunk_size: ...
  target_pose_delta_index: ...
  stats_path: ...
  normalization_mode: ...
  normalization_mask: [...]
  action_padding_reduction: fixed_flow_samples_and_horizon

stage1:
  visual_disabled: true
  initialization_strategy: ...
  reset_modules: [...]
  module_fingerprints_before: {...}
  module_fingerprints_after: {...}
  trainable_modules: [...]
  goal_encoder: ...
  num_goal_tokens: ...
  action_loss: ...

stage2:
  goal_inference_module: ...
  num_latents: ...
  num_pose_latents: ...
  num_layer_groups: ...
  raw_image_visible_to_action_head: ...
  pose_loss_weight: ...
  trainable_modules: [...]

inference:
  requires_future_goal: false
  raw_image_mask_matches_training: ...
  num_flow_steps: ...
  action_steps_executed: ...
```

---

## 14. 最终对齐原则

“迁移 V3”有两种不同目标，必须在实验名中区分：

### Exact-V3 topology

精确复刻：

- Stage1 4 oracle-goal tokens
- Stage2 100/8 recurrent latents
- 6 groups
- self → semantic → visual
- raw image 对 action head mask
- `L_flow + 0.3 L_pose`

用于判断代码移植是否对齐。

### Two-stage goal-prior method

只保持：

- oracle-goal Stage1
- visual-goal Stage2
- Stage1 action prior 继承
- future-goal auxiliary supervision
- inference 无 future goal

允许目标模型使用原生 conditioning、flow recipe 和 action topology。

对于 ImageWAM 和 π0.5，建议先完成 target-native 的最小两阶段版本，再实现
Exact-V3 topology。这样可以把“方法是否有效”和“V3 某个网络细节是否可迁移”分开验证。

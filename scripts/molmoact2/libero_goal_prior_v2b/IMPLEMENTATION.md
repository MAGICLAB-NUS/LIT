# Goal-Pose Prior v2b (scheme-2a) — 实现说明

## 与 v2 的关系

v2b 保留 v2 的递归 latent 思路，但同时改造 **pose 读出**与**逐层聚合器**。原 v2
在 step600 后 pose loss 明显加速，最终 step1000 达到 flow=0.213 / pose=0.164，
说明 100-token 表示可学习；但它没有 token 间直接通信，且同一套 cross-attention
跨 36 层共享。v2b 将其升级为两组功能 token + 六组 depth-specific aggregator。

## 核心改动：两组功能特化 token

100 个 latent token 按行切成两组：

- **pose 组**：前 `num_semantic_visual_pose_tokens`（默认 8）个 token。仅这组被 L_pose
  监督，经 v1 已验证的 concat 读出头（`8×768 -> MLP -> 8D`）解码目标位姿。
- **context 组**：其余 92 个 token，聚合语义-视觉上下文。

两组都逐层递归、都把投影后的 KV 拼进 action expert context——即 pose 组既是被监督的
显式 goal-pose prior，又真正因果地条件化 AE（scheme-2a）。这与 v1 失败模式（AE 唯一且
狭窄地依赖 pose、eval 时 pose 偏就整体错）不同：AE 同时吃 pose 组 + 92 个 context token。

pose/context 的功能特化来自「L_pose 仅监督前 P 个 token + pose 专用 concat 读出头」；
全局 latent self-attention 又允许两组在每层双向交换信息，因此 pose 梯度也能直接塑形
context tokens。

## 六组逐层数据流

```text
Q^share_l = SelfAttention(Q_{l-1})                         # 全部 100 token
Q^sem_l   = SemanticCrossAttention(Q^share_l, H_l[language/state])
Q_l       = VisualCrossAttention(Q^sem_l, H_l[image patches])
```

36 层分成 6 个连续组：0–5、6–11、12–17、18–23、24–29、30–35。每组拥有独立的
self/semantic/visual attention 与 latent→AE K/V projection，组内 6 层共享参数。
初始 learnable query bank 只有一份，`Q_l` 跨组连续传递且不重置。该设计约 170M
聚合器参数，相比 36 层完全独立所需的约 1B 更节省。

`Q_l`（全部 100 个）投影为 synthetic K/V 追加到该层 VLM KV，AE 单向读取。最后一层
`Q_36` 中前 8 个（pose 组）经 concat 头解码 normalized 8D state
`[position(3), axis-angle(3), gripper(2)]`。

```text
L = L_flow + pose_recon_loss_weight * L_pose      # v2b 默认 weight=0.3
```

## 关键差异一览（v2 -> v2b）

- 新增配置 `num_semantic_visual_pose_tokens`（默认 8），校验 `1 <= P < N`。
- 新增 `semantic_visual_enable_self_attention`（兼容默认 false）和
  `semantic_visual_num_layer_groups`（兼容默认 1）；v2b 显式设置 true / 6。
- pose 读出头：删除 `_SemanticVisualPoseDecoder`（单向量池化），改用
  `LayerNorm + _GoalPoseDecoder(num_tokens=P, hidden_size=768, ...)`。专用 LayerNorm
  补偿 recurrent latent 未经过 VLM final norm 的尺度差异。
- L_pose 只对 `hidden_states[:, :P, :]` 计算，仍断言全部 100 token 在场。
- `pose_recon_loss_weight` 默认降到 `0.3`（脚本层设置）。
- 训练与推理使用相同 layer→group 映射；全部 100 token 的 AE KV 拼接语义不变。
- non-RTC continuous inference 也必须走 policy-owned prefill/denoise；直接调用
  upstream `generate_actions_from_inputs` 会绕过 learned queries、synthetic KV 与
  `mask_image_from_action_expert`。路由由 `_uses_policy_continuous_generation`
  显式保护，并有回归单测。
- 默认 false / 1 时保留旧 v2 group-0 模块名与 state-dict 路径。

## 关键实现文件

- `lerobot/src/lerobot/policies/molmoact2/configuration_molmoact2.py`
- `lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py`
- `lerobot/tests/policies/molmoact2/test_molmoact2.py`
- `scripts/libero_goal_prior_v2b/train_stage2.sh`
- `scripts/libero_eval/eval_libero_v2b_checkpoint.sh`

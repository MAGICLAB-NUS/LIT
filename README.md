<div align="center">

# Latent Interface Training

### Breaking the Vision–Action Shortcut for Generalizable Robot Foundation Models

[Jianman Lin](https://github.com/jianmanlincjx)<sup>1\*</sup> · Shailesh Shailesh<sup>2\*</sup> · Zhongyi Luo<sup>3</sup> · Jiafei Duan<sup>2†</sup>

<sup>1</sup>South China University of Technology · <sup>2</sup>National University of Singapore · <sup>3</sup>Nanyang Technological University

[![Project page](https://img.shields.io/badge/Project-Page-1F4E9C)](https://magiclab-nus.github.io/LIT/)
[![Paper](https://img.shields.io/badge/Paper-PDF-B23A32)](https://magiclab-nus.github.io/LIT/static/paper/LIT.pdf)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Real--robot%20checkpoints-shailes--h%2FMolmoact2--LIT-f7c843)](https://huggingface.co/shailes-h/Molmoact2-LIT)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Real--robot%20data-yam__bimanual__manipulation-f7c843)](https://huggingface.co/datasets/shailes-h/yam_bimanual_manipulation)
[![License](https://img.shields.io/badge/License-Apache%202.0-2ea44f)](./LICENSE)

</div>

---

<div align="center">

<img src="docs/static/images/framework.png" width="92%" alt="Latent Interface Training: Stage 1 learns an action prior without images; Stage 2 routes vision through a pose-supervised latent interface.">

</div>

LIT keeps a robot foundation model's backbone and action expert as they are and changes only the interface
between them. Direct conditioning on visual tokens is removed; a small set of learnable latent tokens —
supervised to reconstruct the terminal SE(3) end-effector pose — becomes the action expert's **only** route
to vision. **Stage 1** learns an action prior from language, state and that pose with no images.
**Stage 2** trains the interface. The same recipe, with the same interface settings, is applied to four
architectures.

<div align="center">

<img src="docs/static/images/results_strip.png" width="88%" alt="Base vs LIT: LIBERO-Plus Overall on π0.5 68.97→79.67, MolmoAct2 63.62→71.92, FAST-WAM 51.44→60.63, ImageWAM 83.02→86.89; real robot (MolmoAct2) lighting 53.3→70.0, camera 30.0→46.7, distractors 50.0→63.3.">


</div>

**This README is organised in three parts:**

1. **[How to integrate LIT into a VLA or WAM](#integrating-lit-into-a-vla-or-wam)** — or into your own framework. Follow the six-step procedure and you can implement it.
2. **[A worked example](#worked-example-molmoact2-train-then-evaluate)** — training and evaluating LIT integrated into MolmoAct2.
3. **[Checkpoints and data](#checkpoints-and-data)** and the **[code repositories](#code)** for all four frameworks.

---

## Integrating LIT into a VLA or WAM

LIT is a framework-agnostic two-stage strategy, not a new model. It applies to any robot foundation model that
combines a pretrained backbone (a VLM or a video model) with an embodiment-specific action expert, and it retains
the backbone and action-expert architectures, the action representation, the prediction horizon and the native
action-generation objective. What changes is *how vision conditions the action expert*: in standard
architectures, backbone visual representations directly condition the action expert; LIT instead routes visual
conditioning **exclusively through a pose-supervised latent interface** to an action expert that was first
pretrained without images. The same recipe, with the same interface settings, was applied unchanged to two VLAs
(π0.5, MolmoAct2) and two WAMs (FAST-WAM, ImageWAM).

Integration is six steps. Each step below names where it lives in the four forks.

| Step | What you do | What it gives you |
| :-: | --- | --- |
| **1** | Identify the **coupling layers** — where backbone representations condition the action expert through the architecture's native mechanism — and remove the direct visual conditioning | vision no longer reaches the action expert directly |
| **2** | Add the **latent interface**: `K = 100` learnable latent tokens that aggregate the backbone's visual and semantic representations and condition the action expert layer-wise | the latent tokens are the action expert's only visual conditioning pathway |
| **3** | Add the **pose-reconstruction objective**: an MLP decoder reconstructs each chunk's terminal SE(3) goal state from the latent tokens (`λ_pose = 0.3`) | the interface is encouraged to retain goal-relevant spatial information |
| **4** | Add the **SE(3) goal encoder** used in Stage 1 only | the action expert can be pretrained on language, state and the terminal pose, without images |
| **5** | Train in **two stages**: spatial-goal-conditioned action pretraining → vision–action interface learning | the reported model |
| **6** | Run **three checks** — the interface is used, the pose is reconstructed, the new parameters train and save | catches the silent failures we hit |

### What the host must provide

| Requirement | Used for |
| --- | --- |
| A pretrained backbone that yields per-layer representations for the image and for language / robot state | the latent tokens aggregate them (Step 2) |
| An action expert conditioned on those representations at identifiable **coupling layers** through a native mechanism — cross-attention, a joint attention mask, or feature injection into a DiT | the one place LIT edits (Steps 1–2) |
| Action chunks of horizon `H` and an end-effector pose in the dataset state | the terminal goal state `g_t = [p_{t+H}; r_{t+H}; q_{t+H}] ∈ ℝ⁸` (position, axis-angle orientation, gripper), the Stage-1 conditioning signal and the Stage-2 reconstruction target; not required at inference |

### Step 1 — Remove the direct visual conditioning

Find the coupling layers and stop backbone visual representations from conditioning the action expert there.
Language and robot state may keep their native path (MolmoAct2) or be routed through the interface as well (π0.5);
what matters is that visual conditioning has no direct route.

<table>
<thead><tr><th width="11%">Host</th><th width="36%">Native conditioning mechanism</th><th width="25%">What changes</th><th width="28%">Reference</th></tr></thead>
<tbody>
<tr><td>MolmoAct2 (VLA)</td><td>Layer-wise cross-attention from the action expert to the VLM token sequence (image, instruction and state tokens)</td><td>Image tokens are masked out of every action-expert attention</td><td><code>mask_image_from_action_expert</code> in <code>lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py</code></td></tr>
<tr><td>π0.5 (VLA)</td><td>One joint self-attention shared by the VLM and the action expert; the expert's tokens attend the VLM's image, language and state tokens</td><td>Attention mask: action-expert tokens cannot attend image tokens (nor language/state tokens, which reach it through the interface)</td><td>attention mask in <code>src/pi05_goal_prior/modeling_pi05_goal_prior.py</code></td></tr>
<tr><td>FAST-WAM (WAM)</td><td>Per-block representations of the Wan2.2 video DiT injected into the ActionDiT</td><td>The direct injection is removed; conditioning comes only from the interface</td><td><code>goal_prior_stage: stage2</code> in <code>configs/model/fastwam_goal_prior_stage2.yaml</code>, wired in <code>fastwam_joint.py</code> / <code>mot.py</code></td></tr>
<tr><td>ImageWAM (WAM)</td><td>Per-block representations of the FLUX.2 Klein DiT fed to the action head</td><td>Same: the direct path is removed</td><td><code>backbones/imagewam.py</code>, <code>configs/model/imagewam_flux2_klein_4b_goal_prior_stage2.yaml</code></td></tr>
</tbody></table>

### Step 2 — Add the latent interface

`K = 100` learnable latent tokens `Z₀` (dimension `d = 768`), shared across inputs. At each coupling layer `ℓ` the
tokens are updated by **self-attention, semantic cross-attention and visual cross-attention** — the latent tokens
are the queries, the backbone's language/state and visual representations at that layer are the keys and values —
and the updated tokens condition the corresponding action-expert layer through the architecture's native
mechanism (as the keys/values of its cross-attention, as the only tokens its rows may attend, or as the injected
features). For parameter efficiency, every `m` consecutive coupling layers share the interface attention parameters
(`inner_dim = 512`, 8 heads, 6 parameter groups in our runs).

Reference: `semantic_visual_recurrent` (MolmoAct2); `goal_prior.py` (Pi05, fastwam); `goal_pose_prior.py` (ImageWAM).

### Step 3 — Add the pose-reconstruction objective

An MLP decoder reconstructs the terminal goal state `g_t` from the final latent tokens:
`L_pose = ‖ĝ_t − g_t‖²`, computed in the preprocessed (quantile-normalised) state space and averaged over valid
targets, and `L_stage2 = L_act + λ_pose · L_pose` with `λ_pose = 0.3`. `g_t` is the same target that conditioned
Stage 1, read from the dataset's state at `t + H` (`H = 10` in our runs). The decoder is training-time only.

Reference: `GoalPoseDecoder` in every fork; the target is built in the data processor
(`target_pose_delta_index` in MolmoAct2's `processor_molmoact2.py`).

### Step 4 — Add the SE(3) goal encoder (Stage 1 only)

A trainable three-layer MLP with GELU activations maps `g_t` to goal tokens, which are concatenated with the
backbone's language/state representations at each coupling layer and condition the action expert through the
native mechanism. It is omitted after Stage 1 and never used at inference — in Stage 2 the latent tokens carry
the spatial information instead.

Reference: `SE3Encoder` / `GoalPoseEncoder` in every fork.

### Step 5 — Train in two stages

```text
Stage 1  Spatial-goal-conditioned action pretraining
         backbone frozen · no images · action expert trained from scratch
         conditioned on language + robot state + SE(3)-encoded terminal pose  →  native action loss (L_prior)
Stage 2  Vision–action interface learning
         action expert initialised from Stage 1 · goal encoder omitted · latent interface + pose decoder added
         full fine-tuning of backbone, action expert, latent tokens, interface attention and decoder:
         L_stage2 = L_act + 0.3 · L_pose
```

Learning rates that mattered: the interface modules and the action expert at `1e-4` with a longer warm-up; the
backbone at `1e-5`. Interface settings were **not** tuned per architecture: `K = 100`, `d = 768`,
`inner_dim = 512`, `λ_pose = 0.3`. At inference the latent interface stays active while the Stage-1 goal encoder
and the Stage-2 decoder are omitted; the policy needs only images, language and robot state and follows the
host's native action sampling.

### Step 6 — Three checks before trusting a run

Each caught a silent failure for us at least once:

- **Is the interface actually the conditioning pathway?** Measure the action expert's attention mass on the latent
  tokens versus everything else (`tools/probe_latent_attention.py` in Pi05, `scripts/audit_goal_prior_v2.py` in
  ImageWAM). Near zero means the direct visual path is still open, or a gate never opened.
- **Do the latent tokens reconstruct the pose?** The normalised pose loss should reach ~`1e-3` on training data;
  decode it and draw it back onto the frame (`scripts/render_pose_videos.sh`) — the reconstructed goal should lead
  the gripper, not trail it.
- **Do the new parameters train and save?** Under bf16 autocast, small gates and freshly added modules can freeze
  or be dropped from the checkpoint; check the parameter count in `train_config.json` and that the new keys load.

---

## Worked example: MolmoAct2, train then evaluate

The MolmoAct2 fork carries the full recipe; the other three forks follow the same steps in their own READMEs.

```bash
git clone --recursive -b feat/libero-goal-prior-v4 https://github.com/jianmanlincjx/Molmoact2.git
git clone https://github.com/MAGICLAB-NUS/LIT.git && cd LIT

export LIT_MOLMOACT2=/path/to/Molmoact2
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus            # github.com/sylvestf/LIBERO-plus
export DATASET_ROOT=/path/to/libero_lerobot_format      # LIBERO in LeRobot format, all four suites
bash scripts/preflight.sh                                # submodule commit, LIBERO-Plus, GPUs
```

**Train** (from the pretrained backbone; the `_v3` launchers' defaults equal the released `train_config.json`):

```bash
cd "$LIT_MOLMOACT2"
bash scripts/libero_goal_prior_v3/train_stage1.sh           # Stage 1: action prior without images
bash scripts/libero_goal_prior_v3/train_stage2.sh           # Stage 2: latent interface + pose reconstruction, from Stage 1
OPTIMIZER_ACTION_EXPERT_LR=1e-4 SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000 \
  bash scripts/libero_goal_prior/train_baseline.sh          # (optional) matched baseline
```

Stage 2 reports the Stage-1 SE(3) encoder as unexpected keys
when it loads — expected; the encoder is training-time only.

**Evaluate** the Stage-2 checkpoint (`outputs/…/checkpoints/<step>/pretrained_model`):

```bash
cd LIT
bash scripts/eval_libero_plus.sh lit <checkpoint-dir>   # LIBERO-Plus: 10,030 tasks, one episode each, seed 1000
bash scripts/eval_libero.sh      lit <checkpoint-dir>   # LIBERO: 4 suites × 50 episodes
```

Both are resumable across GPUs and print the per-axis / per-suite result when they finish.
`scripts/aggregate.py <results-root>` re-aggregates later; `--pair <a> <b>` compares two runs on the tasks both finished.

## Two things that change the numbers

- **`LIBERO_PLUS_FIX_LANG=1`** — set by the scripts. Without it the LIBERO-Plus harness feeds the
  perturbation parameters to the policy as its instruction, and every axis drops by several points.
- **Overall = arithmetic mean over the seven perturbation axes** (Camera 1,599 · Noise 1,601 · Lighting 1,142 ·
  Background 1,076 · Robot 1,550 · Layout 1,525 · Language 1,537 tasks). The task-weighted rate is about two
  points lower; `aggregate.py` prints both and labels the reported one.

## Checkpoints and data

Released on Hugging Face, public, no login needed:

| | Repository | Contents | Download |
| --- | --- | --- | --- |
| **Real-robot checkpoints** | [shailes-h/Molmoact2-LIT](https://huggingface.co/shailes-h/Molmoact2-LIT) | the MolmoAct2 models used in the real-robot experiments — LIT and the matched baseline, trained on the demonstrations below and evaluated on the three real tasks | `hf download shailes-h/Molmoact2-LIT` |
| **Real-robot data** | [shailes-h/yam_bimanual_manipulation](https://huggingface.co/datasets/shailes-h/yam_bimanual_manipulation) | YAM dual-arm platform, three tasks (blocks → box, dust-pan wipe, egg transfer), 292 successful episodes, three cameras, LeRobot v3.0, 4.8 GB | `hf download shailes-h/yam_bimanual_manipulation --repo-type dataset` |

## Code

One fork per framework. Every README has the same three parts — **① evaluate the released checkpoint**,
**② train, then evaluate** (from the pretrained base), **③ how LIT is integrated in that framework** —
and keeps the upstream README as `README_upstream.md`. Use the pinned branch and commit.

| Framework | Repository | Branch | Commit | Verified on a fresh machine |
| --- | --- | --- | --- | --- |
| MolmoAct2 | [Molmoact2](https://github.com/jianmanlincjx/Molmoact2) + [lerobot](https://github.com/jianmanlincjx/lerobot) submodule | `feat/libero-goal-prior-v4` | `bf8ca94` / `c802e8a6` | ✅ weights load; LIBERO and LIBERO-Plus rollouts |
| π0.5 | [Pi05](https://github.com/jianmanlincjx/Pi05) | `main` | `b227a65` | ✅ baseline and LIT weights load; rollouts |
| FAST-WAM | [fastwam](https://github.com/jianmanlincjx/fastwam) | `feat/goal-pose-prior` | `64dcd74` | ✅ LIT weights load; rollouts (environment recipe in its README) |
| ImageWAM | [ImageWAM](https://github.com/jianmanlincjx/ImageWAM) | `feat/goal-prior-bottleneck-fix` | `1792d10` | ✅ environment builds from `uv.lock` (torch 2.7.1+cu118, CUDA visible); evaluation not re-run here — needs the gated FLUX.2-dev autoencoder |

## Citation

```bibtex
@article{lin2026lit,
  title   = {Breaking the Vision--Action Shortcut: Latent Interface Training
             for Generalizable Robot Foundation Models},
  author  = {Lin, Jianman and Shailesh, Shailesh and Luo, Zhongyi and Duan, Jiafei},
  journal = {Under review},
  year    = {2026}
}
```

<div align="center">

# Latent Interface Training

### Breaking the Vision–Action Shortcut for Generalizable Robot Foundation Models

[Jianman Lin](https://github.com/jianmanlincjx)<sup>1\*</sup> · Shailesh Shailesh<sup>2\*</sup> · Zhongyi Luo<sup>3</sup> · Jiafei Duan<sup>2†</sup>

<sup>1</sup>South China University of Technology · <sup>2</sup>National University of Singapore · <sup>3</sup>Nanyang Technological University

[![Project page](https://img.shields.io/badge/Project-Page-1F4E9C)](https://jianmanlincjx.github.io/LIT/)
[![Paper](https://img.shields.io/badge/Paper-PDF-B23A32)](https://jianmanlincjx.github.io/LIT/static/paper/LIT.pdf)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-linjianman%2FLIT-f7c843)](https://huggingface.co/linjianman/LIT)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Real--robot%20data-yam__bimanual__manipulation-f7c843)](https://huggingface.co/datasets/chinchinati/yam_bimanual_manipulation)
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

<sub>LIBERO-Plus: zero-shot on all 10,030 tasks, one episode each; Overall is the mean over the seven perturbation axes.
Real robot: three tasks on the MolmoAct2 backbone, 10 rollouts per task per OOD condition. In-distribution LIBERO success is preserved or improved on every architecture.
Per-axis numbers are on the <a href="https://jianmanlincjx.github.io/LIT/#plus">project page</a>; <code>results/plot_results_strip.py</code> redraws this figure.</sub>

</div>

**Contents** &nbsp; [Code](#code) · [Checkpoints](#checkpoints) · [Quick start](#quick-start) · [Two things that change the numbers](#two-things-that-change-the-numbers) · [Integrating LIT](#integrating-lit-into-your-own-model) · [Citation](#citation)

---

## Code

One fork per framework. Every README has the same three parts — **① evaluate the released checkpoint**,
**② train, then evaluate** (from the pretrained base), **③ how LIT is integrated in that framework** —
and keeps the upstream README as `README_upstream.md`. Use the pinned branch and commit.

| Framework | Repository | Branch | Commit | Verified on a fresh machine |
| --- | --- | --- | --- | --- |
| MolmoAct2 | [Molmoact2](https://github.com/jianmanlincjx/Molmoact2) + [lerobot](https://github.com/jianmanlincjx/lerobot) submodule | `feat/libero-goal-prior-v4` | `acebade` / `b966b8c0` | ✅ weights load; LIBERO and LIBERO-Plus rollouts |
| π0.5 | [Pi05](https://github.com/jianmanlincjx/Pi05) | `main` | `b227a65` | ✅ baseline and LIT weights load; rollouts |
| FAST-WAM | [fastwam](https://github.com/jianmanlincjx/fastwam) | `feat/goal-pose-prior` | `64dcd74` | ✅ LIT weights load; rollouts (environment recipe in its README) |
| ImageWAM | [ImageWAM](https://github.com/jianmanlincjx/ImageWAM) | `feat/goal-prior-bottleneck-fix` | `1792d10` | ✅ environment builds from `uv.lock` (torch 2.7.1+cu118, CUDA visible); evaluation not re-run here — needs the gated FLUX.2-dev autoencoder |

## Checkpoints

Stage-2 models (the ones in the tables) and the Stage-1 action priors they start from are on Hugging Face:
**[linjianman/LIT](https://huggingface.co/linjianman/LIT)** (public, no login needed; 43 GB for the four Stage-2 models, 84 GB with Stage 1).

```bash
hf download linjianman/LIT --include "*/lit_stage2/*" --local-dir LIT_ckpt        # the four reported models
hf download linjianman/LIT --include "molmoact2/*" --local-dir LIT_ckpt           # one framework, both stages
```

| Directory | Format | How to use it |
| --- | --- | --- |
| `molmoact2/lit_stage2`, `pi05/lit_stage2` | LeRobot policy directory (`config.json` + `model.safetensors` + normalisers) | `--policy.path <dir>` |
| `fastwam/lit_stage2`, `imagewam/lit_stage2` | `model.pt` + `config.yaml` + `dataset_stats.json` | `ckpt=<dir>/model.pt` `dataset_stats_path=<dir>/dataset_stats.json` |
| `*/lit_stage1` | same layout as the Stage 2 of that framework | start Stage 2 from it and skip Stage 1 — each fork's README ② gives the variable (`POLICY_PATH`, `STAGE1`, `resume=`, `STAGE1_CHECKPOINT`) |

The MolmoAct2 / π0.5 baselines we fine-tuned are available on request; the FAST-WAM and ImageWAM
baselines are the authors' released weights.

**Real-robot data.** The demonstrations behind the real-robot results are public:
[chinchinati/yam_bimanual_manipulation](https://huggingface.co/datasets/chinchinati/yam_bimanual_manipulation)
— YAM dual-arm platform, three tasks (blocks → box, dust-pan wipe, egg transfer), 292 successful episodes,
three cameras, LeRobot v3.0, 4.8 GB (`hf download chinchinati/yam_bimanual_manipulation --repo-type dataset`).

## Quick start

MolmoAct2, from a clean machine. The other three forks follow the same three steps in their own READMEs.

```bash
git clone --recursive -b feat/libero-goal-prior-v4 https://github.com/jianmanlincjx/Molmoact2.git
git clone https://github.com/jianmanlincjx/LIT.git && cd LIT

export LIT_MOLMOACT2=/path/to/Molmoact2
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus            # github.com/sylvestf/LIBERO-plus
bash scripts/preflight.sh LIT_ckpt/molmoact2/lit_stage2  # submodule commit, LIBERO-Plus, GPUs, checkpoint
```

<details open>
<summary><b>① Evaluate the released checkpoint</b></summary>

```bash
bash scripts/eval_libero_plus.sh lit LIT_ckpt/molmoact2/lit_stage2   # LIBERO-Plus: 10,030 tasks, seed 1000
bash scripts/eval_libero.sh      lit LIT_ckpt/molmoact2/lit_stage2   # LIBERO: 4 suites × 50 episodes
```

Both are resumable across GPUs and print the per-axis / per-suite result when they finish.
`scripts/aggregate.py <results-root>` re-aggregates later; `--pair <a> <b>` compares two runs on the tasks both finished.

</details>

<details>
<summary><b>② Train, then evaluate</b></summary>

```bash
export DATASET_ROOT=/path/to/libero_lerobot_format
cd "$LIT_MOLMOACT2"
OPTIMIZER_ACTION_EXPERT_LR=1e-4 SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000 \
  bash scripts/libero_goal_prior/train_baseline.sh          # baseline, 30K steps, batch 32/GPU
bash scripts/libero_goal_prior_v3/train_stage1.sh           # Stage 1: no images, 10K steps, batch 128/GPU
bash scripts/libero_goal_prior_v3/train_stage2.sh           # Stage 2: 100 latents (8 pose-supervised), 30K steps, from Stage 1
```

The `_v3` launchers are the ones the paper's checkpoints were trained with (their defaults equal the
released `train_config.json`); `scripts/libero_goal_prior/` and `_v4/` are earlier / variant recipes.
Batch sizes are per GPU; the paper used 7 GPUs. Then evaluate `outputs/…/checkpoints/030000/pretrained_model` as in ①. Stage 2 reports the Stage-1
SE(3) encoder as unexpected keys when it loads — expected; the encoder is training-time only.

</details>

## Two things that change the numbers

- **`LIBERO_PLUS_FIX_LANG=1`** — set by the scripts. Without it the LIBERO-Plus harness feeds the
  perturbation parameters to the policy as its instruction, and every axis drops by several points.
- **Overall = arithmetic mean over the seven perturbation axes** (Camera 1,599 · Noise 1,601 · Lighting 1,142 ·
  Background 1,076 · Robot 1,550 · Layout 1,525 · Language 1,537 tasks). The task-weighted rate is about two
  points lower; `aggregate.py` prints both and labels the reported one.

## Integrating LIT into your own model

LIT is a change to the *interface* between a backbone and an action expert, not to either of them. It fits
any architecture in which an action expert is conditioned on backbone token features — cross-attention
(MolmoAct2), a shared self-attention over a joint sequence (π0.5), or feature injection into a DiT (FAST-WAM,
ImageWAM). Four pieces, in the order we would add them:

| # | Piece | What to build | Reference implementations |
| :-: | --- | --- | --- |
| 1 | **Firewall** | Find where visual features enter the action expert and close that path: mask image tokens out of the action expert's attention, or drop the visual feature injection. Language and state may stay (MolmoAct2) or go too (π0.5, `mask_language_from_action_expert`) — vision must have no direct path. | `mask_image_from_action_expert` in MolmoAct2's `modeling_molmoact2.py`; attention mask in Pi05 `modeling_pi05_goal_prior.py` |
| 2 | **Latent interface** | `N = 100` learnable tokens. They read the backbone's visual *and* semantic features (a small self-/cross-attention stack, or appended to the backbone sequence so the backbone contextualises them layer by layer) and are handed to the action expert at exactly the coupling point you closed in step 1 — layer-wise if the host conditions layer-wise. | `semantic_visual_recurrent` (MolmoAct2); `goal_prior.py` in Pi05 / fastwam; `goal_pose_prior.py` in ImageWAM |
| 3 | **Spatial supervision** | Reserve 8 of the latents. A 3-layer MLP (`inner_dim = 512`) decodes them to the **terminal pose of the current action chunk**: end-effector position + axis-angle rotation + gripper, read from the dataset's state at `t + H` (`H` = chunk length, 10 here) and quantile-normalised to `[-1, 1]`. Loss: MSE, weight `λ = 0.3`. | `GoalPoseDecoder` in every fork; target built in MolmoAct2's `processor_molmoact2.py` (`target_pose_delta_index`) |
| 4 | **SE(3) encoder (Stage 1 only)** | A 3-layer MLP that turns the same 8-d target into a few conditioning tokens for the action expert. Used only while training without images; deleted afterwards. | `SE3Encoder` / `GoalPoseEncoder` in every fork |

Then train in two stages, keeping the host's action representation, chunk length and generation objective untouched:

```text
Stage 1   backbone frozen · no images · action expert from scratch
          condition on  language + state + SE(3)-encoded terminal pose  →  native action loss      (10K steps)
Stage 2   init action expert from Stage 1 · drop the SE(3) encoder · enable latents + pose decoder
          fine-tune everything:  L = L_action + 0.3 · L_pose                                          (30K steps)
```

Learning rates that mattered for us: new modules (latents, aggregator) and the action expert at `1e-4` with
`5K` warmup, the backbone at `1e-5`. Interface settings were **not** tuned per architecture:
`num_latents=100`, `num_pose_tokens=8`, `latent_dim=768`, `inner_dim=512`, `lambda_pose=0.3`.

Three checks before you trust a run — each caught a silent failure for us at least once:

- **Is the interface used at all?** Measure the action expert's attention mass on the latent tokens vs. everything
  else (`tools/probe_latent_attention.py` in Pi05, `scripts/audit_goal_prior_v2.py` in ImageWAM). Near zero
  means the firewall leaks or a gate never opened.
- **Do the latents carry the pose?** Normalised pose loss should reach ~`1e-3` on training data; decode it and draw it
  back onto the frame (`scripts/render_pose_videos.sh`) — the red marker should lead the gripper, not trail it.
- **New parameters actually train and save.** Under bf16 autocast, small gates and freshly added modules can freeze
  or be dropped from the checkpoint; check the parameter count in `train_config.json` and that the new keys load.

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

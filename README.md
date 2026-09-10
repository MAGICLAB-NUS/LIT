# Latent Interface Training (LIT)

Reproduction package for **"Breaking the Vision–Action Shortcut: Latent Interface Training for
Generalizable Robot Foundation Models."**

LIT keeps a robot foundation model's backbone and Action Expert (AE) untouched and changes only the
interface between them. Direct conditioning on backbone visual tokens is removed; a small set of
learnable spatial-cue tokens becomes the only pathway through which visual information reaches the AE.
Those cues are supervised to reconstruct the terminal SE(3) goal, so what crosses the interface is
geometry rather than appearance.

Training has two stages:

- **Stage 1** — the backbone is frozen and the AE learns a spatial-goal-to-action mapping from the
  language instruction, the robot state and the terminal SE(3) end-effector pose, with no visual input.
  The AE is initialised from scratch so no existing vision–action coupling is inherited.
- **Stage 2** — learnable cue tokens aggregate goal-relevant information from the visual and semantic
  backbone representations and condition the AE layer by layer. A lightweight decoder reconstructs the
  terminal pose from the final cues. The whole model is fine-tuned jointly.

## Where the code lives

The method is a localised change to four existing codebases, so each one is kept as its own fork rather
than vendored here. Pin these exact commits to reproduce the reported numbers:

| Framework | Repository | Branch | Commit |
| --- | --- | --- | --- |
| MolmoAct2 | [`jianmanlincjx/Molmoact2`](https://github.com/jianmanlincjx/Molmoact2) | `feat/libero-goal-prior-v4` | `95062b0` |
| ↳ policy library (submodule) | [`jianmanlincjx/lerobot`](https://github.com/jianmanlincjx/lerobot) | `feat/libero-goal-prior-v4` | `b966b8c0` |
| FAST-WAM | [`jianmanlincjx/fastwam`](https://github.com/jianmanlincjx/fastwam) | `feat/goal-pose-prior` | `12e4573` |
| ImageWAM | [`jianmanlincjx/ImageWAM`](https://github.com/jianmanlincjx/ImageWAM) | `feat/goal-prior-bottleneck-fix` | `66f4b91` |
| π0.5 | [`jianmanlincjx/pi05`](https://github.com/jianmanlincjx/pi05) | `main` | `085a699` |

Each fork carries a `REPRODUCE.md` covering baseline training, LIT training, and the two evaluation
protocols for that framework.

> **On the MolmoAct2 submodule.** `Molmoact2` drives `lerobot` as a git submodule. Clone with
> `--recursive` and confirm the submodule sits at `b966b8c0` — the ablation switches
> (`LIT_ALLOW_OPEN_VISUAL_PATH`, `enable_ae_pose_head`) live there.

## What is in this repository

```
scripts/
  launchers/     training, evaluation and aggregation entry points used for the paper
  molmoact2/     the MolmoAct2 training and evaluation shell scripts, as run
```

### Training (MolmoAct2)

```bash
scripts/molmoact2/libero_goal_prior/train_baseline.sh    # baseline, no LIT
scripts/molmoact2/libero_goal_prior/train_stage1.sh      # Stage 1: visual-free action prior
scripts/molmoact2/libero_goal_prior/train_stage2.sh      # Stage 2: latent interface
```

Stage 2 is initialised from the Stage-1 checkpoint. The SE(3) encoder is used only in Stage 1 and is
discarded afterwards, so it appears as unexpected keys when the Stage-2 run loads the Stage-1 weights;
this is expected.

### Evaluation

**LIBERO (in-distribution)** — four suites, 50 episodes per task, official horizons, seed 1000,
2,000 episodes per model:

```bash
scripts/molmoact2/libero_eval/eval_libero_v4_official_8gpu.sh <checkpoint>
scripts/launchers/eval_id.sh <name> <checkpoint>          # 8-GPU wrapper + shard retry
scripts/launchers/collect_id.py <name>                     # per-suite and overall success
```

**LIBERO-Plus (out-of-distribution)** — all 10,030 tasks, one episode per task, seed 1000:

```bash
scripts/molmoact2/libero_eval/build_libero_plus_manifest.py
scripts/molmoact2/libero_eval/eval_libero_plus_checkpoint_suites.sh <checkpoint>
scripts/launchers/eval_abl_sliced.sh <name> <checkpoint>   # shard across GPUs, resumable
scripts/launchers/collect_plus.py                          # per-axis aggregation
```

Set `LIBERO_PLUS_FIX_LANG=1`. Without it the harness derives each instruction from the task filename,
which feeds the perturbation parameters to the model as the instruction and depresses every axis.

### Aggregating results

`collect_abl.py` pairs an ablation arm against the baseline and the full model task by task, so every
axis is compared on the same denominator even while a run is still in progress. `show_abl.py` renders
the comparison.

Success rates are reported per perturbation axis. The seven axes carry unequal task counts — Camera
1,599 · Noise 1,601 · Lighting 1,142 · Background 1,076 · Robot 1,550 · Layout 1,525 · Language 1,537,
10,030 in total — so the arithmetic mean over axes and the task-weighted rate differ by roughly two
points. The paper reports the arithmetic mean over the seven axes.

## Checkpoints

Released separately; links will be added here.

## Citation

```bibtex
@inproceedings{lit2026,
  title     = {Breaking the Vision--Action Shortcut: Latent Interface Training
               for Generalizable Robot Foundation Models},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```

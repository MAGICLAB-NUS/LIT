# Latent Interface Training (LIT)

Code, checkpoints and reproduction steps for **Breaking the Vision–Action Shortcut: Latent Interface
Training for Generalizable Robot Foundation Models**. Project page: https://jianmanlincjx.github.io/LIT/

LIT keeps a robot foundation model's backbone and action expert untouched and changes only the interface
between them: direct conditioning on visual tokens is removed, and a small set of learnable latent tokens —
supervised to reconstruct the terminal SE(3) end-effector pose — becomes the action expert's only route to
vision. Stage 1 learns an action prior from language, state and that pose with no images; Stage 2 trains
the interface.

## Code

One fork per framework. Each carries a `REPRODUCE.md` with its training and evaluation commands and the
checkpoint that produced each reported number. Use the pinned branch/commit.

| Framework | Repository | Branch | Commit | Verified on a fresh machine |
| --- | --- | --- | --- | --- |
| MolmoAct2 | [Molmoact2](https://github.com/jianmanlincjx/Molmoact2) (+ [lerobot](https://github.com/jianmanlincjx/lerobot) submodule) | `feat/libero-goal-prior-v4` | `e6d4dfe` / `b966b8c0` | weights load; LIBERO and LIBERO-Plus rollouts ✓ |
| π0.5 | [Pi05](https://github.com/jianmanlincjx/Pi05) | `main` | `b992444` | baseline and LIT weights load; rollouts ✓ |
| FAST-WAM | [fastwam](https://github.com/jianmanlincjx/fastwam) | `feat/goal-pose-prior` | `fc15028` | LIT weights load; 2/2 LIBERO rollouts ✓ (env recipe in its REPRODUCE.md) |
| ImageWAM | [ImageWAM](https://github.com/jianmanlincjx/ImageWAM) | `feat/goal-prior-bottleneck-fix` | `07e0026` | in progress |

## Checkpoints

Released on Hugging Face at **https://huggingface.co/linjianman/LIT** (stage-2 models; ModelScope mirror to follow), one directory per model:

```
molmoact2/lit_stage2     LeRobot policy dir — pass the directory to --policy.path
pi05/lit_stage2          LeRobot policy dir
fastwam/lit_stage2       model.pt + config.yaml + dataset_stats.json
imagewam/lit_stage2      model.pt + config.yaml + dataset_stats.json
```

`lit_stage2` is the model reported in the tables. Stage-1 action priors and the MolmoAct2 / π0.5 baselines
we fine-tuned are available on request; FAST-WAM and ImageWAM baselines are the authors' released weights.

```bash
hf download linjianman/LIT --local-dir LIT_ckpt      # all four, 41 GB
hf download linjianman/LIT --include "molmoact2/*" --local-dir LIT_ckpt
```

## Quick start (MolmoAct2)

```bash
git clone --recursive -b feat/libero-goal-prior-v4 https://github.com/jianmanlincjx/Molmoact2.git
git clone https://github.com/jianmanlincjx/LIT.git && cd LIT

export LIT_MOLMOACT2=/path/to/Molmoact2
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus     # https://github.com/sylvestf/LIBERO-plus
bash scripts/preflight.sh /path/to/molmoact2/lit_stage2   # checks submodule commit, LIBERO-Plus, GPUs, and the checkpoint
```

**Evaluate a released checkpoint**

```bash
bash scripts/eval_libero_plus.sh lit /path/to/molmoact2/lit_stage2   # LIBERO-Plus, 10,030 tasks, seed 1000
bash scripts/eval_libero.sh      lit /path/to/molmoact2/lit_stage2   # LIBERO, 4 suites x 50 episodes
```

Both are resumable and print the per-axis / per-suite result when done. `scripts/aggregate.py <results-root>`
re-aggregates later; `--pair <a> <b>` compares two runs on the tasks both finished.

**Train, then evaluate**

```bash
export DATASET_ROOT=/path/to/libero_lerobot_format
cd "$LIT_MOLMOACT2"
bash scripts/libero_goal_prior/train_baseline.sh   # baseline
bash scripts/libero_goal_prior/train_stage1.sh     # Stage 1, 10K steps, no images
bash scripts/libero_goal_prior/train_stage2.sh     # Stage 2, 30K steps, from the Stage-1 checkpoint
```

Then evaluate `outputs/.../checkpoints/030000/pretrained_model` as above. Stage 2 reports the Stage-1
SE(3) encoder as unexpected keys when it loads — expected; the encoder is training-time only.

For π0.5, FAST-WAM and ImageWAM follow the `REPRODUCE.md` in each fork; the protocols are the same.

## Two things that change the numbers

- **`LIBERO_PLUS_FIX_LANG=1`** (set by the scripts). Without it the LIBERO-Plus harness feeds the
  perturbation parameters to the policy as its instruction, and every axis drops by several points.
- **Overall is the arithmetic mean over the seven perturbation axes** (Camera 1,599 · Noise 1,601 ·
  Lighting 1,142 · Background 1,076 · Robot 1,550 · Layout 1,525 · Language 1,537 tasks). The
  task-weighted rate is about two points lower; `aggregate.py` prints both and labels the reported one.

## Layout

```
scripts/
  env.sh preflight.sh          paths and pre-run checks
  eval_libero.sh eval_libero_plus.sh aggregate.py
  render_pose_videos.sh        rollouts with the decoded pose drawn back onto the frame (project-page clips)
  molmoact2/                   the MolmoAct2 train/eval scripts at the pinned commit
  paper_runs/                  launchers exactly as run for the paper (machine-specific; provenance only)
docs/                          project page
```

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

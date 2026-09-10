# Latent Interface Training (LIT)

Reproduction package for **"Breaking the Vision–Action Shortcut: Latent Interface Training for
Generalizable Robot Foundation Models."**

LIT keeps a robot foundation model's backbone and Action Expert (AE) untouched and changes only the
interface between them. Direct conditioning on backbone visual tokens is removed; a small set of
learnable spatial-cue tokens becomes the only pathway through which visual information reaches the AE.
Those cues are supervised to reconstruct the terminal SE(3) goal, so what crosses the interface is
geometry rather than appearance.

- **Stage 1** — the backbone is frozen and the AE learns a spatial-goal-to-action mapping from the
  language instruction, the robot state and the terminal SE(3) end-effector pose, with no visual input.
  The AE starts from scratch, so no existing vision–action coupling is inherited.
- **Stage 2** — learnable cue tokens aggregate goal-relevant information from the visual and semantic
  backbone representations and condition the AE layer by layer. A lightweight decoder reconstructs the
  terminal pose from the final cues. The whole model is fine-tuned jointly.

## Where the code lives

The method is a localised change to four existing codebases, so each stays its own fork rather than
being vendored here. Pin these commits:

| Framework | Repository | Branch | Commit | Verified on a fresh machine |
| --- | --- | --- | --- | --- |
| MolmoAct2 | [`jianmanlincjx/Molmoact2`](https://github.com/jianmanlincjx/Molmoact2) | `feat/libero-goal-prior-v4` | `33d59ae` | released weights load; LIBERO and LIBERO-Plus rollouts run (2026-09-10) |
| ↳ policy library (submodule) | [`jianmanlincjx/lerobot`](https://github.com/jianmanlincjx/lerobot) | `feat/libero-goal-prior-v4` | `b966b8c0` | — |
| π0.5 | [`jianmanlincjx/Pi05`](https://github.com/jianmanlincjx/Pi05) | `main` | `d5f13e6` | baseline and LIT weights load; 2/2 LIBERO rollouts each (2026-09-10) |
| FAST-WAM | [`jianmanlincjx/fastwam`](https://github.com/jianmanlincjx/fastwam) | `feat/goal-pose-prior` | `9450718` | environment build in progress |
| ImageWAM | [`jianmanlincjx/ImageWAM`](https://github.com/jianmanlincjx/ImageWAM) | `feat/goal-prior-bottleneck-fix` | `07e0026` | environment build in progress |

`Molmoact2` drives `lerobot` as a git submodule, so clone it recursively. The ablation switches
(`LIT_ALLOW_OPEN_VISUAL_PATH`, `enable_ae_pose_head`) live in the submodule.

`Pi05` is a small package on top of **upstream LeRobot 0.6.2** (stock `pi05` policy, untouched):
importing `pi05_goal_prior` registers the `pi05_goal_prior` policy type. Released π0.5 checkpoints
carry `"type": "pi05"` (baseline) or `"type": "pi05_goal_prior"` (LIT) in `config.json`, and
`lerobot_eval` loads either directly.

Project page: **https://jianmanlincjx.github.io/LIT/**

```bash
git clone --recursive -b feat/libero-goal-prior-v4 \
  https://github.com/jianmanlincjx/Molmoact2.git
```

The commands below are for MolmoAct2. The other three forks each carry a `REPRODUCE.md` with the
equivalent steps for that framework.

## Setup

Three paths have to be pointed at real directories. Export them, or copy `scripts/env.local.sh.example`
to `scripts/env.local.sh` and edit it.

```bash
export LIT_MOLMOACT2=/path/to/Molmoact2          # the checkout above
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus     # contains libero/libero/benchmark/
export DATASET_ROOT=/path/to/libero_lerobot      # training only
```

Then check the environment before spending GPU hours:

```bash
bash scripts/preflight.sh /path/to/checkpoint
```

It verifies the submodule commit, that `task_classification.json` really lists 10,030 tasks, that the
LIBERO-plus assets are present, how many GPUs are visible, and which LIT switches the checkpoint was
trained with — so a baseline checkpoint cannot be mistaken for a LIT one.

## 1. Evaluate a released checkpoint

Released checkpoints are flat directories: `config.json`, `model.safetensors` and the
preprocessor/postprocessor files sit directly inside. Pass that directory.

```bash
# out-of-distribution: all 10,030 LIBERO-Plus tasks, 1 episode each, seed 1000
bash scripts/eval_libero_plus.sh lit_full /path/to/lit_full_stage2_030000

# in-distribution: 4 suites x 50 episodes per task, official horizons, seed 1000
bash scripts/eval_libero.sh      lit_full /path/to/lit_full_stage2_030000
```

Both print the aggregated result when they finish, and both are resumable — re-running the same run
name skips tasks that already have a result. Results and logs go to `$LIT_WORK`
(`_work/` next to this repository by default).

To aggregate again later, or to compare two runs task by task:

```bash
python3 scripts/aggregate.py            _work/libero_plus/lit_full
python3 scripts/aggregate.py --libero   _work/libero/lit_full
python3 scripts/aggregate.py --pair     _work/libero_plus/baseline _work/libero_plus/lit_full
```

`--pair` restricts both runs to the tasks they have both finished, so the comparison stays honest
while a run is still in progress.

## 2. Train, then evaluate

```bash
cd "$LIT_MOLMOACT2"
bash scripts/libero_goal_prior/train_baseline.sh    # baseline, no LIT
bash scripts/libero_goal_prior/train_stage1.sh      # Stage 1: visual-free action prior
bash scripts/libero_goal_prior/train_stage2.sh      # Stage 2: latent interface
```

Stage 2 reads the Stage-1 checkpoint from `STAGE1_OUTPUT_DIR` (default: the Stage-1 output path for
the same seed). The SE(3) encoder exists only in Stage 1 and is dropped afterwards, so Stage 2 reports
it as unexpected keys when it loads the Stage-1 weights — that is expected, not a failed load.

Then evaluate the resulting checkpoint exactly as in section 1, pointing at
`outputs/.../checkpoints/NNNNNN/pretrained_model`.

## Reading the numbers

Success rates are reported per perturbation axis. The seven axes carry unequal task counts —
Camera 1,599 · Noise 1,601 · Lighting 1,142 · Background 1,076 · Robot 1,550 · Layout 1,525 ·
Language 1,537, 10,030 in total — so the arithmetic mean over axes and the task-weighted rate differ
by roughly two points. **The paper reports the arithmetic mean over the seven axes**; `aggregate.py`
prints both and labels which one is reported.

`LIBERO_PLUS_FIX_LANG=1` is set by `eval_libero_plus.sh` and should stay set. Without it the harness
derives each instruction from the task filename, which hands the perturbation parameters to the model
as its instruction and depresses every axis by several points.

## Layout

```
scripts/
  env.sh              the three paths, plus the checks the entry points share
  preflight.sh        verify the environment and a checkpoint before running
  eval_libero.sh      in-distribution LIBERO, 8 shards, retries missing shards
  eval_libero_plus.sh LIBERO-Plus, sliced across GPUs, resumable
  aggregate.py        per-axis / per-suite aggregation and paired comparison
  molmoact2/          the MolmoAct2 training and evaluation scripts, at the pinned commit
  render_pose_videos.sh
                      re-render rollouts with the decoded terminal pose drawn back onto the
                      frame (the clips on the project page); one line per clip in JOBS —
                      `gpu envtype suite task_id label`, envtype `libero` or `libero_plus`
  paper_runs/         the launchers as they were actually run for the paper (machine-specific
                      paths; kept for provenance, not meant to run elsewhere)
docs/                 the project page (GitHub Pages, served from /docs)
```

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

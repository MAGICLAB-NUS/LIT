# DROID Goal-Pose Prior reproduction

This directory contains the fail-closed data preparation and two-stage training
entry points for the DROID adaptation. The source dataset is never modified.

## Pinned inputs

- Dataset: [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)
- Dataset revision: `0eabc778f959c54b8c5aa3626cc1128d2d2e54d4`
- Stage-1 VLM: [`allenai/Molmo2-ER`](https://huggingface.co/allenai/Molmo2-ER), revision `dab22564403d2607855bb1fffb0721285b445081`
- Architecture template: [`allenai/MolmoAct2`](https://huggingface.co/allenai/MolmoAct2), revision `e432d85f6e039edca44afb93c262f3084ab72a9c`, with a newly randomized continuous Action Expert

Initialize the submodule environment and download both pinned model snapshots:

```bash
git submodule update --init --recursive
(cd lerobot && uv sync --extra molmoact2)

lerobot/.venv/bin/hf download allenai/MolmoAct2 \
  --revision e432d85f6e039edca44afb93c262f3084ab72a9c \
  --local-dir Checkpoint/MolmoAct2

lerobot/.venv/bin/hf download allenai/Molmo2-ER \
  --revision dab22564403d2607855bb1fffb0721285b445081 \
  --local-dir Checkpoint/Molmo2-ER
```

The launchers verify these revisions from the downloaded metadata and refuse a
different or unverifiable local snapshot.

The DROID dataset is expected at `/data0/JM/dataset/droid_1.0.1`. A mirror can
be used for the initial Hub download, but the recorded revision must match:

```bash
HF_ENDPOINT=https://hf-mirror.com \
lerobot/.venv/bin/hf download lerobot/droid_1.0.1 \
  --repo-type dataset \
  --revision 0eabc778f959c54b8c5aa3626cc1128d2d2e54d4 \
  --local-dir /data0/JM/dataset/droid_1.0.1 \
  --max-workers 2
```

## Dataset construction

`prepare_dataset.py` creates a derived LeRobot v3 view with:

- unchanged 8-D `observation.state` and 8-D `action`;
- `observation.ee_pose = [xyz, axis-angle, gripper]`, derived from the recorded
  Cartesian wrist-site RPY and gripper value;
- original row, episode, task, timestamp, and video identities;
- a `valid_anchor_indices.parquet` training manifest; and
- state, future-goal, and action statistics computed only from valid anchors.

Videos are symlinked to the source and are never decoded or re-encoded during
construction. Data parquet files are mirrored because the new 7-D feature must
be materialized.

The anchor rule is deterministic:

1. require `is_episode_successful=true`;
2. resolve `task_index` through `meta/tasks.parquet` and require non-empty
   canonical task text;
3. require current state, 15 actions, and the `t+15` goal to remain contiguous
   in the same episode;
4. apply the DROID/OpenPI joint-velocity idle segmentation pinned as
   `Physical-Intelligence/openpi@c23745b5`: consecutive commands with
   per-joint delta below `1e-3` are idle, idle runs of at least seven frames are
   removed, retained runs shorter than 16 frames are removed; and
5. trim the final 15 frames of each retained run so the complete action/goal
   horizon remains non-idle.

Step 5 is the explicit 15-step-horizon adaptation of OpenPI's original
`filter_last_n_in_ranges=10`; the segmentation itself is unchanged. Both the
upstream value and the local adaptation are recorded in provenance. Changing
them creates a different dataset recipe and must be reported as a separate
experiment.

Build a 100-episode smoke view first:

```bash
lerobot/.venv/bin/python scripts/droid_goal_prior/prepare_dataset.py \
  --source-root /data0/JM/dataset/droid_1.0.1 \
  --output-root /data0/JM/dataset/droid_1.0.1_goal_pose_smoke_v3 \
  --max-episodes 100 \
  --smoke-anchors 128

lerobot/.venv/bin/python scripts/droid_goal_prior/prepare_dataset.py \
  --source-root /data0/JM/dataset/droid_1.0.1 \
  --output-root /data0/JM/dataset/droid_1.0.1_goal_pose_smoke_v3 \
  --verify-only
```

After the smoke view passes, build the complete view by omitting the limit:

```bash
lerobot/.venv/bin/python scripts/droid_goal_prior/prepare_dataset.py \
  --source-root /data0/JM/dataset/droid_1.0.1 \
  --output-root /data0/JM/dataset/droid_1.0.1_goal_pose
```

The build uses a sibling staging directory and is published atomically only
after strict verification. Re-running the same completed configuration is
idempotent. The script refuses to overwrite incomplete output or output built
with a different configuration.

## Required dataset metadata

The derived dataset defaults to
`/data0/JM/dataset/droid_1.0.1_goal_pose`. Both stages require the
`valid_anchor_indices.parquet` path and validate it against
`goal_pose_provenance.json`.

The provenance file is produced by `prepare_dataset.py`. Its relevant contract
is:

```json
{
  "format_version": 1,
  "recipe_version": 3,
  "config": {
    "source_root": "/data0/JM/dataset/droid_1.0.1",
    "output_root": "/data0/JM/dataset/droid_1.0.1_goal_pose",
    "revision": "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4",
    "horizon": 15
  },
  "source_read_only": true,
  "artifact_sha256": {
    "valid_anchor_indices.parquet": "<sha256>",
    "meta/stats.json": "<sha256>"
  }
}
```

`meta/info.json` must expose 8-D `observation.state`, 8-D `action`, and 7-D
`observation.ee_pose`. `meta/stats.json` must contain finite, shape-correct,
monotonic statistics for all three features. The scripts recompute and compare
the provenance, manifest, and stats SHA-256 hashes before launch. The
preparation verifier additionally reconstructs every anchor from source rows,
checks every original parquet column byte-semantically through Arrow equality,
recomputes every EE pose and statistic, and verifies that `videos` resolves to
the original source directory.

## Launch

The complete recipe currently contains **17,259,872 valid training anchors from
74,546 episodes** after success, language, horizon, and non-idle filtering.
Training lengths below are based on anchor exposure, rather than copying the
LIBERO schedule.

### Smoke test

Run **100 steps per stage** on the 100-episode smoke view. This validates data
decoding, forward/backward, checkpoint serialization, and Stage-1-to-Stage-2
loading; it is not intended to produce a useful policy.

```bash
DATASET_ROOT=/data0/JM/dataset/droid_1.0.1_goal_pose_smoke_v3 \
SAMPLE_MANIFEST_PATH=/data0/JM/dataset/droid_1.0.1_goal_pose_smoke_v3/valid_anchor_indices.parquet \
SMOKE_RUN=true \
SMOKE_STEPS=100 \
bash scripts/droid_goal_prior/train_stage1.sh

DATASET_ROOT=/data0/JM/dataset/droid_1.0.1_goal_pose_smoke_v3 \
SAMPLE_MANIFEST_PATH=/data0/JM/dataset/droid_1.0.1_goal_pose_smoke_v3/valid_anchor_indices.parquet \
SMOKE_RUN=true \
STAGE1_SMOKE_STEPS=100 \
SMOKE_STEPS=100 \
bash scripts/droid_goal_prior/train_stage2.sh
```

The smoke defaults are already 100 steps, so the explicit step variables above
mainly document the Stage-1/Stage-2 checkpoint contract.

### Formal training

The schedule is chosen by total sample exposure, not raw optimizer steps:

```text
data-equivalent epochs = steps × batch_size_per_gpu × number_of_gpus / 17,259,872
```

The strongest directly comparable public reference is OpenPI's
[`pi05_full_droid_finetune`](https://github.com/Physical-Intelligence/openpi/blob/c23745b5/examples/droid/README_train.md):
it uses **8 H100 GPUs, global batch 256, and 100k steps** (25.6M samples,
reported as approximately one original-DROID epoch) when initialized from
pi0.5. OpenPI recommends **240k steps at the same global batch** (61.44M
samples, approximately three original-DROID epochs) when initialized only from
PaliGemma. The original
[DROID diffusion-policy experiment](https://arxiv.org/html/2403.12945) used one
A100, batch 128, and 25k steps, but it trained a much smaller task-specific
policy with 50% DROID sampling, so its 25k step count is not an appropriate
budget for full-DROID 5B-parameter adaptation.

Our recommended 8-GPU schedule is therefore:

- **Stage 1: 50,000 steps, batch size 128 per GPU.** The global batch is 1,024.
  Because the continuous Action Expert is randomly initialized and this stage
  is relatively inexpensive, it receives 51.2M anchor samples, or about
  **2.97 data-equivalent epochs**.
- **Stage 2: 180,000 steps, batch size 24 per GPU.** The global batch is 192.
  Stage 2 inherits the trained action prior but must still adapt the visual
  pathway to DROID. It receives 34.56M anchor samples, or about **2.00
  data-equivalent epochs**, midway between OpenPI's one-epoch pretrained recipe
  and three-epoch weaker-initialization recipe. Checkpoints are retained every
  30k steps through 180k.

The pinned optimization recipe is:

- **Stage 1:** Action Expert and goal encoder LR `5e-5`, 500-step warmup, then
  cosine decay to `5e-6` over 50k steps.
- **Stage 2:** VLM/ViT/connector LR `1e-5`; Action Expert and semantic-visual
  modules LR `1e-4`; all groups use a 5k-step warmup and cosine decay to 10% of
  their peak LR over 180k steps.
- **Both stages:** AdamW with betas `(0.9, 0.95)`, epsilon `1e-6`, weight decay
  `0`, and global gradient clipping at `1.0`.

Stage 1 follows OpenPI's full-DROID optimization scale (`5e-5`, AdamW,
gradient clip `1.0`, and approximately 1% warmup), while retaining cosine decay
for the randomly initialized Action Expert. Stage 2 uses the component-wise
scale already exercised by the LIBERO v3 high-LR experiment and independently used by
[StarVLA](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/starVLA_guideline.md):
`1e-5` for the visual-language backbone and `1e-4` for action modules. The
longer 5k warmup protects the inherited Stage-1 prior during joint visual
adaptation. These are architectural analogies, not claims of exact optimizer
equivalence across models.

Run Stage 1 to completion before starting Stage 2:

```bash
DATASET_ROOT=/data0/JM/dataset/droid_1.0.1_goal_pose \
SAMPLE_MANIFEST_PATH=/data0/JM/dataset/droid_1.0.1_goal_pose/valid_anchor_indices.parquet \
STEPS=50000 \
BATCH_SIZE=128 \
bash scripts/droid_goal_prior/train_stage1.sh

DATASET_ROOT=/data0/JM/dataset/droid_1.0.1_goal_pose \
SAMPLE_MANIFEST_PATH=/data0/JM/dataset/droid_1.0.1_goal_pose/valid_anchor_indices.parquet \
STAGE1_FORMAL_STEPS=50000 \
STEPS=180000 \
BATCH_SIZE=24 \
bash scripts/droid_goal_prior/train_stage2.sh
```

Stage 2 accepts the matching
`lerobot/outputs/droid_goal_prior/seed_<seed>/stage1/checkpoints/050000/pretrained_model`
checkpoint by default and validates its training configuration, exact processor
stats/masks, Action Expert fingerprint, and missing/unexpected-key allowlist.
The scripts use these formal values by default; the explicit variables make the
intended schedule auditable in shell history.

The verified 80 GB GPU defaults are `BATCH_SIZE=128` per GPU for Stage 1 and
`BATCH_SIZE=24` per GPU for Stage 2. Stage-1 `bs=256` reduced throughput despite
ample memory. Stage-2 `bs=32` ran in a single process but OOMed during the first
DDP backward with three camera streams; `bs=24` completed 7-GPU DDP with about
64.7 GB peak memory per GPU. These values remain overridable through
`BATCH_SIZE`.

`SMOKE_STEPS`, `SMOKE_BATCH_SIZE`, `SMOKE_NUM_WORKERS`, `SMOKE_SAVE_FREQ`,
`SMOKE_LOG_FREQ`, and `SMOKE_SAVE_CHECKPOINT` override the smoke defaults.

To inspect a command without a prepared dataset or checkpoint:

```bash
DRY_RUN=true VALIDATE_DATASET=false \
bash scripts/droid_goal_prior/train_stage1.sh \
  --sample-manifest /path/to/future_manifest.parquet

DRY_RUN=true VALIDATE_DATASET=false VALIDATE_STAGE1=false \
bash scripts/droid_goal_prior/train_stage2.sh \
  --sample-manifest /path/to/future_manifest.parquet
```

Validation bypasses are rejected unless `DRY_RUN=true`.

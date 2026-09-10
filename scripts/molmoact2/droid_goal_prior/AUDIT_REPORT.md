# DROID formal-training audit report

Audit date: 2026-08-02  
Dataset recipe: v3  
Decision: **GO for formal Stage 1; GO for Stage 2 after the validated Stage-1 checkpoint is complete.**

This audit used the existing `lerobot/.venv` environment. It did not rebuild an
environment and did not repeat either training smoke run.

## Scope and evidence

The audited source is `/data0/JM/dataset/droid_1.0.1` at revision
`0eabc778f959c54b8c5aa3626cc1128d2d2e54d4`. The derived dataset is
`/data0/JM/dataset/droid_1.0.1_goal_pose`.

Machine-readable evidence is retained in:

- `audit_dataset_report.json`
- `audit_normalization_report.json`
- `dataloader_golden_report.json`

The relevant artifact hashes are:

- manifest:
  `9138c8d590658f29edb9e7af1f25f288dcd6202f3f3702323c1c3b6a40b02d6b`
- `meta/stats.json`:
  `c418b303c77aaeb70c82ffe365a2a874779edf08754fd51db4b2bf5451410726`
- `meta/info.json`:
  `f63cb73223d0cb614bc691d19d5449cad1063f155760d80869108633ee179942`

## Data integrity: PASS

`prepare_dataset.py --verify-only` completed over the full recipe:

- 86 data parquet files;
- 27,630,375 source and derived frames;
- 95,658 episodes, including 78,864 successful episodes; and
- 17,259,872 valid anchors from 74,546 episodes.

The verifier checked every data parquet, not a sample. For every retained row it
confirmed:

- source and derived row counts and recorded SHA-256 hashes;
- Arrow equality of every original column;
- exact `float32` reconstruction of
  `observation.ee_pose = [xyz, axis-angle, gripper]`;
- RPY-to-axis-angle rotation round-trip (`atol=2e-6`);
- finite pose values;
- contiguous global index, frame index, episode storage, and 15 Hz timestamps;
- exact episode metadata pairing;
- exact regeneration of the anchor manifest; and
- exact regeneration of manifest-aligned state, action-window, and future-goal
  statistics.

The independent audit additionally streamed all 17,259,872 manifest rows and
found no duplicate, ordering, schema, identity, task, or range discrepancy. It
checked 96 deterministic ordinary/episode-boundary/parquet-boundary anchors
directly against source and derived windows. Every checked sample had:

- current state at `t`;
- 15 actions at `[t, t+14]`;
- goal pose at exactly `t+15`;
- one episode and contiguous timestamps; and
- a non-empty canonical language task.

All 95,658 episode metadata records were inventoried across the three expected
cameras. They reference 784 unique video files, with zero missing files. The
derived `videos` directory is a symlink to the immutable source video tree, so
no video was split, decoded, or re-encoded.

## OpenPI non-idle parity: PASS

The audit contains an independent transcription of
`Physical-Intelligence/openpi@c23745b5`'s DROID segmentation. Across 100
episodes containing qualifying idle runs:

- idle threshold: per-joint command delta `< 1e-3`;
- minimum idle run: 7 frames;
- minimum retained run: 16 frames;
- upstream terminal trim: 10 frames; and
- local terminal trim: 15 frames.

The local mask exactly matched the independently reconstructed trim-15 mask.
Its difference from upstream was exactly the additional five terminal frames
required by this project's 15-step action/goal horizon.

## Normalization: PASS

The normalization audit rescanned all exact training samples:

- state: 17,259,872 vectors at `t`;
- action: 258,898,080 vectors from 15-step windows; and
- goal pose: 17,259,872 vectors at `t+15`.

All dimensions were finite and their counts, minima, maxima, and q01/q99 values
matched `meta/stats.json`. Every normalized non-gripper dimension had the
expected approximately 2% total q01/q99 tail clipping. In particular, goal-pose
`x`, `y`, and `z` each had clipping fraction `0.0200000324`; there is no
pathological Z clipping like the earlier incorrect-normalization failure.

Forward QUANTILES normalization, clamp to `[-1, 1]`, and inverse restoration
were tested over all values. The maximum numerical round-trip error was below
`5e-16`.

The metadata-derived masks are:

- state: seven normalized joints plus one raw gripper;
- action: seven normalized joints plus one raw gripper; and
- goal pose: six normalized SE(3) dimensions plus one raw gripper.

All three raw gripper dimensions remained in `[0, 1]`. They are not quantile
normalized and are therefore not distorted by dataset-dependent gripper
quantiles.

## Stage-1 to Stage-2 lineage: PASS

The launch path now fails closed on checkpoint topology:

- Stage 2 may miss only newly introduced `semantic_visual_*` parameters;
- Stage 2 may receive only the obsolete Stage-1 `goal_se3_encoder.*` parameters
  as unexpected keys;
- any Action Expert missing/unexpected key aborts loading; and
- any other topology difference aborts loading.

The loader computes the same deterministic Action Expert fingerprint directly
from small safetensors slices and from the loaded module. A mismatch aborts
before training. This avoids silently starting Stage 2 with a partially loaded
or reinitialized action prior.

`train_stage2.sh` also validates that the Stage-1 manifest path is the current
full-recipe manifest and compares every saved state/action/goal normalization
tensor and mask in both processor state files against the current dataset
statistics. The real Stage-1 smoke checkpoint passed this lineage dry run.

## Real dataloader-to-processor golden batch: PASS

The golden check initialized the complete 17,259,872-anchor manifest and chose
the first, middle, and last public samples. Their absolute indices were
`167`, `13,764,704`, and `27,630,359`, spanning episodes `1`, `47,798`, and
`95,657`.

The decoded raw shapes were state `(3, 8)`, action `(3, 15, 8)`, and goal
feature `(3, 2, 7)`. The Stage-2 processor produced:

- state `(3, 8)`;
- padded action `(3, 15, 32)`;
- extracted `t+15` goal `(3, 7)`; and
- non-empty three-camera visual tensors.

State, action, and goal values matched independent q01/q99 normalization,
gripper masking, and clamping with maximum absolute error exactly `0.0`. No
action or goal row was padded, and only action dimensions 8 through 31 were
marked as model padding.

## Regression results

The focused DROID and existing MolmoAct2/LIBERO regression suite completed with
`99 passed`. It covers preparation fixtures, failure injection, manifest
mapping and strict validation, normalization contracts, Stage-1/Stage-2 key
allowlisting, Action Expert fingerprinting, and the existing independent-goal
feature behavior.

## Non-blocking warnings

LeRobot reports `goal_pose_prior` as an unknown extension field in
`meta/info.json`. It preserves and ignores that extension while reading all
standard features; the launch preflight and audit scripts validate the field
directly. Transformers also emits processor deprecation warnings. Neither
warning changes data, tensor values, or model inputs.

## Final decision and launch conditions

**GO** means the audited recipe, manifest, stats, sampling, processor behavior,
and two-stage lineage are suitable for expensive training. It is conditional
on leaving the following immutable:

- source revision and recipe-v3 provenance;
- the three artifact hashes listed above;
- horizon 15 and the recorded OpenPI non-idle parameters;
- Stage-1 initialization from pinned Molmo2-ER with a randomized Action Expert;
  and
- Stage 2 loading the exact Stage-1 terminal checkpoint selected by the launch
  script.

Any change to those values is a new recipe and requires rerunning the audit.

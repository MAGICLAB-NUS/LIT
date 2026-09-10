#!/usr/bin/env python3
"""Golden-check one real DROID manifest batch through the Stage-2 processor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

import lerobot.policies.molmoact2.configuration_molmoact2  # noqa: F401
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

GOAL_KEY = "observation.ee_pose"


def _expected_normalized(
    values: torch.Tensor,
    stats: dict[str, Any],
    *,
    gripper_index: int,
) -> torch.Tensor:
    q01 = torch.as_tensor(stats["q01"], dtype=values.dtype)
    q99 = torch.as_tensor(stats["q99"], dtype=values.dtype)
    normalized = 2.0 * (values - q01) / (q99 - q01) - 1.0
    normalized[..., gripper_index] = values[..., gripper_index]
    return normalized.clamp(-1.0, 1.0)


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.to(dtype=torch.float64) - expected.to(dtype=torch.float64)).abs().max())


def validate_batch(
    checkpoint: Path,
    *,
    sample_positions: list[int] | None = None,
) -> dict[str, Any]:
    train_config_path = checkpoint / "train_config.json"
    cfg = TrainPipelineConfig.from_pretrained(train_config_path)
    cfg.dataset.image_transforms.enable = True
    dataset = make_dataset(cfg)
    if len(dataset) < 3:
        raise ValueError("golden validation requires at least three manifest anchors")

    positions = sample_positions or [0, len(dataset) // 2, len(dataset) - 1]
    if len(set(positions)) != len(positions) or any(
        position < 0 or position >= len(dataset) for position in positions
    ):
        raise ValueError(f"invalid public manifest positions: {positions}")

    manifest = dataset.reader._sample_manifest_rows
    if manifest is None:
        raise RuntimeError("dataset did not load the configured sample manifest")
    items = [dataset[position] for position in positions]
    for position, item in zip(positions, items, strict=True):
        expected_identity = (
            int(manifest.indices[position]),
            int(manifest.episode_indices[position]),
            int(manifest.frame_indices[position]),
        )
        actual_identity = (
            int(item["index"]),
            int(item["episode_index"]),
            int(item["frame_index"]),
        )
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"manifest identity mismatch at public position {position}: "
                f"{actual_identity} != {expected_identity}"
            )

    batch = default_collate(items)
    for camera_key in dataset.meta.camera_keys:
        if batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
    raw_state = batch[OBS_STATE].clone()
    raw_action = batch[ACTION].clone()
    raw_goal = batch[GOAL_KEY][:, -1].clone()

    policy = cfg.policy
    policy.pretrained_path = checkpoint
    policy.disable_visual_input = False
    policy.enable_goal_pose = True
    policy.goal_pose_feature_key = GOAL_KEY
    policy.goal_token_source = "learnable_queries"
    policy.goal_conditioning_mode = "semantic_visual_recurrent"
    policy.mask_image_from_action_expert = True
    policy.enable_pose_reconstruction = True

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.input_features, **policy.output_features},
                "norm_map": policy.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.output_features,
                "norm_map": policy.normalization_mapping,
            }
        },
    )
    processed = preprocessor(batch)

    expected_state = _expected_normalized(
        raw_state, dataset.meta.stats[OBS_STATE], gripper_index=7
    )
    expected_action = _expected_normalized(
        raw_action, dataset.meta.stats[ACTION], gripper_index=7
    )
    expected_goal = _expected_normalized(
        raw_goal, dataset.meta.stats[GOAL_KEY], gripper_index=6
    )
    errors = {
        "state": _max_error(processed[OBS_STATE], expected_state),
        "action": _max_error(processed[ACTION][..., :8], expected_action),
        "goal_pose": _max_error(processed["goal_pose"], expected_goal),
    }
    if any(error > 1e-6 for error in errors.values()):
        raise RuntimeError(f"processor normalization mismatch: {errors}")
    if bool(processed["action_horizon_is_pad"].any()):
        raise RuntimeError("manifest batch unexpectedly contains padded action rows")
    if bool(processed["goal_pose_is_pad"].any()):
        raise RuntimeError("manifest batch unexpectedly contains a padded t+15 goal")
    if processed[ACTION].shape != (len(positions), 15, 32):
        raise RuntimeError(f"unexpected padded action shape: {processed[ACTION].shape}")
    if not bool(processed["action_dim_is_pad"][:, :8].logical_not().all()):
        raise RuntimeError("real DROID action dimensions were marked as padding")
    if not bool(processed["action_dim_is_pad"][:, 8:].all()):
        raise RuntimeError("padded MolmoAct2 action dimensions were marked as real")

    visual_keys = {
        key: list(value.shape)
        for key, value in processed.items()
        if key in {"pixel_values", "image_token_pooling", "image_grids", "image_num_crops"}
        and torch.is_tensor(value)
    }
    if not visual_keys:
        raise RuntimeError("Stage-2 processed batch contains no visual model inputs")
    for key in ("input_ids", "attention_mask", "goal_pose"):
        if key not in processed or not torch.isfinite(processed[key]).all():
            raise RuntimeError(f"processed batch has invalid {key}")

    return {
        "status": "passed",
        "dataset_frames": dataset.meta.total_frames,
        "manifest_anchors": len(dataset),
        "manifest_episodes": dataset.num_episodes,
        "public_positions": positions,
        "absolute_indices": [int(manifest.indices[position]) for position in positions],
        "episode_indices": [
            int(manifest.episode_indices[position]) for position in positions
        ],
        "frame_indices": [int(manifest.frame_indices[position]) for position in positions],
        "raw_shapes": {
            "state": list(raw_state.shape),
            "action": list(raw_action.shape),
            "goal_feature": list(batch[GOAL_KEY].shape),
        },
        "processed_shapes": {
            "state": list(processed[OBS_STATE].shape),
            "action": list(processed[ACTION].shape),
            "goal_pose": list(processed["goal_pose"].shape),
        },
        "visual_inputs": visual_keys,
        "max_abs_error": errors,
        "normalization_masks": {
            OBS_STATE: [True] * 7 + [False],
            ACTION: [True] * 7 + [False],
            GOAL_KEY: [True] * 6 + [False],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_batch(args.checkpoint.expanduser().resolve())
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

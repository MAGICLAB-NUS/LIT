#!/usr/bin/env python3
"""Fail closed unless a checkpoint is the V4 8/8 hard-bottleneck policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED: dict[str, Any] = {
    "type": "molmoact2",
    "enable_goal_pose": True,
    "goal_conditioning_mode": "semantic_visual_recurrent",
    "goal_token_source": "learnable_queries",
    "num_goal_tokens": 8,
    "num_semantic_visual_tokens": 8,
    "num_semantic_visual_pose_tokens": 8,
    "semantic_visual_hidden_dim": 768,
    "semantic_visual_enable_self_attention": True,
    "semantic_visual_num_layer_groups": 6,
    "mask_image_from_action_expert": True,
    "enable_pose_reconstruction": True,
    "pose_recon_loss_weight": 0.3,
    "chunk_size": 10,
    "n_action_steps": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_path", type=Path)
    args = parser.parse_args()

    config_path = args.policy_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read checkpoint config: {config_path}: {error}") from error

    mismatches = [
        f"{key}: expected {expected!r}, got {config.get(key)!r}"
        for key, expected in EXPECTED.items()
        if config.get(key) != expected
    ]
    if mismatches:
        raise SystemExit(
            "Refusing to evaluate: checkpoint is not V4 8/8 hard bottleneck:\n  "
            + "\n  ".join(mismatches)
        )

    print(
        "[eval-v4] verified hard bottleneck:"
        f" tokens={config['num_semantic_visual_tokens']}"
        f" pose_tokens={config['num_semantic_visual_pose_tokens']}"
        f" goal_tokens={config['num_goal_tokens']}"
        f" groups={config['semantic_visual_num_layer_groups']}"
        f" hidden={config['semantic_visual_hidden_dim']}"
    )


if __name__ == "__main__":
    main()

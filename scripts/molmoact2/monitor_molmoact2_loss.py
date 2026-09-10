#!/usr/bin/env python3
"""Continuously parse and visualize MolmoAct2 flow/pose training metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FLOAT = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
PROGRESS_PATTERN = re.compile(r"Training:.*?\|\s*(?P<step>\d+)/(?P<total_steps>\d+)\s*\[")
TOKEN_PATTERN = re.compile(rf"\b(?P<name>[A-Za-z_]+):(?P<value>{FLOAT})")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
REQUIRED = {
    "epch",
    "loss",
    "grdn",
    "lr",
    "mem_gb",
    "updt_s",
    "data_s",
    "lr_vlm",
    "lr_vit",
    "lr_connector",
    "lr_action_expert",
    "flow",
}
FIELDS = (
    "step",
    "total_steps",
    "epoch",
    "loss",
    "flow_loss",
    "pose_loss",
    "weighted_pose_loss",
    "grad_norm",
    "lr",
    "lr_vlm",
    "lr_vit",
    "lr_connector",
    "lr_action_expert",
    "lr_semantic_visual",
    "memory_gb",
    "update_s",
    "data_s",
)


def parse_metrics(log_path: Path, pose_weight: float) -> list[dict[str, float]]:
    """Parse carriage-return tqdm logs and keep the latest record for every exact step."""
    text = ANSI_PATTERN.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    records_by_step: dict[int, dict[str, float]] = {}
    for segment in text.splitlines():
        if " flow:" not in segment or " loss:" not in segment:
            continue
        progress = PROGRESS_PATTERN.search(segment)
        if progress is None:
            continue
        tokens = {match.group("name"): float(match.group("value")) for match in TOKEN_PATTERN.finditer(segment)}
        if not REQUIRED.issubset(tokens):
            continue
        record = {
            "step": int(progress.group("step")),
            "total_steps": int(progress.group("total_steps")),
            "epoch": tokens["epch"],
            "loss": tokens["loss"],
            "flow_loss": tokens["flow"],
            "pose_loss": tokens.get("pose", float("nan")),
            "weighted_pose_loss": pose_weight * tokens.get("pose", float("nan")),
            "grad_norm": tokens["grdn"],
            "lr": tokens["lr"],
            "lr_vlm": tokens["lr_vlm"],
            "lr_vit": tokens["lr_vit"],
            "lr_connector": tokens["lr_connector"],
            "lr_action_expert": tokens["lr_action_expert"],
            "lr_semantic_visual": tokens.get("lr_semantic_visual", float("nan")),
            "memory_gb": tokens["mem_gb"],
            "update_s": tokens["updt_s"],
            "data_s": tokens["data_s"],
        }
        if all(math.isfinite(tokens[name]) for name in REQUIRED):
            records_by_step[int(record["step"])] = record
    return [records_by_step[step] for step in sorted(records_by_step)]


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(records: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def render_plot(
    records: list[dict[str, float]],
    path: Path,
    *,
    smooth_window: int,
    title: str,
) -> None:
    steps = [int(record["step"]) for record in records]

    def values(name: str) -> list[float]:
        return [record[name] for record in records]

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    latest = records[-1]
    figure.suptitle(
        f"{title}\nstep={steps[-1]}/{int(latest['total_steps'])} · "
        f"flow={latest['flow_loss']:.4f} · pose={latest['pose_loss']:.4f} · "
        f"points={len(records)}",
        fontsize=14,
    )

    loss_axis = axes[0, 0]
    for name, label, color in (
        ("loss", "total loss", "tab:blue"),
        ("flow_loss", "flow loss", "tab:orange"),
    ):
        raw = values(name)
        loss_axis.plot(steps, raw, color=color, alpha=0.16, linewidth=0.8)
        loss_axis.plot(
            steps,
            moving_average(raw, smooth_window),
            color=color,
            linewidth=1.8,
            label=f"{label} (MA{smooth_window})",
        )
    loss_axis.set(title="Policy losses", xlabel="Optimizer step", ylabel="Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    pose_axis = axes[0, 1]
    pose = values("pose_loss")
    weighted_pose = values("weighted_pose_loss")
    pose_axis.plot(steps, pose, color="tab:green", alpha=0.15, linewidth=0.8)
    pose_axis.plot(
        steps,
        moving_average(pose, smooth_window),
        color="tab:green",
        linewidth=1.8,
        label=f"pose loss (MA{smooth_window})",
    )
    pose_axis.plot(
        steps,
        moving_average(weighted_pose, smooth_window),
        color="tab:red",
        linewidth=1.5,
        label=f"weighted pose (MA{smooth_window})",
    )
    pose_axis.set(title="Pose supervision", xlabel="Optimizer step", ylabel="Loss")
    pose_axis.grid(alpha=0.25)
    pose_axis.legend()

    grad_axis = axes[1, 0]
    gradients = values("grad_norm")
    grad_axis.plot(steps, gradients, color="tab:purple", alpha=0.16, linewidth=0.8)
    grad_axis.plot(
        steps,
        moving_average(gradients, smooth_window),
        color="tab:purple",
        linewidth=1.8,
        label=f"grad norm (MA{smooth_window})",
    )
    grad_axis.set(title="Gradient and learning rate", xlabel="Optimizer step", ylabel="Grad norm")
    grad_axis.grid(alpha=0.25)
    lr_axis = grad_axis.twinx()
    lr_axis.plot(steps, values("lr"), color="tab:cyan", linewidth=1.3, label="main LR")
    lr_axis.plot(steps, values("lr_vit"), color="tab:olive", linewidth=1.1, label="ViT LR")
    lr_axis.set_ylabel("Learning rate")
    lines = grad_axis.lines[-1:] + lr_axis.lines
    grad_axis.legend(lines, [line.get_label() for line in lines], loc="best")

    runtime_axis = axes[1, 1]
    runtime_axis.plot(
        steps,
        moving_average(values("update_s"), smooth_window),
        color="tab:brown",
        linewidth=1.8,
        label=f"update time (MA{smooth_window})",
    )
    runtime_axis.plot(
        steps,
        moving_average(values("data_s"), smooth_window),
        color="tab:pink",
        linewidth=1.4,
        label=f"data time (MA{smooth_window})",
    )
    runtime_axis.set(title="Runtime and memory", xlabel="Optimizer step", ylabel="Seconds")
    runtime_axis.grid(alpha=0.25)
    memory_axis = runtime_axis.twinx()
    memory_axis.plot(steps, values("memory_gb"), color="tab:gray", alpha=0.8, label="memory")
    memory_axis.set_ylabel("GPU memory (GB)")
    lines = runtime_axis.lines + memory_axis.lines
    runtime_axis.legend(lines, [line.get_label() for line in lines], loc="best")

    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    os.replace(temporary, path)


def update_outputs(args: argparse.Namespace) -> int:
    records = parse_metrics(args.log, args.pose_weight)
    if args.min_step > 0:
        records = [record for record in records if record["step"] >= args.min_step]
    if not records:
        print(f"[loss-monitor] no metrics at/after step {args.min_step}: {args.log}", flush=True)
        return 0

    render_plot(records, args.output, smooth_window=args.smooth_window, title=args.title)
    write_csv(records, args.csv)
    latest = records[-1]
    atomic_json(
        {
            "log": str(args.log),
            "plot": str(args.output),
            "csv": str(args.csv),
            "min_step": args.min_step,
            "smooth_window": args.smooth_window,
            "pose_weight": args.pose_weight,
            "latest": latest,
            "updated_at": time.time(),
        },
        args.latest_json,
    )
    print(
        "[loss-monitor] "
        f"step={int(latest['step'])}/{int(latest['total_steps'])} "
        f"loss={latest['loss']:.5f} flow={latest['flow_loss']:.5f} "
        f"pose={latest['pose_loss']:.5f} weighted_pose={latest['weighted_pose_loss']:.5f} "
        f"grad={latest['grad_norm']:.3f} update={latest['update_s']:.2f}s "
        f"plot={args.output}",
        flush=True,
    )
    return int(latest["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--latest-json", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--pose-weight", type=float, default=0.3)
    parser.add_argument("--title", default="MolmoAct2 v2b training")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.csv is None:
        args.csv = args.output.with_suffix(".csv")
    if args.latest_json is None:
        args.latest_json = args.output.with_suffix(".latest.json")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be positive")
    if args.min_step < 0:
        raise ValueError("--min-step must be non-negative")

    while True:
        try:
            update_outputs(args)
        except FileNotFoundError:
            print(f"[loss-monitor] waiting for log: {args.log}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

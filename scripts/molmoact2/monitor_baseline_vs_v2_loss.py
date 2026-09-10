#!/usr/bin/env python3
"""Continuously overlay baseline and v2b MolmoAct2 training curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from monitor_molmoact2_loss import FIELDS, moving_average, parse_metrics


def finite_values(records: list[dict[str, float]], name: str) -> tuple[list[int], list[float]]:
    points = [
        (int(record["step"]), record[name])
        for record in records
        if math.isfinite(record[name])
    ]
    return [point[0] for point in points], [point[1] for point in points]


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(
    baseline: list[dict[str, float]],
    v2: list[dict[str, float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run", *FIELDS))
        writer.writeheader()
        for label, records in (("baseline", baseline), ("v2b", v2)):
            for record in records:
                writer.writerow({"run": label, **record})
    os.replace(temporary, path)


def add_smoothed_curve(axis, records, name, label, color, window):
    steps, raw = finite_values(records, name)
    if not steps:
        return
    axis.plot(steps, raw, color=color, alpha=0.12, linewidth=0.7)
    axis.plot(
        steps,
        moving_average(raw, window),
        color=color,
        linewidth=1.9,
        label=f"{label} (MA{window})",
    )


def render(
    baseline: list[dict[str, float]],
    v2: list[dict[str, float]],
    path: Path,
    *,
    smooth_window: int,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    baseline_latest = baseline[-1] if baseline else None
    v2_latest = v2[-1] if v2 else None
    baseline_text = (
        f"baseline step={int(baseline_latest['step'])} flow={baseline_latest['flow_loss']:.4f}"
        if baseline_latest
        else "baseline waiting"
    )
    v2_text = (
        f"v2b step={int(v2_latest['step'])} flow={v2_latest['flow_loss']:.4f}"
        if v2_latest
        else "v2b waiting"
    )
    figure.suptitle(f"{title}\n{baseline_text} · {v2_text}", fontsize=14)

    flow_axis = axes[0, 0]
    add_smoothed_curve(flow_axis, baseline, "flow_loss", "baseline flow", "tab:blue", smooth_window)
    add_smoothed_curve(flow_axis, v2, "flow_loss", "v2b flow", "tab:orange", smooth_window)
    flow_axis.set_yscale("log")
    flow_axis.set(title="Action flow loss", xlabel="Visual training step", ylabel="Flow loss (log)")
    flow_axis.grid(alpha=0.25)
    flow_axis.legend()

    pose_axis = axes[0, 1]
    add_smoothed_curve(pose_axis, v2, "pose_loss", "v2b pose", "tab:green", smooth_window)
    add_smoothed_curve(
        pose_axis,
        v2,
        "weighted_pose_loss",
        "v2b weighted pose",
        "tab:red",
        smooth_window,
    )
    pose_axis.set(
        title="v2b auxiliary pose supervision",
        xlabel="Visual training step",
        ylabel="Pose loss (log)",
    )
    pose_axis.set_yscale("log")
    pose_axis.grid(alpha=0.25)
    pose_axis.legend()

    grad_axis = axes[1, 0]
    add_smoothed_curve(grad_axis, baseline, "grad_norm", "baseline grad", "tab:blue", smooth_window)
    add_smoothed_curve(grad_axis, v2, "grad_norm", "v2b grad", "tab:orange", smooth_window)
    grad_axis.set_yscale("log")
    grad_axis.set(title="Gradient norm", xlabel="Visual training step", ylabel="Grad norm (log)")
    grad_axis.grid(alpha=0.25)
    grad_axis.legend()

    runtime_axis = axes[1, 1]
    add_smoothed_curve(
        runtime_axis,
        baseline,
        "update_s",
        "baseline update time",
        "tab:blue",
        smooth_window,
    )
    add_smoothed_curve(
        runtime_axis,
        v2,
        "update_s",
        "v2b update time",
        "tab:orange",
        smooth_window,
    )
    runtime_axis.set(title="Training throughput", xlabel="Visual training step", ylabel="Seconds / step")
    runtime_axis.grid(alpha=0.25)
    runtime_axis.legend()

    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    os.replace(temporary, path)


def update(args: argparse.Namespace) -> None:
    baseline = (
        parse_metrics(args.baseline_log, pose_weight=0.0)
        if args.baseline_log.exists()
        else []
    )
    v2 = parse_metrics(args.v2_log, pose_weight=args.pose_weight) if args.v2_log.exists() else []
    baseline = [record for record in baseline if record["step"] >= args.min_step]
    v2 = [record for record in v2 if record["step"] >= args.min_step]
    if args.baseline_max_step is not None:
        baseline = [record for record in baseline if record["step"] <= args.baseline_max_step]
    if args.v2_max_step is not None:
        v2 = [record for record in v2 if record["step"] <= args.v2_max_step]
    if not baseline and not v2:
        print("[loss-compare] waiting for both logs", flush=True)
        return

    render(
        baseline,
        v2,
        args.output,
        smooth_window=args.smooth_window,
        title=args.title,
    )
    write_csv(baseline, v2, args.csv)
    payload = {
        "baseline_log": str(args.baseline_log),
        "v2_log": str(args.v2_log),
        "plot": str(args.output),
        "csv": str(args.csv),
        "min_step": args.min_step,
        "baseline_max_step": args.baseline_max_step,
        "v2_max_step": args.v2_max_step,
        "smooth_window": args.smooth_window,
        "latest": {
            "baseline": baseline[-1] if baseline else None,
            "v2b": v2[-1] if v2 else None,
        },
        "updated_at": time.time(),
    }
    atomic_write_json(payload, args.latest_json)
    baseline_status = (
        f"step={int(baseline[-1]['step'])} flow={baseline[-1]['flow_loss']:.5f}"
        if baseline
        else "waiting"
    )
    v2_status = (
        f"step={int(v2[-1]['step'])} flow={v2[-1]['flow_loss']:.5f}"
        if v2
        else "waiting"
    )
    print(
        f"[loss-compare] baseline {baseline_status} | v2b {v2_status} | plot={args.output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--v2-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--latest-json", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-step", type=int, default=1000)
    parser.add_argument("--baseline-max-step", type=int)
    parser.add_argument("--v2-max-step", type=int)
    parser.add_argument("--pose-weight", type=float, default=0.3)
    parser.add_argument("--title", default="MolmoAct2 baseline vs v2b")
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

    while True:
        update(args)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

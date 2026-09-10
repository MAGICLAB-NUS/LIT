#!/usr/bin/env python3
"""Continuously compare baseline, v2b, and live v3 MolmoAct2 training curves."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "lerobot/outputs/libero_goal_prior_v3/seed_1000"

RUN_STYLES = {
    "baseline": ("baseline", "tab:blue"),
    "v2b": ("v2b (old quantiles)", "tab:orange"),
    "v3": ("v3 (fixed quantiles)", "tab:purple"),
}


def finite_values(
    records: list[dict[str, float]], name: str
) -> tuple[list[int], list[float]]:
    points = [
        (int(record["step"]), record[name])
        for record in records
        if math.isfinite(record[name])
    ]
    return [point[0] for point in points], [point[1] for point in points]


def add_smoothed_curve(
    axis,
    records: list[dict[str, float]],
    name: str,
    label: str,
    color: str,
    window: int,
) -> None:
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


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(runs: dict[str, list[dict[str, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run", *FIELDS))
        writer.writeheader()
        for run, records in runs.items():
            for record in records:
                writer.writerow({"run": run, **record})
    os.replace(temporary, path)


def latest_text(run: str, records: list[dict[str, float]]) -> str:
    if not records:
        return f"{run} waiting"
    latest = records[-1]
    pose = latest["pose_loss"]
    pose_text = f" pose={pose:.4f}" if math.isfinite(pose) else ""
    return (
        f"{run} step={int(latest['step'])} "
        f"flow={latest['flow_loss']:.4f}{pose_text}"
    )


def render(
    runs: dict[str, list[dict[str, float]]],
    path: Path,
    *,
    smooth_window: int,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    statuses = " · ".join(latest_text(run, records) for run, records in runs.items())
    figure.suptitle(f"{title}\n{statuses}", fontsize=13)

    flow_axis = axes[0, 0]
    for run, records in runs.items():
        label, color = RUN_STYLES[run]
        add_smoothed_curve(
            flow_axis,
            records,
            "flow_loss",
            f"{label} flow",
            color,
            smooth_window,
        )
    flow_axis.set_yscale("log")
    flow_axis.set(
        title="Action flow loss",
        xlabel="Training step",
        ylabel="Flow loss (log)",
    )
    flow_axis.grid(alpha=0.25)
    flow_axis.legend()

    pose_axis = axes[0, 1]
    pose_styles = {
        "v2b": ("tab:green", "tab:red"),
        "v3": ("tab:cyan", "tab:pink"),
    }
    for run in ("v2b", "v3"):
        records = runs[run]
        raw_color, weighted_color = pose_styles[run]
        label = RUN_STYLES[run][0]
        add_smoothed_curve(
            pose_axis,
            records,
            "pose_loss",
            f"{label} pose",
            raw_color,
            smooth_window,
        )
        add_smoothed_curve(
            pose_axis,
            records,
            "weighted_pose_loss",
            f"{label} weighted pose",
            weighted_color,
            smooth_window,
        )
    pose_axis.set_yscale("log")
    pose_axis.set(
        title="Auxiliary pose supervision",
        xlabel="Stage-2 visual training step",
        ylabel="Pose loss (log)",
    )
    pose_axis.grid(alpha=0.25)
    pose_axis.legend()

    grad_axis = axes[1, 0]
    for run, records in runs.items():
        label, color = RUN_STYLES[run]
        add_smoothed_curve(
            grad_axis,
            records,
            "grad_norm",
            f"{label} grad",
            color,
            smooth_window,
        )
    grad_axis.set_yscale("log")
    grad_axis.set(
        title="Gradient norm",
        xlabel="Training step",
        ylabel="Gradient norm (log)",
    )
    grad_axis.grid(alpha=0.25)
    grad_axis.legend()

    runtime_axis = axes[1, 1]
    for run, records in runs.items():
        label, color = RUN_STYLES[run]
        add_smoothed_curve(
            runtime_axis,
            records,
            "update_s",
            f"{label} update time",
            color,
            smooth_window,
        )
    runtime_axis.set(
        title="Training throughput",
        xlabel="Training step",
        ylabel="Seconds / step",
    )
    runtime_axis.grid(alpha=0.25)
    runtime_axis.legend()

    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    os.replace(temporary, path)


def load_runs(args: argparse.Namespace) -> dict[str, list[dict[str, float]]]:
    specifications = {
        "baseline": (args.baseline_log, 0.0, args.baseline_max_step),
        "v2b": (args.v2b_log, args.pose_weight, args.v2b_max_step),
        "v3": (args.v3_log, args.pose_weight, args.v3_max_step),
    }
    runs: dict[str, list[dict[str, float]]] = {}
    for run, (log_path, pose_weight, max_step) in specifications.items():
        records = (
            parse_metrics(log_path, pose_weight=pose_weight)
            if log_path.exists()
            else []
        )
        records = [record for record in records if record["step"] >= args.min_step]
        if max_step is not None:
            records = [record for record in records if record["step"] <= max_step]
        runs[run] = records
    return runs


def update(args: argparse.Namespace) -> None:
    runs = load_runs(args)
    if not any(runs.values()):
        print("[loss-compare-v3] waiting for training logs", flush=True)
        return

    render(
        runs,
        args.output,
        smooth_window=args.smooth_window,
        title=args.title,
    )
    write_csv(runs, args.csv)
    atomic_write_json(
        {
            "logs": {
                "baseline": str(args.baseline_log),
                "v2b": str(args.v2b_log),
                "v3": str(args.v3_log),
            },
            "plot": str(args.output),
            "csv": str(args.csv),
            "min_step": args.min_step,
            "max_steps": {
                "baseline": args.baseline_max_step,
                "v2b": args.v2b_max_step,
                "v3": args.v3_max_step,
            },
            "smooth_window": args.smooth_window,
            "latest": {
                run: records[-1] if records else None
                for run, records in runs.items()
            },
            "updated_at": time.time(),
        },
        args.latest_json,
    )
    print(
        "[loss-compare-v3] "
        + " | ".join(latest_text(run, records) for run, records in runs.items())
        + f" | plot={args.output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=ROOT
        / "lerobot/outputs/libero_goal_prior/seed_1000/libero_baseline.console.log",
    )
    parser.add_argument(
        "--v2b-log",
        type=Path,
        default=ROOT / "lerobot/outputs/libero_goal_prior_v2b/seed_1000/stage2.console.log",
    )
    parser.add_argument(
        "--v3-log",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "stage2.console.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "baseline_vs_v2b_vs_v3_live.png",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--latest-json", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--baseline-max-step", type=int, default=30_000)
    parser.add_argument("--v2b-max-step", type=int, default=20_000)
    parser.add_argument("--v3-max-step", type=int, default=30_000)
    parser.add_argument("--pose-weight", type=float, default=0.3)
    parser.add_argument(
        "--title",
        default="MolmoAct2 baseline vs v2b vs v3 (live)",
    )
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

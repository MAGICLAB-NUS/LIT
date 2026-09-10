#!/usr/bin/env python3
"""Continuously compare per-task LIBERO success rates across checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import time
from typing import Any


SUITE_ORDER = {
    "libero_spatial": 0,
    "libero_object": 1,
    "libero_10": 2,
    "libero_goal": 3,
}


@dataclass(frozen=True)
class Checkpoint:
    label: str
    root: Path


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def task_key(suite: str, task_id: int) -> str:
    return f"{suite}:{task_id}"


def metric(successes: int, episodes: int) -> dict[str, int | float]:
    return {
        "successes": successes,
        "episodes": episodes,
        "pc_success": 100.0 * successes / episodes if episodes else 0.0,
    }


def discover_suite_dirs(root: Path) -> list[Path]:
    """Support both suite/ and sharded gpu_*/suite/ layouts."""
    dirs: list[Path] = []
    seen: set[Path] = set()
    patterns = (
        "*/live_eval.json",
        "*/eval_info.json",
        "gpu_*/**/live_eval.json",
        "gpu_*/**/eval_info.json",
    )
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            directory = path.parent
            if directory.name.startswith("gpu_"):
                continue
            if directory in seen:
                continue
            seen.add(directory)
            dirs.append(directory)
    return dirs


def ingest_completed_tasks(
    tasks: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    *,
    final: bool,
) -> None:
    for task in payload.get("per_task", []) or []:
        task_suite = str(task.get("task_group") or "")
        if not task_suite:
            continue
        task_id = int(task["task_id"])
        key = task_key(task_suite, task_id)
        successes = [
            bool(value) for value in task.get("metrics", {}).get("successes", [])
        ]
        success_count = sum(successes)
        episodes = len(successes)
        if episodes <= 0:
            continue
        existing = tasks.get(key)
        if existing and not existing.get("is_lower_bound") and not final:
            continue
        tasks[key] = {
            "suite": task_suite,
            "task_id": task_id,
            "status": "final" if final else str(payload.get("status", "running")),
            "is_lower_bound": False,
            **metric(success_count, episodes),
        }


def ingest_running_tasks(
    tasks: dict[str, dict[str, Any]],
    running_tasks: list[dict[str, Any]],
) -> None:
    for running in running_tasks:
        task_suite = str(running.get("task_group") or "")
        if not task_suite:
            continue
        task_id = int(running["task_id"])
        key = task_key(task_suite, task_id)
        if key in tasks and not tasks[key].get("is_lower_bound"):
            continue
        finished = int(running.get("finished_rollouts") or 0)
        successes = int(running.get("successes_so_far") or 0)
        if finished <= 0:
            continue
        tasks[key] = {
            "suite": task_suite,
            "task_id": task_id,
            "status": "running",
            "is_lower_bound": True,
            "finished_rollouts": finished,
            "step": int(running.get("step", 0)),
            "max_steps": int(running.get("max_steps", 0)),
            **metric(successes, finished),
        }


def collect_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    suite_statuses: dict[str, list[str]] = {}

    for suite_dir in discover_suite_dirs(checkpoint.root):
        suite = suite_dir.name
        process = load_json(suite_dir / "process_status.json") or {}
        final_info = load_json(suite_dir / "eval_info.json")
        live = load_json(suite_dir / "live_eval.json") or {}
        realtime = load_json(suite_dir / "realtime_accuracy.json") or {}

        if final_info is not None:
            status = "final"
            ingest_completed_tasks(tasks, final_info, final=True)
        else:
            status = str(
                live.get("status")
                or process.get("status")
                or realtime.get("status")
                or "pending"
            )
            ingest_completed_tasks(tasks, live, final=status == "final")
            running = list(live.get("running_tasks") or [])
            if not running:
                running = list(realtime.get("running_tasks") or [])
            ingest_running_tasks(tasks, running)

        suite_statuses.setdefault(suite, []).append(status)

    suite_status: dict[str, str] = {}
    for suite, statuses in suite_statuses.items():
        if statuses and all(item == "final" for item in statuses):
            suite_status[suite] = "final"
        elif any(item == "failed" for item in statuses):
            suite_status[suite] = "failed"
        elif any(item == "running" for item in statuses):
            suite_status[suite] = "running"
        else:
            suite_status[suite] = statuses[0] if statuses else "pending"

    total_successes = sum(int(task["successes"]) for task in tasks.values())
    total_episodes = sum(int(task["episodes"]) for task in tasks.values())
    return {
        "label": checkpoint.label,
        "root": str(checkpoint.root),
        "tasks": tasks,
        "suite_status": suite_status,
        "completed": metric(total_successes, total_episodes),
    }


def row_sort_key(key: str) -> tuple[int, int, str]:
    suite, task_id = key.rsplit(":", 1)
    return (SUITE_ORDER.get(suite, 99), int(task_id), suite)


def build_payload(checkpoints: list[Checkpoint]) -> dict[str, Any]:
    collected = [collect_checkpoint(checkpoint) for checkpoint in checkpoints]
    keys = sorted(
        {key for result in collected for key in result["tasks"]},
        key=row_sort_key,
    )
    rows = []
    for key in keys:
        suite, task_id = key.rsplit(":", 1)
        values = {
            result["label"]: result["tasks"].get(key)
            for result in collected
        }
        rows.append({"suite": suite, "task_id": int(task_id), "checkpoints": values})
    return {
        "checkpoints": [
            {
                key: value
                for key, value in result.items()
                if key != "tasks"
            }
            for result in collected
        ],
        "rows": rows,
        "updated_at": time.time(),
    }


def format_rate(value: dict[str, Any] | None) -> str:
    if value is None:
        return "—"
    suffix = " lower bound" if value.get("is_lower_bound") else ""
    return (
        f"{value['pc_success']:.2f}% "
        f"({value['successes']}/{value['episodes']}){suffix}"
    )


def render_html(payload: dict[str, Any], refresh_seconds: int) -> str:
    labels = [checkpoint["label"] for checkpoint in payload["checkpoints"]]
    summaries = []
    for checkpoint in payload["checkpoints"]:
        complete = checkpoint["completed"]
        summaries.append(
            "<div class='summary'>"
            f"<h2>{escape(checkpoint['label'])}</h2>"
            f"<strong>{complete['pc_success']:.2f}%</strong>"
            f"<span>{complete['successes']}/{complete['episodes']} completed rollouts</span>"
            "</div>"
        )

    header = "".join(
        f"<th>{escape(label)} success</th>" for label in labels
    )
    body_rows = []
    for row in payload["rows"]:
        cells = []
        for label in labels:
            value = row["checkpoints"].get(label)
            status_class = "running" if value and value.get("status") == "running" else ""
            cells.append(f"<td class='{status_class}'>{escape(format_rate(value))}</td>")

        body_rows.append(
            "<tr>"
            f"<td>{escape(row['suite'])}</td>"
            f"<td>{row['task_id']}</td>"
            + "".join(cells)
            + "</tr>"
        )

    updated = datetime.fromtimestamp(payload["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <title>LIBERO checkpoint task comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    .caption {{ color: #666; margin-bottom: 20px; }}
    .summaries {{ display: flex; gap: 24px; margin: 20px 0; }}
    .summary {{ border: 1px solid #ddd; padding: 12px 18px; min-width: 220px; }}
    .summary h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .summary strong {{ display: block; font-size: 24px; }}
    .summary span {{ color: #666; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead {{ position: sticky; top: 0; background: #f5f5f5; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .running {{ color: #8a5a00; }}
  </style>
</head>
<body>
  <h1>LIBERO per-task checkpoint comparison</h1>
  <div class="caption">Updated {updated} · auto-refresh every {refresh_seconds}s · running rates are lower bounds</div>
  <div class="summaries">{''.join(summaries)}</div>
  <table>
    <thead><tr><th>Suite</th><th>Task ID</th>{header}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""


def _format_cell(value: dict[str, Any] | None) -> str:
    if value is None or not value.get("episodes"):
        return "—"
    if value.get("is_lower_bound"):
        return (
            f"≥{value['pc_success']:.2f}% "
            f"({value['successes']}/{value['episodes']}, running)"
        )
    return f"{value['pc_success']:.2f}% ({value['successes']}/{value['episodes']})"


def _aggregate_by_suite(
    payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """suite -> label -> metric dict (successes/episodes/pc_success)."""
    labels = [checkpoint["label"] for checkpoint in payload["checkpoints"]]
    by_suite: dict[str, dict[str, list[int]]] = {}
    for row in payload["rows"]:
        suite = str(row["suite"])
        suite_bucket = by_suite.setdefault(
            suite, {label: [0, 0, False] for label in labels}
        )
        for label in labels:
            value = row["checkpoints"].get(label)
            if value is None or not value.get("episodes"):
                continue
            suite_bucket[label][0] += int(value["successes"])
            suite_bucket[label][1] += int(value["episodes"])
            if value.get("is_lower_bound"):
                suite_bucket[label][2] = True
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for suite, label_counts in by_suite.items():
        result[suite] = {}
        for label, (successes, episodes, is_lower_bound) in label_counts.items():
            if not episodes:
                result[suite][label] = None  # type: ignore[assignment]
                continue
            cell = metric(successes, episodes)
            if is_lower_bound:
                cell["is_lower_bound"] = True
            result[suite][label] = cell
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    labels = [checkpoint["label"] for checkpoint in payload["checkpoints"]]
    updated = datetime.fromtimestamp(payload["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# LIBERO suite comparison",
        "",
        f"Updated: `{updated}` · running success rates are lower bounds",
        "",
        "Protocol: official LIBERO (50 ep/task, seed 1000) when comparing "
        + " / ".join(f"`{label}`" for label in labels)
        + ". Running cells are lower bounds from partial rollouts.",
        "",
        "## Overall",
        "",
    ]
    overall_header = ["Overall", *[label for label in labels]]
    show_delta = len(labels) == 2
    if show_delta:
        overall_header.append(f"Δ {labels[0]} − {labels[1]}")
    lines.extend(
        [
            "| " + " | ".join(overall_header) + " |",
            "| " + " | ".join(["---"] * len(overall_header)) + " |",
        ]
    )
    overall_cells = ["All suites"]
    rates: list[float | None] = []
    for checkpoint in payload["checkpoints"]:
        complete = checkpoint["completed"]
        overall_cells.append(_format_cell(complete))
        rates.append(
            float(complete["pc_success"]) if complete.get("episodes") else None
        )
    if show_delta:
        if rates[0] is not None and rates[1] is not None:
            overall_cells.append(f"{rates[0] - rates[1]:+.2f} pp")
        else:
            overall_cells.append("—")
    lines.append("| " + " | ".join(overall_cells) + " |")

    suite_header = ["Suite", *labels]
    if show_delta:
        suite_header.append(f"Δ {labels[0]} − {labels[1]}")
    lines.extend(
        [
            "",
            "## Per-suite",
            "",
            "| " + " | ".join(suite_header) + " |",
            "| " + " | ".join(["---"] * len(suite_header)) + " |",
        ]
    )
    by_suite = _aggregate_by_suite(payload)
    suites = sorted(by_suite, key=lambda suite: SUITE_ORDER.get(suite, 99))
    for suite in suites:
        cells = [f"`{suite}`"]
        suite_rates: list[float | None] = []
        for label in labels:
            value = by_suite[suite].get(label)
            cells.append(_format_cell(value))
            suite_rates.append(
                float(value["pc_success"])
                if value is not None and value.get("episodes")
                else None
            )
        if show_delta:
            if suite_rates[0] is not None and suite_rates[1] is not None:
                cells.append(f"{suite_rates[0] - suite_rates[1]:+.2f} pp")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_csv(payload: dict[str, Any], path: Path) -> None:
    labels = [checkpoint["label"] for checkpoint in payload["checkpoints"]]
    fields = ["suite", "task_id"]
    for label in labels:
        fields.extend(
            [
                f"{label}_status",
                f"{label}_successes",
                f"{label}_episodes",
                f"{label}_pc_success",
                f"{label}_is_lower_bound",
            ]
        )
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            output: dict[str, Any] = {
                "suite": row["suite"],
                "task_id": row["task_id"],
            }
            for label in labels:
                value = row["checkpoints"].get(label)
                if value:
                    for field in ("status", "successes", "episodes", "pc_success", "is_lower_bound"):
                        output[f"{label}_{field}"] = value.get(field)
            writer.writerow(output)
    os.replace(temporary, path)


def parse_checkpoint(value: str) -> Checkpoint:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/path/to/eval/root") from error
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must contain both label and path")
    return Checkpoint(label=label, root=Path(path))


def update(args: argparse.Namespace) -> None:
    payload = build_payload(args.checkpoint)
    if args.output.suffix.lower() in {".md", ".markdown"}:
        # Keep the same inode so Cursor/VS Code Markdown previews do not lose the
        # open document when the live table refreshes.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(payload), encoding="utf-8")
    else:
        atomic_text(args.output, render_html(payload, int(args.interval)))
    atomic_text(args.json, json.dumps(payload, indent=2) + "\n")
    write_csv(payload, args.csv)
    row_count = len(payload["rows"])
    summary = " | ".join(
        f"{checkpoint['label']}={checkpoint['completed']['pc_success']:.2f}%"
        for checkpoint in payload["checkpoints"]
    )
    print(f"[checkpoint-compare] tasks={row_count} {summary} output={args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("at least two --checkpoint values are required")
    if args.csv is None:
        args.csv = args.output.with_suffix(".csv")
    if args.json is None:
        args.json = args.output.with_suffix(".json")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    while True:
        update(args)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

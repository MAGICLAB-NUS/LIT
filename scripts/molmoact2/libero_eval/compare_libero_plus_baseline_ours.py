#!/usr/bin/env python3
"""Compare baseline vs ours on the shared LIBERO-plus balanced task set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SUITE_ORDER = ("libero_spatial", "libero_object", "libero_10", "libero_goal")


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


def metric(successes: int, episodes: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "episodes": episodes,
        "pc_success": 100.0 * successes / episodes if episodes else 0.0,
    }


def collect_side(root: Path) -> dict[str, Any]:
    tasks: dict[tuple[str, int], dict[str, Any]] = {}
    suite_status: dict[str, str] = {}
    total_successes = 0
    total_episodes = 0

    for suite in SUITE_ORDER:
        live_path = root / suite / "live_eval.json"
        live = load_json(live_path) or {}
        suite_status[suite] = str(live.get("status", "pending"))

        for task in live.get("per_task", []) or []:
            task_id = int(task["task_id"])
            successes = [
                bool(value) for value in task.get("metrics", {}).get("successes", [])
            ]
            success_count = sum(successes)
            episodes = len(successes)
            total_successes += success_count
            total_episodes += episodes
            tasks[(suite, task_id)] = {
                "status": "final",
                "is_lower_bound": False,
                **metric(success_count, episodes),
            }

        realtime = load_json(root / suite / "realtime_accuracy.json") or {}
        running = list(live.get("running_tasks", []) or [])
        running.extend(realtime.get("running_tasks", []) or [])
        for item in running:
            task_id = int(item["task_id"])
            key = (suite, task_id)
            if key in tasks:
                continue
            successes = int(item.get("successes_so_far", 0))
            episodes = int(item.get("total_rollouts", 0))
            tasks[key] = {
                "status": "running",
                "is_lower_bound": True,
                "finished_rollouts": int(item.get("finished_rollouts", 0)),
                **metric(successes, episodes),
            }

    return {
        "tasks": tasks,
        "suite_status": suite_status,
        "completed": metric(total_successes, total_episodes),
    }


def format_rate(value: dict[str, Any] | None) -> str:
    if value is None:
        return "—"
    if value.get("is_lower_bound"):
        return (
            f"≥{value['pc_success']:.2f}% "
            f"({value['successes']}/{value['episodes']}, running)"
        )
    return f"{value['pc_success']:.2f}% ({value['successes']}/{value['episodes']})"


def format_delta(
    baseline: dict[str, Any] | None, ours: dict[str, Any] | None
) -> str:
    if (
        baseline
        and ours
        and not baseline.get("is_lower_bound")
        and not ours.get("is_lower_bound")
    ):
        return f"{ours['pc_success'] - baseline['pc_success']:+.2f} pp"
    return "—"


def suite_completed(
    tasks: dict[tuple[str, int], dict[str, Any]], suite: str
) -> dict[str, Any]:
    successes = 0
    episodes = 0
    for (task_suite, _), value in tasks.items():
        if task_suite != suite or value.get("is_lower_bound"):
            continue
        successes += int(value["successes"])
        episodes += int(value["episodes"])
    return metric(successes, episodes)


def category_completed(
    tasks: dict[tuple[str, int], dict[str, Any]],
    tasks_meta: dict[tuple[str, int], dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    successes = 0
    episodes = 0
    for key, value in tasks.items():
        if value.get("is_lower_bound"):
            continue
        meta = tasks_meta.get(key)
        if meta is None or meta.get("category") != category:
            continue
        successes += int(value["successes"])
        episodes += int(value["episodes"])
    return metric(successes, episodes)


def render_markdown(
    *,
    baseline_root: Path,
    ours_root: Path,
    baseline: dict[str, Any],
    ours: dict[str, Any],
    manifest: dict[str, Any],
    run_manifest: dict[str, Any] | None,
) -> str:
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tasks_meta = {
        (task["suite"], int(task["task_id"])): task for task in manifest["tasks"]
    }
    baseline_ckpt = (run_manifest or {}).get("policy_path", "baseline")
    ours_run = load_json(ours_root / "run_manifest.json") or {}
    ours_ckpt = ours_run.get("policy_path", "ours")

    lines = [
        "# LIBERO-plus baseline vs ours",
        "",
        f"Updated: `{updated}` · unfinished / in-progress task rates are lower bounds",
        "",
        "## Evaluation setup",
        "",
        "- **Benchmark**: LIBERO-plus",
        f"- **Protocol**: `{manifest.get('phase', 'base_category')}` "
        f"(exclude `{', '.join(manifest.get('excluded_categories', []))}`)",
        f"- **Selection / rollout seed**: `{manifest.get('selection_seed')}`",
        f"- **Task sample**: `{manifest.get('samples_per_cell')}` variants per "
        f"(base skill × perturbation category), prefer difficulty≈"
        f"`{manifest.get('target_difficulty', 3)}`",
        f"- **Episodes per task**: `{manifest.get('episodes_per_task')}`",
        f"- **Total tasks / rollouts**: `{manifest.get('num_tasks')}` / "
        f"`{manifest.get('num_rollouts')}`",
        "- **Included categories**: "
        + ", ".join(f"`{c}`" for c in manifest.get("included_categories", [])),
        "- **Suites**: "
        + ", ".join(f"`{s}`" for s in manifest.get("suites", list(SUITE_ORDER))),
        f"- **baseline checkpoint**: `{baseline_ckpt}`",
        f"- **ours checkpoint**: `{ours_ckpt}`",
        "- **Task set**: identical shared `task_manifest.json` for both sides",
        "- **Eval order**: shuffled within each suite (same seed both sides)",
        "",
        "## Overall completed rollouts",
        "",
        f"- **baseline**: {baseline['completed']['pc_success']:.2f}% "
        f"({baseline['completed']['successes']}/{baseline['completed']['episodes']})",
        f"- **ours**: {ours['completed']['pc_success']:.2f}% "
        f"({ours['completed']['successes']}/{ours['completed']['episodes']})",
        "",
        "## Per-suite completed rollouts",
        "",
        "| Suite | baseline | ours | Δ ours − baseline |",
        "| --- | --- | --- | --- |",
    ]

    for suite in SUITE_ORDER:
        b = suite_completed(baseline["tasks"], suite)
        o = suite_completed(ours["tasks"], suite)
        if b["episodes"] and o["episodes"]:
            delta = f"{o['pc_success'] - b['pc_success']:+.2f} pp"
        else:
            delta = "—"
        lines.append(
            f"| `{suite}` "
            f"({baseline['suite_status'].get(suite, '?')} / "
            f"{ours['suite_status'].get(suite, '?')}) | "
            f"{b['pc_success']:.2f}% ({b['successes']}/{b['episodes']}) | "
            f"{o['pc_success']:.2f}% ({o['successes']}/{o['episodes']}) | "
            f"{delta} |"
        )

    categories = list(manifest.get("included_categories") or [])
    if not categories:
        categories = sorted(
            {
                str(meta.get("category"))
                for meta in tasks_meta.values()
                if meta.get("category")
            }
        )
    lines.extend(
        [
            "",
            "## Per-category completed rollouts",
            "",
            "| Category | baseline | ours | Δ ours − baseline |",
            "| --- | --- | --- | --- |",
        ]
    )
    for category in categories:
        b = category_completed(baseline["tasks"], tasks_meta, category)
        o = category_completed(ours["tasks"], tasks_meta, category)
        if b["episodes"] and o["episodes"]:
            delta = f"{o['pc_success'] - b['pc_success']:+.2f} pp"
        else:
            delta = "—"
        lines.append(
            f"| `{category}` | "
            f"{b['pc_success']:.2f}% ({b['successes']}/{b['episodes']}) | "
            f"{o['pc_success']:.2f}% ({o['successes']}/{o['episodes']}) | "
            f"{delta} |"
        )

    for suite in SUITE_ORDER:
        suite_tasks = [
            tasks_meta[key] for key in tasks_meta if key[0] == suite
        ]
        # Prefer eval_order (shuffled schedule); fall back to task_id.
        suite_tasks.sort(
            key=lambda item: (
                item.get("eval_order", 10**9),
                int(item["task_id"]),
            )
        )
        lines.extend(
            [
                "",
                f"## `{suite}` per-task success rate",
                "",
                "| Eval# | Task ID | Category | Diff | baseline | ours | Δ ours − baseline |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for meta in suite_tasks:
            task_id = int(meta["task_id"])
            b = baseline["tasks"].get((suite, task_id))
            o = ours["tasks"].get((suite, task_id))
            eval_order = meta.get("eval_order", "—")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(eval_order),
                        str(task_id),
                        meta.get("category", ""),
                        str(meta.get("difficulty_level", "")),
                        format_rate(b),
                        format_rate(o),
                        format_delta(b, o),
                    ]
                )
                + " |"
            )

    lines.append("")
    return "\n".join(lines)


def write_csv(
    path: Path,
    *,
    baseline: dict[str, Any],
    ours: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    fields = [
        "suite",
        "task_id",
        "category",
        "difficulty_level",
        "name",
        "baseline_status",
        "baseline_successes",
        "baseline_episodes",
        "baseline_pc_success",
        "baseline_is_lower_bound",
        "ours_status",
        "ours_successes",
        "ours_episodes",
        "ours_pc_success",
        "ours_is_lower_bound",
        "delta_ours_minus_baseline_pp",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for meta in sorted(
            manifest["tasks"],
            key=lambda item: (
                SUITE_ORDER.index(item["suite"]),
                item.get("eval_order", 10**9),
                int(item["task_id"]),
            ),
        ):
            suite = meta["suite"]
            task_id = int(meta["task_id"])
            b = baseline["tasks"].get((suite, task_id))
            o = ours["tasks"].get((suite, task_id))
            row: dict[str, Any] = {
                "suite": suite,
                "task_id": task_id,
                "category": meta.get("category"),
                "difficulty_level": meta.get("difficulty_level"),
                "name": meta.get("name"),
            }
            for prefix, value in (("baseline", b), ("ours", o)):
                if value:
                    for field in (
                        "status",
                        "successes",
                        "episodes",
                        "pc_success",
                        "is_lower_bound",
                    ):
                        row[f"{prefix}_{field}"] = value.get(field)
            if (
                b
                and o
                and not b.get("is_lower_bound")
                and not o.get("is_lower_bound")
            ):
                row["delta_ours_minus_baseline_pp"] = (
                    o["pc_success"] - b["pc_success"]
                )
            writer.writerow(row)
    os.replace(temporary, path)


def update(args: argparse.Namespace) -> None:
    manifest = load_json(args.baseline_root / "task_manifest.json")
    if manifest is None:
        raise FileNotFoundError(args.baseline_root / "task_manifest.json")
    run_manifest = load_json(args.baseline_root / "run_manifest.json")
    baseline = collect_side(args.baseline_root)
    ours = collect_side(args.ours_root)
    markdown = render_markdown(
        baseline_root=args.baseline_root,
        ours_root=args.ours_root,
        baseline=baseline,
        ours=ours,
        manifest=manifest,
        run_manifest=run_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    write_csv(
        args.csv,
        baseline=baseline,
        ours=ours,
        manifest=manifest,
    )
    print(
        f"[plus-compare] baseline={baseline['completed']['pc_success']:.2f}% "
        f"ours={ours['completed']['pc_success']:.2f}% "
        f"completed_eps="
        f"{baseline['completed']['episodes']}/{ours['completed']['episodes']} "
        f"output={args.output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(
            "/data2/JM/Code/molmoact2/lerobot/outputs/libero_plus_eval/baseline"
        ),
    )
    parser.add_argument(
        "--ours-root",
        type=Path,
        default=Path("/data2/JM/Code/molmoact2/lerobot/outputs/libero_plus_eval/ours"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data2/JM/Code/molmoact2/lerobot/outputs/libero_plus_eval/"
            "baseline_vs_ours_task_comparison.md"
        ),
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.csv is None:
        args.csv = args.output.with_suffix(".csv")
    while True:
        update(args)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

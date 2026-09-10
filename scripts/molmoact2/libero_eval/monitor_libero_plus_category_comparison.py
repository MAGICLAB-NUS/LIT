#!/usr/bin/env python3
"""Continuously compare baseline and V3 by LIBERO-Plus category."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect_completed(root: Path) -> dict[tuple[str, int], tuple[int, int]]:
    """Collect completed rollout counts from direct or 8-GPU-sharded outputs."""
    records: dict[tuple[str, int], tuple[int, int]] = {}
    paths = list(root.glob("*/live_eval.json"))
    paths.extend(root.glob("gpu_*/**/live_eval.json"))
    for path in sorted(set(paths)):
        payload = load_json(path) or {}
        for task in payload.get("per_task", []) or []:
            suite = str(task.get("task_group") or "")
            if not suite:
                continue
            task_id = int(task["task_id"])
            successes = [
                bool(value)
                for value in task.get("metrics", {}).get("successes", [])
            ]
            if successes:
                records[(suite, task_id)] = (sum(successes), len(successes))
    return records


def rate(successes: int, episodes: int) -> str:
    if not episodes:
        return "—"
    return f"{100.0 * successes / episodes:.2f}% ({successes}/{episodes})"


def aggregate(
    keys: set[tuple[str, int]],
    records: dict[tuple[str, int], tuple[int, int]],
) -> tuple[int, int]:
    successes = 0
    episodes = 0
    for key in keys:
        success_count, episode_count = records[key]
        successes += success_count
        episodes += episode_count
    return successes, episodes


def best(
    baseline: tuple[int, int],
    v3: tuple[int, int],
) -> str:
    available = [
        (label, value)
        for label, value in (("Baseline", baseline), ("V3", v3))
        if value[1]
    ]
    if not available:
        return "—"
    best_rate = max(value[0] / value[1] for _, value in available)
    labels = [
        label
        for label, value in available
        if abs(value[0] / value[1] - best_rate) < 1e-12
    ]
    return f"{' / '.join(labels)} · {100.0 * best_rate:.2f}%"


def gap(v3: tuple[int, int], baseline: tuple[int, int]) -> str:
    if not v3[1] or not baseline[1]:
        return "—"
    value = 100.0 * v3[0] / v3[1] - 100.0 * baseline[0] / baseline[1]
    return f"{value:+.2f} pp"


def comparison_row(
    label: str,
    expected: set[tuple[str, int]],
    matched_keys: set[tuple[str, int]],
    baseline: dict[tuple[str, int], tuple[int, int]],
    v3: dict[tuple[str, int], tuple[int, int]],
) -> str:
    matched = matched_keys & expected
    baseline_value = aggregate(matched, baseline)
    v3_value = aggregate(matched, v3)
    return (
        f"| {label} | {len(matched)}/{len(expected)} | "
        f"{rate(*baseline_value)} | {rate(*v3_value)} | "
        f"{best(baseline_value, v3_value)} | {gap(v3_value, baseline_value)} |"
    )


def render(
    baseline_root: Path,
    v3_root: Path,
    manifest: dict[str, Any],
) -> str:
    baseline = collect_completed(baseline_root)
    v3 = collect_completed(v3_root)
    metadata = {
        (str(task["suite"]), int(task["task_id"])): task
        for task in manifest["tasks"]
    }
    categories = list(manifest.get("included_categories") or [])
    if not categories:
        categories = sorted(
            {
                str(task.get("category"))
                for task in manifest["tasks"]
                if task.get("category")
            }
        )
    suites = list(manifest.get("suites") or [])
    if not suites:
        suites = sorted({str(task["suite"]) for task in manifest["tasks"]})

    expected_by_category: dict[str, set[tuple[str, int]]] = defaultdict(set)
    expected_by_suite: dict[str, set[tuple[str, int]]] = defaultdict(set)
    expected_by_suite_category: dict[
        tuple[str, str], set[tuple[str, int]]
    ] = defaultdict(set)
    for key, task in metadata.items():
        category = str(task["category"])
        suite = str(task["suite"])
        expected_by_category[category].add(key)
        expected_by_suite[suite].add(key)
        expected_by_suite_category[(suite, category)].add(key)

    baseline_keys = set(baseline) & set(metadata)
    v3_keys = set(v3) & set(metadata)
    matched_keys = baseline_keys & v3_keys
    language_keys = expected_by_category.get("Language Instructions", set())
    non_language_keys = set(metadata) - language_keys

    header = (
        "| Scope | Matched progress | Baseline | V3 | Best | V3 − Baseline |"
    )
    separator = "| --- | --- | --- | --- | --- | --- |"
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# LIBERO-Plus comparison: baseline vs V3",
        "",
        f"Updated: `{updated}`",
        "",
        "Protocol: full LIBERO-Plus, 10,030 tasks, 1 episode/task, seed 1000.",
        "Both model columns use the exact intersection of completed task IDs across "
        "the baseline and V3 runs.",
        "",
        "## Overall",
        "",
        header,
        separator,
        comparison_row(
            "w/ Language",
            set(metadata),
            matched_keys,
            baseline,
            v3,
        ),
        comparison_row(
            "w/o Language",
            non_language_keys,
            matched_keys,
            baseline,
            v3,
        ),
    ]

    lines.extend(
        [
            "",
            "## Per-category",
            "",
            header.replace("Scope", "Category"),
            separator,
        ]
    )
    for category in categories:
        lines.append(
            comparison_row(
                category,
                expected_by_category[category],
                matched_keys,
                baseline,
                v3,
            )
        )

    lines.extend(["", "## Per-suite × category"])
    for suite in suites:
        lines.extend(
            [
                "",
                f"### `{suite}`",
                "",
                header.replace("Scope", "Category"),
                separator,
            ]
        )
        lines.append(
            comparison_row(
                "Avg",
                expected_by_suite[suite],
                matched_keys,
                baseline,
                v3,
            )
        )
        for category in categories:
            lines.append(
                comparison_row(
                    category,
                    expected_by_suite_category[(suite, category)],
                    matched_keys,
                    baseline,
                    v3,
                )
            )

    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- baseline: `{baseline_root}`",
            f"- V3: `{v3_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def update(args: argparse.Namespace) -> None:
    baseline_manifest = load_json(args.baseline_root / "task_manifest.json")
    v3_manifest = load_json(args.v3_root / "task_manifest.json")
    if baseline_manifest is None:
        raise FileNotFoundError(args.baseline_root / "task_manifest.json")
    if v3_manifest is None:
        raise FileNotFoundError(args.v3_root / "task_manifest.json")

    baseline_tasks = [
        (task["suite"], int(task["task_id"]), task.get("category"))
        for task in baseline_manifest["tasks"]
    ]
    v3_tasks = [
        (task["suite"], int(task["task_id"]), task.get("category"))
        for task in v3_manifest["tasks"]
    ]
    if baseline_tasks != v3_tasks:
        raise ValueError("Baseline and V3 task manifests do not match")

    text = render(args.baseline_root, args.v3_root, baseline_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    temporary_output.write_text(text, encoding="utf-8")
    temporary_output.replace(args.output)
    print(f"[plus-category-compare] output={args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        update(args)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

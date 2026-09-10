#!/usr/bin/env python3
"""Compare Baseline vs V3 on LIBERO-PRO using matched completed task IDs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_ORDER = ("libero_goal", "libero_spatial", "libero_10", "libero_object")
PERT_ORDER = ("object", "swap", "lan", "task")
PERT_LABEL = {"object": "Obj", "swap": "Pos", "lan": "Sem", "task": "Task"}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def collect_completed(
    root: Path,
) -> dict[tuple[str, int], tuple[int, int]]:
    """Collect completed per-task success counts keyed by PRO suite and task ID."""
    records: dict[tuple[str, int], tuple[int, int]] = {}
    paths = list(root.glob("*/live_eval.json"))
    paths.extend(root.glob("*/eval_info.json"))
    for path in sorted(paths):
        payload = load_json(path) or {}
        suite_from_path = path.parent.name
        for task in payload.get("per_task", []) or []:
            suite = str(task.get("task_group") or suite_from_path)
            task_id = int(task["task_id"])
            successes = [
                bool(value)
                for value in task.get("metrics", {}).get("successes", [])
            ]
            if successes:
                records[(suite, task_id)] = (sum(successes), len(successes))
    return records


def metric(
    keys: set[tuple[str, int]],
    records: dict[tuple[str, int], tuple[int, int]],
) -> tuple[int, int]:
    successes = sum(records[key][0] for key in keys)
    episodes = sum(records[key][1] for key in keys)
    return successes, episodes


def rate(value: tuple[int, int]) -> str:
    successes, episodes = value
    if not episodes:
        return "—"
    return f"{100.0 * successes / episodes:.2f}% ({successes}/{episodes})"


def best(baseline: tuple[int, int], v3: tuple[int, int]) -> str:
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
    label_cells: list[str],
    expected: set[tuple[str, int]],
    matched_keys: set[tuple[str, int]],
    baseline: dict[tuple[str, int], tuple[int, int]],
    v3: dict[tuple[str, int], tuple[int, int]],
) -> str:
    matched = matched_keys & expected
    baseline_value = metric(matched, baseline)
    v3_value = metric(matched, v3)
    cells = [
        *label_cells,
        f"{len(matched)}/{len(expected)}",
        rate(baseline_value),
        rate(v3_value),
        best(baseline_value, v3_value),
        gap(v3_value, baseline_value),
    ]
    return "| " + " | ".join(cells) + " |"


def render(
    baseline_root: Path,
    v3_root: Path,
) -> str:
    baseline = collect_completed(baseline_root)
    v3 = collect_completed(v3_root)

    expected_cells = [
        f"{base}_{perturbation}"
        for base in BASE_ORDER
        for perturbation in PERT_ORDER
    ]
    expected_keys = {
        (suite, task_id)
        for suite in expected_cells
        for task_id in range(10)
    }
    matched_keys = (set(baseline) & set(v3) & expected_keys)

    header = (
        "| Scope | Matched progress | Baseline | V3 | Best | V3 − Baseline |"
    )
    separator = "| --- | --- | --- | --- | --- | --- |"
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# LIBERO-PRO comparison: Baseline vs V3",
        "",
        f"Updated: `{updated}`",
        "",
        "Protocol: public Total16, 4 base suites × {Obj, Pos, Sem, Task}, "
        "10 tasks/cell × 50 episodes/task, seed 1000.",
        "Both model columns use the exact intersection of completed suite/task IDs "
        "across the baseline and V3 runs.",
        "",
        "## Overall Total16",
        "",
        header,
        separator,
        comparison_row(
            ["Total16"],
            expected_keys,
            matched_keys,
            baseline,
            v3,
        ),
        "",
        "## Per perturbation",
        "",
        header.replace("Scope", "Perturbation"),
        separator,
    ]

    for perturbation in PERT_ORDER:
        suites = {f"{base}_{perturbation}" for base in BASE_ORDER}
        expected = {key for key in expected_keys if key[0] in suites}
        lines.append(
            comparison_row(
                [PERT_LABEL[perturbation]],
                expected,
                matched_keys,
                baseline,
                v3,
            )
        )

    cell_header = (
        "| Base suite | Perturbation | Matched progress | Baseline | V3 | "
        "Best | V3 − Baseline |"
    )
    cell_separator = "| --- | --- | --- | --- | --- | --- | --- |"
    lines.extend(
        [
            "",
            "## Per cell",
            "",
            cell_header,
            cell_separator,
        ]
    )
    for base in BASE_ORDER:
        for perturbation in PERT_ORDER:
            suite = f"{base}_{perturbation}"
            expected = {(suite, task_id) for task_id in range(10)}
            lines.append(
                comparison_row(
                    [base, PERT_LABEL[perturbation]],
                    expected,
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
            f"- Baseline: `{baseline_root}`",
            f"- V3: `{v3_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def update(args: argparse.Namespace) -> None:
    text = render(args.baseline_root, args.v3_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    temporary_output.write_text(text, encoding="utf-8")
    temporary_output.replace(args.output)
    print(f"[pro-compare] output={args.output}", flush=True)


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

#!/usr/bin/env python3
"""Continuously compare baseline, V3-25k, and V3-30k on LIBERO-Plus."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from monitor_libero_plus_category_comparison import (
    aggregate,
    collect_completed,
    load_json,
    rate,
)


def best(
    values: list[tuple[str, tuple[int, int]]],
) -> str:
    available = [(label, value) for label, value in values if value[1]]
    if not available:
        return "—"
    best_rate = max(value[0] / value[1] for _, value in available)
    labels = [
        label
        for label, value in available
        if abs(value[0] / value[1] - best_rate) < 1e-12
    ]
    return f"{' / '.join(labels)} · {100.0 * best_rate:.2f}%"


def gap(
    model: tuple[int, int],
    baseline: tuple[int, int],
) -> str:
    if not model[1] or not baseline[1]:
        return "—"
    value = 100.0 * model[0] / model[1] - 100.0 * baseline[0] / baseline[1]
    return f"{value:+.2f} pp"


def comparison_row(
    label: str,
    expected: set[tuple[str, int]],
    matched_keys: set[tuple[str, int]],
    baseline: dict[tuple[str, int], tuple[int, int]],
    v3_25k: dict[tuple[str, int], tuple[int, int]],
    v3_30k: dict[tuple[str, int], tuple[int, int]],
) -> str:
    matched = matched_keys & expected
    baseline_value = aggregate(matched, baseline)
    v3_25k_matched = aggregate(matched, v3_25k)
    v3_30k_matched = aggregate(matched, v3_30k)
    return (
        f"| {label} | {len(matched)}/{len(expected)} | "
        f"{rate(*baseline_value)} | {rate(*v3_25k_matched)} | "
        f"{rate(*v3_30k_matched)} | "
        f"{best([('Baseline', baseline_value), ('V3-25k', v3_25k_matched), ('V3-30k', v3_30k_matched)])} | "
        f"{gap(v3_30k_matched, baseline_value)} |"
    )


def render(
    baseline_root: Path,
    v3_25k_root: Path,
    v3_30k_root: Path,
    manifest: dict[str, Any],
) -> str:
    baseline = collect_completed(baseline_root)
    v3_25k = collect_completed(v3_25k_root)
    v3_30k = collect_completed(v3_30k_root)
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

    metadata_keys = set(metadata)
    baseline_keys = set(baseline) & metadata_keys
    v3_25k_keys = set(v3_25k) & metadata_keys
    v3_30k_keys = set(v3_30k) & metadata_keys
    matched_keys = baseline_keys & v3_25k_keys & v3_30k_keys
    language_keys = expected_by_category.get("Language Instructions", set())
    non_language_keys = metadata_keys - language_keys

    header = (
        "| Scope | Matched progress | Baseline | V3-25k matched | "
        "V3-30k matched | Best | V3-30k − Baseline |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- |"
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# LIBERO-Plus comparison: baseline vs V3-25k vs V3-30k",
        "",
        f"Updated: `{updated}`",
        "",
        "Protocol: full LIBERO-Plus, 10,030 tasks, 1 episode/task, seed 1000.",
        "All model columns use the exact intersection of completed task IDs across "
        "the three runs.",
        "",
        "## Overall",
        "",
        header,
        separator,
        comparison_row(
            "w/ Language",
            metadata_keys,
            matched_keys,
            baseline,
            v3_25k,
            v3_30k,
        ),
        comparison_row(
            "w/o Language",
            non_language_keys,
            matched_keys,
            baseline,
            v3_25k,
            v3_30k,
        ),
        "",
        "## Per-category",
        "",
        header.replace("Scope", "Category"),
        separator,
    ]
    for category in categories:
        lines.append(
            comparison_row(
                category,
                expected_by_category[category],
                matched_keys,
                baseline,
                v3_25k,
                v3_30k,
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
                v3_25k,
                v3_30k,
            )
        )
        for category in categories:
            lines.append(
                comparison_row(
                    category,
                    expected_by_suite_category[(suite, category)],
                    matched_keys,
                    baseline,
                    v3_25k,
                    v3_30k,
                )
            )

    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- baseline: `{baseline_root}`",
            f"- V3-25k: `{v3_25k_root}`",
            f"- V3-30k: `{v3_30k_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def manifest_tasks(manifest: dict[str, Any]) -> list[tuple[str, int, Any]]:
    return [
        (task["suite"], int(task["task_id"]), task.get("category"))
        for task in manifest["tasks"]
    ]


def update(args: argparse.Namespace) -> None:
    roots = (args.baseline_root, args.v3_25k_root, args.v3_30k_root)
    manifests = [load_json(root / "task_manifest.json") for root in roots]
    for root, manifest in zip(roots, manifests, strict=True):
        if manifest is None:
            raise FileNotFoundError(root / "task_manifest.json")
    baseline_manifest, v3_25k_manifest, v3_30k_manifest = manifests
    assert baseline_manifest is not None
    assert v3_25k_manifest is not None
    assert v3_30k_manifest is not None
    expected_tasks = manifest_tasks(baseline_manifest)
    if manifest_tasks(v3_25k_manifest) != expected_tasks:
        raise ValueError("Baseline and V3-25k task manifests do not match")
    if manifest_tasks(v3_30k_manifest) != expected_tasks:
        raise ValueError("Baseline and V3-30k task manifests do not match")

    text = render(
        args.baseline_root,
        args.v3_25k_root,
        args.v3_30k_root,
        baseline_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    temporary_output.write_text(text, encoding="utf-8")
    temporary_output.replace(args.output)
    print(f"[plus-threeway-compare] output={args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--v3-25k-root", type=Path, required=True)
    parser.add_argument("--v3-30k-root", type=Path, required=True)
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

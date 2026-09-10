#!/usr/bin/env python3
"""Build a deterministic LIBERO-Plus evaluation manifest.

Protocols:
  - full (default for public comparison): every task in task_classification.json,
    including Language Instructions. Matches the official LIBERO-Plus paper setting
    of evaluating the complete 10,030-task suite.
  - base_category: cover every base skill × non-language perturbation category,
    sampling N variants per cell (prefer mid difficulty), then shuffle within suite.
  - balanced: legacy suite × category × difficulty grid sampling (language excluded).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUITES = ("libero_object", "libero_10", "libero_goal", "libero_spatial")
NON_LANGUAGE_CATEGORIES = (
    "Background Textures",
    "Camera Viewpoints",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
)
CATEGORIES = NON_LANGUAGE_CATEGORIES  # backward-compatible alias for subsample protocols
ALL_CATEGORIES = NON_LANGUAGE_CATEGORIES + ("Language Instructions",)
LEVELS = (1, 2, 3, 4, 5)
LANGUAGE_CATEGORY = "Language Instructions"
TARGET_DIFFICULTY = 3
_BASE_SUFFIX_PATTERNS = (
    r"_table_\d+$",
    r"_tb_\d+$",
    r"_view_.*$",
    r"_light_.*$",
    r"_noise_.*$",
    r"_initstate_.*$",
    r"_language_.*$",
    r"_add_.*$",
    r"_level.*$",
)


def load_classification(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = set(SUITES) - payload.keys()
    if missing:
        raise ValueError(f"Classification is missing suites: {sorted(missing)}")
    for suite in SUITES:
        ids = [entry["id"] for entry in payload[suite]]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"{suite}: classification IDs must be contiguous and one-based")
    return payload


def base_skill_name(name: str) -> str:
    skill = name
    changed = True
    while changed:
        changed = False
        for pattern in _BASE_SUFFIX_PATTERNS:
            stripped = re.sub(pattern, "", skill)
            if stripped != skill:
                skill = stripped
                changed = True
    return skill


def select_balanced(
    data: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    samples_per_cell: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for suite_index, suite in enumerate(SUITES):
        for category_index, category in enumerate(CATEGORIES):
            selected_ids: set[int] = set()
            deficits: list[tuple[int, int]] = []
            for level in LEVELS:
                candidates = [
                    entry
                    for entry in data[suite]
                    if entry["category"] == category and entry.get("difficulty_level") == level
                ]
                if not candidates:
                    raise ValueError(f"No tasks for {suite}/{category}/level-{level}")
                count = min(samples_per_cell, len(candidates))
                cell_seed = seed + suite_index * 10_000 + category_index * 100 + level
                sampled = random.Random(cell_seed).sample(candidates, count)
                for entry in sampled:
                    selected.append(_record(suite, entry, level, base_skill=base_skill_name(entry["name"])))
                    selected_ids.add(entry["id"])
                if count < samples_per_cell:
                    deficits.append((level, samples_per_cell - count))

            for target_level, count in deficits:
                available = [
                    entry
                    for entry in data[suite]
                    if entry["category"] == category and entry["id"] not in selected_ids
                ]
                rng = random.Random(
                    seed + suite_index * 10_000 + category_index * 100 + target_level + 50
                )
                rng.shuffle(available)
                available.sort(
                    key=lambda entry: abs((entry.get("difficulty_level") or 99) - target_level)
                )
                backfill = available[:count]
                if len(backfill) != count:
                    raise ValueError(f"Cannot backfill {suite}/{category}/level-{target_level}")
                for entry in backfill:
                    selected.append(
                        _record(
                            suite,
                            entry,
                            target_level,
                            base_skill=base_skill_name(entry["name"]),
                        )
                    )
                    selected_ids.add(entry["id"])
    return sorted(selected, key=lambda item: (item["suite"], item["task_id"]))


def _suite_base_skills(entries: list[dict[str, Any]]) -> list[str]:
    """Prefer base skills that exist under all non-language categories."""
    by_base: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        if entry["category"] == LANGUAGE_CATEGORY:
            continue
        by_base[base_skill_name(entry["name"])].add(entry["category"])
    full = sorted(base for base, cats in by_base.items() if set(CATEGORIES) <= cats)
    if len(full) >= 10:
        return full[:10] if len(full) > 10 else full
    # Fall back to the most complete bases.
    ranked = sorted(
        by_base.items(),
        key=lambda item: (-len(set(CATEGORIES) & item[1]), item[0]),
    )
    return [base for base, _ in ranked[:10]]


def _pick_near_difficulty(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    target_level: int = TARGET_DIFFICULTY,
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)}")
    rng = random.Random(seed)
    by_dist: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in candidates:
        dist = abs((entry.get("difficulty_level") or 99) - target_level)
        by_dist[dist].append(entry)
    ordered: list[dict[str, Any]] = []
    for dist in sorted(by_dist):
        group = list(by_dist[dist])
        rng.shuffle(group)
        ordered.extend(group)
    return ordered[:count]


def select_full(
    data: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Official LIBERO-Plus protocol: every classified task, all 7 categories."""
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        for entry in data[suite]:
            category = entry["category"]
            if category not in ALL_CATEGORIES:
                raise ValueError(f"Unknown category in {suite}: {category}")
            selected.append(
                _record(
                    suite,
                    entry,
                    sampling_level=int(entry.get("difficulty_level") or TARGET_DIFFICULTY),
                    base_skill=base_skill_name(entry["name"]),
                )
            )
    return selected


def select_base_category(
    data: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    samples_per_cell: int,
) -> list[dict[str, Any]]:
    """10 base skills × 6 categories × N variants (prefer mid difficulty)."""
    selected: list[dict[str, Any]] = []
    for suite_index, suite in enumerate(SUITES):
        entries = [entry for entry in data[suite] if entry["category"] != LANGUAGE_CATEGORY]
        bases = _suite_base_skills(entries)
        if len(bases) != 10:
            raise ValueError(f"{suite}: expected 10 base skills, found {len(bases)}: {bases}")

        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            base = base_skill_name(entry["name"])
            if base in bases and entry["category"] in NON_LANGUAGE_CATEGORIES:
                buckets[(base, entry["category"])].append(entry)

        for base_index, base in enumerate(bases):
            for category_index, category in enumerate(NON_LANGUAGE_CATEGORIES):
                candidates = buckets[(base, category)]
                if not candidates:
                    raise ValueError(f"No tasks for {suite}/{base}/{category}")
                cell_seed = (
                    seed
                    + suite_index * 100_000
                    + base_index * 1_000
                    + category_index * 10
                    + samples_per_cell
                )
                picked = _pick_near_difficulty(
                    candidates, count=samples_per_cell, seed=cell_seed
                )
                for entry in picked:
                    selected.append(
                        _record(
                            suite,
                            entry,
                            sampling_level=TARGET_DIFFICULTY,
                            base_skill=base,
                        )
                    )
    return selected


def _record(
    suite: str,
    entry: dict[str, Any],
    sampling_level: int,
    *,
    base_skill: str,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "classification_id": entry["id"],
        "task_id": entry["id"] - 1,
        "name": entry["name"],
        "base_skill": base_skill,
        "category": entry["category"],
        "difficulty_level": entry.get("difficulty_level"),
        "sampling_level": sampling_level,
    }


def shuffle_within_suites(
    tasks: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    by_suite: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITES}
    for task in tasks:
        by_suite[task["suite"]].append(task)

    ordered: list[dict[str, Any]] = []
    for suite_index, suite in enumerate(SUITES):
        suite_tasks = list(by_suite[suite])
        rng = random.Random(seed + 1_000_000 + suite_index)
        rng.shuffle(suite_tasks)
        for eval_index, task in enumerate(suite_tasks):
            record = dict(task)
            record["eval_order"] = eval_index
            ordered.append(record)
    return ordered


def write_manifest(
    output: Path,
    *,
    classification: Path,
    seed: int,
    samples_per_cell: int,
    episodes_per_task: int,
    protocol: str,
) -> None:
    data = load_classification(classification)
    if protocol == "full":
        tasks = select_full(data)
        expected = sum(len(data[suite]) for suite in SUITES)
        phase = "full"
        protocol_version = 4
        excluded_categories: list[str] = []
        included_categories = list(ALL_CATEGORIES)
        if episodes_per_task != 1:
            raise ValueError(
                "Official full LIBERO-Plus protocol requires episodes_per_task=1 "
                f"(got {episodes_per_task})."
            )
    elif protocol == "base_category":
        tasks = select_base_category(data, seed=seed, samples_per_cell=samples_per_cell)
        expected = len(SUITES) * 10 * len(NON_LANGUAGE_CATEGORIES) * samples_per_cell
        phase = "base_category"
        protocol_version = 3
        excluded_categories = [LANGUAGE_CATEGORY]
        included_categories = list(NON_LANGUAGE_CATEGORIES)
    elif protocol == "balanced":
        tasks = select_balanced(data, seed=seed, samples_per_cell=samples_per_cell)
        expected = len(SUITES) * len(NON_LANGUAGE_CATEGORIES) * len(LEVELS) * samples_per_cell
        phase = "balanced"
        protocol_version = 2
        excluded_categories = [LANGUAGE_CATEGORY]
        included_categories = list(NON_LANGUAGE_CATEGORIES)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} tasks, selected {len(tasks)}")
    if protocol != "full" and any(
        task["category"] == LANGUAGE_CATEGORY or "_language_" in task["name"] for task in tasks
    ):
        raise ValueError("Language-perturbed task leaked into the subsampled manifest")
    # Deduplicate by task_id within suite (same variant should not appear twice).
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for task in tasks:
        key = (task["suite"], task["task_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)
    if len(unique) != len(tasks):
        raise ValueError(
            f"Duplicate task_ids in selection: {len(tasks)} -> {len(unique)} unique"
        )

    # Keep official full evaluation in classification order; only subsample protocols shuffle.
    if protocol == "full":
        tasks = unique
        for eval_index, task in enumerate(tasks):
            task["eval_order"] = eval_index
        eval_order = "classification_order"
    else:
        tasks = shuffle_within_suites(unique, seed=seed)
        eval_order = "shuffled_within_suite"

    manifest = {
        "protocol_version": protocol_version,
        "phase": phase,
        "selection_seed": seed,
        "rollout_seed": seed,
        "eval_order": eval_order,
        "samples_per_cell": samples_per_cell if protocol != "full" else None,
        "target_difficulty": TARGET_DIFFICULTY if protocol != "full" else None,
        "episodes_per_task": episodes_per_task,
        "excluded_categories": excluded_categories,
        "included_categories": included_categories,
        "suites": list(SUITES),
        "num_tasks": len(tasks),
        "num_rollouts": len(tasks) * episodes_per_task,
        "tasks": tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)

    print(
        f"[manifest] protocol={phase} tasks={len(tasks)} "
        f"rollouts={manifest['num_rollouts']} "
        f"suites={dict(Counter(task['suite'] for task in tasks))} "
        f"categories={dict(Counter(task['category'] for task in tasks))}"
    )
    print(f"[manifest] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--samples-per-cell", type=int, default=2)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument(
        "--protocol",
        choices=("full", "base_category", "balanced"),
        default="full",
    )
    args = parser.parse_args()
    write_manifest(
        args.output,
        classification=args.classification,
        seed=args.seed,
        samples_per_cell=args.samples_per_cell,
        episodes_per_task=args.episodes_per_task,
        protocol=args.protocol,
    )


if __name__ == "__main__":
    main()

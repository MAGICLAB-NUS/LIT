#!/usr/bin/env python3
"""Live-update V4 in-dist, LIBERO-Plus, and LIBERO-PRO benchmark rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


SUITES = {
    "libero_spatial": "Sp",
    "libero_object": "Ob",
    "libero_goal": "Go",
    "libero_10": "Long",
}
V4_ROW_PREFIX = "| **Ours v4-25k** |"
V4_NOTE = (
    "- **V4 LIBERO Ckpt**: "
    "`libero_goal_prior_v4/seed_1000/stage2/checkpoints/025000` "
    "（8/8 hard bottleneck；in-dist / Plus / PRO 同步评测）"
)
LEGACY_V4_NOTE = (
    "- **V4 LIBERO Ckpt**: "
    "`libero_goal_prior_v4/seed_1000/stage2/checkpoints/025000` "
    "（8/8 hard bottleneck；本次仅评测 LIBERO in-dist）"
)
PROTOCOL_NOTE = (
    "- **V3/V4 eval parity**: in-dist `40 tasks × 50 ep, bs10`; "
    "Plus `10030 tasks × 1 ep, bs1`（task manifest 完全一致）；"
    "PRO `16 cells × 500 ep, bs1`；均为 seed 1000"
)
EXPECTED_POLICY_SUFFIX = (
    "libero_goal_prior_v4/seed_1000/stage2/"
    "checkpoints/025000/pretrained_model"
)
PLUS_CATEGORIES = {
    "Camera Viewpoints": ("Cam", 1599),
    "Robot Initial States": ("Robot", 1550),
    "Language Instructions": ("Lang", 1537),
    "Light Conditions": ("Light", 1142),
    "Background Textures": ("Bg", 1076),
    "Sensor Noise": ("Noise", 1601),
    "Objects Layout": ("Layout", 1525),
}
PRO_PERTURBATIONS = {
    "object": "Obj",
    "swap": "Pos",
    "lan": "Sem",
    "task": "Task",
}
PRO_SUITES = [
    f"{base}_{perturbation}"
    for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for perturbation in PRO_PERTURBATIONS
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(eval_root: Path) -> None:
    manifest = load_json(eval_root / "run_manifest.json")
    expected_suites = set(SUITES)
    errors = []
    if manifest.get("protocol") != "openvla_public_50ep":
        errors.append(f"protocol={manifest.get('protocol')!r}")
    if manifest.get("episodes_per_task") != 50:
        errors.append(f"episodes_per_task={manifest.get('episodes_per_task')!r}")
    if str(manifest.get("official_horizons")).lower() != "true":
        errors.append(f"official_horizons={manifest.get('official_horizons')!r}")
    if set(manifest.get("suites", [])) != expected_suites:
        errors.append(f"suites={manifest.get('suites')!r}")
    policy_path = str(manifest.get("policy_path", ""))
    if not policy_path.endswith(EXPECTED_POLICY_SUFFIX):
        errors.append(f"policy_path={policy_path!r}")
    if errors:
        raise RuntimeError("Refusing non-official or wrong-checkpoint results: " + ", ".join(errors))


def try_load_json(path: Path) -> dict | None:
    try:
        return load_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect_scores(eval_root: Path) -> tuple[dict[str, dict[str, int | float]], bool]:
    scores: dict[str, dict[str, int | float]] = {}
    all_final = True
    for suite in SUITES:
        suite_dirs = []
        legacy_dir = eval_root / suite
        if legacy_dir.is_dir():
            suite_dirs.append(legacy_dir)
        suite_dirs.extend(sorted(eval_root.glob(f"gpu_*/{suite}")))

        shard_finals = []
        success_count = 0
        episode_count = 0
        completed_task_ids: set[int] = set()
        running_payloads = []
        for suite_dir in suite_dirs:
            status = try_load_json(suite_dir / "process_status.json") or {}
            final_info = try_load_json(suite_dir / "eval_info.json")
            live_info = try_load_json(suite_dir / "live_eval.json") or {}
            shard_finals.append(status.get("status") == "final" and final_info is not None)
            payload = final_info or live_info
            for task in payload.get("per_task", []):
                task_id = int(task.get("task_id", -1))
                if task_id in completed_task_ids:
                    raise RuntimeError(f"{suite}: duplicate completed task_id={task_id}")
                completed_task_ids.add(task_id)
                values = task.get("metrics", {}).get("successes", [])
                success_count += sum(bool(value) for value in values)
                episode_count += len(values)
            running_payloads.extend(live_info.get("running_tasks", []))

        is_final = bool(suite_dirs) and all(shard_finals)
        all_final = all_final and is_final
        for running in running_payloads:
            if int(running.get("task_id", -1)) in completed_task_ids:
                continue
            episode_count += int(running.get("finished_rollouts") or 0)
            success_count += int(running.get("successes_so_far") or 0)

        if is_final:
            if completed_task_ids != set(range(10)):
                raise RuntimeError(
                    f"{suite}: expected task_ids 0-9, got {sorted(completed_task_ids)}"
                )
            if episode_count != 500:
                raise RuntimeError(f"{suite}: expected 500 episodes, got {episode_count}")
        scores[suite] = {
            "successes": success_count,
            "episodes": episode_count,
            "pc_success": 100.0 * success_count / episode_count if episode_count else 0.0,
        }
    return scores, all_final


def update_summary(
    summary_path: Path,
    scores: dict[str, dict[str, int | float]],
    *,
    final: bool,
) -> str:
    text = summary_path.read_text(encoding="utf-8")
    text = text.replace(LEGACY_V4_NOTE, V4_NOTE)
    if V4_NOTE not in text:
        v3_ckpt_line = (
            "- **Ckpt**: "
            "`libero_goal_prior_v3/seed_1000/stage2/checkpoints/025000`"
        )
        if v3_ckpt_line not in text:
            raise RuntimeError("Could not find V3 checkpoint line in summary")
        text = text.replace(v3_ckpt_line, f"{v3_ckpt_line}\n{V4_NOTE}", 1)
    if PROTOCOL_NOTE not in text:
        text = text.replace(V4_NOTE, f"{V4_NOTE}\n{PROTOCOL_NOTE}", 1)

    total_episodes = sum(int(score["episodes"]) for score in scores.values())
    total_successes = sum(int(score["successes"]) for score in scores.values())

    def format_suite(suite: str) -> str:
        score = scores[suite]
        episodes = int(score["episodes"])
        if final:
            return f"**{float(score['pc_success']):.1f}**"
        if not episodes:
            return "— (0/500)"
        return f"**{float(score['pc_success']):.1f}** ({episodes}/500)"

    if final:
        average = sum(float(scores[suite]["pc_success"]) for suite in SUITES) / len(SUITES)
        init_cell = "**VL+rand**"
        average_cell = f"**{average:.1f}**"
    else:
        init_cell = f"**VL+rand · live {total_episodes}/2000**"
        average_cell = (
            f"**{100.0 * total_successes / total_episodes:.1f}** ({total_episodes}/2000)"
            if total_episodes
            else "— (0/2000)"
        )
    row = (
        f"{V4_ROW_PREFIX} {init_cell} | "
        f"{format_suite('libero_spatial')} | "
        f"{format_suite('libero_object')} | "
        f"{format_suite('libero_goal')} | "
        f"{format_suite('libero_10')} | "
        f"{average_cell} |"
    )
    lines = text.splitlines()
    existing = next(
        (i for i, line in enumerate(lines) if line.startswith("| **Ours v4-25k** |")),
        None,
    )
    if existing is not None:
        lines[existing] = row
    else:
        v3_row = next(
            (
                i
                for i, line in enumerate(lines)
                if line.startswith("| **Ours v3-25k** |")
            ),
            None,
        )
        if v3_row is None:
            raise RuntimeError("Could not find V3 result row in summary")
        lines.insert(v3_row + 1, row)
    updated = "\n".join(lines) + "\n"
    if updated != summary_path.read_text(encoding="utf-8"):
        summary_path.write_text(updated, encoding="utf-8")
    return row


def validate_plus_manifest(eval_root: Path) -> None:
    run = load_json(eval_root / "run_manifest.json")
    task = load_json(eval_root / "task_manifest.json")
    errors = []
    expected_run = {
        "benchmark": "libero_plus",
        "protocol": "full",
        "episodes_per_task": 1,
        "eval_batch_size": 1,
        "eval_seed": 1000,
        "num_gpus": 8,
        "excluded_categories": [],
    }
    for key, expected in expected_run.items():
        if run.get(key) != expected:
            errors.append(f"{key}={run.get(key)!r}, expected {expected!r}")
    if not str(run.get("policy_path", "")).endswith(EXPECTED_POLICY_SUFFIX):
        errors.append(f"policy_path={run.get('policy_path')!r}")
    expected_task = {
        "phase": "full",
        "selection_seed": 1000,
        "rollout_seed": 1000,
        "episodes_per_task": 1,
        "excluded_categories": [],
        "num_tasks": 10030,
        "num_rollouts": 10030,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            errors.append(f"task_manifest.{key}={task.get(key)!r}, expected {expected!r}")
    counts: dict[str, int] = {category: 0 for category in PLUS_CATEGORIES}
    for item in task.get("tasks", []):
        category = str(item.get("category"))
        if category in counts:
            counts[category] += 1
    for category, (_, expected) in PLUS_CATEGORIES.items():
        if counts[category] != expected:
            errors.append(f"{category} tasks={counts[category]}, expected {expected}")

    # The completed V3 run lives beside the V4 model directory. Byte identity
    # proves task IDs, order, category labels, and rollout seeds all match.
    reference = eval_root.parent.parent / "libero_plus_full_seed_1000" / "task_manifest.json"
    if reference.is_file():
        v3_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        v4_hash = hashlib.sha256((eval_root / "task_manifest.json").read_bytes()).hexdigest()
        if v3_hash != v4_hash:
            errors.append(f"Plus task manifest differs from V3: {v4_hash} != {v3_hash}")
    if errors:
        raise RuntimeError("Refusing mismatched V4 Plus results: " + ", ".join(errors))


def validate_pro_manifest(eval_root: Path) -> None:
    run = load_json(eval_root / "run_manifest.json")
    errors = []
    expected = {
        "benchmark": "libero_pro",
        "protocol": "official_paper_50ep",
        "suites": PRO_SUITES,
        "task_ids": None,
        "episodes_per_task": 50,
        "eval_batch_size": 1,
        "eval_seed": 1000,
        "horizons": {"spatial": 220, "object": 280, "goal": 300, "10": 520},
    }
    for key, wanted in expected.items():
        if run.get(key) != wanted:
            errors.append(f"{key}={run.get(key)!r}, expected {wanted!r}")
    if not str(run.get("policy_path", "")).endswith(EXPECTED_POLICY_SUFFIX):
        errors.append(f"policy_path={run.get('policy_path')!r}")

    reference_path = (
        eval_root.parent.parent / "stage2_025000" / "full50_seed_1000" / "run_manifest.json"
    )
    if reference_path.is_file():
        reference = load_json(reference_path)
        for key in expected:
            if reference.get(key) != run.get(key):
                errors.append(f"PRO {key} differs from V3 reference")
    if errors:
        raise RuntimeError("Refusing mismatched V4 PRO results: " + ", ".join(errors))


def metric_from_payload(payload: dict | None) -> dict[str, int | float]:
    payload = payload or {}
    successes = int(payload.get("successes") or 0)
    episodes = int(payload.get("episodes") or 0)
    return {
        "successes": successes,
        "episodes": episodes,
        "pc_success": 100.0 * successes / episodes if episodes else 0.0,
    }


def format_live_percent(
    score: dict[str, int | float],
    expected: int,
    *,
    final: bool,
) -> str:
    episodes = int(score["episodes"])
    if final:
        return f"**{float(score['pc_success']):.1f}**"
    if not episodes:
        return f"— (0/{expected})"
    return f"**{float(score['pc_success']):.1f}** ({episodes}/{expected})"


def upsert_section_row(
    summary_path: Path,
    *,
    section_heading: str,
    next_heading: str | None,
    row_prefix: str,
    v3_prefix: str,
    row: str,
) -> None:
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith(section_heading)),
        None,
    )
    if start is None:
        raise RuntimeError(f"Could not find summary section {section_heading!r}")
    end = len(lines)
    if next_heading is not None:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].startswith(next_heading)
            ),
            len(lines),
        )
    existing = next(
        (
            index
            for index in range(start + 1, end)
            if lines[index].startswith(row_prefix)
        ),
        None,
    )
    if existing is not None:
        lines[existing] = row
    else:
        v3_row = next(
            (
                index
                for index in range(start + 1, end)
                if lines[index].startswith(v3_prefix)
            ),
            None,
        )
        if v3_row is None:
            raise RuntimeError(f"Could not find V3 row in {section_heading!r}")
        lines.insert(v3_row + 1, row)
    updated = "\n".join(lines) + "\n"
    if updated != summary_path.read_text(encoding="utf-8"):
        summary_path.write_text(updated, encoding="utf-8")


def update_plus_summary(summary_path: Path, eval_root: Path) -> tuple[str, bool]:
    payload = try_load_json(eval_root / "live_summary.json") or {}
    overall = metric_from_payload(payload.get("overall"))
    final = payload.get("status") == "final" and int(overall["episodes"]) == 10030
    categories = payload.get("per_category") or {}
    cells = [
        format_live_percent(
            metric_from_payload(categories.get(category)),
            expected,
            final=final,
        )
        for category, (_, expected) in PLUS_CATEGORIES.items()
    ]
    total = format_live_percent(overall, 10030, final=final)
    init = "**VLM+rand AE**" if final else f"**VLM+rand AE · live {int(overall['episodes'])}/10030**"
    row = f"| **Ours v4-25k** | {init} | " + " | ".join([*cells, total]) + " |"
    upsert_section_row(
        summary_path,
        section_heading="## 2.",
        next_heading="## 3.",
        row_prefix="| **Ours v4-25k** |",
        v3_prefix="| **Ours v3-25k** |",
        row=row,
    )
    return row, final


def collect_pro_scores(
    eval_root: Path,
) -> tuple[dict[str, dict[str, int | float]], bool]:
    scores: dict[str, dict[str, int | float]] = {}
    all_final = True
    for suite in PRO_SUITES:
        suite_dir = eval_root / suite
        status = try_load_json(suite_dir / "process_status.json") or {}
        final_info = try_load_json(suite_dir / "eval_info.json")
        live_info = try_load_json(suite_dir / "live_eval.json") or {}
        payload = final_info or live_info
        completed_task_ids: set[int] = set()
        successes = 0
        episodes = 0
        for task in payload.get("per_task", []):
            task_id = int(task.get("task_id", -1))
            if task_id in completed_task_ids:
                raise RuntimeError(f"{suite}: duplicate completed task_id={task_id}")
            completed_task_ids.add(task_id)
            values = task.get("metrics", {}).get("successes", [])
            successes += sum(bool(value) for value in values)
            episodes += len(values)
        for running in live_info.get("running_tasks", []):
            if int(running.get("task_id", -1)) in completed_task_ids:
                continue
            successes += int(running.get("successes_so_far") or 0)
            episodes += int(running.get("finished_rollouts") or 0)
        suite_final = status.get("status") == "final" and final_info is not None
        all_final = all_final and suite_final
        if suite_final:
            if completed_task_ids != set(range(10)):
                raise RuntimeError(
                    f"{suite}: expected task_ids 0-9, got {sorted(completed_task_ids)}"
                )
            if episodes != 500:
                raise RuntimeError(f"{suite}: expected 500 episodes, got {episodes}")
        scores[suite] = {
            "successes": successes,
            "episodes": episodes,
            "pc_success": 100.0 * successes / episodes if episodes else 0.0,
        }
    return scores, all_final


def update_pro_summary(summary_path: Path, eval_root: Path) -> tuple[str, bool]:
    per_suite, all_suites_final = collect_pro_scores(eval_root)
    perturbation_scores: dict[str, dict[str, int | float]] = {}
    total_successes = 0
    total_episodes = 0
    for perturbation in PRO_PERTURBATIONS:
        successes = 0
        episodes = 0
        for suite in PRO_SUITES:
            if not suite.endswith(f"_{perturbation}"):
                continue
            score = per_suite[suite]
            successes += int(score["successes"])
            episodes += int(score["episodes"])
        perturbation_scores[perturbation] = {
            "successes": successes,
            "episodes": episodes,
            "pc_success": 100.0 * successes / episodes if episodes else 0.0,
        }
        total_successes += successes
        total_episodes += episodes
    final = all_suites_final and total_episodes == 8000

    def format_rate(score: dict[str, int | float]) -> str:
        episodes = int(score["episodes"])
        rate = float(score["pc_success"]) / 100.0
        if final:
            return f"**{rate:.2f}**"
        if not episodes:
            return "— (0/2000)"
        return f"**{rate:.2f}** ({episodes}/2000)"

    cells = [format_rate(perturbation_scores[key]) for key in PRO_PERTURBATIONS]
    if final:
        total_cell = f"**{total_successes / total_episodes:.2f}**"
    elif total_episodes:
        total_cell = f"**{total_successes / total_episodes:.2f}** ({total_episodes}/8000)"
    else:
        total_cell = "— (0/8000)"
    row = (
        "| V4 | **Ours v4-25k** | "
        + " | ".join([*cells, "—", total_cell])
        + " |"
    )
    upsert_section_row(
        summary_path,
        section_heading="## 3.",
        next_heading=None,
        row_prefix="| V4 | **Ours v4-25k** |",
        v3_prefix="| 5 | **Ours v3-25k** |",
        row=row,
    )
    return row, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--plus-root", type=Path)
    parser.add_argument("--pro-root", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    validate_manifest(args.eval_root)
    if args.plus_root is not None:
        validate_plus_manifest(args.plus_root)
    if args.pro_root is not None:
        validate_pro_manifest(args.pro_root)
    while True:
        scores, in_dist_final = collect_scores(args.eval_root)
        rows = [update_summary(args.summary, scores, final=in_dist_final)]
        all_final = in_dist_final
        if args.plus_root is not None:
            plus_row, plus_final = update_plus_summary(args.summary, args.plus_root)
            rows.append(plus_row)
            all_final = all_final and plus_final
        if args.pro_root is not None:
            pro_row, pro_final = update_pro_summary(args.summary, args.pro_root)
            rows.append(pro_row)
            all_final = all_final and pro_final
        print("[integrate] " + " || ".join(rows), flush=True)
        if all_final:
            return
        if not args.wait:
            raise SystemExit("Evaluation is not complete")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

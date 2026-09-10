#!/usr/bin/env python3
"""Monitor LIBERO-PRO eval runs and refresh live_summary.md / .json / .csv.

Layout expected (from eval_libero_pro_checkpoint.sh):

  <eval_root>/
    run_manifest.json
    libero_spatial_object/
      process_status.json
      live_eval.json      # while running
      eval_info.json      # when finished
      videos/             # optional
    ...

Suite names encode base + perturbation, e.g. libero_10_swap → base=libero_10, pert=swap (Pos).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PERTURBATION_ORDER = ("object", "swap", "lan", "task", "env")
PERTURBATION_LABEL = {
    "object": "Obj",
    "swap": "Pos",
    "lan": "Sem",
    "task": "Task",
    "env": "Env",
    "unknown": "?",
}
# Match official README suite order: Goal → Spatial → 10 → Object
BASE_ORDER = ("libero_goal", "libero_spatial", "libero_10", "libero_object")

# Official GitHub README leaderboard (Zhou et al.). Rates in [0,1].
# Env is author-only (not in HF packs); open-source rows use None → shown as "-".
# Total16 = mean of Obj/Pos/Sem/Task over 4 bases (16 cells) — open-source compare key.
OFFICIAL_LEADERBOARD: list[dict[str, Any]] = [
    {
        "model": "OpenVLA",
        "source": "paper",
        "goal": [0.96, 0.00, 0.98, 0.00, 0.98],
        "spatial": [0.97, 0.00, 0.97, 0.00, 0.89],
        "libero_10": [0.81, 0.00, 0.96, 0.00, 0.85],
        "object": [0.98, 0.00, 0.98, 0.00, 0.00],
        "total_readme": 0.52,
    },
    {
        "model": "Pi0",
        "source": "paper",
        "goal": [0.94, 0.00, 0.93, 0.00, 0.39],
        "spatial": [0.95, 0.00, 0.97, 0.00, 0.60],
        "libero_10": [0.79, 0.00, 0.82, 0.00, 0.27],
        "object": [0.94, 0.00, 0.90, 0.00, 0.29],
        "total_readme": 0.44,
    },
    {
        "model": "Pi0.5",
        "source": "paper",
        "goal": [0.97, 0.38, 0.97, 0.00, 0.46],
        "spatial": [0.97, 0.20, 0.97, 0.01, 0.46],
        "libero_10": [0.92, 0.08, 0.93, 0.01, 0.46],
        "object": [0.98, 0.17, 0.96, 0.01, 0.73],
        "total_readme": 0.53,
    },
    {
        "model": "MolmoAct",
        "source": "opensource",
        "goal": [0.68, 0.00, 0.85, 0.00, None],
        "spatial": [0.90, 0.00, 0.88, 0.00, None],
        "libero_10": [0.54, 0.00, 0.74, 0.06, None],
        "object": [0.92, 0.06, 0.96, 0.00, None],
        "total_readme": 0.41,
    },
    {
        "model": "NORA",
        "source": "opensource",
        "goal": [0.58, 0.00, 0.88, 0.00, None],
        "spatial": [0.92, 0.00, 0.91, 0.00, None],
        "libero_10": [0.46, 0.00, 0.74, 0.00, None],
        "object": [0.86, 0.00, 0.92, 0.00, None],
        "total_readme": 0.40,
    },
    {
        "model": "x-VLA",
        "source": "opensource",
        "goal": [0.68, 0.01, 0.98, 0.09, None],
        "spatial": [0.97, 0.00, 0.96, 0.00, None],
        "libero_10": [0.62, 0.00, 0.95, 0.10, None],
        "object": [0.89, 0.02, 0.98, 0.08, None],
        "total_readme": 0.46,
    },
]

_BASE_KEY = {
    "libero_goal": "goal",
    "libero_spatial": "spatial",
    "libero_10": "libero_10",
    "libero_object": "object",
}
_PERT_IDX = {"object": 0, "swap": 1, "lan": 2, "task": 3, "env": 4}


def total16(cells_by_base: dict[str, list[float | None]]) -> float | None:
    """Mean of Obj/Pos/Sem/Task (first 4) over 4 bases; skip None / missing."""
    vals: list[float] = []
    for key in ("goal", "spatial", "libero_10", "object"):
        row = cells_by_base.get(key) or []
        for value in row[:4]:
            if value is not None:
                vals.append(float(value))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt_rate(value: float | None, incomplete: bool = False) -> str:
    if value is None:
        return "—"
    mark = "*" if incomplete else ""
    return f"{value:.2f}{mark}"


def ours_cells_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Build open-source-style cells from live matrix (rates in [0,1])."""
    matrix = summary.get("matrix") or {}
    out: dict[str, list[float | None]] = {
        "goal": [None] * 5,
        "spatial": [None] * 5,
        "libero_10": [None] * 5,
        "object": [None] * 5,
    }
    incomplete: set[tuple[str, str]] = set()
    for base, perts in matrix.items():
        key = _BASE_KEY.get(base)
        if key is None:
            continue
        for pert, cell in perts.items():
            idx = _PERT_IDX.get(pert)
            if idx is None:
                continue
            if not cell or int(cell.get("episodes") or 0) == 0:
                continue
            out[key][idx] = float(cell["rate"])
            if cell.get("status") != "final":
                incomplete.add((key, pert))
    t16 = total16(out)
    return {
        "model": f"Ours ({summary.get('checkpoint_label') or 'live'})",
        "source": "live",
        "status": summary.get("status"),
        "goal": out["goal"],
        "spatial": out["spatial"],
        "libero_10": out["libero_10"],
        "object": out["object"],
        "total16": t16,
        "incomplete": sorted(f"{b}.{p}" for b, p in incomplete),
        "rollouts": summary.get("overall", {}).get("episodes", 0),
        "expected": summary.get("expected_episodes", 0),
        "updated_at_iso": summary.get("updated_at_iso"),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _metric(successes: list[bool]) -> dict[str, Any]:
    n_ok = sum(1 for value in successes if value)
    n = len(successes)
    return {
        "successes": n_ok,
        "episodes": n,
        "pc_success": (100.0 * n_ok / n) if n else 0.0,
        "rate": (n_ok / n) if n else 0.0,
    }


def split_suite(name: str) -> tuple[str, str]:
    for pert in PERTURBATION_ORDER:
        suffix = f"_{pert}"
        if name.endswith(suffix):
            return name[: -len(suffix)], pert
    return name, "unknown"


def expected_tasks(manifest: dict[str, Any] | None, suite: str) -> int:
    if not manifest:
        return 10
    task_ids = manifest.get("task_ids")
    if isinstance(task_ids, list):
        return max(len(task_ids), 1)
    # Full suite: 10 tasks unless overridden elsewhere.
    return 10


def expected_episodes(manifest: dict[str, Any] | None, suite: str) -> int:
    if not manifest:
        return 0
    ep = int(manifest.get("episodes_per_task") or 0)
    return ep * expected_tasks(manifest, suite)


def _successes_from_payload(payload: dict[str, Any]) -> tuple[list[bool], int]:
    """Return flattened successes and completed-task count."""
    per_task = payload.get("per_task") or []
    successes: list[bool] = []
    for task in per_task:
        metrics = task.get("metrics") or {}
        raw = metrics.get("successes") or []
        successes.extend(bool(value) for value in raw)
    completed = int(payload.get("completed_tasks") or len(per_task))
    if not successes and "overall" in payload:
        overall = payload["overall"] or {}
        # Final eval_info may only expose aggregates.
        n = int(overall.get("n_episodes") or 0)
        rate = overall.get("pc_success")
        if n and rate is not None:
            n_ok = int(round(float(rate) / 100.0 * n))
            successes = [True] * n_ok + [False] * max(n - n_ok, 0)
            completed = max(completed, len(per_task) or expected_tasks(None, ""))
    return successes, completed


def _tail_progress(eval_log: Path) -> str | None:
    if not eval_log.is_file():
        return None
    try:
        text = eval_log.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    # Prefer the last tqdm-looking line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-80:]):
        if "running_success_rate" in line or "Stepping through eval" in line or "Running rollout" in line:
            # Collapse ANSI / carriage returns.
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line).split("\r")[-1]
            return clean[:160]
    if lines:
        return lines[-1][:160]
    return None


def collect(eval_root: Path) -> dict[str, Any]:
    manifest = _load_json(eval_root / "run_manifest.json") or {}
    planned = list(manifest.get("suites") or [])
    if not planned:
        planned = sorted(
            path.name
            for path in eval_root.iterdir()
            if path.is_dir() and path.name.startswith("libero_")
        )

    ep_per_task = int(manifest.get("episodes_per_task") or 0)
    pert_map = dict(manifest.get("perturbation_map") or PERTURBATION_LABEL)

    per_suite: dict[str, Any] = {}
    by_base: dict[str, list[bool]] = defaultdict(list)
    by_pert: dict[str, list[bool]] = defaultdict(list)
    all_successes: list[bool] = []
    videos_found = 0

    for suite in planned:
        suite_dir = eval_root / suite
        process = _load_json(suite_dir / "process_status.json") or {}
        live = _load_json(suite_dir / "live_eval.json") or {}
        final = _load_json(suite_dir / "eval_info.json") or {}

        if final:
            payload = final
            status = "final"
        elif live:
            payload = live
            status = "final" if live.get("status") == "final" else process.get("status", "running")
        else:
            payload = {}
            status = process.get("status", "pending" if not suite_dir.exists() else "running")

        successes, completed_tasks = _successes_from_payload(payload)
        base, pert = split_suite(suite)
        n_tasks_expected = expected_tasks(manifest, suite)
        n_eps_expected = expected_episodes(manifest, suite)
        metric = _metric(successes)
        video_dir = suite_dir / "videos"
        n_videos = len(list(video_dir.glob("*.mp4"))) if video_dir.is_dir() else 0
        videos_found += n_videos

        per_suite[suite] = {
            "status": status,
            "base": base,
            "perturbation": pert,
            "perturbation_label": pert_map.get(pert, PERTURBATION_LABEL.get(pert, pert)),
            "gpu": process.get("gpu"),
            "completed_tasks": completed_tasks,
            "total_tasks": n_tasks_expected,
            "expected_episodes": n_eps_expected,
            "videos": n_videos,
            "progress_hint": _tail_progress(suite_dir / "eval.log") if status == "running" else None,
            **metric,
        }
        by_base[base].extend(successes)
        by_pert[pert].extend(successes)
        all_successes.extend(successes)

    # Leaderboard-style matrix: base × pert → rate
    matrix: dict[str, dict[str, Any]] = {}
    for suite, payload in per_suite.items():
        base = payload["base"]
        pert = payload["perturbation"]
        matrix.setdefault(base, {})[pert] = {
            "pc_success": payload["pc_success"],
            "rate": payload["rate"],
            "episodes": payload["episodes"],
            "status": payload["status"],
        }

    statuses = [payload["status"] for payload in per_suite.values()]
    if statuses and all(status == "final" for status in statuses):
        overall_status = "final"
    elif any(status == "failed" for status in statuses):
        overall_status = "failed"
    elif any(status == "running" for status in statuses):
        overall_status = "running"
    else:
        overall_status = "pending"

    completed_suites = sum(1 for status in statuses if status == "final")
    total_suites = len(planned)
    overall = _metric(all_successes)
    expected_total = sum(expected_episodes(manifest, suite) for suite in planned)

    return {
        "benchmark": "libero_pro",
        "status": overall_status,
        "eval_root": str(eval_root),
        "checkpoint_label": manifest.get("checkpoint_label"),
        "protocol": manifest.get("protocol"),
        "episodes_per_task": ep_per_task,
        "completed_suites": completed_suites,
        "total_suites": total_suites,
        "expected_episodes": expected_total,
        "videos": videos_found,
        "overall": overall,
        "per_suite": per_suite,
        "per_base": {
            base: _metric(by_base.get(base, []))
            for base in sorted(set(by_base) | set(BASE_ORDER), key=lambda x: (BASE_ORDER.index(x) if x in BASE_ORDER else 99, x))
        },
        "per_perturbation": {
            pert: {
                **_metric(by_pert.get(pert, [])),
                "label": pert_map.get(pert, PERTURBATION_LABEL.get(pert, pert)),
            }
            for pert in PERTURBATION_ORDER
            if pert in by_pert or any(split_suite(s)[1] == pert for s in planned)
        },
        "matrix": matrix,
        "updated_at": time.time(),
        "updated_at_iso": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines: list[str] = [
        "# LIBERO-PRO live summary",
        "",
        f"- **Updated**: `{summary['updated_at_iso']}`",
        f"- **Root**: `{summary['eval_root']}`",
        f"- **Status**: `{summary['status']}` · suites `{summary['completed_suites']}/{summary['total_suites']}`",
        f"- **Protocol**: `{summary.get('protocol')}` · eps/task `{summary.get('episodes_per_task')}`",
        f"- **Rollouts**: `{overall['successes']}/{overall['episodes']}`"
        + (f" (expected ~{summary['expected_episodes']})" if summary.get("expected_episodes") else "")
        + f" · **acc {overall['pc_success']:.2f}%** · rate `{overall['rate']:.3f}`",
        f"- **Videos saved**: `{summary.get('videos', 0)}`",
        "",
        "## Leaderboard matrix (acc % over completed rollouts)",
        "",
    ]

    perts_present = [
        pert
        for pert in PERTURBATION_ORDER
        if pert in summary["per_perturbation"]
    ]
    header = ["Base"] + [PERTURBATION_LABEL.get(p, p) for p in perts_present] + ["Mean"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + [":---:"] * (len(header) - 1)) + " |")

    bases = [b for b in BASE_ORDER if b in summary["matrix"]] + [
        b for b in summary["matrix"] if b not in BASE_ORDER
    ]
    for base in bases:
        row = [f"`{base}`"]
        rates: list[float] = []
        for pert in perts_present:
            cell = summary["matrix"].get(base, {}).get(pert)
            if not cell or cell["episodes"] == 0:
                row.append("—")
            else:
                mark = "" if cell["status"] == "final" else "*"
                row.append(f"{cell['pc_success']:.1f}{mark}")
                rates.append(cell["rate"])
        mean = (100.0 * sum(rates) / len(rates)) if rates else None
        row.append(f"{mean:.1f}" if mean is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Column means
    col = ["**Mean**"]
    col_rates: list[float] = []
    for pert in perts_present:
        payload = summary["per_perturbation"].get(pert)
        if not payload or payload["episodes"] == 0:
            col.append("—")
        else:
            col.append(f"**{payload['pc_success']:.1f}**")
            col_rates.append(payload["rate"])
    total_mean = (100.0 * sum(col_rates) / len(col_rates)) if col_rates else overall["pc_success"]
    col.append(f"**{total_mean:.1f}**")
    lines.append("| " + " | ".join(col) + " |")
    lines.append("")
    lines.append("\\* = suite still running / incomplete. Acc over finished rollouts only.")
    lines.append("")

    lines.extend(
        [
            "## Per perturbation",
            "",
            "| Perturbation | Label | Rollouts | Success | Acc |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for pert, payload in summary["per_perturbation"].items():
        lines.append(
            f"| `{pert}` | {payload.get('label', pert)} | {payload['episodes']} | "
            f"{payload['successes']} | {payload['pc_success']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Per base suite",
            "",
            "| Base | Rollouts | Success | Acc |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for base, payload in summary["per_base"].items():
        lines.append(
            f"| `{base}` | {payload['episodes']} | {payload['successes']} | {payload['pc_success']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Per suite detail",
            "",
            "| Suite | Status | GPU | Tasks | Rollouts | Acc | Videos | Hint |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for suite, payload in summary["per_suite"].items():
        hint = (payload.get("progress_hint") or "").replace("|", "/")
        if len(hint) > 48:
            hint = hint[:45] + "..."
        lines.append(
            f"| `{suite}` | {payload['status']} | {payload.get('gpu', '—')} | "
            f"{payload['completed_tasks']}/{payload['total_tasks']} | "
            f"{payload['episodes']}/{payload['expected_episodes'] or '—'} | "
            f"{payload['pc_success']:.2f}% | {payload['videos']} | {hint or '—'} |"
        )

    lines.extend(["", "---", f"_Auto-refreshed by `monitor_libero_pro.py`._", ""])
    return "\n".join(lines)


def render_leaderboard_markdown(summary: dict[str, Any]) -> str:
    """Open-source protocol board + live Ours row (Env = —)."""
    ours = ours_cells_from_summary(summary)
    lines: list[str] = [
        "# LIBERO-PRO Leaderboard (open-source protocol)",
        "",
        f"- **Updated**: `{summary['updated_at_iso']}`",
        f"- **Protocol**: 50 episodes/task · HF packs Obj/Pos/Sem/Task · Env often `-`",
        f"- **Total16**: mean of 16 cells (4 bases × Obj/Pos/Sem/Task) — primary open-source metric",
        f"- **Ours run**: `{summary.get('status')}` · "
        f"rollouts `{ours['rollouts']}/{ours['expected']}` · ckpt `{summary.get('checkpoint_label')}`",
        "",
        "Source: [LIBERO-PRO README](https://github.com/Zxy-MLlab/LIBERO-PRO) leaderboard. "
        "`*` = suite still running (partial rollouts).",
        "",
        "## Ranking by Total16",
        "",
        # Compact so Total16 stays visible in narrow panes.
        "| # | Model | Src | Total16 | PaperΣ | Env |",
        "| -: | --- | --- | -: | -: | --- |",
    ]

    ranked: list[tuple[str, str, float | None, float | None, str]] = []
    for row in OFFICIAL_LEADERBOARD:
        cells = {
            "goal": row["goal"],
            "spatial": row["spatial"],
            "libero_10": row["libero_10"],
            "object": row["object"],
        }
        t16 = total16(cells)
        has_env = any(v is not None for v in (row["goal"][4], row["spatial"][4], row["libero_10"][4], row["object"][4]))
        ranked.append(
            (
                row["model"],
                row["source"],
                t16,
                row.get("total_readme"),
                "yes" if has_env else "—",
            )
        )
    ranked.append(
        (
            ours["model"],
            "live*",
            ours["total16"],
            None,
            "—",
        )
    )
    ranked_sorted = sorted(
        ranked,
        key=lambda item: (-1.0 if item[2] is None else -item[2], item[0]),
    )
    def _short_pro_model(name: str) -> str:
        return (
            name.replace("(stage2_025000)", "v3-25k")
            .replace("Ours ", "Ours ")
            .strip()
        )

    for i, (model, source, t16, treadme, env) in enumerate(ranked_sorted, start=1):
        t16_s = f"{t16:.2f}" if t16 is not None else "—"
        tr_s = f"{treadme:.2f}" if treadme is not None else "—"
        bold = "**" if source.startswith("live") else ""
        src = "live*" if source.startswith("live") else source.replace("opensource", "os").replace("paper", "paper")
        lines.append(
            f"| {i} | {bold}{_short_pro_model(model)}{bold} | {src} | "
            f"{bold}{t16_s}{bold} | {tr_s} | {env} |"
        )

    # Suite-grouped tables (official README order). Object is its own block so it
    # is not clipped off the right edge of a 21-column mega-table.
    suite_blocks = (
        ("LIBERO-Goal", "goal"),
        ("LIBERO-Spatial", "spatial"),
        ("LIBERO-10", "libero_10"),
        ("LIBERO-Object", "object"),
    )
    all_rows: list[tuple[dict[str, Any], bool]] = [
        (row, False) for row in OFFICIAL_LEADERBOARD
    ] + [(ours, True)]

    lines.extend(["", "## Per-suite matrix (rates in [0, 1], official order)", ""])
    for title, base_key in suite_blocks:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Model | Obj | Pos | Sem | Task | Env | μ4 |")
        lines.append("| --- | -: | -: | -: | -: | -: | -: |")
        for payload, live in all_rows:
            incomplete = set(payload.get("incomplete") or [])
            vals = payload[base_key]
            cells: list[str] = []
            mean_vals: list[float] = []
            for pert, idx in (("object", 0), ("swap", 1), ("lan", 2), ("task", 3), ("env", 4)):
                incomplete_flag = live and f"{base_key}.{pert}" in incomplete
                value = vals[idx] if idx < len(vals) else None
                cells.append(_fmt_rate(value, incomplete_flag))
                if idx < 4 and value is not None:
                    mean_vals.append(float(value))
            mean4 = (sum(mean_vals) / len(mean_vals)) if mean_vals else None
            mean_s = _fmt_rate(mean4, live and bool(mean_vals))
            raw_name = _short_pro_model(payload["model"])
            name = f"**{raw_name}**" if live else raw_name
            if live and mean4 is not None:
                mean_s = f"**{mean_s}**"
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {mean_s} |")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Suites (official): **Goal → Spatial → 10 → Object**. Each has Obj/Pos/Sem/Task/(Env).",
            "- **Open-source submissions** (MolmoAct / NORA / x-VLA): Env = `-`, Total = Total16.",
            "- **Paper baselines** (OpenVLA / Pi0 / Pi0.5): README Total averages 20 cells including Env; "
            "Total16 recomputed here for apples-to-apples comparison.",
            "- **Ours**: Goal-Pose Prior v3-25k · `env.type=libero_pro` · seed 1000 · batch_size=1 · "
            "HF Obj/Pos/Sem/Task only. Wave-2 Goal/10 pending until wave-1 finishes.",
            "",
            "---",
            "_Auto-refreshed by `monitor_libero_pro.py` → `leaderboard.md`._",
            "",
        ]
    )
    return "\n".join(lines)


def render_text(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        f"LIBERO-PRO  {summary['eval_root']}",
        f"Status: {summary['status']}  suites {summary['completed_suites']}/{summary['total_suites']}  "
        f"rollouts {overall['successes']}/{overall['episodes']}  "
        f"acc {overall['pc_success']:.2f}%  videos {summary.get('videos', 0)}",
        "",
        "Per perturbation:",
    ]
    for pert, payload in summary["per_perturbation"].items():
        lines.append(
            f"  {payload.get('label', pert):<4} ({pert:<6})  "
            f"rollouts {payload['episodes']:>5}  success {payload['successes']:>4}  "
            f"acc {payload['pc_success']:6.2f}%"
        )
    lines.append("")
    lines.append("Per suite:")
    for suite, payload in summary["per_suite"].items():
        lines.append(
            f"  {suite:<28} {payload['status']:<8} "
            f"tasks {payload['completed_tasks']:>2}/{payload['total_tasks']:<2}  "
            f"ep {payload['episodes']:>4}/{payload['expected_episodes'] or '-':<4}  "
            f"acc {payload['pc_success']:6.2f}%"
        )
    return "\n".join(lines)


def write_outputs(eval_root: Path, summary: dict[str, Any]) -> None:
    eval_root.mkdir(parents=True, exist_ok=True)

    md_path = eval_root / "live_summary.md"
    md_tmp = md_path.with_suffix(".md.tmp")
    md_tmp.write_text(render_markdown(summary), encoding="utf-8")
    md_tmp.replace(md_path)

    lb_path = eval_root / "leaderboard.md"
    lb_tmp = lb_path.with_suffix(".md.tmp")
    lb_tmp.write_text(render_leaderboard_markdown(summary), encoding="utf-8")
    lb_tmp.replace(lb_path)

    json_path = eval_root / "live_summary.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    # Attach open-source board snapshot for consumers.
    payload = dict(summary)
    payload["leaderboard_ours"] = ours_cells_from_summary(summary)
    payload["leaderboard_official"] = [
        {
            **row,
            "total16": total16(
                {
                    "goal": row["goal"],
                    "spatial": row["spatial"],
                    "libero_10": row["libero_10"],
                    "object": row["object"],
                }
            ),
        }
        for row in OFFICIAL_LEADERBOARD
    ]
    json_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    json_tmp.replace(json_path)

    csv_path = eval_root / "live_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("dimension", "name", "label", "successes", "episodes", "pc_success", "status"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "dimension": "overall",
                "name": "overall",
                "label": "",
                "status": summary["status"],
                **{k: summary["overall"][k] for k in ("successes", "episodes", "pc_success")},
            }
        )
        for name, payload in summary["per_perturbation"].items():
            writer.writerow(
                {
                    "dimension": "perturbation",
                    "name": name,
                    "label": payload.get("label", ""),
                    "status": "",
                    **{k: payload[k] for k in ("successes", "episodes", "pc_success")},
                }
            )
        for name, payload in summary["per_base"].items():
            writer.writerow(
                {
                    "dimension": "base",
                    "name": name,
                    "label": "",
                    "status": "",
                    **{k: payload[k] for k in ("successes", "episodes", "pc_success")},
                }
            )
        for name, payload in summary["per_suite"].items():
            writer.writerow(
                {
                    "dimension": "suite",
                    "name": name,
                    "label": payload.get("perturbation_label", ""),
                    "status": payload.get("status", ""),
                    **{k: payload[k] for k in ("successes", "episodes", "pc_success")},
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="PRO eval root containing run_manifest.json and suite dirs",
    )
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    eval_root = args.eval_root
    if eval_root is None:
        default = Path(
            "/data2/JM/Code/molmoact2/lerobot/outputs/libero_eval_pro/"
            "stage2_025000/full50_seed_1000"
        )
        video = default.parent / "video_samples_seed_1000"
        if (default / "run_manifest.json").is_file():
            eval_root = default
        elif (video / "run_manifest.json").is_file():
            eval_root = video
        else:
            raise SystemExit("Pass --eval-root; no default PRO run found")

    while True:
        summary = collect(eval_root)
        write_outputs(eval_root, summary)
        if not args.once and os.isatty(1):
            print("\033[2J\033[H", end="")
        print(render_text(summary), flush=True)
        print(f"\n[wrote] {eval_root / 'live_summary.md'}", flush=True)
        print(f"[wrote] {eval_root / 'leaderboard.md'}", flush=True)
        if args.once or summary["status"] in {"final", "failed"}:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

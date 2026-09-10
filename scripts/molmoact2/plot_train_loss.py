#!/usr/bin/env python3
"""Parse LeRobot train logs and plot loss / lr curves."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(
    r"step:(?P<step>\d+)\s+"
    r"smpl:(?P<samples>[^\s]+)\s+"
    r"ep:(?P<episodes>[^\s]+)\s+"
    r"epch:(?P<epochs>[0-9.]+)\s+"
    r"(?:.*?)?loss:(?P<loss>[0-9.eE+-]+)"
    r"(?:.*?lr:(?P<lr>[0-9.eE+-]+))?"
)


def parse_log(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        row = {
            "step": int(m.group("step")),
            "epochs": float(m.group("epochs")),
            "loss": float(m.group("loss")),
        }
        if m.group("lr") is not None:
            row["lr"] = float(m.group("lr"))
        # Keep last value per step (in case of duplicates across ranks).
        if rows and rows[-1]["step"] == row["step"]:
            rows[-1] = row
        else:
            rows.append(row)
    return rows


def save_metrics_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def plot_metrics(rows: list[dict], out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    lrs = [r["lr"] for r in rows if "lr" in r]

    n_rows = 2 if lrs else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 6 if lrs else 4), sharex=True)
    if n_rows == 1:
        axes = [axes]
    else:
        axes = list(axes)
    ax0 = axes[0]
    ax0.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="#1f77b4")
    ax0.set_ylabel("loss")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    if lrs:
        ax1 = axes[1]
        ax1.plot(steps[: len(lrs)], lrs, marker="o", markersize=3, linewidth=1.5, color="#ff7f0e")
        ax1.set_ylabel("lr")
        ax1.set_xlabel("step")
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax0.set_xlabel("step")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Train stdout/stderr log file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for metrics.jsonl + loss.png")
    parser.add_argument("--title", type=str, default="MolmoAct2 LIBERO train")
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    rows = parse_log(text)
    if not rows:
        raise SystemExit(f"No step/loss lines found in {args.log}")

    out_dir = args.out_dir or args.log.parent
    jsonl_path = out_dir / "metrics.jsonl"
    png_path = out_dir / "loss.png"
    save_metrics_jsonl(rows, jsonl_path)
    plot_metrics(rows, png_path, title=args.title)
    print(f"[plot] parsed {len(rows)} points from {args.log}")
    print(f"[plot] wrote {jsonl_path}")
    print(f"[plot] wrote {png_path}")
    print(f"[plot] last: step={rows[-1]['step']} loss={rows[-1]['loss']}")


if __name__ == "__main__":
    main()

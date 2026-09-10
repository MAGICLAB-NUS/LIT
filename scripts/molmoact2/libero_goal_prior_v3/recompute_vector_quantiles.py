#!/usr/bin/env python3
"""Recompute q01/q99 (and mean/std/min/max) for vector features from parquet.

LIBERO local stats.json had badly wrong observation.state quantiles (Z q01/q99
≈ [0.64, 0.88] while real EE height spans ≈ [0.04, 1.27]), so QUANTILES
normalization clipped ~90%+ of Z targets to ±1. This script fixes that without
touching training code: backup stats.json, overwrite vector-feature stats from
data/*.parquet, keep image/video stats unchanged.

Usage:
  source scripts/activate_train_env.sh
  python scripts/libero_goal_prior_v3/recompute_vector_quantiles.py
  python scripts/libero_goal_prior_v3/recompute_vector_quantiles.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_ROOT = Path("/data2/JM/dataset/libero_lerobot_format")
DEFAULT_KEYS = ("observation.state", "action")
DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def _load_vectors(root: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {root / 'data'}")
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for path in files:
        table = pq.read_table(path, columns=list(keys))
        for key in keys:
            col = table.column(key)
            arr = np.stack(
                [np.asarray(col[i].as_py(), dtype=np.float64).reshape(-1) for i in range(len(col))],
                axis=0,
            )
            buckets[key].append(arr)
    return {k: np.concatenate(v, axis=0) for k, v in buckets.items()}


def _feature_stats(data: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Match LeRobot stats.json list layout for 1-D vector features."""
    assert data.ndim == 2, data.shape
    quantiles = np.quantile(data, DEFAULT_QUANTILES, axis=0)
    out: dict[str, list[float] | list[int]] = {
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "count": [int(data.shape[0])],
    }
    for q, row in zip(DEFAULT_QUANTILES, quantiles, strict=True):
        out[f"q{int(q * 100):02d}"] = row.tolist()
    return out


def _outside_frac(data: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> dict[str, float]:
    per_dim = ((data < q01) | (data > q99)).mean(axis=0)
    any_dim = ((data < q01) | (data > q99)).any(axis=1).mean()
    report = {f"dim{i}_outside": float(per_dim[i]) for i in range(min(3, data.shape[1]))}
    report["any_dim_outside"] = float(any_dim)
    if data.shape[1] > 2:
        report["z_outside"] = float(per_dim[2])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(DEFAULT_KEYS),
        help="Vector feature keys to recompute (default: observation.state action)",
    )
    parser.add_argument(
        "--backup-name",
        type=str,
        default="",
        help="Backup filename under meta/. Empty -> stats.pre_v3_<timestamp>.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print old vs new quantiles without writing stats.json",
    )
    args = parser.parse_args()

    root: Path = args.dataset_root
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)

    keys = tuple(args.keys)
    print(f"[v3-stats] loading vectors from {root} keys={keys}", flush=True)
    data = _load_vectors(root, keys)
    old = json.loads(stats_path.read_text(encoding="utf-8"))
    new = dict(old)

    for key in keys:
        arr = data[key]
        recomputed = _feature_stats(arr)
        old_q01 = np.asarray(old[key]["q01"], dtype=np.float64)
        old_q99 = np.asarray(old[key]["q99"], dtype=np.float64)
        new_q01 = np.asarray(recomputed["q01"], dtype=np.float64)
        new_q99 = np.asarray(recomputed["q99"], dtype=np.float64)
        print(f"\n[v3-stats] {key} shape={arr.shape}", flush=True)
        print(f"  OLD q01[:3]={old_q01[:3]}  q99[:3]={old_q99[:3]}", flush=True)
        print(f"  NEW q01[:3]={new_q01[:3]}  q99[:3]={new_q99[:3]}", flush=True)
        print(f"  OLD coverage {_outside_frac(arr, old_q01, old_q99)}", flush=True)
        print(f"  NEW coverage {_outside_frac(arr, new_q01, new_q99)}", flush=True)
        new[key] = recomputed

    if args.dry_run:
        print("\n[v3-stats] dry-run: not writing stats.json", flush=True)
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_name = args.backup_name or f"stats.pre_v3_{ts}.json"
    backup_path = root / "meta" / backup_name
    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)
        print(f"\n[v3-stats] backed up {stats_path} -> {backup_path}", flush=True)
    else:
        print(f"\n[v3-stats] backup already exists: {backup_path}", flush=True)

    # Also keep a stable alias for docs / restore.
    alias = root / "meta" / "stats.pre_v3_bad_quantiles.json"
    if not alias.exists():
        shutil.copy2(backup_path, alias)
        print(f"[v3-stats] alias {alias}", flush=True)

    stats_path.write_text(json.dumps(new, indent=4) + "\n", encoding="utf-8")
    note = {
        "updated_at_utc": ts,
        "keys": list(keys),
        "backup": str(backup_path),
        "alias_backup": str(alias),
        "reason": "Fix bad QUANTILES for goal-pose prior v3 (state Z was clipped).",
    }
    (root / "meta" / "stats.v3_recompute_note.json").write_text(
        json.dumps(note, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[v3-stats] wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit DROID goal-prior normalization over the exact training samples.

This audit intentionally consumes the saved q01/q99 values instead of
recomputing quantiles.  The expensive exact-statistics recomputation belongs to
``prepare_dataset.py --verify-only``; this script measures the clipping behavior
of the normalization that training actually applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from .prepare_dataset import (
        ACTION_KEY,
        EE_POSE_KEY,
        HORIZON,
        RECIPE_VERSION,
        STATE_KEY,
        FilePlan,
        NonIdleConfig,
        _ManifestCursor,
        _anchor_columns,
        _episode_arrays,
        _episode_tables,
        _scan_columns,
        _sha256,
        _valid_task_indices,
        anchor_mask,
    )
except ImportError:  # Direct execution: python scripts/.../audit_normalization.py
    from prepare_dataset import (  # type: ignore[no-redef]
        ACTION_KEY,
        EE_POSE_KEY,
        HORIZON,
        RECIPE_VERSION,
        STATE_KEY,
        FilePlan,
        NonIdleConfig,
        _ManifestCursor,
        _anchor_columns,
        _episode_arrays,
        _episode_tables,
        _scan_columns,
        _sha256,
        _valid_task_indices,
        anchor_mask,
    )


DEFAULT_DATASET_ROOT = Path("/data0/JM/dataset/droid_1.0.1_goal_pose")
FEATURE_DIMS = {
    STATE_KEY: 8,
    ACTION_KEY: 8,
    EE_POSE_KEY: 7,
}
FEATURE_ORDER = (STATE_KEY, ACTION_KEY, EE_POSE_KEY)


class AuditContractError(ValueError):
    """Raised when metadata cannot safely define the normalization audit."""


def _flatten_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        names: list[str] = []
        for value in raw_names.values():
            if isinstance(value, (list, tuple)):
                names.extend(str(item) for item in value)
            elif value is not None:
                names.append(str(value))
        return names
    if isinstance(raw_names, (list, tuple)):
        return [str(item) for item in raw_names]
    return [] if raw_names is None else [str(raw_names)]


def normalization_mask_from_feature_meta(
    feature_meta: dict[str, Any], expected_dim: int
) -> tuple[list[str], np.ndarray]:
    """Derive the MolmoAct2 normalization mask from metadata feature names."""

    names = _flatten_names(feature_meta.get("names"))
    if len(names) != expected_dim:
        raise AuditContractError(
            f"metadata names have length {len(names)}, expected {expected_dim}"
        )
    mask = np.asarray(["gripper" not in name.lower() for name in names], dtype=bool)
    gripper_indices = np.flatnonzero(~mask)
    if len(gripper_indices) != 1:
        raise AuditContractError(
            f"metadata must identify exactly one gripper dimension, got {gripper_indices.tolist()}"
        )
    return names, mask


def _stat_vector(
    feature: str, stats: dict[str, Any], statistic: str, expected_dim: int
) -> np.ndarray:
    if statistic not in stats:
        raise AuditContractError(f"{feature} stats are missing {statistic}")
    values = np.asarray(stats[statistic], dtype=np.float64)
    if values.shape != (expected_dim,):
        raise AuditContractError(
            f"{feature} {statistic} shape is {values.shape}, expected {(expected_dim,)}"
        )
    if not np.isfinite(values).all():
        raise AuditContractError(f"{feature} {statistic} contains NaN or Inf")
    return values


def _stat_count(feature: str, stats: dict[str, Any], expected_count: int) -> int:
    raw_count = stats.get("count")
    if (
        not isinstance(raw_count, list)
        or len(raw_count) != 1
        or isinstance(raw_count[0], bool)
        or not isinstance(raw_count[0], int)
    ):
        raise AuditContractError(f"{feature} stats count must be one integer")
    count = int(raw_count[0])
    if count != expected_count:
        raise AuditContractError(
            f"{feature} stats count is {count}, expected {expected_count}"
        )
    return count


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    dim: int
    count: int
    names: list[str]
    normalize_mask: np.ndarray
    q01: np.ndarray
    q99: np.ndarray
    saved_min: np.ndarray
    saved_max: np.ndarray


def validate_feature_stats(
    feature: str,
    stats: dict[str, Any],
    feature_meta: dict[str, Any],
    *,
    expected_dim: int,
    expected_count: int,
) -> FeatureSpec:
    """Validate shape, count, names, and the saved q01/q99 interval."""

    shape = feature_meta.get("shape")
    if shape != [expected_dim]:
        raise AuditContractError(
            f"{feature} metadata shape is {shape!r}, expected {[expected_dim]}"
        )
    names, mask = normalization_mask_from_feature_meta(feature_meta, expected_dim)
    count = _stat_count(feature, stats, expected_count)
    q01 = _stat_vector(feature, stats, "q01", expected_dim)
    q99 = _stat_vector(feature, stats, "q99", expected_dim)
    saved_min = _stat_vector(feature, stats, "min", expected_dim)
    saved_max = _stat_vector(feature, stats, "max", expected_dim)
    invalid_width = np.flatnonzero(q99 <= q01)
    if len(invalid_width):
        raise AuditContractError(
            f"{feature} has q99 <= q01 at dimensions {invalid_width.tolist()}"
        )
    invalid_interval = np.flatnonzero((q01 < saved_min) | (q99 > saved_max))
    if len(invalid_interval):
        raise AuditContractError(
            f"{feature} q01/q99 leave min/max at dimensions {invalid_interval.tolist()}"
        )
    return FeatureSpec(
        key=feature,
        dim=expected_dim,
        count=count,
        names=names,
        normalize_mask=mask,
        q01=q01,
        q99=q99,
        saved_min=saved_min,
        saved_max=saved_max,
    )


class FeatureAccumulator:
    """Bounded-memory per-dimension audit against one feature's saved stats."""

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self.count = np.zeros(spec.dim, dtype=np.int64)
        self.finite = np.zeros(spec.dim, dtype=np.int64)
        self.below = np.zeros(spec.dim, dtype=np.int64)
        self.above = np.zeros(spec.dim, dtype=np.int64)
        self.clamp_loss = np.zeros(spec.dim, dtype=np.int64)
        self.minimum = np.full(spec.dim, np.inf, dtype=np.float64)
        self.maximum = np.full(spec.dim, -np.inf, dtype=np.float64)
        self.roundtrip_max = np.zeros(spec.dim, dtype=np.float64)

    def update(self, raw_values: np.ndarray) -> None:
        values = np.asarray(raw_values)
        if values.ndim != 2 or values.shape[1] != self.spec.dim:
            raise ValueError(
                f"{self.spec.key} scan shape is {values.shape}, "
                f"expected (N, {self.spec.dim})"
            )
        if not len(values):
            return
        values = values.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        self.count += len(values)
        self.finite += finite.sum(axis=0, dtype=np.int64)
        self.below += ((values < self.spec.q01) & finite).sum(axis=0, dtype=np.int64)
        self.above += ((values > self.spec.q99) & finite).sum(axis=0, dtype=np.int64)
        for dimension in range(self.spec.dim):
            finite_values = values[finite[:, dimension], dimension]
            if len(finite_values):
                self.minimum[dimension] = min(
                    self.minimum[dimension], float(finite_values.min())
                )
                self.maximum[dimension] = max(
                    self.maximum[dimension], float(finite_values.max())
                )

        dimensions = np.flatnonzero(self.spec.normalize_mask)
        if not len(dimensions):
            return
        selected = values[:, dimensions]
        selected_finite = finite[:, dimensions]
        q01 = self.spec.q01[dimensions]
        q99 = self.spec.q99[dimensions]
        normalized = 2.0 * (selected - q01) / (q99 - q01) - 1.0
        clamped = np.clip(normalized, -1.0, 1.0)
        self.clamp_loss[dimensions] += (
            (normalized != clamped) & selected_finite
        ).sum(axis=0, dtype=np.int64)
        restored = (clamped + 1.0) * (q99 - q01) / 2.0 + q01
        expected = np.clip(selected, q01, q99)
        error = np.where(selected_finite, np.abs(restored - expected), 0.0)
        self.roundtrip_max[dimensions] = np.maximum(
            self.roundtrip_max[dimensions], error.max(axis=0)
        )

    def report(
        self,
        *,
        outside_fraction_limit: float,
        xyz_hard_limit: float,
        roundtrip_atol: float,
    ) -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        dimensions: list[dict[str, Any]] = []
        for index, name in enumerate(self.spec.names):
            count = int(self.count[index])
            finite = int(self.finite[index])
            below = int(self.below[index])
            above = int(self.above[index])
            outside = below + above
            clamp_loss = int(self.clamp_loss[index])
            denominator = count if count else 1
            outside_fraction = outside / denominator
            clamp_loss_fraction = clamp_loss / denominator
            normalize = bool(self.spec.normalize_mask[index])
            minimum = (
                float(self.minimum[index])
                if np.isfinite(self.minimum[index])
                else None
            )
            maximum = (
                float(self.maximum[index])
                if np.isfinite(self.maximum[index])
                else None
            )
            roundtrip_error = (
                float(self.roundtrip_max[index]) if normalize else None
            )
            dimension = {
                "index": index,
                "name": name,
                "normalize": normalize,
                "count": count,
                "finite": finite,
                "finite_fraction": finite / denominator,
                "below_q01": below,
                "above_q99": above,
                "outside": outside,
                "outside_fraction": outside_fraction,
                "clamp_loss": clamp_loss,
                "clamp_loss_fraction": clamp_loss_fraction,
                "min": minimum,
                "max": maximum,
                "q01": float(self.spec.q01[index]),
                "q99": float(self.spec.q99[index]),
                "roundtrip_max_abs_error": roundtrip_error,
            }
            dimensions.append(dimension)

            prefix = f"{self.spec.key}[{index}] ({name})"
            if count != self.spec.count:
                errors.append(f"{prefix} count is {count}, expected {self.spec.count}")
            if finite != count:
                errors.append(f"{prefix} contains {count - finite} NaN/Inf values")
            if minimum != float(self.spec.saved_min[index]):
                errors.append(
                    f"{prefix} min is {minimum}, stats record "
                    f"{float(self.spec.saved_min[index])}"
                )
            if maximum != float(self.spec.saved_max[index]):
                errors.append(
                    f"{prefix} max is {maximum}, stats record "
                    f"{float(self.spec.saved_max[index])}"
                )
            if normalize:
                if outside_fraction > outside_fraction_limit:
                    errors.append(
                        f"{prefix} outside fraction {outside_fraction:.8f} exceeds "
                        f"{outside_fraction_limit:.8f}"
                    )
                if name.lower() in {"x", "y", "z"} and outside_fraction > xyz_hard_limit:
                    errors.append(
                        f"{prefix} xyz outside fraction {outside_fraction:.8f} exceeds "
                        f"hard limit {xyz_hard_limit:.8f}"
                    )
                if self.roundtrip_max[index] > roundtrip_atol:
                    errors.append(
                        f"{prefix} round-trip error {self.roundtrip_max[index]:.3e} "
                        f"exceeds {roundtrip_atol:.3e}"
                    )
            elif minimum is not None and maximum is not None:
                if minimum < -1.0 or maximum > 1.0:
                    errors.append(
                        f"{prefix} unnormalized gripper range "
                        f"[{minimum}, {maximum}] leaves [-1, 1]"
                    )

        gripper_dimensions = [
            {
                "index": item["index"],
                "name": item["name"],
                "min": item["min"],
                "max": item["max"],
            }
            for item in dimensions
            if not item["normalize"]
        ]
        return (
            {
                "shape": [self.spec.dim],
                "names": self.spec.names,
                "normalization_mask": self.spec.normalize_mask.tolist(),
                "stats_count": self.spec.count,
                "scanned_samples": int(self.count[0]),
                "gripper_dimensions": gripper_dimensions,
                "dimensions": dimensions,
            },
            errors,
        )


def audit_feature_values(
    values: np.ndarray,
    stats: dict[str, Any],
    feature_meta: dict[str, Any],
    *,
    feature: str = "feature",
    outside_fraction_limit: float = 0.05,
    xyz_hard_limit: float = 0.15,
    roundtrip_atol: float = 1e-6,
) -> dict[str, Any]:
    """Small-array entry point used by unit tests and focused diagnostics."""

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"expected a rank-2 value array, got {values.shape}")
    spec = validate_feature_stats(
        feature,
        stats,
        feature_meta,
        expected_dim=values.shape[1],
        expected_count=len(values),
    )
    accumulator = FeatureAccumulator(spec)
    accumulator.update(values)
    report, errors = accumulator.report(
        outside_fraction_limit=outside_fraction_limit,
        xyz_hard_limit=xyz_hard_limit,
        roundtrip_atol=roundtrip_atol,
    )
    return {"passed": not errors, "errors": errors, **report}


def _expected_sample_counts(anchor_count: int, horizon: int) -> dict[str, int]:
    return {
        STATE_KEY: anchor_count,
        ACTION_KEY: anchor_count * horizon,
        EE_POSE_KEY: anchor_count,
    }


def _validate_full_recipe(
    provenance: dict[str, Any], info: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if provenance.get("recipe_version") != RECIPE_VERSION:
        errors.append(
            f"recipe_version is {provenance.get('recipe_version')!r}, "
            f"expected {RECIPE_VERSION}"
        )
    config = provenance.get("config")
    if not isinstance(config, dict):
        return [*errors, "provenance config is missing"]
    if config.get("horizon") != HORIZON:
        errors.append(f"provenance horizon is not {HORIZON}")
    if config.get("max_episodes") is not None or config.get("max_files") is not None:
        errors.append("audit requires the full recipe, not a limited build")
    goal_prior = info.get("goal_pose_prior")
    if not isinstance(goal_prior, dict):
        errors.append("meta/info.json is missing goal_pose_prior")
    else:
        if goal_prior.get("horizon") != HORIZON:
            errors.append(f"metadata goal horizon is not {HORIZON}")
        if goal_prior.get("partial_build") is not False:
            errors.append("audit requires goal_pose_prior.partial_build=false")
        if goal_prior.get("anchor_manifest") != "valid_anchor_indices.parquet":
            errors.append("metadata points to an unexpected anchor manifest")
        if goal_prior.get("target_feature") != EE_POSE_KEY:
            errors.append("metadata points to an unexpected goal-pose feature")
    definition = provenance.get("anchor_definition")
    if isinstance(definition, dict):
        if definition.get("goal_offset") != HORIZON:
            errors.append(f"provenance goal offset is not {HORIZON}")
        if definition.get("action_window") != "[t:t+15]":
            errors.append("provenance action window is not [t:t+15]")
    else:
        errors.append("provenance anchor_definition is missing")
    return errors


def _artifact_report(
    root: Path, provenance: dict[str, Any], relative_path: str
) -> tuple[dict[str, Any], str | None]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "exists": False}, f"missing {relative_path}"
    actual = _sha256(path)
    expected = (provenance.get("artifact_sha256") or {}).get(relative_path)
    matches = isinstance(expected, str) and actual == expected
    report = {
        "path": relative_path,
        "exists": True,
        "sha256": actual,
        "provenance_sha256": expected,
        "sha256_matches_provenance": matches,
    }
    error = None if matches else f"{relative_path} hash does not match provenance"
    return report, error


def _base_report(
    root: Path,
    *,
    outside_fraction_limit: float,
    xyz_hard_limit: float,
    roundtrip_atol: float,
) -> dict[str, Any]:
    return {
        "audit_version": 1,
        "dataset_root": str(root),
        "status": "failed",
        "passed": False,
        "sample_definition": {
            STATE_KEY: "state@t",
            ACTION_KEY: "action[t:t+14] inclusive (15 rows)",
            EE_POSE_KEY: "ee_pose@t+15",
        },
        "thresholds": {
            "non_gripper_outside_fraction_max": outside_fraction_limit,
            "xyz_outside_fraction_hard_max": xyz_hard_limit,
            "roundtrip_max_abs_error": roundtrip_atol,
            "gripper_raw_range": [-1.0, 1.0],
        },
        "artifacts": {},
        "features": {},
        "scan": {"episodes_scanned": 0, "anchors_scanned": 0},
        "errors": [],
    }


def audit_normalization(
    root: Path,
    *,
    outside_fraction_limit: float = 0.05,
    xyz_hard_limit: float = 0.15,
    roundtrip_atol: float = 1e-6,
    chunk_anchors: int = 8192,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Run a fail-closed audit over every full-recipe v3 training sample."""

    root = Path(root)
    report = _base_report(
        root,
        outside_fraction_limit=outside_fraction_limit,
        xyz_hard_limit=xyz_hard_limit,
        roundtrip_atol=roundtrip_atol,
    )
    errors: list[str] = report["errors"]
    if chunk_anchors <= 0:
        errors.append("chunk_anchors must be positive")
        return report
    if outside_fraction_limit < 0 or xyz_hard_limit < 0 or roundtrip_atol < 0:
        errors.append("audit thresholds must be non-negative")
        return report

    provenance_path = root / "goal_pose_provenance.json"
    info_path = root / "meta/info.json"
    stats_path = root / "meta/stats.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        metadata_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot load audit metadata: {exc}")
        return report

    errors.extend(_validate_full_recipe(provenance, info))
    for relative in (
        "valid_anchor_indices.parquet",
        "meta/info.json",
        "meta/stats.json",
    ):
        artifact, artifact_error = _artifact_report(root, provenance, relative)
        report["artifacts"][relative] = artifact
        if artifact_error:
            errors.append(artifact_error)

    anchor_count = provenance.get("anchor_count")
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or anchor_count <= 0:
        errors.append("provenance anchor_count must be a positive integer")
        return report
    report["recipe_version"] = provenance.get("recipe_version")
    report["horizon"] = HORIZON
    report["provenance_anchor_count"] = anchor_count

    manifest_path = root / "valid_anchor_indices.parquet"
    try:
        manifest_rows = pq.ParquetFile(manifest_path).metadata.num_rows
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read anchor manifest: {exc}")
        return report
    report["manifest_rows"] = manifest_rows
    if manifest_rows != anchor_count:
        errors.append(
            f"anchor manifest has {manifest_rows} rows, expected {anchor_count}"
        )

    expected_counts = _expected_sample_counts(anchor_count, HORIZON)
    feature_meta = info.get("features")
    if not isinstance(feature_meta, dict):
        errors.append("meta/info.json features are missing")
        return report
    specs: dict[str, FeatureSpec] = {}
    for feature in FEATURE_ORDER:
        if feature not in metadata_stats:
            errors.append(f"meta/stats.json is missing {feature}")
            continue
        if feature not in feature_meta:
            errors.append(f"meta/info.json is missing {feature}")
            continue
        try:
            specs[feature] = validate_feature_stats(
                feature,
                metadata_stats[feature],
                feature_meta[feature],
                expected_dim=FEATURE_DIMS[feature],
                expected_count=expected_counts[feature],
            )
        except (AuditContractError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if errors or len(specs) != len(FEATURE_ORDER):
        return report

    config = provenance["config"]
    try:
        non_idle = NonIdleConfig(**config["non_idle"])
        plans = [
            FilePlan(item["path"], int(item["output_rows"]))
            for item in provenance["files"]
        ]
        if not plans:
            raise AuditContractError("provenance contains no data files")
        valid_task_indices = _valid_task_indices(root)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(f"invalid scan provenance: {exc}")
        return report

    accumulators = {
        feature: FeatureAccumulator(spec) for feature, spec in specs.items()
    }
    manifest = _ManifestCursor(manifest_path)
    anchors_scanned = 0
    episodes_scanned = 0
    try:
        for episodes_scanned, episode_table in enumerate(
            _episode_tables(root, plans, _scan_columns()), start=1
        ):
            arrays = _episode_arrays(episode_table)
            mask, _ = anchor_mask(
                arrays,
                non_idle,
                horizon=HORIZON,
                valid_task_indices=valid_task_indices,
            )
            expected_manifest = _anchor_columns(arrays, mask)
            manifest.consume(expected_manifest)
            anchor_rows = np.flatnonzero(mask)
            anchors_scanned += len(anchor_rows)
            for start in range(0, len(anchor_rows), chunk_anchors):
                rows = anchor_rows[start : start + chunk_anchors]
                accumulators[STATE_KEY].update(arrays[STATE_KEY][rows])
                accumulators[EE_POSE_KEY].update(
                    arrays[EE_POSE_KEY][rows + HORIZON]
                )
                action_rows = rows[:, None] + np.arange(HORIZON, dtype=np.int64)
                accumulators[ACTION_KEY].update(
                    arrays[ACTION_KEY][action_rows].reshape(-1, FEATURE_DIMS[ACTION_KEY])
                )
            if progress_every and episodes_scanned % progress_every == 0:
                print(
                    f"[audit] episodes={episodes_scanned} anchors={anchors_scanned}",
                    file=sys.stderr,
                    flush=True,
                )
        manifest.finish()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"training-sample scan failed: {exc}")

    report["scan"] = {
        "episodes_scanned": episodes_scanned,
        "anchors_scanned": anchors_scanned,
        "chunk_anchors": chunk_anchors,
    }
    if anchors_scanned != anchor_count:
        errors.append(
            f"scan found {anchors_scanned} anchors, expected {anchor_count}"
        )

    for feature in FEATURE_ORDER:
        feature_report, feature_errors = accumulators[feature].report(
            outside_fraction_limit=outside_fraction_limit,
            xyz_hard_limit=xyz_hard_limit,
            roundtrip_atol=roundtrip_atol,
        )
        report["features"][feature] = feature_report
        errors.extend(feature_errors)

    report["passed"] = not errors
    report["status"] = "passed" if report["passed"] else "failed"
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit full-recipe DROID q01/q99 normalization."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. The report is always printed to stdout.",
    )
    parser.add_argument("--outside-fraction-limit", type=float, default=0.05)
    parser.add_argument("--xyz-hard-limit", type=float, default=0.15)
    parser.add_argument("--roundtrip-atol", type=float, default=1e-6)
    parser.add_argument("--chunk-anchors", type=int, default=8192)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Emit progress to stderr every N episodes; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_normalization(
        args.dataset_root,
        outside_fraction_limit=args.outside_fraction_limit,
        xyz_hard_limit=args.xyz_hard_limit,
        roundtrip_atol=args.roundtrip_atol,
        chunk_anchors=args.chunk_anchors,
        progress_every=args.progress_every,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    sys.stdout.write(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

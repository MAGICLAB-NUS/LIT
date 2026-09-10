#!/usr/bin/env python3
"""Prepare a read-only DROID v3 dataset for the 15-step goal-pose prior.

The derived dataset mirrors source parquet paths, adds ``observation.ee_pose``,
and keeps every original value unchanged.  Videos are linked, never decoded.
Builds happen in a sibling staging directory and are published only after a
strict verification pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.transform import Rotation

DEFAULT_SOURCE_ROOT = Path("/data0/JM/dataset/droid_1.0.1")
DEFAULT_OUTPUT_ROOT = Path("/data0/JM/dataset/droid_1.0.1_goal_pose")
DEFAULT_REVISION = "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4"
RECIPE_VERSION = 3
HORIZON = 15
NON_IDLE_ALGORITHM_SOURCE = (
    "Physical-Intelligence/openpi@c23745b5:"
    "examples/droid/compute_droid_nonidle_ranges.py"
)
EE_POSE_KEY = "observation.ee_pose"
STATE_KEY = "observation.state"
CARTESIAN_KEY = "observation.state.cartesian_position"
GRIPPER_KEY = "observation.state.gripper_position"
ACTION_KEY = "action"
ACTION_JOINT_VELOCITY_KEY = "action.joint_velocity"
LANGUAGE_KEYS = (
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
)
IDENTITY_KEYS = (
    "index",
    "episode_index",
    "frame_index",
    "task_index",
    "timestamp",
)
ANCHOR_SCHEMA = pa.schema(
    [
        pa.field("index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("timestamp", pa.float32()),
    ]
)
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


@dataclass(frozen=True)
class NonIdleConfig:
    """Pinned DROID non-idle segmentation parameters."""

    joint_velocity_delta: float = 1e-3
    min_idle_len: int = 7
    min_non_idle_len: int = 16
    trim_last_steps: int = HORIZON


@dataclass(frozen=True)
class BuildConfig:
    source_root: str
    output_root: str
    revision: str
    max_episodes: int | None
    max_files: int | None
    video_symlink: str
    smoke_anchors: int
    horizon: int
    non_idle: NonIdleConfig


@dataclass(frozen=True)
class FilePlan:
    relative_path: str
    rows: int


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _config_fingerprint(config: BuildConfig) -> str:
    encoded = json.dumps(
        {"recipe_version": RECIPE_VERSION, "config": asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _config_dict_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"recipe_version": RECIPE_VERSION, "config": config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: BuildConfig) -> None:
    if config.horizon != HORIZON:
        raise ValueError(f"horizon is fixed at {HORIZON}")
    if config.smoke_anchors < 0:
        raise ValueError("--smoke-anchors must be non-negative")
    if (
        not np.isfinite(config.non_idle.joint_velocity_delta)
        or config.non_idle.joint_velocity_delta < 0
    ):
        raise ValueError("joint-velocity delta threshold must be finite and non-negative")
    if config.non_idle.min_idle_len < 1 or config.non_idle.min_non_idle_len < 1:
        raise ValueError("non-idle segment lengths must be positive")
    if config.non_idle.trim_last_steps != config.horizon:
        raise ValueError("non-idle ranges must trim exactly one action/goal horizon")


def _vector_column(table: pa.Table, key: str, width: int) -> np.ndarray:
    column = table.column(key).combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        values = column.values.to_numpy(zero_copy_only=False)
        array = values.reshape(len(column), column.type.list_size)
    elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        offsets = column.offsets.to_numpy(zero_copy_only=False)
        lengths = np.diff(offsets)
        if not np.all(lengths == width):
            raise ValueError(f"{key} has non-{width} rows")
        values = column.values.to_numpy(zero_copy_only=False)
        array = values.reshape(len(column), width)
    else:
        raise TypeError(f"{key} must be a list column, got {column.type}")
    if array.shape != (len(table), width):
        raise ValueError(f"{key} shape is {array.shape}, expected {(len(table), width)}")
    return np.asarray(array)


def rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
    """Vectorized intrinsic XYZ Euler-to-rotation-vector conversion."""

    rpy = np.asarray(rpy)
    if rpy.ndim != 2 or rpy.shape[1] != 3:
        raise ValueError(f"expected RPY shape (N, 3), got {rpy.shape}")
    return Rotation.from_euler("xyz", rpy).as_rotvec().astype(np.float32)


def build_ee_pose(cartesian: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Build float32 ``[xyz, rotation-vector, gripper]`` rows."""

    cartesian = np.asarray(cartesian)
    gripper = np.asarray(gripper).reshape(-1)
    if cartesian.ndim != 2 or cartesian.shape[1] != 6:
        raise ValueError(f"expected Cartesian shape (N, 6), got {cartesian.shape}")
    if len(cartesian) != len(gripper):
        raise ValueError("Cartesian and gripper row counts differ")
    pose = np.empty((len(cartesian), 7), dtype=np.float32)
    pose[:, :3] = cartesian[:, :3]
    pose[:, 3:6] = rpy_to_rotvec(cartesian[:, 3:6])
    pose[:, 6] = gripper
    if not np.isfinite(pose).all():
        raise ValueError("non-finite value while constructing observation.ee_pose")
    return pose


def add_ee_pose(table: pa.Table) -> pa.Table:
    """Return a table with one vectorized float32 EE-pose column appended."""

    if EE_POSE_KEY in table.column_names:
        raise ValueError(f"source already contains {EE_POSE_KEY}")
    cartesian = _vector_column(table, CARTESIAN_KEY, 6)
    gripper = table.column(GRIPPER_KEY).combine_chunks().to_numpy(zero_copy_only=False)
    pose = build_ee_pose(cartesian, gripper)
    arrow_pose = pa.FixedSizeListArray.from_arrays(
        pa.array(pose.reshape(-1), type=pa.float32()), 7
    )
    return table.append_column(EE_POSE_KEY, arrow_pose)


def _check_source_revision(source_root: Path, revision: str) -> None:
    marker = source_root / ".cache/huggingface/download/meta/info.json.metadata"
    if not marker.is_file():
        raise FileNotFoundError(f"cannot verify source revision: {marker}")
    first_line = marker.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line != revision:
        raise ValueError(f"source revision is {first_line}, expected {revision}")


def _source_parquets(source_root: Path) -> list[Path]:
    """Resolve data files from episode metadata, excluding stale local files."""

    info = json.loads((source_root / "meta/info.json").read_text(encoding="utf-8"))
    template = info.get("data_path")
    if not isinstance(template, str):
        raise ValueError("meta/info.json has no data_path template")

    references: set[tuple[int, int]] = set()
    episode_rows = 0
    for episode_path in sorted(
        (source_root / "meta/episodes").glob("chunk-*/*.parquet")
    ):
        table = pq.read_table(
            episode_path, columns=["data/chunk_index", "data/file_index"]
        )
        references.update(
            zip(
                map(int, table.column("data/chunk_index").to_pylist()),
                map(int, table.column("data/file_index").to_pylist()),
                strict=True,
            )
        )
        episode_rows += len(table)
    if episode_rows != int(info["total_episodes"]):
        raise ValueError(
            f"episode metadata has {episode_rows} rows, "
            f"expected {info['total_episodes']}"
        )
    if not references:
        raise ValueError("episode metadata references no data parquet files")

    files = [
        source_root
        / template.format(chunk_index=chunk_index, file_index=file_index)
        for chunk_index, file_index in sorted(references)
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"episode metadata references missing data files: {missing[:3]}")

    discovered = set((source_root / "data").glob("chunk-*/*.parquet"))
    ignored = sorted(discovered.difference(files))
    if ignored:
        print(
            f"[prepare] ignoring {len(ignored)} unreferenced local parquet files; "
            "episode metadata is authoritative",
            flush=True,
        )
    frame_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if frame_rows != int(info["total_frames"]):
        raise ValueError(
            f"referenced data files have {frame_rows} rows, "
            f"expected {info['total_frames']}"
        )
    return files


def plan_source_files(
    source_root: Path,
    max_files: int | None = None,
    max_episodes: int | None = None,
) -> list[FilePlan]:
    """Select complete leading rows without renumbering any identifier."""

    if max_files is not None and max_episodes is not None:
        raise ValueError("--max-files and --max-episodes are mutually exclusive")
    if max_files is not None and max_files <= 0:
        raise ValueError("--max-files must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    files = _source_parquets(source_root)
    if max_files is not None:
        files = files[:max_files]
        plans = [
            FilePlan(str(path.relative_to(source_root)), pq.ParquetFile(path).metadata.num_rows)
            for path in files
        ]
        final = files[-1]
        is_last = pq.read_table(final, columns=["is_last"]).column("is_last").to_numpy(
            zero_copy_only=False
        )
        complete_rows = np.flatnonzero(is_last)
        if not complete_rows.size:
            raise ValueError(f"--max-files={max_files} contains no complete episode")
        plans[-1] = FilePlan(plans[-1].relative_path, int(complete_rows[-1]) + 1)
        return plans
    if max_episodes is None:
        return [
            FilePlan(str(path.relative_to(source_root)), pq.ParquetFile(path).metadata.num_rows)
            for path in files
        ]

    plans: list[FilePlan] = []
    seen: set[int] = set()
    target_episode: int | None = None
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "is_last"])
        episodes = table.column("episode_index").to_numpy(zero_copy_only=False)
        is_last = table.column("is_last").to_numpy(zero_copy_only=False)
        rows = len(table)
        if target_episode is None:
            for episode in episodes:
                episode_int = int(episode)
                if episode_int not in seen:
                    seen.add(episode_int)
                    if len(seen) == max_episodes:
                        target_episode = episode_int
                        break
        if target_episode is not None:
            target_rows = np.flatnonzero(episodes == target_episode)
            if target_rows.size:
                completed = target_rows[is_last[target_rows]]
                if completed.size:
                    rows = int(completed[0]) + 1
                    plans.append(FilePlan(str(path.relative_to(source_root)), rows))
                    return plans
            elif plans:
                raise ValueError(
                    f"episode {target_episode} did not terminate before {path.relative_to(source_root)}"
                )
        plans.append(FilePlan(str(path.relative_to(source_root)), rows))
    raise ValueError(f"source contains fewer than {max_episodes} complete episodes")


def _selected_episode_max(source_root: Path, plans: list[FilePlan]) -> int:
    final = plans[-1]
    episodes = pq.read_table(
        source_root / final.relative_path, columns=["episode_index"]
    ).column("episode_index")
    if final.rows <= 0:
        raise ValueError("selected data prefix is empty")
    return int(episodes[final.rows - 1].as_py())


def _copy_metadata(
    source_root: Path,
    output_root: Path,
    selected_episode_max: int,
    partial: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Copy v3 metadata, filtering episode rows only for a smoke prefix."""

    source_meta = source_root / "meta"
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sorted(source_meta.iterdir()):
        if source.is_file():
            destination = output_meta / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination.relative_to(output_root)))

    episode_records: list[dict[str, Any]] = []
    source_episodes = source_meta / "episodes"
    if not source_episodes.is_dir():
        raise FileNotFoundError(source_episodes)
    for source in sorted(source_episodes.glob("chunk-*/*.parquet")):
        relative = source.relative_to(source_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if partial:
            source_table = pq.read_table(source)
            episode_ids = source_table.column("episode_index").to_numpy(zero_copy_only=False)
            keep = int(np.searchsorted(episode_ids, selected_episode_max, side="right"))
            if keep == 0:
                break
            output_table = source_table.slice(0, keep)
            pq.write_table(output_table, destination, compression="zstd")
            source_rows = len(source_table)
            output_rows = len(output_table)
        else:
            shutil.copy2(source, destination)
            source_rows = pq.ParquetFile(source).metadata.num_rows
            output_rows = source_rows
        copied.append(str(relative))
        episode_records.append(
            {
                "path": str(relative),
                "source_rows": source_rows,
                "output_rows": output_rows,
                "source_sha256": _sha256(source),
                "output_sha256": _sha256(destination),
            }
        )
        if partial and int(output_table.column("episode_index")[-1].as_py()) >= selected_episode_max:
            break
    if not episode_records:
        raise ValueError("no episode metadata rows were copied")
    last_episode = int(
        pq.read_table(
            output_root / episode_records[-1]["path"], columns=["episode_index"]
        ).column("episode_index")[-1].as_py()
    )
    if partial and last_episode != selected_episode_max:
        raise ValueError(
            f"episode metadata ended at {last_episode}, expected {selected_episode_max}"
        )
    return copied, episode_records


def _create_video_symlink(
    source_root: Path, staging_root: Path, final_root: Path, mode: str
) -> str:
    source_videos = source_root / "videos"
    if not source_videos.is_dir():
        raise FileNotFoundError(source_videos)
    link = staging_root / "videos"
    if mode == "absolute":
        target = str(source_videos.resolve())
    elif mode == "relative":
        target = os.path.relpath(source_videos.resolve(), start=final_root.resolve())
    else:
        raise ValueError(f"unknown video symlink mode: {mode}")
    link.symlink_to(target, target_is_directory=True)
    return target


def _convert_files(
    source_root: Path, output_root: Path, plans: list[FilePlan]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_number, plan in enumerate(plans, start=1):
        source = source_root / plan.relative_path
        destination = output_root / plan.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        if plan.rows > len(table):
            raise ValueError(f"planned rows exceed source rows for {source}")
        table = table.slice(0, plan.rows)
        converted = add_ee_pose(table)
        pq.write_table(converted, destination, compression="zstd")
        records.append(
            {
                "path": plan.relative_path,
                "source_rows": pq.ParquetFile(source).metadata.num_rows,
                "output_rows": plan.rows,
                "source_sha256": _sha256(source),
                "output_sha256": _sha256(destination),
            }
        )
        if file_number == 1 or file_number % 25 == 0 or file_number == len(plans):
            print(
                f"[prepare] converted parquet files {file_number}/{len(plans)}",
                flush=True,
            )
    return records


def _episode_tables(
    root: Path, plans: list[FilePlan], columns: list[str]
) -> Iterator[pa.Table]:
    carry: pa.Table | None = None
    for plan in plans:
        table = pq.read_table(root / plan.relative_path, columns=columns).slice(0, plan.rows)
        if not len(table):
            continue
        episode_ids = table.column("episode_index").to_numpy(zero_copy_only=False)
        boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, len(table)]
        groups = [table.slice(int(start), int(stop - start)) for start, stop in zip(starts, stops)]
        if carry is not None:
            carry_id = int(carry.column("episode_index")[0].as_py())
            first_id = int(groups[0].column("episode_index")[0].as_py())
            if carry_id == first_id:
                groups[0] = pa.concat_tables([carry, groups[0]])
            else:
                yield carry
        yield from groups[:-1]
        carry = groups[-1]
    if carry is not None:
        yield carry


def _episode_arrays(table: pa.Table) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        key: table.column(key).combine_chunks().to_numpy(zero_copy_only=False)
        for key in IDENTITY_KEYS
    }
    arrays["is_episode_successful"] = (
        table.column("is_episode_successful").combine_chunks().to_numpy(zero_copy_only=False)
    )
    arrays[STATE_KEY] = _vector_column(table, STATE_KEY, 8).astype(np.float32, copy=False)
    arrays[EE_POSE_KEY] = _vector_column(table, EE_POSE_KEY, 7).astype(np.float32, copy=False)
    arrays[ACTION_KEY] = _vector_column(table, ACTION_KEY, 8).astype(np.float32, copy=False)
    arrays[ACTION_JOINT_VELOCITY_KEY] = _vector_column(
        table, ACTION_JOINT_VELOCITY_KEY, 7
    ).astype(np.float32, copy=False)
    for key in LANGUAGE_KEYS:
        arrays[key] = np.asarray(table.column(key).to_pylist(), dtype=object)
    return arrays


def _valid_task_indices(root: Path) -> set[int]:
    """Return task IDs whose canonical LeRobot task string is non-empty."""

    path = root / "meta/tasks.parquet"
    table = pq.read_table(path)
    if "task_index" not in table.column_names:
        raise ValueError(f"{path} is missing task_index")
    text_key = "task" if "task" in table.column_names else "__index_level_0__"
    if text_key not in table.column_names:
        raise ValueError(f"{path} has no canonical task-text column")
    task_indices = table.column("task_index").to_pylist()
    task_texts = table.column(text_key).to_pylist()
    if len(task_indices) != len(set(task_indices)):
        raise ValueError(f"{path} contains duplicate task_index values")
    return {
        int(task_index)
        for task_index, task in zip(task_indices, task_texts, strict=True)
        if task is not None and str(task).strip()
    }


def _true_segments(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    differences = np.diff(padded.astype(np.int8))
    return np.flatnonzero(differences == 1), np.flatnonzero(differences == -1)


def non_idle_anchor_mask(
    joint_velocities: np.ndarray,
    config: NonIdleConfig,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Apply the pinned DROID/OpenPI idle segmentation to anchor start frames."""

    joint_velocities = np.asarray(joint_velocities, dtype=np.float32)
    if joint_velocities.ndim != 2 or joint_velocities.shape[1] != 7:
        raise ValueError(
            f"expected joint velocity shape (N, 7), got {joint_velocities.shape}"
        )
    anchor_count = max(0, len(joint_velocities) - horizon)
    anchors = np.zeros(anchor_count, dtype=bool)
    if anchor_count == 0:
        return anchors

    is_idle = np.concatenate(
        (
            [False],
            np.all(
                np.abs(joint_velocities[1:] - joint_velocities[:-1])
                < config.joint_velocity_delta,
                axis=1,
            ),
        )
    )
    idle_starts, idle_ends = _true_segments(is_idle)
    keep = np.ones(len(joint_velocities), dtype=bool)
    for start, end in zip(idle_starts, idle_ends, strict=True):
        if end - start >= config.min_idle_len:
            keep[start:end] = False

    keep_starts, keep_ends = _true_segments(keep)
    for start, end in zip(keep_starts, keep_ends, strict=True):
        if end - start < config.min_non_idle_len:
            continue
        anchor_end = min(end - config.trim_last_steps, anchor_count)
        if anchor_end > start:
            anchors[start:anchor_end] = True
    return anchors


def anchor_mask(
    arrays: dict[str, np.ndarray],
    non_idle: NonIdleConfig,
    horizon: int = HORIZON,
    valid_task_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-anchor mask and `(N, horizon, 8)` action windows."""

    episode = arrays["episode_index"]
    frame = arrays["frame_index"]
    index = arrays["index"]
    timestamp = arrays["timestamp"]
    state = arrays[STATE_KEY]
    pose = arrays[EE_POSE_KEY]
    actions = arrays[ACTION_KEY]
    joint_velocities = arrays[ACTION_JOINT_VELOCITY_KEY]
    if not np.all(episode == episode[0]):
        raise ValueError("episode scanner emitted mixed episode IDs")
    if not np.all(np.diff(frame) == 1) or not np.all(np.diff(index) == 1):
        raise ValueError(f"non-contiguous indices in episode {int(episode[0])}")
    if not np.all(np.diff(timestamp) > 0):
        raise ValueError(f"non-increasing timestamps in episode {int(episode[0])}")
    for key, values in (
        (STATE_KEY, state),
        (EE_POSE_KEY, pose),
        (ACTION_KEY, actions),
        (ACTION_JOINT_VELOCITY_KEY, joint_velocities),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {key} in episode {int(episode[0])}")
    task_indices = arrays["task_index"]
    if not np.all(task_indices == task_indices[0]):
        raise ValueError(f"task_index changes within episode {int(episode[0])}")

    successful = arrays["is_episode_successful"]
    if not np.all(successful == successful[0]):
        raise ValueError(f"inconsistent success flag in episode {int(episode[0])}")
    length = len(index)
    count = max(0, length - horizon)
    if count == 0:
        return np.zeros(0, dtype=bool), np.empty((0, horizon, 8), dtype=np.float32)
    eligible = np.full(count, bool(successful[0]), dtype=bool)
    if valid_task_indices is None:
        eligible &= np.fromiter(
            (
                any(
                    arrays[key][row] is not None and str(arrays[key][row]).strip()
                    for key in LANGUAGE_KEYS
                )
                for row in range(count)
            ),
            dtype=bool,
            count=count,
        )
    else:
        eligible &= np.isin(task_indices[:count], list(valid_task_indices))

    windows = sliding_window_view(actions, horizon, axis=0)[:count].transpose(0, 2, 1)
    moving = non_idle_anchor_mask(joint_velocities, non_idle, horizon)
    return eligible & moving, windows


def _anchor_columns(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "index": arrays["index"][: len(mask)][mask].astype(np.int64, copy=False),
        "episode_index": arrays["episode_index"][: len(mask)][mask].astype(np.int64, copy=False),
        "frame_index": arrays["frame_index"][: len(mask)][mask].astype(np.int64, copy=False),
        "task_index": arrays["task_index"][: len(mask)][mask].astype(np.int64, copy=False),
        "timestamp": arrays["timestamp"][: len(mask)][mask].astype(np.float32, copy=False),
    }


def _scan_columns() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *IDENTITY_KEYS,
                "is_episode_successful",
                *LANGUAGE_KEYS,
                STATE_KEY,
                EE_POSE_KEY,
                ACTION_KEY,
                ACTION_JOINT_VELOCITY_KEY,
            ]
        )
    )


def _write_anchor_artifacts(
    root: Path,
    plans: list[FilePlan],
    non_idle: NonIdleConfig,
    smoke_count: int,
) -> tuple[int, list[int]]:
    manifest_path = root / "valid_anchor_indices.parquet"
    writer = pq.ParquetWriter(manifest_path, ANCHOR_SCHEMA, compression="zstd")
    valid_task_indices = _valid_task_indices(root)
    if not valid_task_indices:
        raise ValueError("canonical task table contains no non-empty language")
    successful_episodes: list[int] = []
    smoke_rows: list[dict[str, Any]] = []
    anchor_count = 0
    try:
        for episode_number, episode_table in enumerate(
            _episode_tables(root, plans, _scan_columns()), start=1
        ):
            arrays = _episode_arrays(episode_table)
            episode_id = int(arrays["episode_index"][0])
            success = arrays["is_episode_successful"]
            if bool(success[0]):
                successful_episodes.append(episode_id)
            mask, _ = anchor_mask(
                arrays, non_idle, valid_task_indices=valid_task_indices
            )
            columns = _anchor_columns(arrays, mask)
            count = len(columns["index"])
            if count:
                batch = pa.Table.from_pydict(columns, schema=ANCHOR_SCHEMA)
                writer.write_table(batch)
                if len(smoke_rows) < smoke_count:
                    take = min(smoke_count - len(smoke_rows), count)
                    smoke_rows.extend(batch.slice(0, take).to_pylist())
                anchor_count += count
            if episode_number % 1000 == 0:
                print(
                    f"[prepare] scanned {episode_number} episodes; "
                    f"valid anchors={anchor_count}",
                    flush=True,
                )
    finally:
        writer.close()
    _json_dump(
        root / "successful_episodes.json",
        {
            "count": len(successful_episodes),
            "episode_indices": successful_episodes,
        },
    )
    _json_dump(
        root / "smoke_anchor_indices.json",
        {
            "selection": "first_valid_anchors_in_global_index_order",
            "requested": smoke_count,
            "count": len(smoke_rows),
            "anchors": smoke_rows,
        },
    )
    return anchor_count, successful_episodes


def _merge_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    chunk: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    chunk64 = np.asarray(chunk, dtype=np.float64)
    chunk_count = len(chunk64)
    if not chunk_count:
        return count, mean, m2
    chunk_mean = chunk64.mean(axis=0)
    chunk_m2 = np.square(chunk64 - chunk_mean).sum(axis=0)
    if count == 0:
        return chunk_count, chunk_mean, chunk_m2
    total = count + chunk_count
    delta = chunk_mean - mean
    merged_mean = mean + delta * (chunk_count / total)
    merged_m2 = m2 + chunk_m2 + delta * delta * (count * chunk_count / total)
    return total, merged_mean, merged_m2


def _quantiles_from_sorted(sorted_values: np.memmap) -> list[float]:
    size = len(sorted_values)
    if size == 0:
        raise ValueError("cannot compute quantiles of an empty feature")
    values: list[float] = []
    for quantile in QUANTILES:
        position = (size - 1) * quantile
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        fraction = position - lower
        value = float(sorted_values[lower]) * (1 - fraction) + float(
            sorted_values[upper]
        ) * fraction
        values.append(value)
    return values


def _memmap_stats(
    data_path: Path,
    shape: tuple[int, int],
    scratch_dir: Path,
    chunk_rows: int = 262_144,
) -> dict[str, list[float] | list[int]]:
    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
    minimum = np.full(shape[1], np.inf, dtype=np.float64)
    maximum = np.full(shape[1], -np.inf, dtype=np.float64)
    count = 0
    mean = np.zeros(shape[1], dtype=np.float64)
    m2 = np.zeros(shape[1], dtype=np.float64)
    for start in range(0, shape[0], chunk_rows):
        chunk = np.asarray(data[start : start + chunk_rows])
        if not np.isfinite(chunk).all():
            raise ValueError(f"non-finite values in stats memmap {data_path}")
        minimum = np.minimum(minimum, chunk.min(axis=0))
        maximum = np.maximum(maximum, chunk.max(axis=0))
        count, mean, m2 = _merge_moments(count, mean, m2, chunk)

    quantile_rows: list[list[float]] = [[] for _ in QUANTILES]
    scratch_path = scratch_dir / f"{data_path.stem}.sorted.f32"
    for dimension in range(shape[1]):
        scratch = np.memmap(scratch_path, dtype=np.float32, mode="w+", shape=(shape[0],))
        for start in range(0, shape[0], chunk_rows):
            stop = min(start + chunk_rows, shape[0])
            scratch[start:stop] = data[start:stop, dimension]
        scratch.flush()
        scratch.sort()
        for row, value in zip(quantile_rows, _quantiles_from_sorted(scratch), strict=True):
            row.append(value)
        del scratch
    scratch_path.unlink(missing_ok=True)
    del data
    result: dict[str, list[float] | list[int]] = {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(m2 / count).tolist(),
        "count": [count],
    }
    for quantile, row in zip(QUANTILES, quantile_rows, strict=True):
        result[f"q{int(quantile * 100):02d}"] = row
    return result


def _compute_anchor_stats(
    root: Path,
    plans: list[FilePlan],
    non_idle: NonIdleConfig,
    anchor_count: int,
    work_dir: Path,
) -> dict[str, Any]:
    if anchor_count <= 0:
        raise ValueError("no valid anchors; refusing to produce empty statistics")
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        STATE_KEY: work_dir / "anchor_state.f32",
        EE_POSE_KEY: work_dir / "anchor_goal_pose.f32",
        ACTION_KEY: work_dir / "anchor_actions.f32",
    }
    maps = {
        STATE_KEY: np.memmap(paths[STATE_KEY], dtype=np.float32, mode="w+", shape=(anchor_count, 8)),
        EE_POSE_KEY: np.memmap(
            paths[EE_POSE_KEY], dtype=np.float32, mode="w+", shape=(anchor_count, 7)
        ),
        ACTION_KEY: np.memmap(
            paths[ACTION_KEY],
            dtype=np.float32,
            mode="w+",
            shape=(anchor_count * HORIZON, 8),
        ),
    }
    valid_task_indices = _valid_task_indices(root)
    offset = 0
    for episode_number, episode_table in enumerate(
        _episode_tables(root, plans, _scan_columns()), start=1
    ):
        arrays = _episode_arrays(episode_table)
        mask, windows = anchor_mask(
            arrays, non_idle, valid_task_indices=valid_task_indices
        )
        count = int(mask.sum())
        if not count:
            continue
        stop = offset + count
        maps[STATE_KEY][offset:stop] = arrays[STATE_KEY][: len(mask)][mask]
        maps[EE_POSE_KEY][offset:stop] = arrays[EE_POSE_KEY][HORIZON:][mask]
        maps[ACTION_KEY][offset * HORIZON : stop * HORIZON] = windows[mask].reshape(-1, 8)
        offset = stop
        if episode_number % 1000 == 0:
            print(
                f"[prepare] materialized stats rows from {episode_number} episodes; "
                f"anchors={offset}/{anchor_count}",
                flush=True,
            )
    if offset != anchor_count:
        raise ValueError(f"anchor scan changed: expected {anchor_count}, got {offset}")
    for mapping in maps.values():
        mapping.flush()
    maps.clear()

    features = {
        STATE_KEY: _memmap_stats(paths[STATE_KEY], (anchor_count, 8), work_dir),
        EE_POSE_KEY: _memmap_stats(paths[EE_POSE_KEY], (anchor_count, 7), work_dir),
        ACTION_KEY: _memmap_stats(
            paths[ACTION_KEY], (anchor_count * HORIZON, 8), work_dir
        ),
    }
    for path in paths.values():
        path.unlink(missing_ok=True)
    return {
        "anchor_count": anchor_count,
        "horizon": HORIZON,
        "action_samples": anchor_count * HORIZON,
        "features": features,
    }


def _update_metadata(
    root: Path,
    stats: dict[str, Any],
    partial: bool,
    total_frames: int,
    total_episodes: int,
) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    features[EE_POSE_KEY] = {
        "dtype": "float32",
        "shape": [7],
        "names": {
            "axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
    }
    info["goal_pose_prior"] = {
        "horizon": HORIZON,
        "partial_build": partial,
        "anchor_manifest": "valid_anchor_indices.parquet",
        "target_feature": EE_POSE_KEY,
    }
    if partial:
        info["total_frames"] = total_frames
        info["total_episodes"] = total_episodes
        info["splits"] = {"train": f"0:{total_episodes}"}
    _json_dump(info_path, info)

    stats_path = root / "meta/stats.json"
    source_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    source_stats.update(stats["features"])
    _json_dump(stats_path, source_stats)
    _json_dump(root / "anchor_stats.json", stats)


def _artifact_hashes(root: Path) -> dict[str, str]:
    names = [
        "successful_episodes.json",
        "valid_anchor_indices.parquet",
        "smoke_anchor_indices.json",
        "anchor_stats.json",
    ]
    names.extend(
        str(path.relative_to(root))
        for path in sorted((root / "meta").iterdir())
        if path.is_file()
    )
    return {name: _sha256(root / name) for name in names}


class _ManifestCursor:
    def __init__(self, path: Path) -> None:
        self._iterator = pq.ParquetFile(path).iter_batches(batch_size=65_536)
        self._batch: pa.RecordBatch | None = None
        self._offset = 0
        self.rows = 0

    def _ensure_batch(self) -> bool:
        while self._batch is None or self._offset == len(self._batch):
            try:
                self._batch = next(self._iterator)
                self._offset = 0
            except StopIteration:
                self._batch = None
                return False
        return True

    def consume(self, expected: dict[str, np.ndarray]) -> None:
        expected_offset = 0
        count = len(expected["index"])
        while expected_offset < count:
            if not self._ensure_batch():
                raise ValueError("anchor manifest ended early")
            assert self._batch is not None
            take = min(count - expected_offset, len(self._batch) - self._offset)
            for key in ANCHOR_SCHEMA.names:
                actual = (
                    self._batch.column(key)
                    .slice(self._offset, take)
                    .to_numpy(zero_copy_only=False)
                )
                wanted = expected[key][expected_offset : expected_offset + take]
                if not np.array_equal(actual, wanted):
                    raise ValueError(f"anchor manifest mismatch in {key}")
            self._offset += take
            expected_offset += take
            self.rows += take

    def finish(self) -> None:
        if self._ensure_batch():
            raise ValueError("anchor manifest contains extra rows")


def _verify_pairing_and_poses(
    source_root: Path,
    output_root: Path,
    files: list[dict[str, Any]],
) -> None:
    previous: tuple[int, int, int, float] | None = None
    finished_episodes: set[int] = set()
    for record in files:
        relative = record["path"]
        source_path = source_root / relative
        output_path = output_root / relative
        if _sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source hash changed: {relative}")
        if _sha256(output_path) != record["output_sha256"]:
            raise ValueError(f"derived hash changed: {relative}")
        if pq.ParquetFile(source_path).metadata.num_rows != int(record["source_rows"]):
            raise ValueError(f"source row-count changed: {relative}")
        if pq.ParquetFile(output_path).metadata.num_rows != int(record["output_rows"]):
            raise ValueError(f"derived row-count changed: {relative}")
        source = pq.read_table(source_path).slice(0, int(record["output_rows"]))
        derived = pq.read_table(output_path)
        if len(source) != len(derived):
            raise ValueError(f"row-count mismatch: {relative}")
        if derived.column_names != [*source.column_names, EE_POSE_KEY]:
            raise ValueError(f"schema pairing mismatch: {relative}")
        if not derived.select(source.column_names).equals(source):
            raise ValueError(f"original columns changed: {relative}")

        cartesian = _vector_column(source, CARTESIAN_KEY, 6)
        gripper = source.column(GRIPPER_KEY).to_numpy(zero_copy_only=False)
        expected_pose = build_ee_pose(cartesian, gripper)
        actual_pose = _vector_column(derived, EE_POSE_KEY, 7)
        if actual_pose.dtype != np.float32:
            raise ValueError(f"{EE_POSE_KEY} is not float32: {relative}")
        if not np.array_equal(actual_pose, expected_pose):
            raise ValueError(f"EE-pose conversion mismatch: {relative}")
        expected_rotation = Rotation.from_euler("xyz", cartesian[:, 3:6]).as_matrix()
        roundtrip_rotation = Rotation.from_rotvec(actual_pose[:, 3:6]).as_matrix()
        if not np.allclose(expected_rotation, roundtrip_rotation, atol=2e-6, rtol=0):
            raise ValueError(f"RPY/rotation-vector roundtrip failed: {relative}")
        if not np.isfinite(actual_pose).all():
            raise ValueError(f"non-finite EE pose: {relative}")

        identities = {
            key: derived.column(key).to_numpy(zero_copy_only=False) for key in IDENTITY_KEYS
        }
        for row in zip(
            identities["episode_index"],
            identities["frame_index"],
            identities["index"],
            identities["timestamp"],
            strict=True,
        ):
            current = (int(row[0]), int(row[1]), int(row[2]), float(row[3]))
            if previous is not None:
                if current[0] == previous[0]:
                    if current[1] != previous[1] + 1 or current[2] != previous[2] + 1:
                        raise ValueError(f"non-contiguous episode {current[0]}")
                    if not np.isclose(
                        current[3] - previous[3], 1 / 15, atol=2e-5, rtol=0
                    ):
                        raise ValueError(f"timestamp discontinuity in episode {current[0]}")
                else:
                    finished_episodes.add(previous[0])
                    if current[0] in finished_episodes:
                        raise ValueError(f"episode {current[0]} is not stored contiguously")
            previous = current


def _verify_episode_metadata(
    source_root: Path,
    output_root: Path,
    records: list[dict[str, Any]],
) -> None:
    previous_episode: int | None = None
    for record in records:
        relative = record["path"]
        source_path = source_root / relative
        output_path = output_root / relative
        if _sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source episode metadata hash changed: {relative}")
        if _sha256(output_path) != record["output_sha256"]:
            raise ValueError(f"derived episode metadata hash changed: {relative}")
        if pq.ParquetFile(source_path).metadata.num_rows != int(record["source_rows"]):
            raise ValueError(f"source episode metadata row count changed: {relative}")
        if pq.ParquetFile(output_path).metadata.num_rows != int(record["output_rows"]):
            raise ValueError(f"derived episode metadata row count changed: {relative}")
        if (
            record["source_sha256"] != record["output_sha256"]
            or int(record["source_rows"]) != int(record["output_rows"])
        ):
            source = pq.read_table(source_path).slice(0, int(record["output_rows"]))
            output = pq.read_table(output_path)
            if not output.equals(source):
                raise ValueError(f"episode metadata pairing mismatch: {relative}")
        episodes = pq.read_table(output_path, columns=["episode_index"]).column(
            "episode_index"
        ).to_numpy(zero_copy_only=False)
        if len(episodes) and previous_episode is not None:
            if int(episodes[0]) != previous_episode + 1:
                raise ValueError("episode metadata is not globally contiguous")
        if len(episodes):
            if not np.all(np.diff(episodes) == 1):
                raise ValueError(f"episode metadata is not contiguous: {relative}")
            previous_episode = int(episodes[-1])


def _assert_stats_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual.keys() != expected.keys():
        raise ValueError("stats top-level keys differ")
    for key in ("anchor_count", "horizon", "action_samples"):
        if actual[key] != expected[key]:
            raise ValueError(f"stats {key} mismatch")
    if actual["features"].keys() != expected["features"].keys():
        raise ValueError("stats feature keys differ")
    for feature in expected["features"]:
        for statistic, wanted in expected["features"][feature].items():
            got = actual["features"][feature].get(statistic)
            if statistic == "count":
                if got != wanted:
                    raise ValueError(f"{feature} {statistic} mismatch")
            elif not np.allclose(got, wanted, atol=1e-7, rtol=1e-7):
                raise ValueError(f"{feature} {statistic} mismatch")


def verify_dataset(
    root: Path,
    source_root: Path | None = None,
    require_success: bool = True,
) -> dict[str, Any]:
    """Fail-closed verification of pairing, anchors, stats, roundtrip, and hashes."""

    provenance_path = root / "goal_pose_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("recipe_version") != RECIPE_VERSION:
        raise ValueError(
            f"unsupported recipe version: {provenance.get('recipe_version')!r}"
        )
    config_dict = provenance["config"]
    if provenance.get("config_fingerprint") != _config_dict_fingerprint(config_dict):
        raise ValueError("provenance config fingerprint mismatch")
    source = source_root or Path(config_dict["source_root"])
    _check_source_revision(source, config_dict["revision"])
    if config_dict["horizon"] != HORIZON:
        raise ValueError(f"unsupported horizon in provenance: {config_dict['horizon']}")
    non_idle = NonIdleConfig(**config_dict["non_idle"])
    loaded_config = BuildConfig(
        source_root=config_dict["source_root"],
        output_root=config_dict["output_root"],
        revision=config_dict["revision"],
        max_episodes=config_dict["max_episodes"],
        max_files=config_dict["max_files"],
        video_symlink=config_dict["video_symlink"],
        smoke_anchors=config_dict["smoke_anchors"],
        horizon=config_dict["horizon"],
        non_idle=non_idle,
    )
    _validate_config(loaded_config)
    plans = [FilePlan(item["path"], int(item["output_rows"])) for item in provenance["files"]]

    if require_success:
        success_path = root / "_SUCCESS.json"
        if not success_path.is_file():
            raise FileNotFoundError("atomic completion marker is missing")
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success["provenance_sha256"] != _sha256(provenance_path):
            raise ValueError("completion marker provenance hash mismatch")
        if success["config_fingerprint"] != provenance["config_fingerprint"]:
            raise ValueError("completion marker config mismatch")
    for relative, digest in provenance["artifact_sha256"].items():
        if _sha256(root / relative) != digest:
            raise ValueError(f"artifact hash changed: {relative}")

    _verify_pairing_and_poses(source, root, provenance["files"])
    _verify_episode_metadata(source, root, provenance["episode_metadata_files"])
    cursor = _ManifestCursor(root / "valid_anchor_indices.parquet")
    valid_task_indices = _valid_task_indices(root)
    successful: list[int] = []
    expected_anchor_count = 0
    smoke_json = json.loads((root / "smoke_anchor_indices.json").read_text(encoding="utf-8"))
    smoke_requested = int(smoke_json["requested"])
    expected_smoke: list[dict[str, Any]] = []
    for episode_table in _episode_tables(root, plans, _scan_columns()):
        arrays = _episode_arrays(episode_table)
        if bool(arrays["is_episode_successful"][0]):
            successful.append(int(arrays["episode_index"][0]))
        mask, _ = anchor_mask(
            arrays, non_idle, valid_task_indices=valid_task_indices
        )
        expected = _anchor_columns(arrays, mask)
        cursor.consume(expected)
        if len(expected_smoke) < smoke_requested and len(expected["index"]):
            take = min(smoke_requested - len(expected_smoke), len(expected["index"]))
            expected_smoke.extend(
                pa.Table.from_pydict(expected, schema=ANCHOR_SCHEMA).slice(0, take).to_pylist()
            )
        expected_anchor_count += int(mask.sum())
    cursor.finish()
    if expected_anchor_count != provenance["anchor_count"]:
        raise ValueError("anchor count does not match provenance")

    successful_json = json.loads((root / "successful_episodes.json").read_text(encoding="utf-8"))
    if successful_json != {"count": len(successful), "episode_indices": successful}:
        raise ValueError("successful episode manifest mismatch")
    expected_smoke_json = {
        "selection": "first_valid_anchors_in_global_index_order",
        "requested": smoke_requested,
        "count": len(expected_smoke),
        "anchors": expected_smoke,
    }
    if smoke_json != expected_smoke_json:
        raise ValueError("smoke anchor manifest mismatch")

    with tempfile.TemporaryDirectory(prefix="droid-goal-prior-verify-") as temporary:
        recomputed_stats = _compute_anchor_stats(
            root, plans, non_idle, expected_anchor_count, Path(temporary)
        )
    saved_stats = json.loads((root / "anchor_stats.json").read_text(encoding="utf-8"))
    _assert_stats_equal(saved_stats, recomputed_stats)
    metadata_stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    for feature, expected in recomputed_stats["features"].items():
        if feature not in metadata_stats:
            raise ValueError(f"meta/stats.json is missing {feature}")
        _assert_stats_equal(
            {
                "anchor_count": expected_anchor_count,
                "horizon": HORIZON,
                "action_samples": expected_anchor_count * HORIZON,
                "features": {feature: metadata_stats[feature]},
            },
            {
                "anchor_count": expected_anchor_count,
                "horizon": HORIZON,
                "action_samples": expected_anchor_count * HORIZON,
                "features": {feature: expected},
            },
        )

    videos = root / "videos"
    if not videos.is_symlink() or videos.resolve() != (source / "videos").resolve():
        raise ValueError("videos is not a symlink to the source videos directory")
    return {
        "files": len(plans),
        "anchors": expected_anchor_count,
        "successful_episodes": len(successful),
    }


def prepare_dataset(config: BuildConfig) -> dict[str, Any]:
    _validate_config(config)
    source_root = Path(config.source_root).resolve()
    output_root = Path(config.output_root).resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("output root must not be the source root or a child of it")
    _check_source_revision(source_root, config.revision)
    fingerprint = _config_fingerprint(config)

    if output_root.exists():
        provenance_path = output_root / "goal_pose_provenance.json"
        success_path = output_root / "_SUCCESS.json"
        if provenance_path.is_file() and success_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("config_fingerprint") != fingerprint:
                raise FileExistsError("completed output exists with a different configuration")
            success = json.loads(success_path.read_text(encoding="utf-8"))
            if success.get("config_fingerprint") != fingerprint:
                raise ValueError("completion marker config mismatch")
            if success.get("provenance_sha256") != _sha256(provenance_path):
                raise ValueError("completion marker provenance hash mismatch")
            return {
                "files": len(provenance["files"]),
                "anchors": int(provenance["anchor_count"]),
                "successful_episodes": int(provenance["successful_episode_count"]),
            }
        raise FileExistsError(f"refusing to replace incomplete output: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        plans = plan_source_files(
            source_root,
            max_files=config.max_files,
            max_episodes=config.max_episodes,
        )
        print(
            f"[prepare] selected {len(plans)} parquet files; "
            f"rows={sum(plan.rows for plan in plans)}",
            flush=True,
        )
        partial = config.max_files is not None or config.max_episodes is not None
        selected_episode_max = (
            _selected_episode_max(source_root, plans) if partial else -1
        )
        copied_meta, episode_metadata_files = _copy_metadata(
            source_root,
            staging_root,
            selected_episode_max=selected_episode_max,
            partial=partial,
        )
        video_target = _create_video_symlink(
            source_root, staging_root, output_root, config.video_symlink
        )
        files = _convert_files(source_root, staging_root, plans)
        anchor_count, successful = _write_anchor_artifacts(
            staging_root, plans, config.non_idle, config.smoke_anchors
        )
        stats_work = staging_root / ".stats-work"
        stats = _compute_anchor_stats(
            staging_root, plans, config.non_idle, anchor_count, stats_work
        )
        shutil.rmtree(stats_work)
        _update_metadata(
            staging_root,
            stats,
            partial,
            total_frames=sum(plan.rows for plan in plans),
            total_episodes=selected_episode_max + 1,
        )
        provenance = {
            "format_version": 1,
            "recipe_version": RECIPE_VERSION,
            "config": asdict(config),
            "config_fingerprint": fingerprint,
            "source_read_only": True,
            "source_revision_verified_from": (
                ".cache/huggingface/download/meta/info.json.metadata"
            ),
            "ee_pose_definition": {
                "source": [
                    CARTESIAN_KEY,
                    GRIPPER_KEY,
                ],
                "value": "[xyz, scipy Rotation.from_euler('xyz', rpy).as_rotvec(), gripper]",
                "dtype": "float32",
                "wrist_site_not_tcp": True,
            },
            "anchor_definition": {
                "goal_offset": HORIZON,
                "action_window": "[t:t+15]",
                "success_required": True,
                "canonical_task_text_required": True,
                "same_episode_and_contiguous_required": True,
                "non_idle_algorithm_source": NON_IDLE_ALGORITHM_SOURCE,
                "source_filter_last_n_in_ranges": 10,
                "horizon_adaptation": (
                    "trim_last_steps is raised from the source value 10 to 15 so "
                    "all 15 actions and the t+15 goal remain inside a retained range"
                ),
                "non_idle_rule": (
                    "segment action.joint_velocity using consecutive-command delta; "
                    "drop long idle and short non-idle ranges; trim one full horizon"
                ),
                "non_idle_parameters": asdict(config.non_idle),
            },
            "files": files,
            "copied_meta": copied_meta,
            "episode_metadata_files": episode_metadata_files,
            "videos_symlink_target": video_target,
            "anchor_count": anchor_count,
            "successful_episode_count": len(successful),
            "artifact_sha256": _artifact_hashes(staging_root),
        }
        _json_dump(staging_root / "goal_pose_provenance.json", provenance)
        result = verify_dataset(staging_root, source_root=source_root, require_success=False)
        _json_dump(
            staging_root / "_SUCCESS.json",
            {
                "status": "complete",
                "config_fingerprint": fingerprint,
                "provenance_sha256": _sha256(staging_root / "goal_pose_provenance.json"),
            },
        )
        os.replace(staging_root, output_root)
        return result
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-episodes", type=int)
    limits.add_argument("--max-files", type=int)
    parser.add_argument("--video-symlink", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--smoke-anchors", type=int, default=128)
    parser.add_argument("--joint-velocity-delta", type=float, default=1e-3)
    parser.add_argument("--min-idle-len", type=int, default=7)
    parser.add_argument("--min-non-idle-len", type=int, default=16)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not build; strictly verify the completed output and source pairing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    non_idle = NonIdleConfig(
        joint_velocity_delta=args.joint_velocity_delta,
        min_idle_len=args.min_idle_len,
        min_non_idle_len=args.min_non_idle_len,
        trim_last_steps=HORIZON,
    )
    config = BuildConfig(
        source_root=str(args.source_root.resolve()),
        output_root=str(args.output_root.resolve()),
        revision=args.revision,
        max_episodes=args.max_episodes,
        max_files=args.max_files,
        video_symlink=args.video_symlink,
        smoke_anchors=args.smoke_anchors,
        horizon=HORIZON,
        non_idle=non_idle,
    )
    if args.verify_only:
        result = verify_dataset(Path(config.output_root), source_root=Path(config.source_root))
    else:
        result = prepare_dataset(config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

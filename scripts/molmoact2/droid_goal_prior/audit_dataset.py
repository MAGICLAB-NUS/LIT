#!/usr/bin/env python3
"""Independent, streaming consistency audit for the DROID goal-prior dataset."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.droid_goal_prior.prepare_dataset import (
    ACTION_JOINT_VELOCITY_KEY,
    ACTION_KEY,
    ANCHOR_SCHEMA,
    CARTESIAN_KEY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    EE_POSE_KEY,
    GRIPPER_KEY,
    HORIZON,
    IDENTITY_KEYS,
    RECIPE_VERSION,
    STATE_KEY,
    FilePlan,
    NonIdleConfig,
    _episode_tables,
    _vector_column,
    build_ee_pose,
    non_idle_anchor_mask,
)

CAMERA_KEYS = (
    "observation.images.wrist_left",
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
)
SAMPLE_COLUMNS = [
    *IDENTITY_KEYS,
    "is_episode_successful",
    STATE_KEY,
    ACTION_KEY,
    CARTESIAN_KEY,
    GRIPPER_KEY,
]


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class Differences:
    """Bounded examples plus an unbounded discrepancy count."""

    def __init__(self, limit: int = 50) -> None:
        self.count = 0
        self.examples: list[dict[str, Any]] = []
        self.limit = limit

    def add(self, category: str, message: str, **context: Any) -> None:
        self.count += 1
        if len(self.examples) < self.limit:
            item = {"category": category, "message": message}
            item.update({key: _native(value) for key, value in context.items()})
            self.examples.append(item)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _row(batch: pa.RecordBatch, offset: int) -> dict[str, Any]:
    return {
        name: _native(batch.column(name)[offset].as_py())
        for name in ANCHOR_SCHEMA.names
    }


class PrioritySample:
    """Streaming fixed-seed priority sample with O(k) memory."""

    def __init__(self, size: int, rng: np.random.Generator) -> None:
        self.size = size
        self.rng = rng
        self._heap: list[tuple[float, int, dict[str, Any]]] = []
        self._serial = 0

    def offer(self, item: dict[str, Any], priority: float | None = None) -> None:
        if self.size <= 0:
            return
        key = float(self.rng.random() if priority is None else priority)
        entry = (-key, self._serial, item)
        self._serial += 1
        if len(self._heap) < self.size:
            heapq.heappush(self._heap, entry)
        elif entry[0] > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def offer_batch(self, batch: pa.RecordBatch) -> None:
        if self.size <= 0 or len(batch) == 0:
            return
        priorities = self.rng.random(len(batch))
        take = min(self.size, len(batch))
        offsets = np.argpartition(priorities, take - 1)[:take]
        for offset in offsets:
            self.offer(_row(batch, int(offset)), float(priorities[offset]))

    def values(self) -> list[dict[str, Any]]:
        return [
            item
            for _, _, item in sorted(self._heap, key=lambda entry: (-entry[0], entry[1]))
        ]


class DataFiles:
    """Ordered data-file catalog and bounded parquet-column cache."""

    def __init__(
        self,
        source_root: Path,
        output_root: Path,
        records: list[dict[str, Any]],
        differences: Differences,
    ) -> None:
        self.source_root = source_root
        self.output_root = output_root
        self.records = records
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.rows: list[int] = []
        self._cache: OrderedDict[tuple[str, int, tuple[str, ...]], pa.Table] = OrderedDict()
        previous_end: int | None = None
        for file_id, record in enumerate(records):
            path = output_root / record["path"]
            expected_rows = int(record["output_rows"])
            try:
                parquet = pq.ParquetFile(path)
                actual_rows = parquet.metadata.num_rows
                if actual_rows != expected_rows:
                    differences.add(
                        "provenance.file_rows",
                        "派生 parquet 行数与 provenance 不符",
                        path=record["path"],
                        expected=expected_rows,
                        actual=actual_rows,
                    )
                if actual_rows == 0:
                    raise ValueError("empty data parquet")
                first = int(parquet.read_row_group(0, columns=["index"])["index"][0].as_py())
                last_group = parquet.metadata.num_row_groups - 1
                last_column = parquet.read_row_group(last_group, columns=["index"])["index"]
                last = int(last_column[-1].as_py())
            except Exception as error:
                differences.add(
                    "provenance.file",
                    "无法读取 provenance 指定的 parquet",
                    path=record.get("path"),
                    error=str(error),
                )
                first = 0 if previous_end is None else previous_end + 1
                last = first + max(expected_rows - 1, 0)
            if previous_end is not None and first != previous_end + 1:
                differences.add(
                    "data.index_contiguity",
                    "相邻派生 parquet 的全局 index 不连续",
                    file_id=file_id,
                    previous_end=previous_end,
                    current_start=first,
                )
            self.starts.append(first)
            self.ends.append(last)
            self.rows.append(expected_rows)
            previous_end = last
        self._end_array = np.asarray(self.ends, dtype=np.int64)

    def file_ids(self, indices: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._end_array, indices, side="left")

    def _table(self, root_kind: str, file_id: int, columns: Iterable[str]) -> pa.Table:
        column_tuple = tuple(columns)
        key = (root_kind, file_id, column_tuple)
        if key in self._cache:
            table = self._cache.pop(key)
            self._cache[key] = table
            return table
        root = self.source_root if root_kind == "source" else self.output_root
        table = pq.read_table(root / self.records[file_id]["path"], columns=list(column_tuple))
        if root_kind == "source":
            table = table.slice(0, self.rows[file_id])
        self._cache[key] = table
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return table

    def identity_rows(
        self, file_id: int, indices: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        table = self._table("output", file_id, IDENTITY_KEYS)
        data_indices = table["index"].combine_chunks().to_numpy(zero_copy_only=False)
        offsets = np.searchsorted(data_indices, indices)
        valid = offsets < len(data_indices)
        if valid.any():
            valid_positions = np.flatnonzero(valid)
            valid[valid_positions] &= data_indices[offsets[valid_positions]] == indices[valid_positions]
        arrays = {
            key: table[key].combine_chunks().to_numpy(zero_copy_only=False)
            for key in IDENTITY_KEYS
        }
        return offsets, arrays

    def read_window(
        self, root_kind: str, start_index: int, count: int, columns: list[str]
    ) -> pa.Table:
        pieces: list[pa.Table] = []
        current = start_index
        remaining = count
        while remaining:
            file_id = int(np.searchsorted(self._end_array, current, side="left"))
            if file_id >= len(self.records) or current < self.starts[file_id]:
                break
            table = self._table(root_kind, file_id, columns)
            indices = table["index"].combine_chunks().to_numpy(zero_copy_only=False)
            offset = int(np.searchsorted(indices, current))
            if offset >= len(indices) or int(indices[offset]) != current:
                break
            take = min(remaining, len(table) - offset)
            pieces.append(table.slice(offset, take))
            current += take
            remaining -= take
        if not pieces:
            return pa.table({key: [] for key in columns})
        return pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]


def _compare_manifest_group(
    batch: pa.RecordBatch,
    positions: np.ndarray,
    file_id: int,
    catalog: DataFiles,
    differences: Differences,
) -> None:
    wanted_indices = batch.column("index").to_numpy(zero_copy_only=False)[positions]
    offsets, arrays = catalog.identity_rows(file_id, wanted_indices)
    if np.any(offsets >= len(arrays["index"])):
        differences.add(
            "manifest.index",
            "manifest index 在派生数据中不存在",
            file_id=file_id,
        )
        return
    actual_indices = arrays["index"][offsets]
    present = actual_indices == wanted_indices
    if not np.all(present):
        differences.add(
            "manifest.index",
            "manifest index 在派生数据中不存在",
            file_id=file_id,
            first_index=int(wanted_indices[np.flatnonzero(~present)[0]]),
        )
        return
    for key in IDENTITY_KEYS:
        expected = batch.column(key).to_numpy(zero_copy_only=False)[positions]
        actual = arrays[key][offsets]
        if not np.array_equal(actual, expected):
            mismatch = int(np.flatnonzero(actual != expected)[0])
            differences.add(
                f"manifest.{key}",
                "manifest 身份字段与派生 parquet 不一致",
                index=int(wanted_indices[mismatch]),
                expected=_native(expected[mismatch]),
                actual=_native(actual[mismatch]),
            )


def audit_manifest(
    path: Path,
    catalog: DataFiles,
    provenance: dict[str, Any],
    seed: int,
    sample_count: int,
    batch_size: int,
    differences: Differences,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != ANCHOR_SCHEMA:
        differences.add(
            "manifest.schema",
            "manifest schema 必须与 recipe v3 schema 完全一致",
            expected=str(ANCHOR_SCHEMA),
            actual=str(parquet.schema_arrow),
        )
    ordinary = PrioritySample(sample_count, np.random.default_rng(seed))
    episode_boundaries = PrioritySample(sample_count, np.random.default_rng(seed + 1))
    parquet_boundaries = PrioritySample(sample_count, np.random.default_rng(seed + 2))
    previous_index: int | None = None
    current_episode: int | None = None
    episode_first: dict[str, Any] | None = None
    episode_last: dict[str, Any] | None = None
    file_first: dict[int, dict[str, Any]] = {}
    file_last: dict[int, dict[str, Any]] = {}
    row_count = 0

    for batch in parquet.iter_batches(batch_size=batch_size, columns=ANCHOR_SCHEMA.names):
        indices = batch.column("index").to_numpy(zero_copy_only=False)
        if len(indices):
            if previous_index is not None and int(indices[0]) <= previous_index:
                differences.add(
                    "manifest.order",
                    "manifest index 非严格递增或重复",
                    previous=previous_index,
                    current=int(indices[0]),
                    row=row_count,
                )
            bad = np.flatnonzero(np.diff(indices) <= 0)
            if bad.size:
                offset = int(bad[0])
                differences.add(
                    "manifest.order",
                    "manifest index 非严格递增或重复",
                    previous=int(indices[offset]),
                    current=int(indices[offset + 1]),
                    row=row_count + offset + 1,
                )
            previous_index = int(indices[-1])

        file_ids = catalog.file_ids(indices)
        invalid = (file_ids >= len(catalog.records))
        valid_file_positions = np.flatnonzero(~invalid)
        if invalid.any():
            differences.add(
                "manifest.index",
                "manifest index 超出 provenance 数据文件范围",
                index=int(indices[np.flatnonzero(invalid)[0]]),
            )
        for file_id in np.unique(file_ids[valid_file_positions]):
            positions = np.flatnonzero(file_ids == file_id)
            if np.any(indices[positions] < catalog.starts[int(file_id)]):
                differences.add(
                    "manifest.index",
                    "manifest index 落在 parquet 文件间隙",
                    index=int(indices[positions][0]),
                )
                continue
            _compare_manifest_group(batch, positions, int(file_id), catalog, differences)
            file_first.setdefault(int(file_id), _row(batch, int(positions[0])))
            file_last[int(file_id)] = _row(batch, int(positions[-1]))

        ordinary.offer_batch(batch)
        episodes = batch.column("episode_index").to_numpy(zero_copy_only=False)
        if len(batch):
            boundaries = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1, len(batch)]
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                first = _row(batch, int(start))
                last = _row(batch, int(stop - 1))
                episode = int(episodes[start])
                if current_episode is None:
                    current_episode, episode_first = episode, first
                elif episode != current_episode:
                    assert episode_first is not None and episode_last is not None
                    episode_boundaries.offer(episode_first)
                    if episode_last["index"] != episode_first["index"]:
                        episode_boundaries.offer(episode_last)
                    current_episode, episode_first = episode, first
                episode_last = last
        row_count += len(batch)

    if episode_first is not None and episode_last is not None:
        episode_boundaries.offer(episode_first)
        if episode_last["index"] != episode_first["index"]:
            episode_boundaries.offer(episode_last)
    for file_id in sorted(file_first):
        parquet_boundaries.offer(file_first[file_id])
        if file_last[file_id]["index"] != file_first[file_id]["index"]:
            parquet_boundaries.offer(file_last[file_id])

    expected_rows = int(provenance.get("anchor_count", -1))
    if row_count != expected_rows:
        differences.add(
            "manifest.row_count",
            "manifest 行数与 provenance anchor_count 不一致",
            expected=expected_rows,
            actual=row_count,
        )
    return (
        {
            "ordinary": ordinary.values(),
            "episode_boundary": episode_boundaries.values(),
            "parquet_boundary": parquet_boundaries.values(),
        },
        {"rows": row_count, "schema": str(parquet.schema_arrow)},
    )


def _tasks(root: Path, differences: Differences) -> dict[int, str]:
    path = root / "meta/tasks.parquet"
    table = pq.read_table(path)
    text_key = "task" if "task" in table.column_names else "__index_level_0__"
    result: dict[int, str] = {}
    for index, text in zip(
        table["task_index"].to_pylist(), table[text_key].to_pylist(), strict=True
    ):
        task_index = int(index)
        if task_index in result:
            differences.add("task.duplicate", "canonical task_index 重复", task_index=task_index)
        result[task_index] = "" if text is None else str(text).strip()
    return result


def _array(table: pa.Table, key: str, width: int) -> np.ndarray:
    return _vector_column(table, key, width).astype(np.float32, copy=False)


def audit_samples(
    samples: dict[str, list[dict[str, Any]]],
    catalog: DataFiles,
    tasks: dict[int, str],
    differences: Differences,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    categories_by_index: dict[int, set[str]] = {}
    anchors: dict[int, dict[str, Any]] = {}
    for category, rows in samples.items():
        for row in rows:
            index = int(row["index"])
            anchors[index] = row
            categories_by_index.setdefault(index, set()).add(category)
    audited: list[dict[str, Any]] = []
    anchors_by_episode: dict[int, list[dict[str, Any]]] = {}
    derived_columns = [*SAMPLE_COLUMNS, EE_POSE_KEY]

    for index in sorted(anchors):
        anchor = anchors[index]
        categories = sorted(categories_by_index[index])
        before = differences.count
        try:
            source = catalog.read_window("source", index, HORIZON + 1, SAMPLE_COLUMNS)
            derived = catalog.read_window("output", index, HORIZON + 1, derived_columns)
            if len(source) != HORIZON + 1 or len(derived) != HORIZON + 1:
                differences.add(
                    "sample.padding",
                    "anchor 没有完整的 15-step action 和 t+15 goal，疑似 padding",
                    index=index,
                    source_rows=len(source),
                    derived_rows=len(derived),
                )
                continue

            source_ids = {
                key: source[key].combine_chunks().to_numpy(zero_copy_only=False)
                for key in IDENTITY_KEYS
            }
            derived_ids = {
                key: derived[key].combine_chunks().to_numpy(zero_copy_only=False)
                for key in IDENTITY_KEYS
            }
            for key in IDENTITY_KEYS:
                if not np.array_equal(source_ids[key], derived_ids[key]):
                    differences.add(
                        f"sample.{key}",
                        "源/派生窗口身份字段不一致",
                        index=index,
                    )
            if not np.all(source_ids["episode_index"] == source_ids["episode_index"][0]):
                differences.add("sample.episode", "anchor 窗口跨 episode", index=index)
            if not np.all(np.diff(source_ids["index"]) == 1):
                differences.add("sample.index", "anchor 窗口 index 不连续", index=index)
            if not np.all(np.diff(source_ids["frame_index"]) == 1):
                differences.add("sample.frame_index", "anchor 窗口 frame 不连续", index=index)
            if not np.all(np.diff(source_ids["timestamp"]) > 0):
                differences.add("sample.timestamp", "anchor 窗口 timestamp 非递增", index=index)

            for key in IDENTITY_KEYS:
                actual = source_ids[key][0]
                expected = anchor[key]
                if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                    differences.add(
                        f"sample.anchor_{key}",
                        "manifest anchor 与源 parquet 不一致",
                        index=index,
                        expected=expected,
                        actual=actual,
                    )

            if not np.array_equal(
                _array(source.slice(0, 1), STATE_KEY, 8),
                _array(derived.slice(0, 1), STATE_KEY, 8),
            ):
                differences.add("sample.state", "state[t] 源/派生不一致", index=index)
            if not np.array_equal(
                _array(source.slice(0, HORIZON), ACTION_KEY, 8),
                _array(derived.slice(0, HORIZON), ACTION_KEY, 8),
            ):
                differences.add(
                    "sample.action", "action[t:t+15] 源/派生不一致", index=index
                )
            goal_source = source.slice(HORIZON, 1)
            expected_pose = build_ee_pose(
                _array(goal_source, CARTESIAN_KEY, 6),
                goal_source[GRIPPER_KEY].combine_chunks().to_numpy(zero_copy_only=False),
            )
            actual_pose = _array(derived.slice(HORIZON, 1), EE_POSE_KEY, 7)
            if not np.array_equal(actual_pose, expected_pose):
                differences.add(
                    "sample.ee_pose",
                    "ee_pose[t+15] 与源 Cartesian+gripper 转换不一致",
                    index=index,
                )

            source_success = source["is_episode_successful"].to_numpy(zero_copy_only=False)
            derived_success = derived["is_episode_successful"].to_numpy(zero_copy_only=False)
            if not np.array_equal(source_success, derived_success) or not np.all(source_success):
                differences.add(
                    "sample.success",
                    "anchor success 标记不为真或源/派生不一致",
                    index=index,
                )
            task_index = int(source_ids["task_index"][0])
            if task_index != int(anchor["task_index"]) or not tasks.get(task_index, ""):
                differences.add(
                    "sample.canonical_task",
                    "anchor 未解析到非空 canonical task",
                    index=index,
                    task_index=task_index,
                )
            episode = int(source_ids["episode_index"][0])
            anchors_by_episode.setdefault(episode, []).append(anchor)
        except Exception as error:
            differences.add(
                "sample.exception",
                "直接源/派生校验失败",
                index=index,
                error=str(error),
            )
        audited.append(
            {
                "index": index,
                "categories": categories,
                "passed": differences.count == before,
            }
        )
    return audited, anchors_by_episode


def _episode_metadata(
    root: Path,
    records: list[dict[str, Any]],
    episode_ids: set[int],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    columns = ["episode_index"]
    for camera in CAMERA_KEYS:
        columns.extend(
            [
                f"videos/{camera}/chunk_index",
                f"videos/{camera}/file_index",
                f"videos/{camera}/from_timestamp",
                f"videos/{camera}/to_timestamp",
            ]
        )
    for record in records:
        table = pq.read_table(root / record["path"], columns=columns)
        episodes = table["episode_index"].to_numpy(zero_copy_only=False)
        positions = np.flatnonzero(np.isin(episodes, list(episode_ids)))
        for position in positions:
            row = table.slice(int(position), 1).to_pylist()[0]
            result[int(row["episode_index"])] = row
        if len(result) == len(episode_ids):
            break
    return result


def audit_videos(
    source_root: Path,
    output_root: Path,
    provenance: dict[str, Any],
    anchors_by_episode: dict[int, list[dict[str, Any]]],
    differences: Differences,
) -> dict[str, Any]:
    info = json.loads((output_root / "meta/info.json").read_text(encoding="utf-8"))
    video_keys = tuple(
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    )
    if set(video_keys) != set(CAMERA_KEYS):
        differences.add(
            "video.camera_keys",
            "数据集必须恰好包含预期三路相机",
            expected=list(CAMERA_KEYS),
            actual=list(video_keys),
        )
    template = info.get("video_path")
    if not isinstance(template, str):
        differences.add("video.template", "meta/info.json 缺少 video_path")
        return {"episodes": 0, "files_checked": 0}
    inventory_files: set[str] = set()
    inventory_episodes = 0
    for record in provenance["episode_metadata_files"]:
        metadata = pq.read_table(
            output_root / record["path"],
            columns=[
                "episode_index",
                *[
                    field
                    for camera in CAMERA_KEYS
                    for field in (
                        f"videos/{camera}/chunk_index",
                        f"videos/{camera}/file_index",
                        f"videos/{camera}/from_timestamp",
                        f"videos/{camera}/to_timestamp",
                    )
                ],
            ],
        )
        inventory_episodes += len(metadata)
        for camera in CAMERA_KEYS:
            prefix = f"videos/{camera}"
            chunks = metadata[f"{prefix}/chunk_index"].to_numpy(zero_copy_only=False)
            files = metadata[f"{prefix}/file_index"].to_numpy(zero_copy_only=False)
            starts = metadata[f"{prefix}/from_timestamp"].to_numpy(zero_copy_only=False)
            stops = metadata[f"{prefix}/to_timestamp"].to_numpy(zero_copy_only=False)
            invalid_times = np.flatnonzero(
                ~np.isfinite(starts) | ~np.isfinite(stops) | (stops < starts)
            )
            if invalid_times.size:
                differences.add(
                    "video.metadata_time",
                    "视频 metadata 时间范围非法",
                    camera=camera,
                    episode_index=int(metadata["episode_index"][int(invalid_times[0])].as_py()),
                )
            for chunk, file_index in np.unique(
                np.column_stack((chunks, files)), axis=0
            ):
                relative = template.format(
                    video_key=camera,
                    chunk_index=int(chunk),
                    file_index=int(file_index),
                )
                inventory_files.add(relative)
    missing_inventory = [
        relative
        for relative in sorted(inventory_files)
        if not (output_root / relative).is_file()
    ]
    for relative in missing_inventory[:50]:
        differences.add(
            "video.inventory_file",
            "episode metadata 引用的视频文件不存在",
            path=relative,
        )

    episode_ids = set(anchors_by_episode)
    source_meta = _episode_metadata(
        source_root, provenance["episode_metadata_files"], episode_ids
    )
    output_meta = _episode_metadata(
        output_root, provenance["episode_metadata_files"], episode_ids
    )
    files_checked = 0
    fps = float(info.get("fps", 15))
    for episode in sorted(episode_ids):
        source_row = source_meta.get(episode)
        output_row = output_meta.get(episode)
        if source_row is None or output_row is None:
            differences.add(
                "video.episode_metadata",
                "采样 episode 缺少 metadata",
                episode_index=episode,
            )
            continue
        for camera in CAMERA_KEYS:
            prefix = f"videos/{camera}"
            fields = (
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            )
            if any(source_row.get(key) != output_row.get(key) for key in fields):
                differences.add(
                    "video.metadata_pairing",
                    "源/派生 episode 视频 metadata 不一致",
                    episode_index=episode,
                    camera=camera,
                )
            try:
                relative = template.format(
                    video_key=camera,
                    chunk_index=int(output_row[fields[0]]),
                    file_index=int(output_row[fields[1]]),
                )
            except Exception as error:
                differences.add(
                    "video.path",
                    "无法按 video_path 构造视频路径",
                    episode_index=episode,
                    camera=camera,
                    error=str(error),
                )
                continue
            for root_kind, root in (("source", source_root), ("output", output_root)):
                if not (root / relative).is_file():
                    differences.add(
                        "video.file",
                        "三相机视频文件不存在",
                        root=root_kind,
                        path=relative,
                        episode_index=episode,
                    )
                files_checked += 1
            start = float(output_row[fields[2]])
            stop = float(output_row[fields[3]])
            for anchor in anchors_by_episode[episode]:
                shifted = start + float(anchor["timestamp"])
                if shifted < start - 1e-7 or shifted > stop + 0.5 / fps:
                    differences.add(
                        "video.timestamp_coverage",
                        "episode metadata 时间范围未覆盖 anchor timestamp",
                        episode_index=episode,
                        camera=camera,
                        anchor_index=int(anchor["index"]),
                        shifted_timestamp=shifted,
                        from_timestamp=start,
                        to_timestamp=stop,
                    )
    return {
        "sampled_episodes": len(episode_ids),
        "sampled_files_checked": files_checked,
        "inventory_episodes": inventory_episodes,
        "inventory_unique_files": len(inventory_files),
        "inventory_missing_files": len(missing_inventory),
    }


def openpi_nonidle_ranges(
    joint_velocities: np.ndarray,
    *,
    joint_velocity_delta: float = 1e-3,
    min_idle_len: int = 7,
    min_non_idle_len: int = 16,
    trim_last_steps: int = 10,
) -> list[tuple[int, int]]:
    """Independent transcription of openpi@c23745b5's DROID range algorithm."""

    velocities = np.asarray(joint_velocities, dtype=np.float32)
    if velocities.ndim != 2 or velocities.shape[1] != 7:
        raise ValueError(f"expected joint velocity shape (N, 7), got {velocities.shape}")
    is_idle = np.hstack(
        [
            np.asarray([False]),
            np.all(
                np.abs(velocities[1:] - velocities[:-1]) < joint_velocity_delta,
                axis=1,
            ),
        ]
    )
    idle_padded = np.concatenate([[False], is_idle, [False]])
    idle_diff = np.diff(idle_padded.astype(int))
    idle_starts = np.where(idle_diff == 1)[0]
    idle_ends = np.where(idle_diff == -1)[0]
    long_idle = (idle_ends - idle_starts) >= min_idle_len
    keep = np.ones(len(velocities), dtype=bool)
    for start, end in zip(idle_starts[long_idle], idle_ends[long_idle], strict=True):
        keep[start:end] = False

    keep_padded = np.concatenate([[False], keep, [False]])
    keep_diff = np.diff(keep_padded.astype(int))
    keep_starts = np.where(keep_diff == 1)[0]
    keep_ends = np.where(keep_diff == -1)[0]
    long_motion = (keep_ends - keep_starts) >= min_non_idle_len
    return [
        (int(start), int(end) - trim_last_steps)
        for start, end in zip(
            keep_starts[long_motion], keep_ends[long_motion], strict=True
        )
    ]


def _ranges_to_anchor_mask(
    ranges: list[tuple[int, int]], episode_length: int, horizon: int = HORIZON
) -> np.ndarray:
    mask = np.zeros(max(0, episode_length - horizon), dtype=bool)
    for start, end in ranges:
        clipped_end = min(end, len(mask))
        if clipped_end > start:
            mask[start:clipped_end] = True
    return mask


def _has_qualifying_idle(velocities: np.ndarray, config: NonIdleConfig) -> bool:
    if len(velocities) < 2:
        return False
    idle = np.hstack(
        [
            np.asarray([False]),
            np.all(
                np.abs(velocities[1:] - velocities[:-1])
                < config.joint_velocity_delta,
                axis=1,
            ),
        ]
    )
    padded = np.concatenate([[False], idle, [False]])
    differences = np.diff(padded.astype(int))
    starts = np.where(differences == 1)[0]
    ends = np.where(differences == -1)[0]
    return bool(np.any(ends - starts >= config.min_idle_len))


def audit_openpi_parity(
    output_root: Path,
    plans: list[FilePlan],
    config: NonIdleConfig,
    requested_episodes: int,
    differences: Differences,
) -> dict[str, Any]:
    checked = 0
    columns = ["episode_index", ACTION_JOINT_VELOCITY_KEY]
    for table in _episode_tables(output_root, plans, columns):
        velocities = _vector_column(table, ACTION_JOINT_VELOCITY_KEY, 7).astype(
            np.float32, copy=False
        )
        if not _has_qualifying_idle(velocities, config):
            continue
        episode = int(table["episode_index"][0].as_py())
        upstream_ranges = openpi_nonidle_ranges(
            velocities,
            joint_velocity_delta=config.joint_velocity_delta,
            min_idle_len=config.min_idle_len,
            min_non_idle_len=config.min_non_idle_len,
            trim_last_steps=10,
        )
        adapted_ranges = openpi_nonidle_ranges(
            velocities,
            joint_velocity_delta=config.joint_velocity_delta,
            min_idle_len=config.min_idle_len,
            min_non_idle_len=config.min_non_idle_len,
            trim_last_steps=15,
        )
        upstream = _ranges_to_anchor_mask(upstream_ranges, len(velocities))
        adapted = _ranges_to_anchor_mask(adapted_ranges, len(velocities))
        local = non_idle_anchor_mask(velocities, config, HORIZON)
        if not np.array_equal(local, adapted):
            differences.add(
                "parity.local",
                "本地 non_idle_anchor_mask 与独立 OpenPI trim15 复刻不一致",
                episode_index=episode,
            )
        expected_trim_only = upstream & ~adapted
        actual_difference = upstream ^ local
        if not np.array_equal(actual_difference, expected_trim_only):
            differences.add(
                "parity.trim",
                "OpenPI 与本地差异不只来自 trim10 到 trim15",
                episode_index=episode,
            )
        checked += 1
        if checked >= requested_episodes:
            break
    if checked < requested_episodes:
        differences.add(
            "parity.coverage",
            "含 qualifying idle 的 parity episode 数不足",
            requested=requested_episodes,
            checked=checked,
        )
    return {
        "requested_idle_episodes": requested_episodes,
        "checked_idle_episodes": checked,
        "upstream_trim": 10,
        "local_trim": 15,
    }


def run_audit(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    seed: int = 20260719,
    samples_per_category: int = 32,
    parity_episodes: int = 100,
    manifest_batch_size: int = 65_536,
) -> dict[str, Any]:
    """Run all audits and return a JSON-serializable report."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    differences = Differences()
    report: dict[str, Any] = {
        "audit": "droid_goal_prior_dataset",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": seed,
        "parameters": {
            "samples_per_category": samples_per_category,
            "parity_episodes": parity_episodes,
            "manifest_batch_size": manifest_batch_size,
        },
    }
    try:
        provenance_path = output_root / "goal_pose_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("recipe_version") != RECIPE_VERSION:
            differences.add(
                "provenance.recipe_version",
                "只支持 recipe v3",
                actual=provenance.get("recipe_version"),
            )
        config_dict = provenance.get("config", {})
        if int(config_dict.get("horizon", -1)) != HORIZON:
            differences.add(
                "provenance.horizon",
                "provenance horizon 不是 15",
                actual=config_dict.get("horizon"),
            )
        if Path(config_dict.get("source_root", "")).resolve() != source_root:
            differences.add(
                "provenance.source_root",
                "CLI source-root 与 provenance 不一致",
                provenance=config_dict.get("source_root"),
            )
        if Path(config_dict.get("output_root", "")).resolve() != output_root:
            differences.add(
                "provenance.output_root",
                "CLI output-root 与 provenance 不一致",
                provenance=config_dict.get("output_root"),
            )
        manifest_path = output_root / "valid_anchor_indices.parquet"
        expected_manifest_hash = provenance.get("artifact_sha256", {}).get(
            "valid_anchor_indices.parquet"
        )
        if expected_manifest_hash and _sha256(manifest_path) != expected_manifest_hash:
            differences.add(
                "provenance.manifest_hash",
                "valid_anchor_indices.parquet 哈希与 provenance 不一致",
            )
        catalog = DataFiles(source_root, output_root, provenance["files"], differences)
        samples, manifest_report = audit_manifest(
            manifest_path,
            catalog,
            provenance,
            seed,
            samples_per_category,
            manifest_batch_size,
            differences,
        )
        report["manifest"] = manifest_report
        report["sample_selection"] = {
            category: len(rows) for category, rows in samples.items()
        }
        tasks = _tasks(output_root, differences)
        audited, anchors_by_episode = audit_samples(samples, catalog, tasks, differences)
        report["samples"] = {
            "unique_anchors": len(audited),
            "passed": sum(bool(item["passed"]) for item in audited),
            "anchors": audited,
        }
        report["videos"] = audit_videos(
            source_root, output_root, provenance, anchors_by_episode, differences
        )
        non_idle = NonIdleConfig(**config_dict["non_idle"])
        plans = [
            FilePlan(record["path"], int(record["output_rows"]))
            for record in provenance["files"]
        ]
        report["openpi_parity"] = audit_openpi_parity(
            output_root, plans, non_idle, parity_episodes, differences
        )
        report["provenance"] = {
            "recipe_version": provenance.get("recipe_version"),
            "anchor_count": provenance.get("anchor_count"),
            "non_idle": asdict(non_idle),
        }
    except Exception as error:
        differences.add("audit.fatal", "审计无法继续", error=str(error))

    report["status"] = "passed" if differences.count == 0 else "failed"
    report["difference_count"] = differences.count
    report["differences"] = differences.examples
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--samples-per-category", type=int, default=32)
    parser.add_argument("--parity-episodes", type=int, default=100)
    parser.add_argument("--manifest-batch-size", type=int, default=65_536)
    parser.add_argument(
        "--report",
        default="-",
        help="JSON report path, or '-' for stdout (default).",
    )
    args = parser.parse_args()
    if args.samples_per_category < 1:
        parser.error("--samples-per-category must be positive")
    if args.parity_episodes < 1:
        parser.error("--parity-episodes must be positive")
    if args.manifest_batch_size < 1:
        parser.error("--manifest-batch-size must be positive")
    return args


def main() -> int:
    args = _parse_args()
    report = run_audit(
        args.source_root,
        args.output_root,
        seed=args.seed,
        samples_per_category=args.samples_per_category,
        parity_episodes=args.parity_episodes,
        manifest_batch_size=args.manifest_batch_size,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report == "-":
        sys.stdout.write(encoded)
    else:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

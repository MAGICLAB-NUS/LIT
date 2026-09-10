#!/usr/bin/env python3
"""Visualize goal-pose labels / predictions on LIBERO agentview images.

Phases:
  gt      — project dataset GT goal + current EE (no policy)
  pred    — load a Stage2 checkpoint and overlay pred vs GT
  rollout — short closed-loop episode with predicted goal overlay

State layout (raw / denormalized): [xyz(3), axis-angle(3), gripper(2)].
Dataset frames are raw; model pose decoder outputs q01/q99-normalized values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WS = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path("/data2/JM/dataset/libero_lerobot_format")
DEFAULT_REPO_ID = "local/libero_lerobot_format"
DEFAULT_CKPT = (
    WS
    / "lerobot/outputs/libero_goal_prior_v2b/seed_1000/stage2/checkpoints/020000/pretrained_model"
)
CAMERA_NAME = "agentview"
COLOR_CURRENT = (255, 90, 40)  # BGR blue-ish
COLOR_GT = (40, 200, 60)  # BGR green
COLOR_PRED = (40, 40, 255)  # BGR red


@dataclass
class CameraCache:
    height: int
    width: int
    transform: np.ndarray  # 4x4 world->pixel
    rotate180: bool


def denormalize_state(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse of QUANTILES normalize to [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    denom = np.maximum(q99 - q01, 1e-6)
    return (x + 1.0) * denom / 2.0 + q01


def normalize_state(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    denom = np.maximum(q99 - q01, 1e-6)
    return np.clip(2.0 * (x - q01) / denom - 1.0, -1.0, 1.0)


def load_state_quantiles(
    dataset_root: Path,
    stats_json: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stats_path = Path(stats_json) if stats_json is not None else dataset_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))["observation.state"]
    return np.asarray(stats["q01"], dtype=np.float64), np.asarray(stats["q99"], dtype=np.float64)


def xyz_for_overlay(
    xyz: np.ndarray,
    *,
    ref_z: float | None,
    xy_only: bool,
) -> np.ndarray:
    """Optionally replace Z with ref_z so overlays compare planar (XY) error only."""
    out = np.asarray(xyz, dtype=np.float64).reshape(-1)[:3].copy()
    if xy_only:
        if ref_z is None:
            raise ValueError("xy_only overlay requires ref_z")
        out[2] = float(ref_z)
    return out


def looks_normalized(state: np.ndarray) -> bool:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    return bool(np.all(np.abs(state) <= 1.05))


def to_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr)


def project_world_to_pixel(
    xyz: np.ndarray,
    transform: np.ndarray,
    height: int,
    width: int,
    *,
    rotate180: bool = False,
    clip: bool = True,
) -> tuple[int, int] | None:
    """Project world XYZ to (u, v) pixel coords (OpenCV: u=col, v=row)."""
    from robosuite.utils.camera_utils import project_points_from_world_to_camera

    pts = np.asarray(xyz, dtype=np.float64).reshape(1, 3)
    # returns (row, col)
    pixels = project_points_from_world_to_camera(pts, transform, height, width)[0]
    row, col = int(pixels[0]), int(pixels[1])
    if rotate180:
        row = height - 1 - row
        col = width - 1 - col
    if clip:
        if not (0 <= col < width and 0 <= row < height):
            return None
        col = int(np.clip(col, 0, width - 1))
        row = int(np.clip(row, 0, height - 1))
    return col, row


def draw_marker(
    image_bgr: np.ndarray,
    uv: tuple[int, int] | None,
    color: tuple[int, int, int],
    label: str,
    radius: int = 6,
) -> None:
    import cv2

    if uv is None:
        return
    u, v = uv
    cv2.circle(image_bgr, (u, v), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image_bgr, (u, v), radius + 2, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(
        image_bgr,
        label,
        (u + 8, v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_axes(
    image_bgr: np.ndarray,
    origin_xyz: np.ndarray,
    axis_angle: np.ndarray,
    transform: np.ndarray,
    height: int,
    width: int,
    *,
    rotate180: bool,
    scale: float = 0.05,
) -> None:
    import cv2

    rot = axis_angle_to_matrix(axis_angle)
    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # x,y,z in BGR-ish
    o_uv = project_world_to_pixel(origin, transform, height, width, rotate180=rotate180)
    if o_uv is None:
        return
    for axis_idx, color in enumerate(colors):
        tip = origin + scale * rot[:, axis_idx]
        t_uv = project_world_to_pixel(tip, transform, height, width, rotate180=rotate180)
        if t_uv is None:
            continue
        cv2.line(image_bgr, o_uv, t_uv, color, 2, cv2.LINE_AA)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def draw_overlay(
    image: np.ndarray,
    *,
    current_xyz: np.ndarray | None,
    gt_xyz: np.ndarray | None,
    pred_xyz: np.ndarray | None,
    transform: np.ndarray,
    rotate180: bool,
    current_aa: np.ndarray | None = None,
    gt_aa: np.ndarray | None = None,
    pred_aa: np.ndarray | None = None,
    title: str = "",
) -> np.ndarray:
    import cv2

    image_bgr = cv2.cvtColor(to_hwc_uint8(image), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    if current_xyz is not None:
        uv = project_world_to_pixel(current_xyz, transform, h, w, rotate180=rotate180)
        draw_marker(image_bgr, uv, COLOR_CURRENT, "cur")
        if current_aa is not None:
            draw_axes(image_bgr, current_xyz, current_aa, transform, h, w, rotate180=rotate180)
    if gt_xyz is not None:
        uv = project_world_to_pixel(gt_xyz, transform, h, w, rotate180=rotate180)
        draw_marker(image_bgr, uv, COLOR_GT, "gt")
        if gt_aa is not None:
            draw_axes(image_bgr, gt_xyz, gt_aa, transform, h, w, rotate180=rotate180)
    if pred_xyz is not None:
        uv = project_world_to_pixel(pred_xyz, transform, h, w, rotate180=rotate180)
        draw_marker(image_bgr, uv, COLOR_PRED, "pred")
        if pred_aa is not None:
            draw_axes(image_bgr, pred_xyz, pred_aa, transform, h, w, rotate180=rotate180)
    if title:
        cv2.putText(
            image_bgr,
            title[:90],
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_task_language_map() -> dict[str, tuple[str, int, str]]:
    from libero.libero.benchmark import get_benchmark_dict

    mapping: dict[str, tuple[str, int, str]] = {}
    for suite_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        suite = get_benchmark_dict()[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            key = str(task.language).strip().lower()
            mapping[key] = (suite_name, task_id, task.name)
    return mapping


def ensure_libero_filesystem() -> Path | None:
    """If ~/.libero points at missing trees, temporarily retarget to a known-good root."""
    good = Path("/data2/JM/Code/MMaDA-VLA-main/LIBERO/libero/libero")
    if not (good / "bddl_files" / "libero_spatial").is_dir():
        return None
    cfg_path = Path.home() / ".libero" / "config.yaml"
    try:
        from libero.libero import get_libero_path

        current = Path(get_libero_path("bddl_files"))
        if current.is_dir() and (current / "libero_spatial").is_dir():
            return None
    except Exception:  # noqa: BLE001
        pass
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "\n".join(
            [
                f"benchmark_root: {good}",
                f"bddl_files: {good / 'bddl_files'}",
                f"init_states: {good / 'init_files'}",
                "datasets: /data2/JM/Code/MMaDA-VLA-main/dataset/libero",
                f"assets: {good / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[libero] retargeted config -> {good}", flush=True)
    return good


def resolve_bddl_root(explicit: Path | None = None) -> Path:
    """Find a usable LIBERO bddl_files root (local configs may point at stale paths)."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_root = os.environ.get("LIBERO_BDDL_ROOT") or os.environ.get("LIBERO_BDDL_FILES")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            Path("/data2/JM/Code/MMaDA-VLA-main/LIBERO/libero/libero/bddl_files"),
            Path("/data2/JM/MMaDA-VLA-main/LIBERO/libero/libero/bddl_files"),
            Path(
                "/data2/JM/Code/molmo_serious/molmoact2-main/third_party/"
                "LIBERO/libero/libero/bddl_files"
            ),
        ]
    )
    try:
        from libero.libero import get_libero_path

        candidates.append(Path(get_libero_path("bddl_files")))
    except Exception:  # noqa: BLE001
        pass
    for candidate in candidates:
        probe = candidate / "libero_spatial"
        if candidate.is_dir() and probe.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate LIBERO bddl_files. Pass --bddl-root or set LIBERO_BDDL_ROOT."
    )


def resolve_task_bddl(
    suite_name: str,
    task_id: int,
    *,
    bddl_root: Path | None = None,
) -> tuple[Any, Path]:
    from libero.libero.benchmark import get_benchmark_dict

    suite = get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    root = resolve_bddl_root(bddl_root)
    path = root / task.problem_folder / task.bddl_file
    if not path.exists():
        raise FileNotFoundError(f"Missing BDDL file: {path}")
    return task, path


def get_agentview_camera(
    suite_name: str,
    task_id: int,
    *,
    height: int = 256,
    width: int = 256,
    cache: dict[tuple[str, int], CameraCache] | None = None,
    rotate180: bool | None = None,
    bddl_root: Path | None = None,
) -> CameraCache:
    key = (suite_name, int(task_id))
    if cache is not None and key in cache:
        return cache[key]

    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    os.environ.setdefault("MUJOCO_GL", "egl")
    _task, bddl = resolve_task_bddl(suite_name, task_id, bddl_root=bddl_root)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=height,
        camera_widths=width,
    )
    try:
        env.reset()
        sim = env.env.sim
        transform = get_camera_transform_matrix(
            sim=sim,
            camera_name=CAMERA_NAME,
            camera_height=height,
            camera_width=width,
        ).astype(np.float64)
    finally:
        env.close()

    cam = CameraCache(
        height=height,
        width=width,
        transform=transform,
        rotate180=True if rotate180 is None else bool(rotate180),
    )
    if cache is not None:
        cache[key] = cam
    return cam


def choose_rotate180(
    image: np.ndarray,
    current_xyz: np.ndarray,
    transform: np.ndarray,
) -> bool:
    """Pick the convention that puts current EE nearer the lower-central image region."""
    h, w = to_hwc_uint8(image).shape[:2]
    scores = []
    for rotate in (False, True):
        uv = project_world_to_pixel(current_xyz, transform, h, w, rotate180=rotate, clip=False)
        if uv is None:
            scores.append((-1e9, rotate))
            continue
        u, v = uv
        in_frame = 0 <= u < w and 0 <= v < h
        # Prefer lower-central (gripper often in lower half for agentview).
        target_u, target_v = w * 0.5, h * 0.65
        dist = math.hypot(u - target_u, v - target_v)
        score = (1000.0 if in_frame else 0.0) - dist
        scores.append((score, rotate))
    scores.sort(reverse=True)
    return bool(scores[0][1])


def save_grid(images: list[np.ndarray], path: Path, cols: int = 4) -> None:
    import cv2

    if not images:
        return
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    h, w = images[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        rgb = to_hwc_uint8(img)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def stratified_indices(dataset, num_samples: int, seed: int, task_map: dict[str, tuple[str, int, str]]) -> list[int]:
    """Sample frames covering as many suites/tasks as possible."""
    rng = np.random.default_rng(seed)
    # Invert task language -> suite for episode lookup via dataset[i]['task'].
    by_suite: dict[str, list[int]] = {
        "libero_spatial": [],
        "libero_object": [],
        "libero_goal": [],
        "libero_10": [],
    }
    # Probe a large random subset once.
    probe_n = min(len(dataset), max(4000, num_samples * 200))
    candidates = rng.choice(len(dataset), size=probe_n, replace=False)
    for idx in candidates.tolist():
        item = dataset[int(idx)]
        is_pad = item.get("observation.state_is_pad")
        if is_pad is not None and bool(np.asarray(is_pad)[-1]):
            continue
        task = str(item.get("task", "")).strip().lower()
        meta = task_map.get(task)
        if meta is None:
            continue
        suite_name = meta[0]
        bucket = by_suite.get(suite_name)
        if bucket is not None and len(bucket) < max(8, num_samples):
            bucket.append(int(idx))
        if all(len(v) >= max(4, num_samples // 4) for v in by_suite.values()):
            break

    selected: list[int] = []
    suites = [s for s, v in by_suite.items() if v]
    if not suites:
        return list(range(min(num_samples, len(dataset))))
    while len(selected) < num_samples and any(by_suite[s] for s in suites):
        for suite_name in suites:
            bucket = by_suite[suite_name]
            if not bucket:
                continue
            pick = int(bucket.pop(int(rng.integers(0, len(bucket)))))
            selected.append(pick)
            if len(selected) >= num_samples:
                break
    return selected


def ensure_lerobot_on_path() -> None:
    src = WS / "lerobot" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def make_dataset(dataset_root: Path, repo_id: str, chunk_size: int):
    ensure_lerobot_on_path()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = 10.0
    return LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_root),
        delta_timestamps={"observation.state": [0.0, float(chunk_size) / fps]},
        video_backend="pyav",
    )


def extract_states(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(item["observation.state"], dtype=np.float64)
    if state.ndim == 1:
        raise ValueError("Expected observation.state with time axis [current, goal].")
    return state[0].copy(), state[-1].copy()


def maybe_denorm(state: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    if looks_normalized(state):
        return denormalize_state(state, q01, q99)
    return np.asarray(state, dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cmd_gt(args: argparse.Namespace) -> None:
    import cv2

    dataset = make_dataset(args.dataset_root, args.repo_id, args.chunk_size)
    q01, q99 = load_state_quantiles(args.dataset_root)
    task_map = build_task_language_map()
    cam_cache: dict[tuple[str, int], CameraCache] = {}
    indices = stratified_indices(dataset, args.num_samples, args.seed, task_map)

    overlays: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    singles = out_dir / "singles"
    singles.mkdir(exist_ok=True)

    for sample_i, idx in enumerate(indices):
        item = dataset[int(idx)]
        image = item["observation.images.image"]
        current_raw, goal_raw = extract_states(item)
        current = maybe_denorm(current_raw, q01, q99)
        goal = maybe_denorm(goal_raw, q01, q99)
        task = str(item.get("task", "")).strip()
        meta = task_map.get(task.lower())
        if meta is None:
            print(f"[gt] skip idx={idx}: unknown task {task!r}", flush=True)
            continue
        suite_name, task_id, task_name = meta
        cam = get_agentview_camera(
            suite_name,
            task_id,
            height=256,
            width=256,
            cache=cam_cache,
            rotate180=args.rotate180,
            bddl_root=args.bddl_root,
        )
        if args.rotate180 is None:
            cam.rotate180 = choose_rotate180(image, current[:3], cam.transform)
            cam_cache[(suite_name, task_id)] = cam

        title = f"{suite_name}#{task_id} idx={idx}"
        overlay = draw_overlay(
            image,
            current_xyz=current[:3],
            gt_xyz=goal[:3],
            pred_xyz=None,
            transform=cam.transform,
            rotate180=cam.rotate180,
            current_aa=current[3:6],
            gt_aa=goal[3:6],
            title=title,
        )
        overlays.append(overlay)
        single_path = singles / f"{sample_i:02d}_{suite_name}_{task_id}.png"
        cv2.imwrite(str(single_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        cur_uv = project_world_to_pixel(
            current[:3], cam.transform, cam.height, cam.width, rotate180=cam.rotate180, clip=False
        )
        gt_uv = project_world_to_pixel(
            goal[:3], cam.transform, cam.height, cam.width, rotate180=cam.rotate180, clip=False
        )
        rows.append(
            {
                "sample": sample_i,
                "dataset_index": idx,
                "suite": suite_name,
                "task_id": task_id,
                "task": task,
                "task_name": task_name,
                "rotate180": cam.rotate180,
                "current_xyz": " ".join(f"{x:.4f}" for x in current[:3]),
                "gt_xyz": " ".join(f"{x:.4f}" for x in goal[:3]),
                "delta_xyz_norm": float(np.linalg.norm(goal[:3] - current[:3])),
                "current_uv": "" if cur_uv is None else f"{cur_uv[0]},{cur_uv[1]}",
                "gt_uv": "" if gt_uv is None else f"{gt_uv[0]},{gt_uv[1]}",
            }
        )
        print(
            f"[gt] {sample_i+1}/{len(indices)} {suite_name}#{task_id} "
            f"rot180={cam.rotate180} Δxyz={rows[-1]['delta_xyz_norm']:.3f}",
            flush=True,
        )

    grid_path = out_dir / "gt_grid.png"
    save_grid(overlays, grid_path, cols=4)
    write_csv(out_dir / "gt_summary.csv", rows)
    print(f"[gt] wrote {grid_path} and {out_dir / 'gt_summary.csv'} ({len(rows)} samples)", flush=True)


def predict_goal_pose_from_policy(policy: Any, batch: dict[str, Any]) -> np.ndarray:
    """Decode Stage2 goal pose (normalized) for one batch using the training path."""
    import torch

    was_training = policy.training
    policy.eval()
    with torch.inference_mode():
        # Reuse flow-matching forward path to get semantic-visual hidden states.
        model_inputs = policy._model_inputs(batch)
        if policy._uses_semantic_visual_conditioning():
            flow_loss, hidden_states = policy._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction="mean",
            )
            del flow_loss
            num_pose = int(policy.config.num_semantic_visual_pose_tokens)
            pose_hidden = hidden_states[:, :num_pose, :]
            pose_hidden = policy.semantic_visual_pose_norm(pose_hidden)
            pred = policy.semantic_visual_pose_decoder(pose_hidden).float()
        else:
            # Force a backbone pass with goal tokens appended via the flow path as well.
            flow_loss, hidden_states = policy._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction="mean",
            )
            del flow_loss
            num_tokens = int(policy.config.num_goal_tokens)
            goal_hidden = hidden_states[:, -num_tokens:, :]
            pred = policy.goal_pose_decoder(goal_hidden).float()
    if was_training:
        policy.train()
    return pred.detach().cpu().numpy()


def build_policy_batch_from_dataset_item(item: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    batch: dict[str, Any] = {}
    # Images: dataset gives CHW float; policy processor usually expects that after packing.
    # Here we call policy.forward-style internals that expect already-processed model batch.
    # So for pred mode we go through the full preprocessor when available.
    image = item["observation.images.image"]
    image2 = item.get("observation.images.image2")
    state = item["observation.state"]
    if hasattr(state, "ndim") and int(state.ndim) == 2:
        # Keep time axis for goal extraction if preprocessor sees raw observation dict.
        pass
    batch["observation.images.image"] = image.unsqueeze(0) if image.ndim == 3 else image
    if image2 is not None:
        batch["observation.images.image2"] = image2.unsqueeze(0) if image2.ndim == 3 else image2
    batch[OBS_STATE] = state.unsqueeze(0) if state.ndim == 2 else state
    if "observation.state_is_pad" in item:
        pad = item["observation.state_is_pad"]
        batch["observation.state_is_pad"] = pad.unsqueeze(0) if pad.ndim == 1 else pad
    if ACTION in item or "action" in item:
        action = item.get(ACTION, item.get("action"))
        if action is not None:
            # Provide a dummy action chunk for flow loss path.
            if action.ndim == 1:
                action = action.unsqueeze(0)
            # Repeat to chunk length if needed.
            batch[ACTION] = action.unsqueeze(0)
    batch["task"] = [str(item.get("task", ""))]
    # Move tensors
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device=device)
    return batch


def prepare_train_like_batch(
    policy: Any,
    preprocessor: Any,
    item: dict[str, Any],
    *,
    device: str,
    chunk_size: int,
    q01: np.ndarray,
    q99: np.ndarray,
) -> dict[str, Any]:
    """Build a processor batch then inject normalized goal_pose for pose decode path."""
    import torch
    from lerobot.utils.constants import ACTION, OBS_STATE

    raw = {
        "observation.images.image": item["observation.images.image"],
        "task": str(item.get("task", "")),
    }
    if "observation.images.image2" in item:
        raw["observation.images.image2"] = item["observation.images.image2"]
    state = item["observation.state"]
    raw[OBS_STATE] = state
    if "observation.state_is_pad" in item:
        raw["observation.state_is_pad"] = item["observation.state_is_pad"]

    # Dummy action chunk required by continuous flow path.
    action = item.get("action")
    if action is None:
        action = torch.zeros(chunk_size, 7, dtype=torch.float32)
    else:
        action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).repeat(chunk_size, 1)
        elif action.shape[0] < chunk_size:
            pad = action[-1:].repeat(chunk_size - action.shape[0], 1)
            action = torch.cat([action, pad], dim=0)
        else:
            action = action[:chunk_size]
    raw[ACTION] = action
    raw["action_is_pad"] = torch.zeros(chunk_size, dtype=torch.bool)

    # Processor expects batch dimension.
    batched = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            batched[key] = value.unsqueeze(0)
        else:
            batched[key] = [value] if key == "task" else value

    processed = preprocessor(batched)
    # Ensure goal_pose exists (normalized). Dataset state is raw; pack step may already
    # have created goal_pose after normalize, but be defensive.
    if "goal_pose" not in processed or processed["goal_pose"] is None:
        current_raw, goal_raw = extract_states(item)
        goal_norm = normalize_state(goal_raw, q01, q99)
        processed["goal_pose"] = torch.as_tensor(goal_norm, dtype=torch.float32).unsqueeze(0)
        processed["goal_pose_is_pad"] = torch.zeros(1, dtype=torch.bool)

    for key, value in list(processed.items()):
        if torch.is_tensor(value):
            processed[key] = value.to(device=device)
    return processed


def load_policy_and_preprocessor(
    checkpoint: Path,
    device: str,
    *,
    dataset_root: Path,
    repo_id: str,
    stats_json: Path | None = None,
):
    ensure_lerobot_on_path()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.pretrained_path = str(checkpoint)
    cfg.device = device
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()
    if getattr(policy.config, "inference_action_mode", None) in (None, ""):
        policy.config.inference_action_mode = "continuous"
    dataset_stats = ds_meta.stats
    if stats_json is not None:
        # Keep vector-feature quantiles consistent with the checkpoint's training stats.
        override = json.loads(Path(stats_json).read_text(encoding="utf-8"))
        dataset_stats = dict(dataset_stats)
        for key in ("observation.state", "action"):
            if key in override:
                dataset_stats[key] = override[key]
        print(f"[pred] using stats override from {stats_json}", flush=True)
    preprocessor, _post = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset_stats,
    )
    return policy, preprocessor


def cmd_pred(args: argparse.Namespace) -> None:
    import cv2
    import torch

    dataset = make_dataset(args.dataset_root, args.repo_id, args.chunk_size)
    q01, q99 = load_state_quantiles(args.dataset_root, getattr(args, "stats_json", None))
    xy_only = bool(getattr(args, "xy_only", False))
    task_map = build_task_language_map()
    cam_cache: dict[tuple[str, int], CameraCache] = {}
    indices = stratified_indices(dataset, args.num_samples, args.seed, task_map)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    policy, preprocessor = load_policy_and_preprocessor(
        Path(args.checkpoint),
        device,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        stats_json=getattr(args, "stats_json", None),
    )

    overlays: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    singles = out_dir / "singles"
    singles.mkdir(exist_ok=True)

    for sample_i, idx in enumerate(indices):
        item = dataset[int(idx)]
        image = item["observation.images.image"]
        current_raw, goal_raw = extract_states(item)
        current = maybe_denorm(current_raw, q01, q99)
        goal = maybe_denorm(goal_raw, q01, q99)
        task = str(item.get("task", "")).strip()
        meta = task_map.get(task.lower())
        if meta is None:
            print(f"[pred] skip idx={idx}: unknown task {task!r}", flush=True)
            continue
        suite_name, task_id, task_name = meta
        cam = get_agentview_camera(
            suite_name,
            task_id,
            cache=cam_cache,
            rotate180=args.rotate180,
            bddl_root=args.bddl_root,
        )
        if args.rotate180 is None:
            cam.rotate180 = choose_rotate180(image, current[:3], cam.transform)
            cam_cache[(suite_name, task_id)] = cam

        batch = prepare_train_like_batch(
            policy,
            preprocessor,
            item,
            device=device,
            chunk_size=args.chunk_size,
            q01=q01,
            q99=q99,
        )
        try:
            pred_norm = predict_goal_pose_from_policy(policy, batch)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[pred] forward failed idx={idx}: {exc}", flush=True)
            continue
        pred = denormalize_state(pred_norm, q01, q99)
        err3d = float(np.linalg.norm(pred[:3] - goal[:3]))
        err_xy = float(np.linalg.norm(pred[:2] - goal[:2]))
        err_z = float(abs(pred[2] - goal[2]))

        # For xy-only viz: project pred/cur at GT height so marker gap = planar error.
        ref_z = float(goal[2])
        pred_draw = xyz_for_overlay(pred[:3], ref_z=ref_z, xy_only=xy_only)
        cur_draw = xyz_for_overlay(current[:3], ref_z=ref_z, xy_only=xy_only)
        gt_draw = goal[:3]

        pred_uv = project_world_to_pixel(
            pred_draw, cam.transform, cam.height, cam.width, rotate180=cam.rotate180, clip=False
        )
        gt_uv = project_world_to_pixel(
            gt_draw, cam.transform, cam.height, cam.width, rotate180=cam.rotate180, clip=False
        )
        pix_err = None
        if pred_uv is not None and gt_uv is not None:
            pix_err = float(math.hypot(pred_uv[0] - gt_uv[0], pred_uv[1] - gt_uv[1]))

        if xy_only:
            title = f"{suite_name}#{task_id} err_xy={err_xy:.3f}m (z ignored)"
        else:
            title = f"{suite_name}#{task_id} err3d={err3d:.3f}m"
        overlay = draw_overlay(
            image,
            current_xyz=cur_draw,
            gt_xyz=gt_draw,
            pred_xyz=pred_draw,
            transform=cam.transform,
            rotate180=cam.rotate180,
            current_aa=None if xy_only else current[3:6],
            gt_aa=None if xy_only else goal[3:6],
            pred_aa=None if xy_only else pred[3:6],
            title=title,
        )
        overlays.append(overlay)
        cv2.imwrite(
            str(singles / f"{sample_i:02d}_{suite_name}_{task_id}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        rows.append(
            {
                "sample": sample_i,
                "dataset_index": idx,
                "suite": suite_name,
                "task_id": task_id,
                "task": task,
                "xy_only": xy_only,
                "err3d": err3d,
                "err_xy": err_xy,
                "err_z": err_z,
                "err_pix": "" if pix_err is None else pix_err,
                "pred_xyz": " ".join(f"{x:.4f}" for x in pred[:3]),
                "gt_xyz": " ".join(f"{x:.4f}" for x in goal[:3]),
            }
        )
        print(
            f"[pred] {sample_i+1}/{len(indices)} {suite_name}#{task_id} "
            f"err_xy={err_xy:.4f} err_z={err_z:.4f} err3d={err3d:.4f} err_pix={pix_err}",
            flush=True,
        )

    grid_name = "pred_vs_gt_xy_grid.png" if xy_only else "pred_vs_gt_grid.png"
    csv_name = "pred_xy_summary.csv" if xy_only else "pred_summary.csv"
    save_grid(overlays, out_dir / grid_name, cols=4)
    write_csv(out_dir / csv_name, rows)
    if rows:
        mean_xy = float(np.mean([r["err_xy"] for r in rows]))
        mean_z = float(np.mean([r["err_z"] for r in rows]))
        mean_3d = float(np.mean([r["err3d"] for r in rows]))
        print(
            f"[pred] mean err_xy={mean_xy:.4f}m  err_z={mean_z:.4f}m  err3d={mean_3d:.4f}m "
            f"over {len(rows)} samples (xy_only={xy_only})",
            flush=True,
        )
    print(f"[pred] wrote {out_dir}", flush=True)


def write_mp4_h264(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    """Write RGB frames as browser/Cursor-friendly H.264 MP4 via ffmpeg."""
    import cv2
    import subprocess
    import tempfile

    if not frames:
        raise ValueError("no frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pose_viz_frames_") as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            bgr = cv2.cvtColor(to_hwc_uint8(frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(tmp_dir / f"frame_{i:05d}.png"), bgr)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path


def cmd_rollout(args: argparse.Namespace) -> None:
    """Short closed-loop rollout with predicted goal overlay on rendered frames."""
    import cv2
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    os.environ.setdefault("MUJOCO_GL", "egl")
    ensure_lerobot_on_path()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    policy, preprocessor = load_policy_and_preprocessor(
        Path(args.checkpoint),
        device,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
    )
    q01, q99 = load_state_quantiles(args.dataset_root)

    task, bddl = resolve_task_bddl(args.suite, args.task_id, bddl_root=args.bddl_root)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.image_size,
        camera_widths=args.image_size,
    )
    obs = env.reset()
    if args.init_state_id is not None and int(args.init_state_id) >= 0:
        try:
            from libero.libero.benchmark import get_benchmark_dict

            suite = get_benchmark_dict()[args.suite]()
            init_states = suite.get_task_init_states(args.task_id)
            obs = env.set_init_state(init_states[int(args.init_state_id)])
        except FileNotFoundError as exc:
            print(f"[rollout] init_states unavailable ({exc}); using env.reset()", flush=True)

    # LIBERO drops objects at reset; wait with a no-op so physics settles before control.
    # Matches lerobot LiberoEnv.num_steps_wait / openpi num_steps_wait.
    dummy = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(int(args.num_steps_wait)):
        step_out = env.step(dummy)
        obs = step_out[0]

    sim = env.env.sim
    transform = get_camera_transform_matrix(
        sim=sim,
        camera_name=CAMERA_NAME,
        camera_height=args.image_size,
        camera_width=args.image_size,
    ).astype(np.float64)
    rotate180 = True if args.rotate180 is None else bool(args.rotate180)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    from lerobot.utils.constants import OBS_STATE

    action_queue: list[np.ndarray] = []
    for step in range(args.max_steps):
        # Build observation for policy.
        agent = obs["agentview_image"][::-1, ::-1].copy()  # match dataset upright convention
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        # Approximate axis-angle from quat if present; else zeros.
        aa = np.zeros(3, dtype=np.float32)
        grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
        state = np.concatenate([eef, aa, grip]).astype(np.float32)

        if not action_queue:
            # One-step pose decode via a synthetic batch using current observation.
            # For rollout we only need pose tokens; still need action placeholders.
            import torch

            fake_item = {
                "observation.images.image": torch.from_numpy(agent).permute(2, 0, 1).float() / 255.0,
                "observation.images.image2": torch.from_numpy(wrist).permute(2, 0, 1).float() / 255.0,
                "observation.state": torch.from_numpy(
                    np.stack([state, state], axis=0)
                ),  # no future GT
                "observation.state_is_pad": torch.tensor([False, True]),
                "action": torch.zeros(args.chunk_size, 7, dtype=torch.float32),
                "task": task.language,
            }
            batch = prepare_train_like_batch(
                policy,
                preprocessor,
                fake_item,
                device=device,
                chunk_size=args.chunk_size,
                q01=q01,
                q99=q99,
            )
            # Override goal pad so loss path still runs; predicted pose is still produced.
            batch["goal_pose_is_pad"] = torch.zeros(1, dtype=torch.bool, device=device)
            try:
                pred_norm = predict_goal_pose_from_policy(policy, batch)[0]
                pred = denormalize_state(pred_norm, q01, q99)
            except Exception as exc:  # noqa: BLE001
                print(f"[rollout] pose decode failed step={step}: {exc}", flush=True)
                pred = None

            # Also get actions for control.
            from copy import deepcopy

            act_batch = deepcopy(batch)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(
                    act_batch, inference_action_mode="continuous"
                )
            actions = chunk[0].detach().cpu().numpy()
            action_queue = [actions[i] for i in range(min(args.n_action_steps, len(actions)))]
        else:
            pred = rows[-1].get("_pred_xyz")
            if pred is not None:
                pred = np.asarray(pred, dtype=np.float64)

        action = action_queue.pop(0)
        # LIBERO expects 7D absolute/relative depending on controller; use as-is.
        step_out = env.step(action.astype(np.float32))
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out
            done = bool(done)

        # Render upright frame and overlay using post-step EE.
        frame = obs["agentview_image"][::-1, ::-1].copy()
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        if args.rotate180 is None and step == 0:
            rotate180 = choose_rotate180(frame, eef, transform)
        overlay = draw_overlay(
            frame,
            current_xyz=eef,
            gt_xyz=None,
            pred_xyz=None if pred is None else pred[:3],
            transform=transform,
            rotate180=rotate180,
            title=f"{args.suite}#{args.task_id} step={step}",
        )
        frames.append(overlay)
        row = {
            "step": step,
            "reward": float(reward),
            "done": bool(done),
            "current_xyz": " ".join(f"{x:.4f}" for x in eef),
            "pred_xyz": "" if pred is None else " ".join(f"{x:.4f}" for x in pred[:3]),
            "_pred_xyz": None if pred is None else pred[:3].tolist(),
        }
        rows.append(row)
        if done:
            break

    env.close()
    video_path = out_dir / f"rollout_{args.suite}_{args.task_id}.mp4"
    if frames:
        write_mp4_h264(frames, video_path, float(args.fps))
        # Also dump a PNG contact sheet so the preview works even if video fails.
        save_grid(frames[: min(16, len(frames))], out_dir / f"rollout_{args.suite}_{args.task_id}_sheet.png", cols=4)
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    write_csv(out_dir / f"rollout_{args.suite}_{args.task_id}.csv", clean_rows)
    print(f"[rollout] wrote {video_path} ({len(frames)} frames, h264)", flush=True)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bddl-root", type=Path, default=None)
    parser.add_argument(
        "--rotate180",
        type=lambda s: {"auto": None, "true": True, "false": False}[s.lower()],
        default="true",
        help="Image/pixel convention relative to MuJoCo camera (auto/true/false). "
        "LIBERO dataset frames are upright vs raw MuJoCo; default true.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gt = sub.add_parser("gt", help="Visualize GT goal pose labels")
    add_shared_args(gt)
    gt.set_defaults(func=cmd_gt)

    pred = sub.add_parser("pred", help="Compare predicted vs GT goal pose")
    add_shared_args(pred)
    pred.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    pred.add_argument("--device", type=str, default="cuda:0")
    pred.add_argument(
        "--xy-only",
        action="store_true",
        help="Ignore Z for overlay/title: project pred/cur at GT height; report err_xy.",
    )
    pred.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Override meta/stats.json for denorm (use pre-v3 backup for old checkpoints).",
    )
    pred.set_defaults(func=cmd_pred)

    rollout = sub.add_parser("rollout", help="Closed-loop rollout with pose overlay")
    rollout.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    rollout.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    rollout.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    rollout.add_argument("--suite", type=str, default="libero_spatial")
    rollout.add_argument("--task-id", type=int, default=0)
    rollout.add_argument("--init-state-id", type=int, default=0)
    rollout.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="No-op settle steps after reset (LIBERO objects fall/slide otherwise).",
    )
    rollout.add_argument("--max-steps", type=int, default=40)
    rollout.add_argument("--chunk-size", type=int, default=10)
    rollout.add_argument("--n-action-steps", type=int, default=5)
    rollout.add_argument("--image-size", type=int, default=256)
    rollout.add_argument("--fps", type=int, default=10)
    rollout.add_argument("--device", type=str, default="cuda:0")
    rollout.add_argument("--bddl-root", type=Path, default=None)
    rollout.add_argument(
        "--rotate180",
        type=lambda s: {"auto": None, "true": True, "false": False}[s.lower()],
        default="auto",
    )
    rollout.add_argument("--output-dir", type=Path, required=True)
    rollout.set_defaults(func=cmd_rollout)
    return parser


def main() -> None:
    ensure_libero_filesystem()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

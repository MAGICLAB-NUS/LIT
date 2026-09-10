#!/usr/bin/env python3
"""LIBERO Stage-1 closed-loop goal-follow probe.

Stage 1 is vision-free and conditions on the chunk-end state
``observation.state[t+H]``. This script:

1. Resets a LIBERO env to a fixed init state.
2. Replays one dataset demo to record oracle EE / state labels.
3. Resets again and runs the Stage-1 checkpoint with oracle goals
   ``goal = recorded_state[min(t+H, T-1)]`` (and optional shuffled goals).
4. Writes overlay videos, 3D trajectory plots, error curves, and a summary JSON.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/libero_goal_prior/eval_stage1_goal_follow.py \\
    --checkpoint lerobot/outputs/libero_goal_prior_v3/seed_1000/stage1/checkpoints/010000/pretrained_model \\
    --suite libero_spatial --task-id 0 --init-state-id 0 \\
    --output-dir lerobot/outputs/libero_stage1_goal_follow/spatial_t0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WS = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path("/data2/JM/dataset/libero_lerobot_format")
DEFAULT_REPO_ID = "local/libero_lerobot_format"
DEFAULT_CKPT = (
    WS
    / "lerobot/outputs/libero_goal_prior_v3/seed_1000/stage1/checkpoints/010000/pretrained_model"
)
CAMERA_NAME = "agentview"


@dataclass
class StepRecord:
    t: int
    cur_xyz: list[float]
    goal_xyz: list[float]
    expert_xyz: list[float]
    pos_err_m: float
    goal_err_m: float
    success: bool


def ensure_lerobot_on_path() -> None:
    src = WS / "lerobot" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def ensure_libero_filesystem() -> None:
    """Reuse the known-good LIBERO asset retarget from viz_goal_pose when needed."""
    good = Path("/data2/JM/Code/MMaDA-VLA-main/LIBERO/libero/libero")
    alt = Path("/data2/JM/Code/molmo_serious/molmoact2-main")
    root = good if (good / "bddl_files" / "libero_spatial").is_dir() else None
    if root is None and (alt / "LIBERO" / "libero" / "libero" / "bddl_files" / "libero_spatial").is_dir():
        root = alt / "LIBERO" / "libero" / "libero"
    if root is None:
        return
    try:
        from libero.libero import get_libero_path

        current = Path(get_libero_path("bddl_files"))
        if current.is_dir() and (current / "libero_spatial").is_dir():
            return
    except Exception:  # noqa: BLE001
        pass
    cfg_path = Path.home() / ".libero" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"assets: {root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Match LiberoProcessorStep._quat2axisangle (dataset / training convention)."""
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep

    q = torch.as_tensor(np.asarray(quat, dtype=np.float32).reshape(4), dtype=torch.float32).unsqueeze(0)
    aa = LiberoProcessorStep()._quat2axisangle(q)[0].detach().cpu().numpy()
    return np.asarray(aa, dtype=np.float32)


def obs_to_state8(obs: dict[str, Any], env: Any) -> np.ndarray:
    """Build the same 8D state used in LIBERO training data / LiberoProcessorStep."""
    del env  # kept for call-site compatibility; orientation comes from quat like training
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(4)
    aa = quat_to_axis_angle(quat)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
    return np.concatenate([eef, aa, grip], axis=0).astype(np.float32)


def to_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def write_mp4_h264(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for frame in frames:
            writer.append_data(to_hwc_uint8(frame))
    finally:
        writer.close()
    return path


def concat_videos_side_by_side(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    *,
    target_h: int = 512,
) -> Path:
    """Horizontally concat two mp4s (camera | traj), resampling to common length."""
    import cv2
    import imageio.v2 as imageio

    def _read(path: Path) -> tuple[list[np.ndarray], float]:
        reader = imageio.get_reader(str(path))
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 10) or 10)
        frames = [np.asarray(f) for f in reader]
        reader.close()
        return frames, fps

    def _resize_h(frame: np.ndarray, h: int) -> np.ndarray:
        frame = to_hwc_uint8(frame)
        H, W = frame.shape[:2]
        if H == h:
            return frame
        return cv2.resize(frame, (int(round(W * h / H)), h), interpolation=cv2.INTER_AREA)

    left, fps_l = _read(left_path)
    right, fps_r = _read(right_path)
    fps = fps_l or fps_r or 10.0
    n = min(len(left), len(right))
    if n == 0:
        raise ValueError(f"empty video for concat: {left_path} / {right_path}")
    if len(left) != len(right):
        idx_l = np.linspace(0, len(left) - 1, n).astype(int)
        idx_r = np.linspace(0, len(right) - 1, n).astype(int)
        left = [left[i] for i in idx_l]
        right = [right[i] for i in idx_r]

    out_frames: list[np.ndarray] = []
    for a, b in zip(left, right):
        a = _resize_h(a, target_h)
        b = _resize_h(b, target_h)
        cat = np.concatenate([a, b], axis=1)
        h, w = cat.shape[:2]
        out_frames.append(cat[: h - h % 2, : w - w % 2])
    return write_mp4_h264(out_frames, out_path, fps)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(axis_angle, dtype=np.float64).reshape(3)).as_matrix()


def project_world_to_pixel(
    xyz: np.ndarray,
    transform: np.ndarray,
    *,
    rotate180: bool,
    height: int,
    width: int,
) -> tuple[int, int] | None:
    import cv2

    pt = np.asarray(xyz, dtype=np.float64).reshape(3)
    hom = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64)
    cam = transform @ hom
    if abs(cam[2]) < 1e-8:
        return None
    u = cam[0] / cam[2]
    v = cam[1] / cam[2]
    if rotate180:
        u = (width - 1) - u
        v = (height - 1) - v
    ui, vi = int(round(u)), int(round(v))
    if ui < -20 or vi < -20 or ui >= width + 20 or vi >= height + 20:
        return None
    return ui, vi


def draw_marker(image_bgr: np.ndarray, uv: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
    import cv2

    u, v = uv
    cv2.circle(image_bgr, (u, v), 7, color, thickness=2, lineType=cv2.LINE_AA)
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


def overlay_frame(
    agent_rgb: np.ndarray,
    *,
    cur: np.ndarray,
    goal: np.ndarray,
    transform: np.ndarray,
    rotate180: bool,
    title: str,
    err_cm: float,
) -> np.ndarray:
    import cv2

    image = to_hwc_uint8(agent_rgb)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    for xyz, color, label in (
        (cur, (255, 90, 40), "cur"),
        (goal, (40, 40, 255), "goal"),
    ):
        uv = project_world_to_pixel(xyz[:3], transform, rotate180=rotate180, height=h, width=w)
        if uv is not None:
            draw_marker(bgr, uv, color, label)
    cv2.putText(
        bgr,
        f"{title}  |cur-goal|={err_cm:.1f}cm",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_policy_processors(checkpoint: Path, device: str, dataset_root: Path, repo_id: str):
    ensure_lerobot_on_path()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.pretrained_path = str(checkpoint)
    cfg.device = device
    if getattr(cfg, "goal_token_source", None) != "se3_encoder":
        raise SystemExit(
            f"Refusing: checkpoint goal_token_source={getattr(cfg, 'goal_token_source', None)!r}; "
            "Stage-1 probe requires se3_encoder."
        )
    if not bool(getattr(cfg, "disable_visual_input", False)):
        print("[warn] disable_visual_input is false; continuing anyway", flush=True)
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()
    if getattr(policy.config, "inference_action_mode", None) in (None, ""):
        policy.config.inference_action_mode = "continuous"
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint),
        dataset_stats=ds_meta.stats,
    )
    return policy, preprocessor, postprocessor, cfg


def find_demo_episode(
    dataset_root: Path,
    repo_id: str,
    language: str,
    *,
    episode_index: int | None,
) -> tuple[Any, int, np.ndarray, np.ndarray]:
    """Return (dataset, ep_idx, states[T,8], actions[T,7])."""
    ensure_lerobot_on_path()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root), video_backend="pyav")
    lang = language.strip().lower()
    if episode_index is not None:
        ep = int(episode_index)
    else:
        ep = None
        # Prefer exact language match via tasks table when available.
        tasks = getattr(dataset.meta, "tasks", None)
        task_index = None
        if tasks is not None:
            for idx, name in enumerate(list(tasks.index)):
                if str(name).strip().lower() == lang:
                    task_index = int(idx)
                    break
        episodes = dataset.meta.episodes
        for candidate in range(int(dataset.meta.total_episodes)):
            start = int(episodes[candidate]["dataset_from_index"])
            item = dataset[start]
            item_task = str(item.get("task", "")).strip().lower()
            if task_index is not None:
                ti = int(item.get("task_index", -1))
                if ti == task_index:
                    ep = candidate
                    break
            elif item_task == lang:
                ep = candidate
                break
        if ep is None:
            raise SystemExit(f"No dataset episode found for language={language!r}")

    from_idx = int(dataset.meta.episodes[ep]["dataset_from_index"])
    to_idx = int(dataset.meta.episodes[ep]["dataset_to_index"])
    states = []
    actions = []
    for i in range(from_idx, to_idx):
        item = dataset[i]
        st = np.asarray(item["observation.state"], dtype=np.float32)
        if st.ndim == 2:
            st = st[0]
        states.append(st.reshape(-1)[:8])
        act = np.asarray(item["action"], dtype=np.float32)
        if act.ndim == 2:
            act = act[0]
        actions.append(act.reshape(-1)[:7])
    return dataset, ep, np.stack(states, axis=0), np.stack(actions, axis=0)


def make_env(suite: str, task_id: int, image_size: int):
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv

    bench = get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    bddl = Path(bench.get_task_bddl_file_path(task_id))
    if not bddl.is_file():
        # Fallback via libero path helper
        from libero.libero import get_libero_path

        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=image_size,
        camera_widths=image_size,
    )
    return env, task, bench


def reset_env(env, bench, task_id: int, init_state_id: int, num_steps_wait: int):
    obs = env.reset()
    init_states = bench.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[int(init_state_id) % len(init_states)])
    dummy = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(int(num_steps_wait)):
        obs = env.step(dummy)[0]
    return obs


def predict_chunk_with_goal(
    *,
    policy,
    preprocessor,
    postprocessor,
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    current_state: np.ndarray,
    goal_state: np.ndarray,
    language: str,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    """Return unnormalized actions (chunk_size, 7)."""
    import torch
    from lerobot.utils.constants import ACTION, OBS_STATE

    policy.reset()
    state_pair = np.stack([current_state, goal_state], axis=0).astype(np.float32)  # (2, 8)
    agent = to_hwc_uint8(agent_rgb)
    wrist = to_hwc_uint8(wrist_rgb)
    raw = {
        "observation.images.image": torch.from_numpy(agent).permute(2, 0, 1).float() / 255.0,
        "observation.images.image2": torch.from_numpy(wrist).permute(2, 0, 1).float() / 255.0,
        OBS_STATE: torch.from_numpy(state_pair),
        "observation.state_is_pad": torch.tensor([False, False]),
        ACTION: torch.zeros(chunk_size, 7, dtype=torch.float32),
        "action_is_pad": torch.zeros(chunk_size, dtype=torch.bool),
        "task": language,
    }
    batched = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            batched[key] = value.unsqueeze(0)
        else:
            batched[key] = [value]
    processed = preprocessor(batched)
    if "goal_pose" not in processed or processed["goal_pose"] is None:
        # Defensive: inject normalized goal via the already-normalized state pair if present,
        # otherwise fall back to raw goal with a warning.
        raise RuntimeError(
            "preprocessor did not produce goal_pose; expected observation.state with time axis "
            "so Stage-1 se3_encoder can condition on the chunk-end state."
        )
    for key, value in list(processed.items()):
        if torch.is_tensor(value):
            processed[key] = value.to(device=device)
    with torch.inference_mode():
        actions = policy.predict_action_chunk(processed)
    actions = postprocessor(actions)
    out = actions[0].detach().float().cpu().numpy()
    return out.astype(np.float32)


def _subsample_indices(n: int, max_points: int = 16) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_points).astype(np.int64))


def pick_index_at_distance(
    states: np.ndarray,
    start_idx: int,
    distance_m: float,
) -> int:
    """Next index along ``states`` whose EE is ~distance_m beyond ``start_idx``."""
    start = np.asarray(states[start_idx, :3], dtype=np.float64)
    target = float(distance_m)
    best_i = min(start_idx + 1, len(states) - 1)
    best_err = abs(np.linalg.norm(states[best_i, :3] - start) - target)
    for i in range(start_idx + 1, len(states)):
        d = float(np.linalg.norm(states[i, :3] - start))
        err = abs(d - target)
        if err < best_err:
            best_err = err
            best_i = i
        # Once we clearly overshoot the target band, stop.
        if d > target * 1.35 and i > start_idx + 2:
            break
    return int(best_i)


def write_trajectory_3d_video(
    out_path: Path,
    follow_xyz: np.ndarray,
    goals_xyz: np.ndarray,
    title: str,
    *,
    fps: float = 10.0,
    stride: int = 2,
) -> Path:
    """Animated 3D video: growing cur path + growing goal trajectory (no expert)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    follow_xyz = np.asarray(follow_xyz, dtype=np.float64)
    goals_xyz = np.asarray(goals_xyz, dtype=np.float64)
    if len(follow_xyz) == 0:
        return out_path

    all_pts = follow_xyz
    if len(goals_xyz):
        all_pts = np.concatenate([follow_xyz, goals_xyz], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.15, 0.02)
    mins = mins - pad
    maxs = maxs + pad

    frames: list[np.ndarray] = []
    indices = list(range(0, len(follow_xyz), max(1, int(stride))))
    if indices[-1] != len(follow_xyz) - 1:
        indices.append(len(follow_xyz) - 1)

    for t in indices:
        fig = plt.figure(figsize=(6.4, 5.4))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot(
            follow_xyz[: t + 1, 0],
            follow_xyz[: t + 1, 1],
            follow_xyz[: t + 1, 2],
            "r-",
            lw=2.0,
            label="cur traj",
        )
        ax.scatter(*follow_xyz[t], c="r", s=50, label="cur now")
        if len(goals_xyz):
            g_end = min(t, len(goals_xyz) - 1)
            ax.plot(
                goals_xyz[: g_end + 1, 0],
                goals_xyz[: g_end + 1, 1],
                goals_xyz[: g_end + 1, 2],
                "b-",
                lw=2.0,
                alpha=0.9,
                label="goal traj",
            )
            g = goals_xyz[g_end]
            ax.scatter(*g, c="b", s=70, marker="*", label="goal now")
            ax.plot(
                [follow_xyz[t, 0], g[0]],
                [follow_xyz[t, 1], g[1]],
                [follow_xyz[t, 2], g[2]],
                "b--",
                lw=1.0,
                alpha=0.7,
            )
            err_cm = float(np.linalg.norm(follow_xyz[t] - g) * 100.0)
        else:
            err_cm = float("nan")
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"{title}\nt={t}  |cur-goal|={err_cm:.1f}cm")
        ax.legend(loc="upper left", fontsize=8)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    return write_mp4_h264(frames, out_path, fps)


def write_trajectory_2d3d_video(
    out_path: Path,
    follow_xyz: np.ndarray,
    goals_xyz: np.ndarray,
    title: str,
    *,
    fps: float = 10.0,
    stride: int = 2,
) -> Path:
    """Animated side-by-side 2D XY + 3D trajectories (cur + goal only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    follow_xyz = np.asarray(follow_xyz, dtype=np.float64)
    goals_xyz = np.asarray(goals_xyz, dtype=np.float64)
    if len(follow_xyz) == 0:
        return out_path

    all_pts = follow_xyz if len(goals_xyz) == 0 else np.concatenate([follow_xyz, goals_xyz], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.15, 0.02)
    mins2 = mins - pad
    maxs2 = maxs + pad

    frames: list[np.ndarray] = []
    indices = list(range(0, len(follow_xyz), max(1, int(stride))))
    if indices[-1] != len(follow_xyz) - 1:
        indices.append(len(follow_xyz) - 1)

    for t in indices:
        fig = plt.figure(figsize=(11.5, 5.2))
        ax2 = fig.add_subplot(1, 2, 1)
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")

        ax2.plot(follow_xyz[: t + 1, 0], follow_xyz[: t + 1, 1], "r-", lw=2.0, label="cur traj")
        ax2.scatter(follow_xyz[t, 0], follow_xyz[t, 1], c="r", s=45, label="cur now")
        if len(goals_xyz):
            g_end = min(t, len(goals_xyz) - 1)
            ax2.plot(
                goals_xyz[: g_end + 1, 0],
                goals_xyz[: g_end + 1, 1],
                "b-",
                lw=2.0,
                label="goal traj",
            )
            ax2.scatter(goals_xyz[g_end, 0], goals_xyz[g_end, 1], c="b", s=55, marker="*", label="goal now")
            err_cm = float(np.linalg.norm(follow_xyz[t] - goals_xyz[g_end]) * 100.0)
        else:
            err_cm = float("nan")
        ax2.set_xlim(mins2[0], maxs2[0])
        ax2.set_ylim(mins2[1], maxs2[1])
        ax2.set_aspect("equal", adjustable="box")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_title("2D XY")
        ax2.legend(loc="best", fontsize=8)

        ax3.plot(
            follow_xyz[: t + 1, 0],
            follow_xyz[: t + 1, 1],
            follow_xyz[: t + 1, 2],
            "r-",
            lw=2.0,
            label="cur traj",
        )
        ax3.scatter(*follow_xyz[t], c="r", s=45, label="cur now")
        if len(goals_xyz):
            g_end = min(t, len(goals_xyz) - 1)
            ax3.plot(
                goals_xyz[: g_end + 1, 0],
                goals_xyz[: g_end + 1, 1],
                goals_xyz[: g_end + 1, 2],
                "b-",
                lw=2.0,
                label="goal traj",
            )
            ax3.scatter(*goals_xyz[g_end], c="b", s=60, marker="*", label="goal now")
        ax3.set_xlim(mins2[0], maxs2[0])
        ax3.set_ylim(mins2[1], maxs2[1])
        ax3.set_zlim(mins2[2], maxs2[2])
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_title("3D")
        ax3.legend(loc="upper left", fontsize=7)

        fig.suptitle(f"{title} | t={t} |cur-goal|={err_cm:.1f}cm", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    return write_mp4_h264(frames, out_path, fps)


def plot_trajectory_2d(
    out_path: Path,
    follow_xyz: np.ndarray,
    goals_xyz: np.ndarray,
    title: str,
    *,
    ax=None,
) -> Any:
    """Top-down XY spatial plot (meters): cur trajectory + goal trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
    else:
        fig = ax.figure

    if len(follow_xyz):
        ax.plot(follow_xyz[:, 0], follow_xyz[:, 1], "r-", lw=2.2, label="cur", zorder=2)
        ax.scatter(follow_xyz[0, 0], follow_xyz[0, 1], c="r", s=45, marker="o", zorder=3)
        ax.scatter(follow_xyz[-1, 0], follow_xyz[-1, 1], c="r", s=55, marker="*", zorder=3)
    if len(goals_xyz):
        ax.plot(goals_xyz[:, 0], goals_xyz[:, 1], "b-", lw=2.0, alpha=0.9, label="goal traj", zorder=2)
        ax.scatter(goals_xyz[0, 0], goals_xyz[0, 1], c="b", s=45, marker="o", zorder=4)
        ax.scatter(goals_xyz[-1, 0], goals_xyz[-1, 1], c="b", s=70, marker="*", zorder=4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"2D XY — {title}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if owns_fig:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return ax


def plot_trajectory_3d(
    out_path: Path,
    follow_xyz: np.ndarray,
    goals_xyz: np.ndarray,
    title: str,
    *,
    ax=None,
) -> Any:
    """3D workspace spatial plot (meters): cur trajectory + goal trajectory curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    owns_fig = ax is None
    if owns_fig:
        fig = plt.figure(figsize=(7.0, 6.0))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = ax.figure

    if len(follow_xyz):
        ax.plot(
            follow_xyz[:, 0],
            follow_xyz[:, 1],
            follow_xyz[:, 2],
            "r-",
            lw=2.2,
            label="cur",
        )
        ax.scatter(*follow_xyz[0], c="r", s=40)
        ax.scatter(*follow_xyz[-1], c="r", s=55, marker="*")
    if len(goals_xyz):
        ax.plot(
            goals_xyz[:, 0],
            goals_xyz[:, 1],
            goals_xyz[:, 2],
            "b-",
            lw=2.0,
            alpha=0.9,
            label="goal traj",
        )
        ax.scatter(*goals_xyz[0], c="b", s=40)
        ax.scatter(*goals_xyz[-1], c="b", s=70, marker="*")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"3D — {title}")
    ax.legend(loc="upper left", fontsize=8)
    if owns_fig:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return ax


def plot_trajectory_2d3d_combined(
    out_path: Path,
    follow_xyz: np.ndarray,
    goals_xyz: np.ndarray,
    title: str,
) -> None:
    """Side-by-side 2D XY + 3D trajectories (cur + goal curves only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 5.8))
    ax2 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_trajectory_2d(out_path, follow_xyz, goals_xyz, title, ax=ax2)
    plot_trajectory_3d(out_path, follow_xyz, goals_xyz, title, ax=ax3)
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectories(
    out_dir: Path,
    mode: str,
    follow_xyz: np.ndarray,
    goals_xyz: np.ndarray,
    title: str,
    *,
    fps: float = 10.0,
) -> None:
    plot_trajectory_2d(out_dir / f"{mode}_traj_2d.png", follow_xyz, goals_xyz, title)
    plot_trajectory_3d(out_dir / f"{mode}_traj_3d.png", follow_xyz, goals_xyz, title)
    plot_trajectory_2d3d_combined(out_dir / f"{mode}_traj_2d3d.png", follow_xyz, goals_xyz, title)
    write_trajectory_3d_video(
        out_dir / f"{mode}_traj_3d.mp4",
        follow_xyz,
        goals_xyz,
        title,
        fps=fps,
        stride=2,
    )
    write_trajectory_2d3d_video(
        out_dir / f"{mode}_traj_2d3d.mp4",
        follow_xyz,
        goals_xyz,
        title,
        fps=fps,
        stride=2,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_mode(
    *,
    mode: str,
    env,
    bench,
    task,
    task_id: int,
    init_state_id: int,
    num_steps_wait: int,
    recorded_states: np.ndarray,
    demo_actions: np.ndarray,
    policy,
    preprocessor,
    postprocessor,
    device: str,
    chunk_size: int,
    n_action_steps: int,
    max_steps: int,
    shuffle_seed: int,
    transform: np.ndarray,
    fixed_goal_index: int | None = None,
    reach_tol_m: float = 0.02,
    waypoint_stride: int = 10,
    hold_steps: int = 100,
    goal_distance_m: float = 0.10,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    obs = reset_env(env, bench, task_id, init_state_id, num_steps_wait)
    language = str(task.language)
    T = int(recorded_states.shape[0])
    horizon = int(chunk_size)

    goal_indices = np.arange(T, dtype=np.int64)
    if mode == "shuffle":
        rng = np.random.default_rng(shuffle_seed)
        goal_indices = rng.permutation(T)

    if mode == "fixed":
        if fixed_goal_index is None:
            fixed_goal_index = min(max(horizon * 4, 40), T - 1)
        fixed_goal_index = int(np.clip(fixed_goal_index, 0, T - 1))
        fixed_goal = recorded_states[fixed_goal_index].copy()
        print(
            f"[stage1-follow] fixed goal index={fixed_goal_index} "
            f"xyz={fixed_goal[:3].tolist()} (held constant)",
            flush=True,
        )
    else:
        fixed_goal = None

    if mode == "waypoint":
        hold = max(1, int(hold_steps))
        dist_m = float(goal_distance_m)
        # Start from a moderately distant waypoint (not too near / not too far).
        waypoint_idx = pick_index_at_distance(recorded_states, 0, dist_m)
        steps_on_goal = 0
        d0 = float(np.linalg.norm(recorded_states[waypoint_idx, :3] - recorded_states[0, :3]) * 100.0)
        print(
            f"[stage1-follow] waypoint: target spacing≈{dist_m * 100:.1f}cm, "
            f"hold each for {hold} steps; start goal_idx={waypoint_idx} "
            f"(~{d0:.1f}cm from start)",
            flush=True,
        )
    else:
        hold = waypoint_idx = steps_on_goal = 0
        dist_m = 0.0

    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    follow_xyz: list[np.ndarray] = []
    goal_xyz_list: list[np.ndarray] = []
    expert_xyz_list: list[np.ndarray] = []
    action_queue: list[np.ndarray] = []
    action_xyz_norms: list[float] = []
    success = False
    reached = False

    steps = int(max_steps) if mode in {"fixed", "waypoint"} else min(int(max_steps), T)
    for t in range(steps):
        cur = obs_to_state8(obs, env)
        expert = recorded_states[min(t, T - 1)]
        if mode == "fixed":
            goal = fixed_goal
            active_goal_idx = int(fixed_goal_index)
        elif mode == "waypoint":
            goal = recorded_states[waypoint_idx]
            active_goal_idx = int(waypoint_idx)
        else:
            goal_t = int(min(t + horizon, T - 1))
            if mode == "shuffle":
                goal_t = int(goal_indices[min(t, T - 1)])
            goal = recorded_states[goal_t]
            active_goal_idx = int(goal_t)

        agent = obs["agentview_image"][::-1, ::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]

        if not action_queue:
            chunk = predict_chunk_with_goal(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                agent_rgb=agent,
                wrist_rgb=wrist,
                current_state=cur,
                goal_state=goal,
                language=language,
                device=device,
                chunk_size=chunk_size,
            )
            for a in chunk[:n_action_steps]:
                action_queue.append(a)
                action_xyz_norms.append(float(np.linalg.norm(a[:3])))

        action = action_queue.pop(0)
        step_out = env.step(action)
        obs = step_out[0]
        try:
            success = bool(env.check_success())
        except Exception:  # noqa: BLE001
            success = False

        pos_err = float(np.linalg.norm(cur[:3] - expert[:3]))
        goal_err = float(np.linalg.norm(cur[:3] - goal[:3]))
        if mode == "fixed" and goal_err <= float(reach_tol_m):
            reached = True
        frame = overlay_frame(
            agent,
            cur=cur,
            goal=goal,
            transform=transform,
            rotate180=True,
            title=f"{mode} t={t} g={active_goal_idx}",
            err_cm=goal_err * 100.0,
        )
        frames.append(frame)
        follow_xyz.append(cur[:3].copy())
        goal_xyz_list.append(np.asarray(goal[:3]).copy())
        expert_xyz_list.append(expert[:3].copy())
        rows.append(
            asdict(
                StepRecord(
                    t=t,
                    cur_xyz=cur[:3].tolist(),
                    goal_xyz=np.asarray(goal[:3]).tolist(),
                    expert_xyz=expert[:3].tolist(),
                    pos_err_m=pos_err,
                    goal_err_m=goal_err,
                    success=success,
                )
            )
        )

        if mode == "waypoint":
            steps_on_goal += 1
            if steps_on_goal >= hold and waypoint_idx < T - 1:
                next_idx = pick_index_at_distance(recorded_states, waypoint_idx, dist_m)
                if next_idx <= waypoint_idx:
                    next_idx = min(waypoint_idx + max(1, int(waypoint_stride)), T - 1)
                waypoint_idx = next_idx
                steps_on_goal = 0
                action_queue.clear()  # replan immediately for the new goal
                d = float(
                    np.linalg.norm(recorded_states[waypoint_idx, :3] - cur[:3]) * 100.0
                )
                print(
                    f"[stage1-follow] waypoint advance -> goal_idx={waypoint_idx} "
                    f"at t={t} (|cur-newgoal|≈{d:.1f}cm)",
                    flush=True,
                )

        if success or reached:
            break

    follow_arr = np.stack(follow_xyz, axis=0) if follow_xyz else np.zeros((0, 3))
    goal_arr = np.stack(goal_xyz_list, axis=0) if goal_xyz_list else np.zeros((0, 3))
    expert_arr = np.stack(expert_xyz_list, axis=0) if expert_xyz_list else np.zeros((0, 3))

    def _mean_cm(a: np.ndarray, b: np.ndarray, n: int | None = None) -> float | None:
        if len(a) == 0:
            return None
        if n is not None:
            a = a[:n]
            b = b[:n]
        return float(np.mean(np.linalg.norm(a - b, axis=1)) * 100.0)

    step_disp_cm = (
        float(np.mean(np.linalg.norm(np.diff(follow_arr, axis=0), axis=1) * 100.0))
        if len(follow_arr) > 1
        else None
    )
    summary = {
        "mode": mode,
        "steps": len(rows),
        "success": bool(success),
        "reached_fixed_goal": bool(reached) if mode == "fixed" else None,
        "fixed_goal_index": int(fixed_goal_index) if mode == "fixed" else None,
        "waypoint_stride": int(waypoint_stride) if mode == "waypoint" else None,
        "hold_steps": int(hold_steps) if mode == "waypoint" else None,
        "goal_distance_m": float(goal_distance_m) if mode == "waypoint" else None,
        "mean_follow_expert_cm": _mean_cm(follow_arr, expert_arr),
        "mean_follow_goal_cm": _mean_cm(follow_arr, goal_arr),
        "mean_follow_expert_cm_first20": _mean_cm(follow_arr, expert_arr, 20),
        "mean_follow_goal_cm_first20": _mean_cm(follow_arr, goal_arr, 20),
        "final_follow_goal_cm": float(np.linalg.norm(follow_arr[-1] - goal_arr[-1]) * 100.0)
        if len(rows)
        else None,
        "final_follow_expert_cm": float(np.linalg.norm(follow_arr[-1] - expert_arr[-1]) * 100.0)
        if len(rows)
        else None,
        "mean_step_disp_cm": step_disp_cm,
        "mean_pred_action_xyz_norm": float(np.mean(action_xyz_norms)) if action_xyz_norms else None,
        "follow_xyz": follow_arr.tolist(),
        "goal_xyz": goal_arr.tolist(),
        "expert_xyz": expert_arr.tolist(),
    }
    _ = demo_actions
    return frames, rows, summary


def record_expert_trajectory(
    env,
    bench,
    task_id: int,
    init_state_id: int,
    num_steps_wait: int,
    demo_actions: np.ndarray,
    demo_states: np.ndarray,
    transform: np.ndarray,
    max_steps: int,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    """Replay dataset actions; return recorded env states + frames + rows."""
    obs = reset_env(env, bench, task_id, init_state_id, num_steps_wait)
    states = []
    frames = []
    rows = []
    steps = min(int(max_steps), int(demo_actions.shape[0]), int(demo_states.shape[0]))
    for t in range(steps):
        cur = obs_to_state8(obs, env)
        states.append(cur)
        agent = obs["agentview_image"][::-1, ::-1]
        frames.append(
            overlay_frame(
                agent,
                cur=cur,
                goal=demo_states[min(t + 10, steps - 1)],
                transform=transform,
                rotate180=True,
                title=f"record t={t}",
                err_cm=float(np.linalg.norm(cur[:3] - demo_states[t, :3]) * 100.0),
            )
        )
        rows.append(
            {
                "t": t,
                "env_xyz": cur[:3].tolist(),
                "demo_xyz": demo_states[t, :3].tolist(),
                "env_demo_err_m": float(np.linalg.norm(cur[:3] - demo_states[t, :3])),
            }
        )
        obs = env.step(demo_actions[t].astype(np.float32))[0]
    return np.stack(states, axis=0), frames, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-state-id", type=int, default=0)
    p.add_argument("--episode-index", type=int, default=None, help="Optional dataset episode override")
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument(
        "--fixed-goal-index",
        type=int,
        default=None,
        help="For mode=fixed: index into recorded_states used as the constant goal.",
    )
    p.add_argument(
        "--reach-tol-m",
        type=float,
        default=0.02,
        help="For mode=fixed: stop when |cur-goal| <= this distance (meters).",
    )
    p.add_argument(
        "--waypoint-stride",
        type=int,
        default=10,
        help="Fallback frame stride if distance-based picking cannot advance.",
    )
    p.add_argument(
        "--hold-steps",
        type=int,
        default=100,
        help="For mode=waypoint: keep each goal for this many control steps before advancing.",
    )
    p.add_argument(
        "--goal-distance-m",
        type=float,
        default=0.10,
        help="For mode=waypoint: approximate EE spacing between successive goals (meters).",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="oracle,shuffle",
        help="Comma-separated: oracle, shuffle, fixed, and/or waypoint",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--skip-record-video", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    ensure_libero_filesystem()
    ensure_lerobot_on_path()

    import torch
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env, task, bench = make_env(args.suite, args.task_id, args.image_size)
    language = str(task.language)
    print(f"[stage1-follow] suite={args.suite} task_id={args.task_id} lang={language!r}", flush=True)

    _, ep_idx, demo_states, demo_actions = find_demo_episode(
        args.dataset_root,
        args.repo_id,
        language,
        episode_index=args.episode_index,
    )
    print(
        f"[stage1-follow] demo episode={ep_idx} T={len(demo_states)} actions={demo_actions.shape}",
        flush=True,
    )

    # Camera transform after a reset.
    obs0 = reset_env(env, bench, args.task_id, args.init_state_id, args.num_steps_wait)
    _ = obs0
    transform = get_camera_transform_matrix(
        sim=env.env.sim,
        camera_name=CAMERA_NAME,
        camera_height=args.image_size,
        camera_width=args.image_size,
    ).astype(np.float64)

    recorded_states, record_frames, record_rows = record_expert_trajectory(
        env,
        bench,
        args.task_id,
        args.init_state_id,
        args.num_steps_wait,
        demo_actions,
        demo_states,
        transform,
        args.max_steps,
    )
    write_csv(out_dir / "record_steps.csv", record_rows)
    np.save(out_dir / "recorded_states.npy", recorded_states)
    if not args.skip_record_video:
        write_mp4_h264(record_frames, out_dir / "record_expert.mp4", args.fps)
    print(
        f"[stage1-follow] recorded {len(recorded_states)} env states; "
        f"mean |env-demo|="
        f"{np.mean(np.linalg.norm(recorded_states[:, :3] - demo_states[: len(recorded_states), :3], axis=1)) * 100:.2f}cm",
        flush=True,
    )

    policy, preprocessor, postprocessor, cfg = load_policy_processors(
        Path(args.checkpoint),
        device,
        args.dataset_root,
        args.repo_id,
    )
    chunk_size = int(getattr(cfg, "chunk_size", args.chunk_size) or args.chunk_size)
    n_action_steps = int(getattr(cfg, "n_action_steps", args.n_action_steps) or args.n_action_steps)

    summaries = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "suite": args.suite,
        "task_id": args.task_id,
        "language": language,
        "init_state_id": args.init_state_id,
        "demo_episode": ep_idx,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "recorded_steps": int(recorded_states.shape[0]),
        "modes": {},
    }

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if mode not in {"oracle", "shuffle", "fixed", "waypoint"}:
            raise SystemExit(f"Unknown mode {mode!r}; expected oracle|shuffle|fixed|waypoint")
        print(f"[stage1-follow] running mode={mode}", flush=True)
        frames, rows, summary = run_mode(
            mode=mode,
            env=env,
            bench=bench,
            task=task,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            num_steps_wait=args.num_steps_wait,
            recorded_states=recorded_states,
            demo_actions=demo_actions,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_steps=args.max_steps,
            shuffle_seed=args.shuffle_seed,
            transform=transform,
            fixed_goal_index=args.fixed_goal_index,
            reach_tol_m=args.reach_tol_m,
            waypoint_stride=args.waypoint_stride,
            hold_steps=args.hold_steps,
            goal_distance_m=args.goal_distance_m,
        )
        write_csv(out_dir / f"{mode}_steps.csv", rows)
        follow_mp4 = out_dir / f"{mode}_follow.mp4"
        write_mp4_h264(frames, follow_mp4, args.fps)
        follow = np.asarray(summary["follow_xyz"], dtype=np.float64)
        goals = np.asarray(summary["goal_xyz"], dtype=np.float64)
        plot_trajectories(
            out_dir,
            mode,
            follow_xyz=follow,
            goals_xyz=goals,
            title=f"{mode}: {args.suite} task{args.task_id}",
            fps=args.fps,
        )
        traj_mp4 = out_dir / f"{mode}_traj_2d3d.mp4"
        if follow_mp4.exists() and traj_mp4.exists():
            concat_videos_side_by_side(
                follow_mp4,
                traj_mp4,
                out_dir / f"{mode}_follow_with_traj.mp4",
            )
        # Drop bulky arrays from JSON summary
        slim = {k: v for k, v in summary.items() if k not in {"follow_xyz", "goal_xyz", "expert_xyz"}}
        summaries["modes"][mode] = slim
        print(
            f"[stage1-follow] {mode}: success={slim['success']} "
            f"mean|follow-goal|={slim['mean_follow_goal_cm']:.2f}cm "
            f"(first20={slim['mean_follow_goal_cm_first20']:.2f}cm) "
            f"final|cur-goal|={slim.get('final_follow_goal_cm')} "
            f"step_disp={slim.get('mean_step_disp_cm')}cm "
            f"action|xyz|={slim.get('mean_pred_action_xyz_norm')} "
            f"reached={slim.get('reached_fixed_goal')}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    env.close()
    print(f"[stage1-follow] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()

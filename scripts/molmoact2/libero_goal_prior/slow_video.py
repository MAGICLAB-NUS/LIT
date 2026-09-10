#!/usr/bin/env python3
"""Create slowed H.264 copies of one rollout video or a directory of videos."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def output_path_for(source: Path, speed: float) -> Path:
    return source.with_name(f"{source.stem}_{speed:g}x{source.suffix}")


def slow_video(source: Path, destination: Path, speed: float, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"[slow-video] skip existing: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"setpts={1.0 / speed:.12g}*PTS",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    print(f"[slow-video] {source} -> {destination} ({speed:g}x)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="An MP4 file or directory searched recursively.")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Output path; valid only for one input file.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found in PATH")
    if not 0.0 < args.speed <= 1.0:
        raise SystemExit(f"--speed must be in (0, 1], got {args.speed}")

    source = args.input.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".mp4":
            raise SystemExit(f"input must be an MP4 file: {source}")
        destination = (
            args.output.expanduser().resolve()
            if args.output is not None
            else output_path_for(source, args.speed)
        )
        slow_video(source, destination, args.speed, overwrite=args.overwrite)
        return

    if args.output is not None:
        raise SystemExit("--output is supported only when input is one MP4 file")
    if not source.is_dir():
        raise SystemExit(f"input does not exist: {source}")

    suffix = f"_{args.speed:g}x"
    videos = sorted(
        path
        for path in source.rglob("*.mp4")
        if not path.stem.endswith(suffix)
    )
    if not videos:
        raise SystemExit(f"no source MP4 files found under {source}")
    for video in videos:
        slow_video(
            video,
            output_path_for(video, args.speed),
            args.speed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

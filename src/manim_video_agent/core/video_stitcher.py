"""
Stitch per-scene MP4 files into a single video using FFmpeg concat demuxer.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def stitch_videos(scene_videos: list[Path], output: Path) -> Path:
    """Concatenate scene MP4s into a single video via FFmpeg concat demuxer.

    All inputs must share the same codec, resolution, and frame rate
    (guaranteed when rendered by the same Manim config).

    Args:
        scene_videos: Ordered list of per-scene MP4 paths.
        output: Destination path for the stitched video.

    Returns:
        The *output* path on success.

    Raises:
        ValueError: If *scene_videos* is empty.
        RuntimeError: If FFmpeg fails.
    """
    if not scene_videos:
        raise ValueError("No scene videos to stitch")

    if len(scene_videos) == 1:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(scene_videos[0].read_bytes())
        return output

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="concat_"
    ) as f:
        for video in scene_videos:
            f.write(f"file '{video.resolve()}'\n")
        concat_list = Path(f.name)

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed (exit {result.returncode}):\n"
                f"{result.stderr[:1000]}"
            )
        logger.info(
            "Stitched %d scenes into %s",
            len(scene_videos),
            output,
        )
        return output
    finally:
        concat_list.unlink(missing_ok=True)

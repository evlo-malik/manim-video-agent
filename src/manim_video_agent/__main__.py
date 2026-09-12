"""Command-line entry point for the video pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from manim_video_agent.core.parallel_pipeline import generate_video_parallel


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manim-video-agent",
        description="Generate a narrated Manim video from a Markdown brief.",
    )
    parser.add_argument("brief", type=Path, help="Markdown input file")
    parser.add_argument("--job", default="video", help="Output job name")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--api-key", help="OpenRouter key; defaults to the environment")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    content = args.brief.read_text(encoding="utf-8")
    result = generate_video_parallel(
        content,
        args.job,
        api_key=args.api_key,
        output_dir=args.output,
    )
    print(result.final_video)


if __name__ == "__main__":
    main()

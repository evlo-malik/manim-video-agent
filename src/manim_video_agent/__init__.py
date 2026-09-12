"""Manim Video Agent — Manim video generation pipeline."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def get_data_dir() -> Path:
    """Locate the data/ directory containing skill files.

    Resolution order:
      1. ``manim_video_agent_data_dir`` from Settings (.env / env var)
      2. Relative to package location (development / editable install)
      3. Relative to current working directory
    """
    from manim_video_agent.config import get_settings

    settings = get_settings()
    if (
        settings.manim_video_agent_data_dir
        and settings.manim_video_agent_data_dir.is_dir()
    ):
        return settings.manim_video_agent_data_dir

    pkg_dir = Path(__file__).resolve().parent
    data_dir = pkg_dir / "data"
    if data_dir.is_dir():
        return data_dir

    raise FileNotFoundError(
        "Cannot find bundled guidance data. Set MANIM_VIDEO_AGENT_DATA_DIR "
        "to a compatible data directory."
    )


def get_skills_dir(skill_name: str = "") -> Path:
    """Return the path to a specific skill data directory."""
    base = get_data_dir() / "skills"
    if skill_name:
        return base / skill_name
    return base

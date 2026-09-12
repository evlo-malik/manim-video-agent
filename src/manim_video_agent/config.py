"""
Centralized configuration using Pydantic Settings.

Reads from environment variables and .env files. All settings are
accessible via the singleton ``get_settings()`` function.

Environment variables map directly to field names (case-insensitive):
    OPENROUTER_API_KEY=sk-or-...
    INWORLD_TTS_API_KEY=...
    MANIM_VIDEO_AGENT_DATA_DIR=/custom/path/to/data
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys -----------------------------------------------------------
    openrouter_api_key: str = ""
    inworld_api_key: str = ""  # Base64 API key from Inworld Portal

    # --- Data paths ----------------------------------------------------------
    manim_video_agent_data_dir: Optional[Path] = None

    # --- OpenRouter defaults -------------------------------------------------
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    default_model: str = "anthropic/claude-sonnet-4.6"

    # --- Composer ------------------------------------------------------------
    research_model: str = "google/gemini-3-flash-preview"
    composer_model: str = "google/gemini-3-flash-preview"
    composer_timeout: int = 300

    # --- Fixer ---------------------------------------------------------------
    fixer_model: str = "anthropic/claude-sonnet-4.6"
    fixer_max_iterations: int = 2

    # --- Reviewer (legacy code-only reviewer) ---------------------------------
    reviewer_model: str = "google/gemini-3-flash-preview"

    # --- Visual Reviewer (render → Gemini vision → Sonnet fix loop) ----------
    visual_reviewer_model: str = "google/gemini-3-flash-preview"
    visual_fixer_model: str = "anthropic/claude-sonnet-4.6"
    visual_review_max_iterations: int = 1

    # --- Video Renderer ------------------------------------------------------
    render_fixer_model: str = "anthropic/claude-sonnet-4.6"
    render_max_retries: int = 2

    # --- TTS -----------------------------------------------------------------
    tts_base_url: str = "https://api.inworld.ai/tts/v1/voice"
    tts_voice_id: str = "Dennis"
    tts_model_id: str = "inworld-tts-1.5-max"

    # --- Narration -----------------------------------------------------------
    narration_model: str = "google/gemini-2.5-flash"

    # --- Parallel scene pipeline ---------------------------------------------
    max_parallel_scenes: int = 3
    max_parallel_renders: int = 3
    scene_render_timeout: int = 600
    max_scene_regen_attempts: int = 2


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    return Settings()

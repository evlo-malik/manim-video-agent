"""
Per-scene plan data models and parsing.

Splits a composed scene plan (with ``---SCENE_BREAK---`` delimiters) into
independent :class:`ScenePlan` units that can be generated, reviewed, and
rendered in parallel.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manim_video_agent.core.composer import ResearchResult

logger = logging.getLogger(__name__)

# Delimiter the composer inserts between scene specifications.
SCENE_BREAK = "---SCENE_BREAK---"


# ---------------------------------------------------------------------------
# Data models (all immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenePlan:
    """Specification for a single scene."""

    index: int  # 1-based scene number
    slug: str  # e.g. "intro", "main_concept"
    title: str  # e.g. "Introduction"
    markdown: str  # full scene specification


@dataclass(frozen=True)
class SharedContext:
    """Visual identity and metadata shared across all scenes."""

    background_color: str
    color_palette: dict[str, str]
    topic_title: str
    total_scenes: int


@dataclass(frozen=True)
class CompositionResult:
    """Output of the scene-aware composition step."""

    plans: list[ScenePlan]
    shared_context: SharedContext
    research: ResearchResult


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Convert *text* to a filesystem-safe slug."""
    return _SLUG_RE.sub("_", text.lower()).strip("_")[:40]


def _extract_title(markdown: str) -> str:
    """Pull the first markdown heading from *markdown*, or fall back."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return "untitled"


def _parse_shared_context_block(block: str) -> SharedContext:
    """Parse the ``## Shared Context`` JSON block emitted by the composer.

    Expected format inside the block::

        ```json
        {
            "background_color": "#1e1e2e",
            "color_palette": {"positive": "#a6e3a1", ...},
            "topic_title": "Fourier Series",
            "total_scenes": 3
        }
        ```
    """
    # Try to extract JSON from fenced code block first
    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", block, re.DOTALL)
    raw = json_match.group(1) if json_match else block

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse shared context JSON — using defaults")
        return SharedContext(
            background_color="#1e1e2e",
            color_palette={
                "positive": "#a6e3a1",
                "negative": "#f38ba8",
                "neutral": "#89dceb",
                "emphasis": "#f9e2af",
            },
            topic_title="Untitled",
            total_scenes=0,
        )

    return SharedContext(
        background_color=data.get("background_color", "#1e1e2e"),
        color_palette=data.get("color_palette", {}),
        topic_title=data.get("topic_title", "Untitled"),
        total_scenes=int(data.get("total_scenes", 0)),
    )


def parse_composition(raw: str) -> tuple[list[ScenePlan], SharedContext]:
    """Split a composed plan into per-scene plans and shared context.

    The composer is expected to emit a ``## Shared Context`` block at the
    top, followed by scene sections separated by ``---SCENE_BREAK---``.

    Returns:
        ``(scene_plans, shared_context)`` tuple.
    """
    # Split off the shared context header
    shared_ctx_marker = "## Shared Context"
    shared_block = ""
    body = raw

    if shared_ctx_marker in raw:
        marker_pos = raw.index(shared_ctx_marker)
        # Find the end of the shared context block (next scene break or next ##)
        rest = raw[marker_pos + len(shared_ctx_marker) :]
        # The shared context block ends at the first SCENE_BREAK
        if SCENE_BREAK in rest:
            break_pos = rest.index(SCENE_BREAK)
            shared_block = rest[:break_pos]
            body = rest[break_pos:]
        else:
            shared_block = rest
            body = ""

    shared_context = _parse_shared_context_block(shared_block)

    # Split remaining body on scene breaks
    raw_scenes = [s.strip() for s in body.split(SCENE_BREAK) if s.strip()]

    plans: list[ScenePlan] = []
    for i, scene_md in enumerate(raw_scenes, start=1):
        title = _extract_title(scene_md)
        slug = _slugify(title)
        plans.append(
            ScenePlan(
                index=i,
                slug=slug,
                title=title,
                markdown=scene_md,
            )
        )

    # Backfill total_scenes if the composer didn't set it
    if shared_context.total_scenes != len(plans):
        shared_context = SharedContext(
            background_color=shared_context.background_color,
            color_palette=shared_context.color_palette,
            topic_title=shared_context.topic_title,
            total_scenes=len(plans),
        )

    return plans, shared_context


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def scene_filename(plan: ScenePlan) -> str:
    """Canonical filename stem for a scene: ``scene_001_intro``."""
    return f"scene_{plan.index:03d}_{plan.slug}"


def write_scene_files(plans: list[ScenePlan], output_dir: Path) -> list[Path]:
    """Write each scene plan as a ``.md`` file.

    Returns the list of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for plan in plans:
        path = output_dir / f"{scene_filename(plan)}.md"
        path.write_text(plan.markdown, encoding="utf-8")
        paths.append(path)
    return paths


def write_shared_context(ctx: SharedContext, output_dir: Path) -> Path:
    """Write shared context as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shared_context.json"
    data = {
        "background_color": ctx.background_color,
        "color_palette": ctx.color_palette,
        "topic_title": ctx.topic_title,
        "total_scenes": ctx.total_scenes,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path

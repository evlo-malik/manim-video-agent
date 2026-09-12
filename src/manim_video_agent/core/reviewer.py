"""
Manim Visual Reviewer Agent

LLM-based service that mentally simulates rendered Manim output and fixes
visual quality issues (overlap, off-screen elements, clutter, etc.).
Single-pass — no iteration. Returns original code if no issues found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from manim_video_agent.config import get_settings


@dataclass(frozen=True)
class ReviewerConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""

    def __post_init__(self):
        settings = get_settings()
        if not self.api_key:
            object.__setattr__(self, "api_key", settings.openrouter_api_key)
        if not self.base_url:
            object.__setattr__(self, "base_url", settings.openrouter_base_url)
        if not self.model:
            object.__setattr__(self, "model", settings.reviewer_model)


REVIEWER_SYSTEM_PROMPT = """\
<role>
You are a Manim visual layout reviewer. You receive Manim code, mentally \
simulate what each scene would look like when rendered, and fix any visual \
quality issues. Output ONLY the corrected Python code — no explanations, \
no markdown fences.
</role>

<manim_frame_dimensions>
- Width: ~14.22 units, Height: 8 units
- Center at (0, 0)
- Safe bounds: x in [-6.5, 6.5], y in [-3.5, 3.5]
- Elements near the edges may be clipped or feel cramped
</manim_frame_dimensions>

<visual_issues_to_check>
1. OFF-SCREEN: Elements positioned outside or partially outside the visible \
frame. Reposition within safe bounds using .to_edge(), .move_to(), or \
coordinate clamping.
2. OVERLAP: Multiple elements placed at the same position unintentionally. \
Spread them out using .next_to(), .arrange(), or VGroup with spacing.
3. TEXT OVERFLOW: Text too wide for its container or the frame. Use \
.scale_to_fit_width(), reduce font_size, or split into multiple lines.
4. CLUTTER: Too many elements visible simultaneously. Stagger appearances \
with FadeIn/FadeOut, or group related elements.
5. BAD SCALE: Elements too small to read or too large relative to the frame. \
Adjust .scale() or font_size.
6. POOR SPACING: Items crammed together or spread too far apart. Use \
buff parameters, .arrange() with appropriate spacing, or manual positioning.
</visual_issues_to_check>

<rules>
- Return ONLY the fixed Python code
- Preserve ALL functionality and animations — only fix visual layout
- Keep same class names and structure
- NEVER remove content or animations
- If the code looks visually fine, return it unchanged
</rules>
"""


class ManimReviewer:
    def __init__(self, config: ReviewerConfig):
        self._config = config

    def review(self, code: str) -> str:
        """Review code for visual issues and return fixed version."""
        prompt = self._build_prompt(code)
        try:
            return self._call_llm(prompt)
        except Exception:
            return code

    def _build_prompt(self, code: str) -> str:
        return f"""<code>
{code}
</code>

Mentally simulate how each scene in this Manim code would look when rendered. \
Check for visual issues (off-screen elements, overlaps, text overflow, \
clutter, bad sizing, poor spacing). Fix any issues you find by adjusting \
positions, scales, font sizes, spacing, and layout. Return the full \
corrected code."""

    def _call_llm(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self._config.base_url,
            headers=headers,
            json={
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=240,
        )

        if response.status_code != 200:
            raise RuntimeError(f"API failed: {response.status_code}")

        content = response.json()["choices"][0]["message"]["content"]
        return self._extract_code(content)

    def _extract_code(self, content: str) -> str:
        lines = content.split("\n")
        in_block = False
        code_lines: list[str] = []

        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                code_lines.append(line)

        return "\n".join(code_lines) if code_lines else content


def review_manim_code(
    code: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Review Manim code for visual issues and return fixed version.

    Args:
        code: Manim source code
        api_key: OpenRouter API key (falls back to settings)
        model: Model to use (falls back to settings)

    Returns:
        Visually corrected code, or original code if no issues found
    """
    kwargs: dict = {}
    if api_key:
        kwargs["api_key"] = api_key
    if model:
        kwargs["model"] = model

    config = ReviewerConfig(**kwargs)

    if not config.api_key:
        raise ValueError(
            "API key required. Set OPENROUTER_API_KEY in .env or environment."
        )

    return ManimReviewer(config).review(code)

"""
Per-scene Manim code generation.

Generates a focused, self-contained Manim script for a single scene plan.
Each scene is an independent file with one Scene subclass named ``Main``,
renderable on its own without cross-scene dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

from manim_video_agent.config import get_settings
from manim_video_agent.core.scene_plan import ScenePlan, SharedContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneCodeResult:
    plan: ScenePlan
    code: str
    model: str
    usage: Optional[dict] = None
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt — modeled on the battle-tested SYSTEM_PROMPT in openrouter.py
# with per-scene architecture (no sub-methods, no clear_scene).
#
# Key design decisions (per Claude prompting best practices):
# - Concrete good/bad code examples for every common Manim pitfall
# - Context for WHY each rule exists (prevents model from "optimizing away")
# - Template structure matches desired output (Python code)
# - No XML tags — reduces markdown-style formatting in output
# ---------------------------------------------------------------------------

PER_SCENE_SYSTEM_PROMPT = """\
You are an expert ManimCE v0.19.0 animator creating one scene of a \
3Blue1Brown-style educational video. You produce a complete, runnable \
Python script with a single Scene class named `Main`. You have no output token limit.

Use ManimCE only (`from manim import *`). The ManimGL fork (manimlib) has \
diverged — many ManimGL parameters cause runtime errors in ManimCE.

ManimCE v0.19.0 breaking changes (vs v0.18.x):
- `Sector()` no longer accepts `inner_radius`/`outer_radius` — use `radius` \
and `angle`. For annular sectors use `AnnularSector`.
- `SurroundingRectangle` now accepts a sequence of Mobjects — pass \
`buff`, `color`, `corner_radius` etc. as keyword arguments, not positional.
- `Code` mobject was rewritten — use `Code.get_styles_list()` instead of \
`Code.styles_list`.
- `ManimColor.from_hex(hex=...)` is now `ManimColor.from_hex(hex_str=...)`.
- `Scene.next_section(type=...)` is now `Scene.next_section(section_type=...)`.
- FFmpeg replaced by pyav internally (no API impact, just FYI).

Generate exactly one Scene class named `Main`. Put all animations in a single \
`construct(self)` method. This scene is part {scene_index} of {total_scenes} — \
rendered independently and stitched together via FFmpeg.
- Do not include `self.clear_scene()` — the scene ends naturally.
- Do not reference objects from other scenes.
- End the scene cleanly (final elements can remain visible).

```python
from manim import *

class Main(Scene):
    def construct(self):
        self.camera.background_color = "{background_color}"

        # ... all animations for this scene ...

        print(f"Scene time: {{self.renderer.time}} seconds")
```

`Text()` is for plain ASCII only — Unicode math symbols (!=, ->, >=, Greek letters) \
render as blank boxes because the font lacks those glyphs. Use `MathTex(r"...")` \
for all math.
```python
# Correct: mixed text + math
VGroup(Text("When", font_size=32), MathTex(r"x \\geq 0")).arrange(RIGHT, buff=0.3)

# Wrong: Unicode in Text — renders blank boxes
Text("When x >= 0")
```

Background: `self.camera.background_color = "{background_color}"` (set once in construct). \
Use these semantic colors consistently: {color_palette_json}

Pacing: `self.wait(1-2)` after major animations. `run_time=1.0-1.5` for standard animations.

Performance: Prefer 2D `Scene` over `ThreeDScene` — only use 3D if the concept genuinely requires it. Keep visuals simple: text, boxes, bullet points, arrows, and basic shapes are preferred over complex geometry. Be mindful of render cost: avoid large per-frame updaters and expensive geometry that could cause timeouts.

Layout — overlap prevention (critical for professional quality):
- Vertical zones: title y in [2.5, 3.5], body y in [-2.0, 2.5], caption y in [-3.5, -2.0]. \
Never place independent element groups in overlapping vertical bands without \
chaining them via `.next_to()`.
- Side-by-side layouts: each column must be <= 6 units wide. Build each side as a VGroup, \
`.arrange(DOWN, buff=0.3)`, `.scale_to_fit_width(6)`, then \
`VGroup(left_col, right_col).arrange(RIGHT, buff=0.5).move_to(ORIGIN)`.
- Chain positioning: always use `.next_to(previous_element, DOWN/RIGHT, buff=0.4)` \
for sequential elements — never hardcode `.move_to()` or `.shift()` with coordinates \
that could collide with existing content.
- Text safety: apply `.scale_to_fit_width(config.frame_width - 2)` to any full-width \
text. For text inside a container, use `.scale_to_fit_width(container.width - 0.5)`.
- Anchor all labels: position labels, annotations, and captions with \
`.next_to(parent_element, direction, buff=0.25)` — never place floating text at \
arbitrary coordinates.
- Group-then-position: build VGroups of related content, `.arrange()` them internally, \
then position the entire group. This prevents individual elements from drifting apart.
- Cleanup: `FadeOut` elements no longer needed before adding new content to a crowded frame.

At the end of construct(), print the cumulative scene time:
```python
print(f"Scene time: {{self.renderer.time}} seconds")
```

Return the complete Python script inside a ```python fence. \
One Scene class named `Main`, immediately runnable."""


def _format_system_prompt(plan: ScenePlan, shared_ctx: SharedContext) -> str:
    return PER_SCENE_SYSTEM_PROMPT.format(
        background_color=shared_ctx.background_color,
        scene_index=plan.index,
        total_scenes=shared_ctx.total_scenes,
        color_palette_json=json.dumps(shared_ctx.color_palette, indent=2),
    )


def _format_user_prompt(plan: ScenePlan, skill_block: str) -> str:
    parts = [
        f"<scene_plan>\n{plan.markdown}\n</scene_plan>",
    ]

    if skill_block:
        parts.append(f"\n<skills>\n{skill_block}\n</skills>")

    parts.append(
        "\nCreate a complete, runnable Manim script for this scene. "
        "Cover all visual elements and animations described in the plan. "
        "Use diagrams, formulas, and visual metaphors to make abstract concepts concrete. "
        "Go beyond the basics — create a polished, fully-featured animation."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_scene_code(
    plan: ScenePlan,
    shared_context: SharedContext,
    skill_block: str = "",
    api_key: str | None = None,
    model: str | None = None,
) -> SceneCodeResult:
    settings = get_settings()
    key = api_key or settings.openrouter_api_key
    selected_model = model or settings.default_model

    system_prompt = _format_system_prompt(plan, shared_context)
    user_prompt = _format_user_prompt(plan, skill_block)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/manim-generator",
        "X-Title": "Manim Scene Generator",
    }

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 16384,
    }

    logger.info(
        "Generating scene %d (%s) with %s...",
        plan.index,
        plan.slug,
        selected_model,
    )

    response = requests.post(
        settings.openrouter_base_url,
        headers=headers,
        json=payload,
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Scene {plan.index} generation failed "
            f"({response.status_code}): {response.text[:500]}"
        )

    data = response.json()
    message = data["choices"][0]["message"]
    raw_content = message.get("content")

    if not raw_content:
        logger.warning(
            "Scene %d (%s): LLM returned null content (model: %s). "
            "Response had reasoning=%s chars. This usually means the model "
            "exhausted output tokens on thinking.",
            plan.index,
            plan.slug,
            selected_model,
            len(message.get("reasoning") or ""),
        )
        raise RuntimeError(
            f"Scene {plan.index} generation returned empty content "
            f"(model may have exhausted output tokens)"
        )

    code = _extract_code(raw_content)

    logger.info(
        "Scene %d (%s) generated: %d lines",
        plan.index,
        plan.slug,
        code.count("\n") + 1,
    )

    return SceneCodeResult(
        plan=plan,
        code=code,
        model=selected_model,
        usage=data.get("usage"),
        reasoning=message.get("reasoning"),
    )


def _strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_code(content: str | None) -> str:
    if not content:
        raise ValueError(
            "LLM returned empty/null content — likely exhausted output tokens "
            "on reasoning. Will retry without reasoning."
        )
    content = _strip_think_tags(content)
    match = re.search(r"```(?:python)?\s*\n(.+?)```", content, re.DOTALL)
    return (match.group(1) if match else content).strip()

"""
OpenRouter API client for Manim code generation.

Handles communication with the OpenRouter API, prompt construction,
and response parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from manim_video_agent.config import get_settings


@dataclass(frozen=True)
class OpenRouterConfig:
    """Configuration for OpenRouter API.

    Defaults are pulled from ``Settings`` so you only need to
    override what changes per-call.
    """

    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    site_url: str = "https://github.com/manim-generator"
    app_name: str = "Manim Generator"

    def __post_init__(self):
        settings = get_settings()
        # object.__setattr__ because frozen dataclass
        if not self.api_key:
            object.__setattr__(self, "api_key", settings.openrouter_api_key)
        if not self.base_url:
            object.__setattr__(self, "base_url", settings.openrouter_base_url)
        if not self.default_model:
            object.__setattr__(self, "default_model", settings.default_model)


@dataclass(frozen=True)
class GenerationResult:
    """Result of code generation."""

    code: str
    model: str
    usage: Optional[dict] = None
    reasoning: Optional[str] = None


class OpenRouterError(Exception):
    """Custom exception for OpenRouter API errors."""


SYSTEM_PROMPT = """\
<role>
You are an expert ManimCE v0.19.0 animator creating 3Blue1Brown-style educational videos. \
You produce complete, runnable Python scripts. You have no output token limit.

Render command:
docker run --rm -v "$(pwd)":/manim -w /manim manimcommunity/manim:v0.19.0 \
  manim -qh lesson_manim.py Main

Use ManimCE only (`from manim import *`). Do not use ManimGL (manimlib) APIs — \
the forks have diverged and many ManimGL parameters cause runtime errors in ManimCE.

ManimCE v0.19.0 breaking changes (vs v0.18.x):
- `Sector()` no longer accepts `inner_radius`/`outer_radius` — use `radius` \
and `angle`. For annular sectors use `AnnularSector`.
- `SurroundingRectangle` now accepts a sequence of Mobjects — pass \
`buff`, `color`, `corner_radius` etc. as keyword arguments, not positional.
- `Code` mobject was rewritten — use `Code.get_styles_list()` instead of \
`Code.styles_list`.
- `ManimColor.from_hex(hex=...)` is now `ManimColor.from_hex(hex_str=...)`.
- `Scene.next_section(type=...)` is now `Scene.next_section(section_type=...)`.
</role>

<architecture>
Generate exactly one Scene class named `Main` with helper methods per section. \
Every section method ends with `self.clear_scene()` except the final one.
```python
from manim import *

class Main(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e2e"
        self.intro()
        self.explanation()
        self.conclusion()

    def clear_scene(self):
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)

    def intro(self):
        title = Text("Topic", font_size=44).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        self.clear_scene()
```
</architecture>

<text_and_latex>
`Text()` is for plain ASCII only — Unicode math symbols (≠, →, ≥, Greek letters) \
render as blank boxes. Use `MathTex(r"...")` for all math.
```python
# Good: mixed text + math
VGroup(Text("When", font_size=32), MathTex(r"x \\geq 0")).arrange(RIGHT, buff=0.3)

# Bad: Unicode in Text — blank boxes
Text("When x ≥ 0")
```
</text_and_latex>

<layout>
Frame: ~14.2 × 8 units. Safe area: x ∈ [-6, 6], y ∈ [-3.5, 3.5]. Keep bottom 15% clear.

Position the first element with `.to_edge()`, then chain subsequent elements with \
`.next_to()`. Use `.arrange()` for groups. Avoid hardcoded coordinates.
```python
# Good: anchor chain
title = Text("Topic", font_size=44).to_edge(UP)
subtitle = Text("Sub", font_size=32).next_to(title, DOWN, buff=0.5)

# Bad: overlapping
title = Text("Topic").to_edge(UP)
subtitle = Text("Sub").to_edge(UP).shift(DOWN)
```

Sizing: MathTex defaults to `.scale(0.8)`. Wide content: `.scale_to_fit_width(config.frame_width - 2)`. \
Text lines under ~60 chars. Font sizes: titles=44, body=28-32, captions=22. \
Max 3-5 elements visible. Reveal lists one-by-one. Max 2 side-by-side columns.
</layout>

<numpy_constants>
ORIGIN, UP, DOWN, LEFT, RIGHT, UL, UR, DL, DR are numpy arrays, NOT Mobjects. \
Never call Mobject methods (.shift(), .scale(), .move_to(), .next_to(), .animate, etc.) on them.
```python
# Bad: ORIGIN is a numpy array, .shift() is a Mobject method — crashes at runtime
center = ORIGIN.copy().shift(DOWN * 0.3)

# Good: use numpy array arithmetic
center = ORIGIN + DOWN * 0.3

# Good: create a Dot, then shift it
center = Dot(ORIGIN).shift(DOWN * 0.3).get_center()
```
</numpy_constants>

<visual_style>
Background: `self.camera.background_color = "#1e1e2e"` (set once in construct). \
Use 3-4 semantic colors: Green=#a6e3a1 (positive), Red=#f38ba8 (negative), \
Teal=#89dceb (neutral), Yellow=#f9e2af (emphasis). \
Pacing: `self.wait(1-2)` after major animations. `run_time=1.0-1.5` for standard animations. \
Always fade out everything between sections.
</visual_style>

<timing_output>
At the end of each section method (before `self.clear_scene()`), print the cumulative renderer time:
```python
print(f"Total time so far: {self.renderer.time} seconds")
```
This tracks total animation time across all sections. Include it in every helper method, \
not in `construct()`.
</timing_output>

<output_format>
Return only the complete Python script — no explanations or markdown fences. \
One Scene class named `Main`, immediately runnable.
</output_format>"""


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(self, config: OpenRouterConfig):
        self._config = config

    def generate_manim_code(
        self,
        content: str,
        model: Optional[str] = None,
        skill_block: str = "",
        scene_plan: Optional[str] = None,
    ) -> GenerationResult:
        """Generate Manim code from markdown content.

        Args:
            content: The markdown content to convert to Manim video
            model: Optional model override
            skill_block: Pre-formatted ManimCE skill rules to append to the system prompt
            scene_plan: Optional scene-by-scene plan to guide code generation

        Returns:
            GenerationResult with generated code

        Raises:
            OpenRouterError: If API call fails
        """
        selected_model = model or self._config.default_model

        headers = self._build_headers()
        payload = self._build_payload(
            content, selected_model, skill_block, scene_plan=scene_plan
        )

        response = requests.post(
            self._config.base_url, headers=headers, json=payload, timeout=240
        )

        if response.status_code != 200:
            raise OpenRouterError(
                f"API request failed: {response.status_code} - {response.text}"
            )

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            body_preview = response.text[:500]
            raise OpenRouterError(
                f"Response returned 200 but body is not valid JSON "
                f"(Content-Length: {response.headers.get('content-length', '?')}, "
                f"received {len(response.text)} chars). "
                f"Body preview: {body_preview!r}"
            ) from e

        return self._parse_response(data, selected_model)

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._config.site_url,
            "X-Title": self._config.app_name,
        }

    def _build_payload(
        self,
        content: str,
        model: str,
        skill_block: str = "",
        scene_plan: Optional[str] = None,
    ) -> dict:
        system_prompt = SYSTEM_PROMPT
        if skill_block:
            system_prompt = system_prompt + "\n\n" + f"<skills>{skill_block}</skills>"

        scene_plan_block = ""
        if scene_plan:
            scene_plan_block = (
                f"\n\n<scene_plan>\n{scene_plan}\n</scene_plan>\n\n"
                "Follow the scene plan above to structure your Manim code. "
                "Each scene in the plan should correspond to a section method."
            )

        user_prompt = f"""<content>
{content}
</content>
{scene_plan_block}
Create a complete Manim video script that explains all key concepts from the content above.

Requirements:
- Cover every major idea with a dedicated visual section
- Use diagrams, formulas, and visual metaphors to make abstract concepts concrete
- Follow all layout, animation, and LaTeX rules from your instructions exactly"""

        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 16384,
        }

    def _parse_response(self, data: dict, model: str) -> GenerationResult:
        try:
            message = data["choices"][0]["message"]
            raw_content = message.get("content")
            usage = data.get("usage")
            reasoning = message.get("reasoning")

            if not raw_content:
                raise OpenRouterError(
                    "LLM returned empty/null content — model may have "
                    "exhausted output tokens on reasoning"
                )

            code = self._extract_code_from_markdown(raw_content)

            return GenerationResult(
                code=code, model=model, usage=usage, reasoning=reasoning
            )
        except (KeyError, IndexError) as e:
            raise OpenRouterError(f"Failed to parse response: {e}")

    def _extract_code_from_markdown(self, content: str) -> str:
        import re

        if not content:
            return ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        lines = content.split("\n")
        in_code_block = False
        code_lines: list[str] = []

        for line in lines:
            if line.strip().startswith("```python"):
                in_code_block = True
                continue
            elif line.strip() == "```" and in_code_block:
                in_code_block = False
                continue

            if in_code_block:
                code_lines.append(line)

        if not code_lines:
            return content

        return "\n".join(code_lines)

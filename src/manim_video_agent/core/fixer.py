"""
Manim Code Fixer Agent

LLM-based service that validates generated Manim code and fixes errors.
Only modifies code when errors are detected — returns original if clean.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Optional

import requests

from manim_video_agent.config import get_settings
from manim_video_agent.core.validator import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    Severity,
    ValidationIssue,
    validate_source,
)


@dataclass(frozen=True)
class FixerConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_iterations: int = 0

    def __post_init__(self):
        settings = get_settings()
        if not self.api_key:
            object.__setattr__(self, "api_key", settings.openrouter_api_key)
        if not self.base_url:
            object.__setattr__(self, "base_url", settings.openrouter_base_url)
        if not self.model:
            object.__setattr__(self, "model", settings.fixer_model)
        if not self.max_iterations:
            object.__setattr__(self, "max_iterations", settings.fixer_max_iterations)


FIXER_SYSTEM_PROMPT = """\
<role>
You are a Manim code fixer. You receive Manim code with detected errors and fix them.
Output ONLY the corrected Python code - no explanations, no markdown fences.
</role>

<manim_frame_dimensions>
- Width: ~14.22 units, Height: 8 units
- Center at (0, 0)
- Safe bounds: x in [-6.5, 6.5], y in [-3.5, 3.5]
</manim_frame_dimensions>

<fixes>
1. TEXT OVERFLOW: Add .scale_to_fit_width(config.frame_width - 2), reduce font_size, or split text
2. OFF-SCREEN: Clamp coordinates, use .to_edge() or .next_to()
3. LARGE FONT: Titles=44, body=28, captions=22
4. UNICODE IN Text(): Replace with MathTex or VGroup arrangement
5. CONTAINER OVERFLOW: Increase container size or reduce content
</fixes>

<rules>
- Return ONLY fixed Python code
- Preserve all functionality
- Keep same class names and structure
- NEVER remove content
</rules>
"""


class ManimFixer:
    def __init__(self, config: FixerConfig):
        self._config = config

    def fix(self, code: str) -> str:
        """Fix Manim code if it has errors. Returns original code if no errors."""
        issues = validate_source(code)
        errors = [i for i in issues if i.severity == Severity.ERROR]

        if not errors:
            return code

        current_code = code
        for _ in range(self._config.max_iterations):
            class_ranges = _get_scene_class_ranges(current_code)
            if not class_ranges:
                prompt = self._build_prompt(current_code, errors)
                try:
                    current_code = self._call_llm(prompt)
                except Exception:
                    break
            else:
                if any(e.scene_name not in class_ranges for e in errors):
                    prompt = self._build_prompt(current_code, errors)
                    try:
                        current_code = self._call_llm(prompt)
                    except Exception:
                        break
                else:
                    for scene_name in _scene_fix_order(errors, class_ranges):
                        scene_errors = [e for e in errors if e.scene_name == scene_name]
                        class_ranges = _get_scene_class_ranges(current_code)
                        if scene_name not in class_ranges:
                            continue
                        scene_code = _extract_scene_class(
                            current_code, class_ranges[scene_name]
                        )
                        prompt = self._build_scene_prompt(
                            scene_name, scene_code, scene_errors
                        )
                        try:
                            fixed_scene = self._call_llm(prompt)
                        except Exception:
                            continue
                        current_code = _replace_scene_class(
                            current_code, class_ranges[scene_name], fixed_scene
                        )

            issues = validate_source(current_code)
            errors = [i for i in issues if i.severity == Severity.ERROR]

            if not errors:
                break

        return current_code

    def _build_prompt(self, code: str, errors: list[ValidationIssue]) -> str:
        errors_text = "\n".join(
            f"- Line {e.line}: {e.message}"
            + (f" (Fix: {e.suggestion})" if e.suggestion else "")
            for e in errors
        )

        return f"""<code>
{code}
</code>

<errors>
{errors_text}
</errors>

<frame>
{FRAME_WIDTH:.2f} x {FRAME_HEIGHT:.2f} units. Safe: x[-6.5, 6.5], y[-3.5, 3.5]
</frame>

Fix all errors. Return only the corrected Python code."""

    def _build_scene_prompt(
        self, scene_name: str, scene_code: str, errors: list[ValidationIssue]
    ) -> str:
        errors_text = "\n".join(
            f"- Line {e.line}: {e.message}"
            + (f" (Fix: {e.suggestion})" if e.suggestion else "")
            for e in errors
        )

        return f"""<scene_name>
{scene_name}
</scene_name>

<scene_code>
{scene_code}
</scene_code>

<errors>
{errors_text}
</errors>

<frame>
{FRAME_WIDTH:.2f} x {FRAME_HEIGHT:.2f} units. Safe: x[-6.5, 6.5], y[-3.5, 3.5]
</frame>

Fix all errors in this scene only. Return ONLY the corrected class code for {scene_name}."""

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
                    {"role": "system", "content": FIXER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=240,
        )

        if response.status_code != 200:
            raise RuntimeError(f"API failed: {response.status_code}")

        raw_content = response.json()["choices"][0]["message"].get("content")
        if not raw_content:
            raise RuntimeError("Fixer LLM returned empty/null content")
        return self._extract_code(raw_content)

    def _extract_code(self, content: str) -> str:
        import re

        if not content:
            return ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
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


def fix_manim_code(
    code: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> list[ValidationIssue]:
    """Collect Manim code errors (deprecated: no longer fixes code).

    Args:
        code: Manim source code
        api_key: Unused (kept for backward compatibility)
        model: Unused (kept for backward compatibility)
        max_iterations: Unused (kept for backward compatibility)

    Returns:
        List of validation errors found in the code
    """
    _ = api_key, model, max_iterations
    return collect_code_errors(code)


def collect_code_errors(code: str) -> list[ValidationIssue]:
    """Collect validation errors in Manim code without fixing."""
    issues = validate_source(code)
    return [i for i in issues if i.severity == Severity.ERROR]


def format_code_error_report(errors: list[ValidationIssue]) -> str:
    """Format validation errors for LLM prompts."""
    if not errors:
        return ""
    lines = []
    for e in errors:
        line = f"- {e.scene_name}:{e.line} {e.message}"
        if e.suggestion:
            line += f" (Fix: {e.suggestion})"
        lines.append(line)
    return "\n".join(lines)


def _get_scene_class_ranges(source: str) -> dict[str, tuple[int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    ranges: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue
        ranges[node.name] = (start, end)
    return ranges


def _extract_scene_class(source: str, line_range: tuple[int, int]) -> str:
    lines = source.splitlines()
    start, end = line_range
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    return "\n".join(lines[start_idx:end_idx])


def _replace_scene_class(
    source: str, line_range: tuple[int, int], new_class_code: str
) -> str:
    lines = source.splitlines()
    start, end = line_range
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    new_lines = new_class_code.splitlines()
    return "\n".join(lines[:start_idx] + new_lines + lines[end_idx:])


def _scene_fix_order(
    errors: list[ValidationIssue], class_ranges: dict[str, tuple[int, int]]
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for e in errors:
        if e.scene_name in seen:
            continue
        seen.add(e.scene_name)
        ordered.append(e.scene_name)
    ordered.sort(key=lambda name: class_ranges.get(name, (0, 0))[0], reverse=True)
    return ordered


# ---------------------------------------------------------------------------
# Method-level extraction / replacement
# ---------------------------------------------------------------------------


def _get_method_ranges(source: str, class_name: str) -> dict[str, tuple[int, int]]:
    """Return ``{method_name: (start_line, end_line)}`` for every method
    defined inside *class_name*.  Line numbers are 1-indexed."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            ranges: dict[str, tuple[int, int]] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(item, "lineno", None)
                    end = getattr(item, "end_lineno", None)
                    if start is not None and end is not None:
                        ranges[item.name] = (start, end)
            return ranges
    return {}


def _extract_method(source: str, line_range: tuple[int, int]) -> str:
    """Extract a method's source lines (preserving original indentation)."""
    lines = source.splitlines()
    start, end = line_range
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    return "\n".join(lines[start_idx:end_idx])


def _replace_method(
    source: str, line_range: tuple[int, int], new_method_code: str
) -> str:
    """Splice *new_method_code* into *source*, replacing the lines at
    *line_range*.  Returns the full source with the method swapped."""
    lines = source.splitlines()
    start, end = line_range
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    new_lines = new_method_code.splitlines()
    return "\n".join(lines[:start_idx] + new_lines + lines[end_idx:])


_FRAME_SUFFIX_RE = re.compile(r"_f\d+$")


def parse_frame_to_method(filename: str) -> str | None:
    """Map a frame filename to the scene method it represents.

    Supports both single-frame and multi-frame naming conventions:
    ``"01_explanation.png"``      → ``"explanation"``
    ``"01_explanation_f02.png"``  → ``"explanation"``
    ``"00_intro_f00.png"``        → ``"intro"``

    Returns ``None`` if the filename doesn't follow the expected pattern.
    """
    name = filename.rsplit(".", 1)[0]  # strip extension
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return _FRAME_SUFFIX_RE.sub("", parts[1])
    return None

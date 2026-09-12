"""
Extract per-scene animation durations from generated Manim code.

Parses ``self.play()``, ``self.wait()``, and ``self.clear_scene()`` calls
within each scene method of the ``Main(Scene)`` class to compute timing.
Uses regex rather than ``ast`` because the generated code includes
monkey-patching preambles that confuse the AST parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Manim default when run_time is omitted from self.play()
DEFAULT_ANIMATION_RUN_TIME: float = 1.0

# clear_scene() does FadeOut(all) + wait(0.5) — round up
CLEAR_SCENE_DURATION: float = 1.0


@dataclass(frozen=True)
class SceneTiming:
    """Duration of a single scene method."""

    method_name: str
    duration_seconds: float


def parse_single_scene_timing(
    manim_code: str,
    scene_id: str | None = None,
) -> SceneTiming | None:
    """Extract timing from a per-scene file where ``construct()`` has all animations.

    Unlike :func:`parse_scene_timings` which looks for sub-method dispatching,
    this computes duration directly from the ``construct()`` method body.
    Used by the parallel pipeline for independent scene files.

    Args:
        manim_code: Source code of the per-scene file.
        scene_id: Unique identifier for this scene (e.g. plan.slug).
            When ``None``, falls back to the class name found in the code.
            In the parallel pipeline every file has ``class Main``, so
            callers MUST pass a unique ``scene_id`` to avoid key collisions
            in narration dicts.

    Returns a :class:`SceneTiming` or ``None`` if ``construct()`` is not found.
    """
    body = _extract_method_body(manim_code, "construct")
    if body is None:
        return None

    duration = _compute_body_duration(body)

    if scene_id is None:
        class_match = re.search(r"class\s+(\w+)\s*\(", manim_code)
        scene_id = class_match.group(1) if class_match else "construct"

    return SceneTiming(method_name=scene_id, duration_seconds=round(duration, 2))


def parse_scene_timings(manim_code: str) -> list[SceneTiming]:
    """Extract per-scene durations from Manim source code.

    Looks for the ``Main(Scene)`` class, discovers which methods are
    called from ``construct()``, then sums animation/wait durations in
    each method body.

    Returns a list of :class:`SceneTiming` in call order.
    """
    scene_methods = _get_construct_calls(manim_code)
    if not scene_methods:
        return []

    timings: list[SceneTiming] = []
    for method_name in scene_methods:
        body = _extract_method_body(manim_code, method_name)
        if body is None:
            continue
        duration = _compute_body_duration(body)
        timings.append(
            SceneTiming(method_name=method_name, duration_seconds=round(duration, 2))
        )

    return timings


def total_duration(timings: list[SceneTiming]) -> float:
    """Sum all scene durations."""
    return round(sum(t.duration_seconds for t in timings), 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches "self.method_name()" inside construct
_CONSTRUCT_CALL_RE = re.compile(r"self\.(\w+)\(\)")

# Matches "self.play(...)" — may span multiple lines
_PLAY_CALL_RE = re.compile(r"self\.play\(", re.DOTALL)

# Matches "run_time=<number>" inside a play call
_RUN_TIME_RE = re.compile(r"run_time\s*=\s*([\d.]+)")

# Matches "self.wait(<number>)" or "self.wait()"
_WAIT_RE = re.compile(r"self\.wait\(\s*([\d.]*)\s*\)")

# Matches "self.clear_scene()"
_CLEAR_SCENE_RE = re.compile(r"self\.clear_scene\(\)")


def _get_construct_calls(code: str) -> list[str]:
    """Return ordered list of method names called from ``construct()``."""
    construct_body = _extract_method_body(code, "construct")
    if construct_body is None:
        return []

    calls: list[str] = []
    for match in _CONSTRUCT_CALL_RE.finditer(construct_body):
        name = match.group(1)
        # Skip non-scene helpers like clear_scene itself
        if name not in ("clear_scene",):
            calls.append(name)
    return calls


def _extract_method_body(code: str, method_name: str) -> str | None:
    """Extract the body of ``def method_name(self, ...)`` at class indent level.

    Returns the body text (everything after the ``def`` line until the next
    method definition at the same indent level), or ``None`` if not found.
    """
    # Match "def method_name(self" with capture of leading indent
    pattern = re.compile(
        rf"^( +)def {re.escape(method_name)}\(self[^)]*\):[^\n]*\n",
        re.MULTILINE,
    )
    match = pattern.search(code)
    if match is None:
        return None

    indent = match.group(1)
    body_start = match.end()

    # Body continues until we hit another def at the same indent or end of class
    end_pattern = re.compile(rf"^{re.escape(indent)}def \w+\(", re.MULTILINE)
    end_match = end_pattern.search(code, body_start)
    body_end = end_match.start() if end_match else len(code)

    return code[body_start:body_end]


def _compute_body_duration(body: str) -> float:
    """Sum durations of play/wait/clear_scene calls in a method body."""
    duration = 0.0

    # Count self.play() calls
    for play_match in _PLAY_CALL_RE.finditer(body):
        # Find the matching closing paren (handle nested parens)
        call_text = _extract_balanced_parens(body, play_match.start())
        rt_match = _RUN_TIME_RE.search(call_text)
        if rt_match:
            duration += float(rt_match.group(1))
        else:
            duration += DEFAULT_ANIMATION_RUN_TIME

    # Count self.wait() calls
    for wait_match in _WAIT_RE.finditer(body):
        val = wait_match.group(1).strip()
        if val:
            duration += float(val)
        else:
            duration += DEFAULT_ANIMATION_RUN_TIME

    # Count self.clear_scene() calls
    duration += len(_CLEAR_SCENE_RE.findall(body)) * CLEAR_SCENE_DURATION

    return duration


def _extract_balanced_parens(text: str, start: int) -> str:
    """Extract text from ``start`` up to and including the balanced closing paren.

    ``start`` should point to the beginning of something like ``self.play(``.
    """
    # Find the opening paren
    paren_pos = text.index("(", start)
    depth = 0
    i = paren_pos
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    # Unbalanced — return best effort
    return text[start:]

"""
Visual frame review pipeline.

Renders last frames via Docker, reviews them with Gemini vision,
and feeds defect reports to Sonnet for code fixes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx
import requests

from manim_video_agent.config import get_settings
from manim_video_agent.core.fixer import (
    _extract_method,
    _extract_scene_class,
    _get_method_ranges,
    _get_scene_class_ranges,
    _replace_method,
    _replace_scene_class,
    _scene_fix_order,
    format_code_error_report,
    parse_frame_to_method,
)

# Gemini 3 Flash pricing per OpenRouter (per token)
_PRICE_INPUT_PER_TOKEN = 0.15 / 1_000_000
_PRICE_OUTPUT_PER_TOKEN = 0.60 / 1_000_000

REVIEW_SYSTEM_PROMPT = """\
<role>
You are a visual quality inspector for Manim-rendered educational video frames. \
You receive a screenshot of a single animation frame and must identify any \
visual defects that would make the frame look broken, unprofessional, or \
unreadable to a viewer.
</role>

<frame_context>
These frames come from ManimCE v0.19.0 educational animations. The background \
is dark (#1e1e2e). Elements use a Catppuccin-inspired color palette. Frames \
capture key moments throughout each scene — some show early states with few \
elements being introduced, while others show fully built compositions. Each \
frame represents a pause point where the viewer is meant to absorb the content.
</frame_context>

<defects_to_detect>
Carefully inspect the image for ALL of the following defect categories:

1. TEXT OVERLAP: Any text or labels sitting on top of each other, making one or \
both unreadable. Includes partially overlapping text where characters bleed into \
adjacent elements.

2. TEXT OVERFLOW: Text extending beyond its container box/rectangle, or text \
running off the visible frame edge. Text that is clipped or truncated.

3. ELEMENT COLLISION: Non-text elements (boxes, arrows, circles, shapes) \
overlapping in a way that obscures content or creates visual confusion. \
Arrows pointing through unrelated elements.

4. OFF-SCREEN CONTENT: Any element partially or fully cut off by the frame \
boundary. Content that appears to continue beyond the visible area.

5. UNREADABLE TEXT: Text too small to read, poor contrast against background, \
or text rendered as blank/garbled characters (common with Unicode in Manim's \
Text() class).

6. BROKEN LAYOUT: Elements that appear randomly scattered rather than \
intentionally arranged. Misaligned columns or rows. Asymmetric layouts that \
should be symmetric.

7. VISUAL CLUTTER: Too many elements crammed together making the frame \
overwhelming and hard to parse. No clear visual hierarchy.

8. MISSING OR EMPTY ELEMENTS: Boxes/containers that appear empty when they \
should contain text. Arrows pointing to nothing. Orphaned labels without \
their corresponding visual element.

9. Z-ORDER ISSUES: Background elements rendering on top of foreground \
elements. Fill colors covering text or important details.

10. SPACING ISSUES: Elements crammed together with no breathing room, or \
excessive gaps that break visual grouping.
</defects_to_detect>

<response_guidelines>
- Set status to "PASS" if no defects found, "FAIL" if any defects found
- Severity levels: "high" (unreadable/broken), "medium" (unprofessional), "low" (minor cosmetic)
- summary: one-sentence overall assessment
- defects array: empty if PASS, populated if FAIL
</response_guidelines>

<guidelines>
- Be precise: describe WHERE in the frame each defect occurs
- Be honest: if the frame looks fine, say PASS — do not invent problems
- Be thorough: check every element in the frame, not just the obvious ones
- Intentional design choices (like overlapping Venn diagram circles) are NOT defects
- Dark/empty backgrounds between elements are normal, not defects
</guidelines>"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "frame_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["PASS", "FAIL"],
                    "description": "PASS if no defects, FAIL if any defects found",
                },
                "defects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": [
                                    "TEXT OVERLAP",
                                    "TEXT OVERFLOW",
                                    "ELEMENT COLLISION",
                                    "OFF-SCREEN CONTENT",
                                    "UNREADABLE TEXT",
                                    "BROKEN LAYOUT",
                                    "VISUAL CLUTTER",
                                    "MISSING OR EMPTY ELEMENTS",
                                    "Z-ORDER ISSUES",
                                    "SPACING ISSUES",
                                ],
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "description": {
                                "type": "string",
                                "description": "Where in the frame and what exactly is wrong",
                            },
                        },
                        "required": ["category", "severity", "description"],
                        "additionalProperties": False,
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "One-sentence overall assessment of the frame",
                },
            },
            "required": ["status", "defects", "summary"],
            "additionalProperties": False,
        },
    },
}

VISUAL_FIXER_SYSTEM_PROMPT = """\
<role>
You are a Manim code fixer. You receive Manim code along with a visual defect \
report and screenshots of the defective frames. Use the screenshots to \
understand exactly what is wrong — they show the actual rendered output. \
Fix the code to resolve ALL reported issues. Output ONLY the corrected Python \
code — no explanations, no markdown fences.
</role>

<manim_frame_dimensions>
- Width: ~14.22 units, Height: 8 units
- Center at (0, 0)
- Safe bounds: x in [-6.5, 6.5], y in [-3.5, 3.5]
- Elements near the edges may be clipped or feel cramped
</manim_frame_dimensions>

<fixes>
1. TEXT OVERLAP: Spread elements with .next_to(), .arrange(), VGroup spacing, \
or reposition manually.
2. TEXT OVERFLOW: Use .scale_to_fit_width(), reduce font_size, or split text.
3. OFF-SCREEN: Clamp coordinates within safe bounds, use .to_edge() or .move_to().
4. ELEMENT COLLISION: Adjust positions, increase buff parameters, rearrange layout.
5. UNREADABLE TEXT: Increase font_size, improve contrast, replace Unicode with \
MathTex or VGroup.
6. BROKEN LAYOUT: Use .arrange(), VGroup, or explicit positioning for alignment.
7. VISUAL CLUTTER: Reduce simultaneous elements, stagger with FadeIn/FadeOut.
8. MISSING ELEMENTS: Ensure all containers have content, all arrows have targets.
9. Z-ORDER: Use bring_to_front/bring_to_back, reorder element creation.
10. SPACING: Adjust buff parameters, use .arrange() with appropriate spacing.
</fixes>

<rules>
- Return ONLY the fixed Python code
- Preserve ALL functionality and animations — only fix layout or errors
- Keep same class names and structure
- NEVER remove content or animations
- If a defect report mentions a specific scene method, focus fixes there
</rules>"""


@dataclass(frozen=True)
class FrameReview:
    filename: str
    status: str
    defects: list[dict]
    summary: str
    cost: float = 0.0
    error: str = ""


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


async def _review_frame(
    client: httpx.AsyncClient, path: Path, api_key: str, model: str
) -> FrameReview:
    """Send a single frame to Gemini for visual review."""
    image_b64 = _encode_image(path)

    try:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Review this Manim frame '{path.name}' for visual defects. "
                                    "Inspect every element carefully."
                                ),
                            },
                        ],
                    },
                ],
                "response_format": RESPONSE_SCHEMA,
                "temperature": 0.2,
            },
            timeout=60,
        )
    except httpx.HTTPError as exc:
        return FrameReview(
            filename=path.name, status="ERROR", defects=[], summary="", error=str(exc)
        )

    if response.status_code != 200:
        return FrameReview(
            filename=path.name,
            status="ERROR",
            defects=[],
            summary="",
            error=f"API returned {response.status_code}: {response.text}",
        )

    data = response.json()
    raw_content = data["choices"][0]["message"].get("content")
    if not raw_content:
        return FrameReview(
            filename=path.name,
            status="ERROR",
            defects=[],
            summary="",
            error="Reviewer LLM returned empty/null content",
        )
    result = json.loads(raw_content)

    usage = data.get("usage", {})
    cost = (
        usage.get("prompt_tokens", 0) * _PRICE_INPUT_PER_TOKEN
        + usage.get("completion_tokens", 0) * _PRICE_OUTPUT_PER_TOKEN
    )

    return FrameReview(
        filename=path.name,
        status=result["status"],
        defects=result["defects"],
        summary=result["summary"],
        cost=cost,
    )


async def review_all_frames(
    frames_dir: Path, api_key: str, model: str | None = None
) -> list[FrameReview]:
    """Review all PNG frames in a directory using Gemini vision (parallel).

    Args:
        frames_dir: Directory containing PNG frame screenshots.
        api_key: OpenRouter API key.
        model: Vision model to use (falls back to settings).

    Returns:
        List of FrameReview results, one per frame.
    """
    if model is None:
        model = get_settings().visual_reviewer_model

    images = sorted(frames_dir.glob("*.png"))
    if not images:
        return []

    async with httpx.AsyncClient() as client:
        tasks = [_review_frame(client, path, api_key, model) for path in images]
        results = await asyncio.gather(*tasks)

    return list(results)


def render_lastframes(code_file: Path, output_dir: Path) -> Path:
    """Render last frames using save_lastframes.py.

    When running inside Docker (MANIM_VIDEO_AGENT_IN_DOCKER=true), calls
    save_lastframes.py directly via subprocess. Otherwise, shells out
    to ``docker compose run`` to render inside a container.

    Args:
        code_file: Path to the Manim Python file to render.
        output_dir: Directory where PNGs will be saved.

    Returns:
        The output directory path.

    Raises:
        RuntimeError: If rendering fails.
    """
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    if settings.manim_video_agent_in_docker:
        cmd = [
            "python",
            "save_lastframes.py",
            str(code_file),
            "--output-dir",
            str(output_dir),
        ]
    else:
        cmd = [
            "docker",
            "compose",
            "run",
            "--rm",
            "manim",
            "python",
            "save_lastframes.py",
            str(code_file),
            "--output-dir",
            str(output_dir),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    stdout = result.stdout or ""
    if stdout:
        print(stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Render failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Check that at least one frame was actually produced
    frame_count = len(list(output_dir.glob("*.png")))
    if frame_count == 0:
        stderr_hint = result.stderr[-500:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"Render exited 0 but produced 0 frames in {output_dir}.\n"
            f"stderr: {stderr_hint}"
        )

    return output_dir, stdout


def format_defect_report(reviews: list[FrameReview]) -> str:
    """Format failed frame reviews into a prompt-friendly defect report.

    Args:
        reviews: List of FrameReview results (only FAIL entries are included).

    Returns:
        Structured defect report string for the fixer LLM.
    """
    failed = [r for r in reviews if r.status == "FAIL"]
    if not failed:
        return ""

    sections: list[str] = []
    for review in failed:
        lines = [f"Frame: {review.filename}", f"  Summary: {review.summary}"]
        for defect in review.defects:
            severity = defect.get("severity", "?").upper()
            category = defect.get("category", "?")
            desc = defect.get("description", "?")
            lines.append(f"  [{severity}] {category}: {desc}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _build_fix_prompt(code: str, defect_report: str, code_error_report: str) -> str:
    return f"""<code>
{code}
</code>

<visual_defect_report>
{defect_report}
</visual_defect_report>

<code_error_report>
{code_error_report}
</code_error_report>

The above reports were generated by reviewing rendered screenshots and/or \
static code validation. Fix all reported issues by adjusting positions, \
scales, font sizes, spacing, and layout, and by fixing any code errors. \
Return only the corrected Python code."""


def _build_scene_fix_prompt(
    scene_name: str,
    scene_code: str,
    defect_report: str,
    code_error_report: str,
) -> str:
    return f"""<scene_name>
{scene_name}
</scene_name>

<scene_code>
{scene_code}
</scene_code>

<visual_defect_report>
{defect_report}
</visual_defect_report>

<code_error_report>
{code_error_report}
</code_error_report>

Fix all reported issues in this scene only. Return ONLY the corrected class \
code for {scene_name}."""


def _build_method_fix_prompt(
    method_name: str,
    method_code: str,
    construct_code: str,
    defect_report: str,
    code_error_report: str,
) -> str:
    return f"""<class_context>
The method below belongs to a Manim Scene class. Here is the construct() \
method for context (DO NOT modify it):
{construct_code}
</class_context>

<method_to_fix>
{method_code}
</method_to_fix>

<visual_defect_report>
{defect_report}
</visual_defect_report>

<code_error_report>
{code_error_report}
</code_error_report>

Fix all reported issues in the `{method_name}` method ONLY. Return ONLY the \
corrected method code (including the `def {method_name}(self):` line), \
preserving the original indentation level."""


def _call_visual_fixer_llm(
    prompt: str,
    api_key: str,
    model: str,
    image_paths: list[Path] | None = None,
) -> str:
    settings = get_settings()

    if image_paths:
        user_content: list[dict] | str = [{"type": "text", "text": prompt}]
        for path in image_paths:
            img_b64 = _encode_image(path)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )
    else:
        user_content = prompt

    response = requests.post(
        settings.openrouter_base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": VISUAL_FIXER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 16384,
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Visual fixer API failed: {response.status_code}: {response.text}"
        )

    raw_content = response.json()["choices"][0]["message"].get("content")
    if not raw_content:
        raise RuntimeError("Visual fixer LLM returned empty/null content")
    return _extract_code(raw_content)


def _group_reviews_by_method(
    reviews: list[FrameReview],
) -> dict[str, list[FrameReview]]:
    """Map method names to their failed FrameReview entries."""
    grouped: dict[str, list[FrameReview]] = {}
    for review in reviews:
        if review.status != "FAIL":
            continue
        method = parse_frame_to_method(review.filename)
        if method is None:
            continue
        grouped.setdefault(method, []).append(review)
    return grouped


def _format_method_defect_report(reviews: list[FrameReview]) -> str:
    """Build a defect report string from a list of reviews for one method."""
    sections: list[str] = []
    for review in reviews:
        lines = [f"Frame: {review.filename}", f"  Summary: {review.summary}"]
        for defect in review.defects:
            severity = defect.get("severity", "?").upper()
            category = defect.get("category", "?")
            desc = defect.get("description", "?")
            lines.append(f"  [{severity}] {category}: {desc}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _collect_image_paths(
    reviews: list[FrameReview], frames_dir: Path | None
) -> list[Path]:
    """Resolve failed frame filenames to image paths on disk."""
    if not frames_dir:
        return []
    paths = []
    for review in reviews:
        path = frames_dir / review.filename
        if path.exists():
            paths.append(path)
    return paths


def fix_from_reports(
    code: str,
    defect_report: str,
    code_errors: list | None,
    api_key: str,
    model: str | None = None,
    failed_reviews: list[FrameReview] | None = None,
    frames_dir: Path | None = None,
) -> str:
    """Call fixer with visual + code error reports to get fixed code.

    When *failed_reviews* is provided and the code has a single scene class,
    the fixer operates at the **method level**: only the methods whose frames
    failed are extracted, fixed individually, and spliced back.  This keeps
    the LLM context small and prevents accidental breakage of passing scenes.

    Falls back to scene-level or full-code fixing when method-scoped fixing
    is not possible (e.g. frame filenames can't be mapped to methods).

    When *frames_dir* is provided, screenshots of the defective frames are
    sent alongside the text report so the fixer can see the actual defects.

    Args:
        code: Current Manim source code.
        defect_report: Structured defect report from visual review.
        code_errors: Validation errors from static code checks.
        api_key: OpenRouter API key.
        model: Model to use for fixing (falls back to settings).
        failed_reviews: Raw FrameReview objects for method-level mapping.
        frames_dir: Directory containing rendered frame PNGs.

    Returns:
        Fixed Manim code.
    """
    if model is None:
        model = get_settings().visual_fixer_model

    code_errors = code_errors or []
    code_error_report = format_code_error_report(code_errors)

    if not defect_report and not code_error_report:
        print("[Fixer] No defects and no code errors — nothing to fix.")
        return code

    class_ranges = _get_scene_class_ranges(code)

    if defect_report:
        # --- Method-scoped fix (single failing method in a single class) ---
        if len(class_ranges) == 1 and failed_reviews:
            scene_name = next(iter(class_ranges.keys()))
            method_ranges = _get_method_ranges(code, scene_name)
            by_method = _group_reviews_by_method(failed_reviews)

            failed_methods = list(by_method.keys())
            unmapped = [m for m in failed_methods if m not in method_ranges]

            if unmapped:
                print(
                    f"[Fixer] Could not map frames to methods: {unmapped} "
                    f"— falling back to scene-level fix."
                )
            elif len(by_method) > 1:
                print(
                    f"[Fixer] {len(by_method)} methods failed "
                    f"({', '.join(failed_methods)}) "
                    f"— using scene-level fix for systemic issues."
                )
            elif len(by_method) == 1:
                method_name = next(iter(by_method))
                reviews_for_method = by_method[method_name]
                defect_count = sum(len(r.defects) for r in reviews_for_method)

                print(
                    f"[Fixer] METHOD-LEVEL fix: {method_name}() "
                    f"({defect_count} defect(s))"
                )
                for review in reviews_for_method:
                    for defect in review.defects:
                        severity = defect.get("severity", "?").upper()
                        category = defect.get("category", "?")
                        desc = defect.get("description", "?")
                        print(f"  [{severity}] {category}: {desc}")

                construct_code = ""
                if "construct" in method_ranges:
                    construct_code = _extract_method(code, method_ranges["construct"])

                method_code = _extract_method(code, method_ranges[method_name])
                print(f"\n[Fixer] Original {method_name}():\n{method_code}\n")

                method_report = _format_method_defect_report(reviews_for_method)
                prompt = _build_method_fix_prompt(
                    method_name,
                    method_code,
                    construct_code,
                    method_report,
                    code_error_report,
                )

                method_images = _collect_image_paths(reviews_for_method, frames_dir)
                if method_images:
                    print(
                        f"[Fixer] Attaching {len(method_images)} "
                        f"screenshot(s) for {method_name}()"
                    )

                try:
                    fixed_method = _call_visual_fixer_llm(
                        prompt,
                        api_key,
                        model,
                        image_paths=method_images or None,
                    )
                except RuntimeError as exc:
                    print(
                        f"[Fixer] Method-level fix failed for "
                        f"{method_name}(): {exc}"
                        f"\n  Falling back to scene-level fix."
                    )
                else:
                    print(f"[Fixer] Fixed {method_name}():\n{fixed_method}\n")
                    return _replace_method(
                        code,
                        method_ranges[method_name],
                        fixed_method,
                    )

        # --- Scene-scoped fix (single class, multiple failures) ------------
        if len(class_ranges) == 1:
            scene_name = next(iter(class_ranges.keys()))
            print(f"[Fixer] SCENE-LEVEL fix: class {scene_name}")
            scene_code = _extract_scene_class(code, class_ranges[scene_name])
            prompt = _build_scene_fix_prompt(
                scene_name, scene_code, defect_report, code_error_report
            )
            scene_images = _collect_image_paths(failed_reviews or [], frames_dir)
            if scene_images:
                print(f"[Fixer] Attaching {len(scene_images)} screenshot(s)")
            fixed_scene = _call_visual_fixer_llm(
                prompt,
                api_key,
                model,
                image_paths=scene_images or None,
            )
            print(f"[Fixer] Fixed class {scene_name}:\n{fixed_scene}\n")
            return _replace_scene_class(code, class_ranges[scene_name], fixed_scene)

        # --- Full-code fix (multiple classes or edge cases) ----------------
        print(
            f"[Fixer] FULL-CODE fix: {len(class_ranges)} class(es) "
            f"({', '.join(class_ranges.keys())})"
        )
        prompt = _build_fix_prompt(code, defect_report, code_error_report)
        all_images = _collect_image_paths(failed_reviews or [], frames_dir)
        if all_images:
            print(f"[Fixer] Attaching {len(all_images)} screenshot(s)")
        fixed_code = _call_visual_fixer_llm(
            prompt,
            api_key,
            model,
            image_paths=all_images or None,
        )
        print(f"[Fixer] Fixed full code:\n{fixed_code}\n")
        return fixed_code

    # No visual defects: fix code errors only, scene-scoped when possible.
    print(f"[Fixer] No visual defects — fixing {len(code_errors)} code error(s) only.")
    if class_ranges and all(e.scene_name in class_ranges for e in code_errors):
        current_code = code
        for scene_name in _scene_fix_order(code_errors, class_ranges):
            class_ranges = _get_scene_class_ranges(current_code)
            if scene_name not in class_ranges:
                continue
            scene_code = _extract_scene_class(current_code, class_ranges[scene_name])
            scene_errors = [e for e in code_errors if e.scene_name == scene_name]
            scene_error_report = format_code_error_report(scene_errors)
            print(
                f"[Fixer] Fixing code errors in class {scene_name}: "
                f"{len(scene_errors)} error(s)"
            )
            for err in scene_errors:
                print(f"  Line {err.line}: {err.message}")
            prompt = _build_scene_fix_prompt(
                scene_name, scene_code, "", scene_error_report
            )
            fixed_scene = _call_visual_fixer_llm(prompt, api_key, model)
            print(f"[Fixer] Fixed class {scene_name}:\n{fixed_scene}\n")
            current_code = _replace_scene_class(
                current_code, class_ranges[scene_name], fixed_scene
            )
        return current_code

    print("[Fixer] FULL-CODE fix for code errors (cannot scope to scenes).")
    prompt = _build_fix_prompt(code, "", code_error_report)
    fixed_code = _call_visual_fixer_llm(prompt, api_key, model)
    print(f"[Fixer] Fixed full code:\n{fixed_code}\n")
    return fixed_code


def fix_from_visual_review(
    code: str, defect_report: str, api_key: str, model: str | None = None
) -> str:
    """Backward-compatible wrapper for visual-only fixes."""
    return fix_from_reports(code, defect_report, None, api_key, model)


def _strip_think_tags(text: str) -> str:
    import re

    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_code(content: str) -> str:
    import re

    if not content:
        return ""
    content = _strip_think_tags(content)
    match = re.search(r"```(?:python)?\s*\n(.+?)```", content, re.DOTALL)
    return (match.group(1) if match else content).strip()


def print_review(review: FrameReview) -> None:
    """Print a single frame review result."""
    if review.error:
        print(f"  {review.filename}: ERROR: {review.error}")
    elif review.status == "PASS":
        print(f"  {review.filename}: PASS")
    else:
        high = sum(1 for d in review.defects if d.get("severity") == "high")
        med = sum(1 for d in review.defects if d.get("severity") == "medium")
        low = sum(1 for d in review.defects if d.get("severity") == "low")
        print(f"  {review.filename}: FAIL ({high} high, {med} medium, {low} low)")

"""
Code generation orchestrator.

Runs the full pipeline: compose scene plan -> generate Manim code -> fix -> visual review.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from manim_video_agent.clients.openrouter import OpenRouterClient, OpenRouterConfig
from manim_video_agent.config import get_settings
from manim_video_agent.core.composer import ComposerConfig, ComposeRequest, VideoComposer
from manim_video_agent.core.fixer import collect_code_errors
from manim_video_agent.core.video_renderer import _fix_render_error
from manim_video_agent.core.visual_reviewer import (
    format_defect_report,
    fix_from_reports,
    print_review,
    render_lastframes,
    review_all_frames,
)
from manim_video_agent.manim.preambles import inject_all_preambles
from manim_video_agent.skills.loader import ManimSkillLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateVideoResult:
    """Result of the full video generation pipeline."""

    code: str
    scene_plan: str | None = None


def write_output_file(file_path: Path, content: str) -> None:
    """Write generated code to output file."""
    file_path.write_text(content, encoding="utf-8")


def _parse_overlap_reports(stdout: str) -> list[str]:
    """Extract [OVERLAP_DETECTED] lines from render stdout (deduplicated)."""
    if not stdout:
        return []
    return list(dict.fromkeys(
        line.strip()
        for line in stdout.splitlines()
        if "[OVERLAP_DETECTED]" in line
    ))


def _format_overlap_report(overlap_lines: list[str]) -> str:
    """Format runtime overlap detections for the fixer LLM."""
    if not overlap_lines:
        return ""
    parts = ["RUNTIME OVERLAP DETECTIONS (exact AABB collision analysis):"]
    for line in overlap_lines:
        parts.append(f"  {line}")
    parts.append(
        "Fix overlaps by adjusting positions with .next_to(), "
        "increasing buff, or rearranging layout."
    )
    return "\n".join(parts)


def _run_visual_review_loop(
    code: str,
    output_file: Path,
    api_key: str,
    max_iterations: int,
    include_code_errors: bool,
    frames_dir: Path | None = None,
) -> str:
    """Render frames, review with Gemini, fix with Sonnet if defects found.

    Args:
        code: Manim source code (clean, without preambles).
        output_file: Path to the Manim file on disk (for Docker to render).
        api_key: OpenRouter API key.
        max_iterations: Maximum review-fix cycles.
        include_code_errors: Whether to run static code checks.
        frames_dir: Directory for rendered frame PNGs.  When ``None``,
            defaults to ``media/lastframes`` (original behavior).

    Returns:
        Final code after all review iterations.
    """
    settings = get_settings()
    frames_dir = frames_dir or Path("media/lastframes")
    current_code = code

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Visual Review Iteration {iteration}/{max_iterations} ---")

        code_errors = collect_code_errors(current_code) if include_code_errors else []

        # Render last frames via Docker (with render-fix retry)
        render_ok = False
        render_stdout = ""
        for render_attempt in range(1, settings.render_max_retries + 2):
            print(f"Rendering last frames (attempt {render_attempt})...")
            write_output_file(output_file, inject_all_preambles(current_code))
            try:
                _, render_stdout = render_lastframes(output_file, frames_dir)
                render_ok = True
                break
            except RuntimeError as exc:
                print(f"Render failed: {exc}")
                if render_attempt > settings.render_max_retries:
                    break
                print("Sending code + traceback to fixer LLM...")
                try:
                    current_code = _fix_render_error(
                        code=current_code,
                        error=str(exc),
                        api_key=api_key,
                        model=settings.render_fixer_model,
                        base_url=settings.openrouter_base_url,
                    )
                    print("Fixer returned corrected code, retrying render...")
                except RuntimeError as fix_exc:
                    print(f"Render fixer failed: {fix_exc}")
                    break

        if not render_ok:
            print("All render attempts failed, skipping visual review.")
            break

        # Review frames with Gemini vision
        print("Reviewing frames with Gemini vision...")
        reviews = asyncio.run(review_all_frames(frames_dir, api_key))

        if not reviews:
            print("WARNING: No frames to review — render produced 0 frames.")
            if code_errors:
                print("Attempting code-only fix for code errors...")
                try:
                    fixed_code = fix_from_reports(
                        current_code, "", code_errors, api_key
                    )
                    current_code = fixed_code
                    continue
                except RuntimeError as fix_exc:
                    print(f"Code-only fixer failed: {fix_exc}")
            break

        for review in reviews:
            print_review(review)

        total_cost = sum(r.cost for r in reviews)
        passed = [r for r in reviews if r.status == "PASS"]
        failed = [r for r in reviews if r.status == "FAIL"]
        errors = [r for r in reviews if r.status == "ERROR"]

        print(
            f"Results: {len(passed)} passed, {len(failed)} failed, "
            f"{len(errors)} errors | Cost: ${total_cost:.4f}"
        )

        # Parse runtime overlap detections from render output
        overlap_lines = _parse_overlap_reports(render_stdout)
        if overlap_lines:
            print(f"\nOverlap checker found {len(overlap_lines)} collision(s):")
            for line in overlap_lines:
                print(f"  {line}")

        if not failed and not code_errors and not overlap_lines:
            print("All frames passed visual review.")
            if frames_dir.exists() and not get_settings().dev_mode:
                shutil.rmtree(frames_dir)
            break

        # Build defect report and fix — frames stay on disk so the
        # fixer can attach failed screenshots alongside the text report.
        defect_report = format_defect_report(reviews)
        overlap_report = _format_overlap_report(overlap_lines)
        if overlap_report:
            defect_report = (
                (defect_report + "\n\n" + overlap_report)
                if defect_report
                else overlap_report
            )
        print(f"\nDefect report:\n{defect_report}")
        print(f"\nSending to {settings.visual_fixer_model} for fixes...")

        try:
            fixed_code = fix_from_reports(
                current_code,
                defect_report,
                code_errors,
                api_key,
                failed_reviews=failed,
                frames_dir=frames_dir,
            )
            current_code = fixed_code
        except RuntimeError as exc:
            print(f"Visual fixer failed: {exc}")
            break
        finally:
            if frames_dir.exists() and not get_settings().dev_mode:
                shutil.rmtree(frames_dir)

    return current_code


def generate_video(
    input_file: Path,
    scenes_file: Path,
    output_file: Path,
    api_key: Optional[str] = None,
    selective_skills: bool = False,
    skip_fixer: bool = True,
) -> GenerateVideoResult:
    """Run the full video generation pipeline.

    Returns a :class:`GenerateVideoResult` containing the final Manim code
    and the optional scene plan (``None`` when plan generation fails).
    """
    settings = get_settings()
    key = api_key or settings.openrouter_api_key
    if not key:
        raise ValueError(
            "API key required. Set OPENROUTER_API_KEY in .env or environment."
        )

    content = input_file.read_text(encoding="utf-8")

    # Phase 1: Compose scene plan (fallback to None on failure)
    scene_plan: str | None = None
    try:
        composer_config = ComposerConfig(api_key=key)
        composer = VideoComposer(composer_config)

        print("Composing scene plan...")
        compose_result = composer.compose(ComposeRequest(topic=content))
        scene_plan = compose_result.plan
        write_output_file(scenes_file, scene_plan)
        print(f"Scene plan written to: {scenes_file}")
    except Exception:
        logger.exception("Scene plan generation failed — continuing without plan")
        print("Scene plan generation failed — continuing without plan")

    # Phase 2: Generate Manim code
    config = OpenRouterConfig(api_key=key)
    client = OpenRouterClient(config)

    print(f"Generating Manim code using {config.default_model}...")
    skill_block = _load_skills(content, selective_skills)
    result = client.generate_manim_code(
        content, skill_block=skill_block, scene_plan=scene_plan
    )

    # Phase 3: Collect code errors for unified fixer
    code_errors = [] if skip_fixer else collect_code_errors(result.code)
    fixed_code = result.code

    # Phase 4: Visual review loop (render → Gemini review → LLM fix)
    print("\nStarting visual review pipeline...")
    final_code = _run_visual_review_loop(
        code=fixed_code,
        output_file=output_file,
        api_key=key,
        max_iterations=settings.visual_review_max_iterations,
        include_code_errors=not skip_fixer,
    )

    write_output_file(output_file, final_code)
    print(f"Generated: {output_file}")

    if result.reasoning:
        print(f"\n--- Reasoning ({len(result.reasoning)} chars) ---")
        print(result.reasoning)
        print("--- End Reasoning ---\n")

    if result.usage:
        print(f"Tokens used: {result.usage}")

    return GenerateVideoResult(code=final_code, scene_plan=scene_plan)


def _load_skills(content: str, selective: bool) -> str:
    """Load ManimCE skill rules for prompt injection."""
    loader = ManimSkillLoader()

    if selective:
        result = loader.select_for_content(content)
        skill_block = loader.format_for_prompt(result.rules)
        print(f"Loaded {len(result.rules)}/{len(loader)} skill rules (selective)")
    else:
        skill_block = loader.format_for_prompt()
        print(f"Loaded {len(loader)} skill rules (all)")

    return skill_block

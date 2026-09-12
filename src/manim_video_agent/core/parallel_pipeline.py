"""
Per-scene parallel video generation pipeline.

Orchestrates: compose → parallel gen+review → sequential narration →
parallel render → stitch → merge audio.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from manim_video_agent.config import get_settings
from manim_video_agent.core.composer import ComposerConfig, ComposeRequest, VideoComposer
from manim_video_agent.core.generator import _run_visual_review_loop, write_output_file
from manim_video_agent.core.narration import (
    PreRenderNarrationResult,
    _get_media_duration,
    compute_scene_extensions,
    generate_scene_narrations,
    inject_single_scene_wait,
    run_post_render_merge,
    synthesize_scenes_with_durations,
)
from manim_video_agent.core.scene_generator import SceneCodeResult, generate_scene_code
from manim_video_agent.core.scene_plan import (
    CompositionResult,
    ScenePlan,
    SharedContext,
    scene_filename,
    write_scene_files,
    write_shared_context,
)
from manim_video_agent.core.timing_parser import SceneTiming, parse_single_scene_timing
from manim_video_agent.core.video_renderer import render_video
from manim_video_agent.core.video_stitcher import stitch_videos
from manim_video_agent.manim.preambles import inject_all_preambles
from manim_video_agent.skills.loader import ManimSkillLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallelPipelineResult:
    """Final output of the parallel pipeline."""

    video_path: str
    scene_count: int
    code_files: list[Path]


def _load_skills(content: str) -> str:
    loader = ManimSkillLoader()
    skill_block = loader.format_for_prompt(include_examples=True)
    logger.info("Loaded %d skill rules (all, with examples)", len(loader))
    return skill_block


def _generate_and_review_scene(
    plan: ScenePlan,
    shared_ctx: SharedContext,
    skill_block: str,
    api_key: str,
    code_dir: Path,
    frames_base_dir: Path,
) -> tuple[ScenePlan, Path]:
    """Generate code for one scene, then run visual review.  Thread-safe."""
    settings = get_settings()
    fname = scene_filename(plan)
    code_file = code_dir / f"{fname}.py"
    frames_dir = frames_base_dir / f"scene_{plan.index:03d}"

    result: SceneCodeResult = generate_scene_code(
        plan=plan,
        shared_context=shared_ctx,
        skill_block=skill_block,
        api_key=api_key,
    )

    reviewed_code = _run_visual_review_loop(
        code=result.code,
        output_file=code_file,
        api_key=api_key,
        max_iterations=settings.visual_review_max_iterations,
        include_code_errors=True,
        frames_dir=frames_dir,
    )

    write_output_file(code_file, reviewed_code)
    logger.info("Scene %d (%s) ready: %s", plan.index, plan.slug, code_file)
    return plan, code_file


def _render_scene(
    code_file: Path,
    video_dir: Path,
    api_key: str,
) -> Path | None:
    """Render one scene file to MP4.  Thread-safe."""
    injected = inject_all_preambles(code_file.read_text(encoding="utf-8"))
    code_file.write_text(injected, encoding="utf-8")

    result = render_video(code_file=code_file, output_dir=video_dir, api_key=api_key)
    if result.success and result.video_path:
        return Path(result.video_path)

    logger.error(
        "Render failed for %s after %d attempt(s): %s",
        code_file.name,
        result.attempts,
        result.error,
    )
    return None


def generate_video_parallel(
    content: str,
    job_id: str,
    api_key: str | None = None,
    output_dir: Path | None = None,
) -> ParallelPipelineResult:
    """Run the full per-scene parallel pipeline.

    Pipeline stages:
    1. Compose scene plan (sequential)
    2. Generate + visual-review per scene (parallel, ThreadPoolExecutor)
    3. Narration: LLM script → TTS → compute extensions (sequential)
    4. Inject wait extensions into scene files
    5. Render each scene to MP4 (parallel)
    6. Stitch scene videos
    7. Build timeline audio + merge onto stitched video
    """
    settings = get_settings()
    key = api_key or settings.openrouter_api_key
    if not key:
        raise ValueError("API key required — set OPENROUTER_API_KEY")

    job_dir = (output_dir or Path("output")) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_file = job_dir / "input.md"
    input_file.write_text(content, encoding="utf-8")

    code_dir = job_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir = job_dir / "scenes"
    frames_base_dir = job_dir / "frames"
    video_dir = job_dir / "videos"
    final_dir = job_dir / "video"

    # ------------------------------------------------------------------ 1
    logger.info("Stage 1: Composing scene plan")
    composer = VideoComposer(ComposerConfig(api_key=key))
    composition: CompositionResult = composer.compose_scenes(
        ComposeRequest(topic=content)
    )

    write_scene_files(composition.plans, scenes_dir)
    write_shared_context(composition.shared_context, job_dir)
    logger.info(
        "Composed %d scenes for '%s'",
        len(composition.plans),
        composition.shared_context.topic_title,
    )

    # ------------------------------------------------------------------ 2
    logger.info("Stage 2: Generating + reviewing scenes (parallel)")
    skill_block = _load_skills(content)
    max_regen = settings.max_scene_regen_attempts

    code_files: dict[int, Path] = {}
    failed_plans: list[ScenePlan] = []

    # --- First pass: parallel generation ---
    with ThreadPoolExecutor(max_workers=settings.max_parallel_scenes) as pool:
        futures = {
            pool.submit(
                _generate_and_review_scene,
                plan,
                composition.shared_context,
                skill_block,
                key,
                code_dir,
                frames_base_dir,
            ): plan
            for plan in composition.plans
        }

        for future in as_completed(futures):
            plan = futures[future]
            try:
                plan, code_file = future.result()
                code_files[plan.index] = code_file
            except Exception:
                logger.exception("Scene %d generation failed (attempt 1)", plan.index)
                failed_plans.append(plan)

    # --- Retry pass: regenerate failed scenes ---
    for attempt in range(2, max_regen + 2):
        if not failed_plans:
            break

        logger.info(
            "Retrying %d failed scene(s) (attempt %d/%d)",
            len(failed_plans),
            attempt,
            max_regen + 1,
        )

        still_failed: list[ScenePlan] = []
        with ThreadPoolExecutor(max_workers=settings.max_parallel_scenes) as pool:
            retry_futures = {
                pool.submit(
                    _generate_and_review_scene,
                    plan,
                    composition.shared_context,
                    skill_block,
                    key,
                    code_dir,
                    frames_base_dir,
                ): plan
                for plan in failed_plans
            }

            for future in as_completed(retry_futures):
                plan = retry_futures[future]
                try:
                    plan, code_file = future.result()
                    code_files[plan.index] = code_file
                    logger.info(
                        "Scene %d (%s) succeeded on attempt %d",
                        plan.index,
                        plan.slug,
                        attempt,
                    )
                except Exception:
                    logger.exception(
                        "Scene %d generation failed (attempt %d)",
                        plan.index,
                        attempt,
                    )
                    still_failed.append(plan)

        failed_plans = still_failed

    ordered_files = [
        code_files[p.index] for p in composition.plans if p.index in code_files
    ]
    if not ordered_files:
        raise RuntimeError("All scene generations failed")

    logger.info(
        "Stage 2 complete: %d/%d scenes ready",
        len(ordered_files),
        len(composition.plans),
    )

    # ------------------------------------------------------------------ 3
    logger.info("Stage 3: Narration (sequential)")
    scene_timings: list[SceneTiming] = []
    timing_map: dict[int, SceneTiming] = {}
    for plan in composition.plans:
        if plan.index not in code_files:
            continue
        code = code_files[plan.index].read_text(encoding="utf-8")
        timing = parse_single_scene_timing(code, scene_id=plan.slug)
        if timing:
            scene_timings.append(timing)
            timing_map[plan.index] = timing

    narration_work_dir = job_dir / "narration"
    narration_work_dir.mkdir(parents=True, exist_ok=True)
    narration_result: PreRenderNarrationResult | None = None

    if scene_timings:
        all_code = "\n\n".join(
            f"# === SCENE: {p.slug} ===\n"
            + code_files[p.index].read_text(encoding="utf-8")
            for p in composition.plans
            if p.index in code_files
        )

        segments = generate_scene_narrations(
            manim_code=all_code,
            original_content=content,
            timings=scene_timings,
            scene_plan="\n".join(p.markdown for p in composition.plans),
        )

        if segments:
            tts_result = synthesize_scenes_with_durations(
                segments,
                scene_timings,
                narration_work_dir,
            )

            if tts_result:
                audio_files, audio_durations = tts_result

                # ---------------------------------------------------------- 4
                extensions = compute_scene_extensions(
                    scene_timings,
                    audio_durations,
                    segments,
                )

                if extensions:
                    logger.info("Stage 4: Injecting wait extensions")
                    for plan in composition.plans:
                        if plan.index not in code_files or plan.index not in timing_map:
                            continue
                        timing = timing_map[plan.index]
                        extra = extensions.get(timing.method_name)
                        if extra:
                            cf = code_files[plan.index]
                            code = cf.read_text(encoding="utf-8")
                            extended = inject_single_scene_wait(code, extra)
                            cf.write_text(extended, encoding="utf-8")

                    # Re-parse timings after extensions so Stage 7
                    # uses the actual (longer) scene durations.
                    scene_timings = []
                    timing_map = {}
                    for plan in composition.plans:
                        if plan.index not in code_files:
                            continue
                        code = code_files[plan.index].read_text(encoding="utf-8")
                        timing = parse_single_scene_timing(code, scene_id=plan.slug)
                        if timing:
                            scene_timings.append(timing)
                            timing_map[plan.index] = timing

                narration_result = PreRenderNarrationResult(
                    extended_code=all_code,
                    audio_files=audio_files,
                    work_dir=narration_work_dir,
                )

    # ------------------------------------------------------------------ 5
    logger.info("Stage 5: Rendering scenes (parallel)")
    video_dir.mkdir(parents=True, exist_ok=True)

    scene_videos: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=settings.max_parallel_renders) as pool:
        render_futures = {
            pool.submit(
                _render_scene,
                code_files[plan.index],
                video_dir,
                key,
            ): plan.index
            for plan in composition.plans
            if plan.index in code_files
        }

        for future in as_completed(render_futures):
            idx = render_futures[future]
            try:
                video_path = future.result()
                if video_path:
                    scene_videos[idx] = video_path
            except Exception:
                logger.exception("Scene %d render failed", idx)

    ordered_videos = [
        scene_videos[p.index] for p in composition.plans if p.index in scene_videos
    ]
    if not ordered_videos:
        raise RuntimeError("All scene renders failed")

    logger.info(
        "Stage 5 complete: %d/%d scenes rendered",
        len(ordered_videos),
        len(composition.plans),
    )

    # ------------------------------------------------------------------ 6
    logger.info("Stage 6: Stitching videos")
    final_dir.mkdir(parents=True, exist_ok=True)
    stitched_path = stitch_videos(ordered_videos, final_dir / "final.mp4")

    # ------------------------------------------------------------------ 7
    final_video = str(stitched_path)
    if narration_result:
        logger.info("Stage 7: Merging narration audio")
        extended_timings = scene_timings
        if extended_timings:
            from manim_video_agent.core.narration import (
                _build_timeline_audio,
                _merge_audio_video,
            )

            timeline_audio = narration_work_dir / "timeline_audio.mp3"
            if _build_timeline_audio(
                narration_result.audio_files, extended_timings, timeline_audio
            ):
                narrated_path = final_dir / "final_narrated.mp4"
                if _merge_audio_video(
                    str(stitched_path), timeline_audio, narrated_path
                ):
                    final_video = str(narrated_path)
                    logger.info("Narrated video: %s", narrated_path)

    logger.info("Pipeline complete: %s", final_video)
    return ParallelPipelineResult(
        video_path=final_video,
        scene_count=len(ordered_videos),
        code_files=ordered_files,
    )

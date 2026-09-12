"""
TTS narration pipeline for generated Manim videos.

Supports two modes:
- **Per-scene** (new): LLM generates narration per scene, TTS synthesizes
  each scene independently, scenes are extended with ``self.wait()`` if
  narration runs longer than the animation, then audio is placed on a
  timeline and merged after render.
- **Full-blob** (legacy): Single narration track synthesized and merged
  after render.

Failures at any stage fall back to the original (silent) video path so
the pipeline never breaks the overall job.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from manim_video_agent.clients.tts import InworldTTSClient
from manim_video_agent.config import get_settings
from manim_video_agent.core.timing_parser import (
    SceneTiming,
    parse_scene_timings,
    total_duration,
)

logger = logging.getLogger(__name__)

# ~2.0 words per second at normal speaking rate
_WORDS_PER_SECOND: float = 2.0

_AGGRESSIVE_WORDS_PER_SECOND: float = 1.6

# Speaking-rate bounds supported by InWorld
_MIN_SPEAKING_RATE: float = 0.8
_MAX_SPEAKING_RATE: float = 1.3

_FULL_NARRATION_BUFFER_SECONDS: float = 2.0

# InWorld TTS character limit per request
_TTS_HARD_LIMIT: int = 1950
_TTS_SPLIT_THRESHOLD: int = 1500

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_text_for_tts(text: str, threshold: int = _TTS_SPLIT_THRESHOLD) -> list[str]:
    """Split *text* into as few chunks as possible, each under *threshold* chars.

    Only splits when the text exceeds the threshold.  Splits greedily at
    sentence boundaries (`.` `!` `?` followed by whitespace) to keep chunks
    natural-sounding.  Falls back to whitespace boundaries if a single
    sentence exceeds the limit.

    Hard rule: no chunk may ever exceed ``_TTS_HARD_LIMIT`` (1950 chars).
    """
    if len(text) <= threshold:
        return [text]

    # Use the stricter of the two limits for actual packing.
    limit = min(threshold, _TTS_HARD_LIMIT)

    sentences = _SENTENCE_BOUNDARY.split(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= limit:
            current = candidate
            continue

        # Current buffer is full — flush it.
        if current:
            chunks.append(current)

        # If this single sentence fits, start a new buffer with it.
        if len(sentence) <= limit:
            current = sentence
            continue

        # Giant sentence — split on whitespace as last resort.
        words = sentence.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word

    if current:
        chunks.append(current)

    # Safety net: force-split any chunk that somehow exceeds the hard limit.
    safe_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= _TTS_HARD_LIMIT:
            safe_chunks.append(chunk)
            continue
        # Brute-force word-level split for the rare edge case.
        words = chunk.split()
        buf = ""
        for word in words:
            candidate = f"{buf} {word}".strip() if buf else word
            if len(candidate) <= _TTS_HARD_LIMIT:
                buf = candidate
            else:
                if buf:
                    safe_chunks.append(buf)
                buf = word
        if buf:
            safe_chunks.append(buf)

    return safe_chunks


# ---------------------------------------------------------------------------
# Part A — Narration script generation (LLM)
# ---------------------------------------------------------------------------

_NARRATION_SYSTEM_PROMPT = """\
You are a friendly, clear educator narrating a math/science animation video.

You will be given:
- The original lesson content (markdown).
- Generated Manim code with scene methods.
- A timing breakdown showing each scene method's animation duration in seconds.
- The total video duration in seconds and a target word count.

Your task: produce a single, continuous narration script for the entire video.
Return only the narration text, with no JSON, no markup, and no timestamps.

Guidelines:
- Keep total word count proportional to the full video duration (~2.0 words/sec) \
so narration finishes before the video ends.
- Be conversational and engaging — like a friendly tutor explaining to a curious student.
- Explain the CONCEPTS and INTUITION behind what is being shown — do NOT describe \
what is visually on screen. Never say "as you can see", "notice how", "watch as", \
"a circle appears", or similar screen-descriptive language.
- GOOD: "The key insight is that any quadratic can be rewritten by completing the square."
- GOOD: "Think of it like folding paper — the fold line is your axis of symmetry."
- BAD: "Now a circle appears on the screen and moves to the right."
- Narration must flow naturally across scenes with smooth transitions and bridge phrases.
- Use simple language, avoid jargon unless the content requires it.
- Do NOT include stage directions, timestamps, or markdown.
- Return ONLY the narration text, no surrounding text or markdown fences.
"""


def generate_full_narration_text(
    manim_code: str,
    original_content: str,
    timings: list[SceneTiming],
    total_override: float | None = None,
) -> str | None:
    settings = get_settings()
    api_key = settings.openrouter_api_key
    if not api_key:
        logger.error("No OpenRouter API key configured — skipping narration")
        return None

    total = total_override if total_override is not None else total_duration(timings)
    effective_total = max(1.0, total - _FULL_NARRATION_BUFFER_SECONDS)
    target_words = int(effective_total * _AGGRESSIVE_WORDS_PER_SECOND)

    timing_summary = "\n".join(
        f"- {t.method_name}: {t.duration_seconds}s" for t in timings
    )

    user_prompt = (
        f"<content>\n{original_content}\n</content>\n\n"
        f"<manim_code>\n{manim_code}\n</manim_code>\n\n"
        f"<timing>\n{timing_summary}\n</timing>\n\n"
        f"<total_duration_seconds>{total}</total_duration_seconds>\n"
        f"<target_words>{target_words}</target_words>\n\n"
        "Produce the full narration script now."
    )

    try:
        response = requests.post(
            settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.narration_model,
                "messages": [
                    {"role": "system", "content": _NARRATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.4,
            },
            timeout=300,
        )

        if response.status_code != 200:
            logger.error(
                "Narration LLM failed (%d): %s",
                response.status_code,
                response.text[:500],
            )
            return None

        raw = response.json()["choices"][0]["message"].get("content")
        if not raw:
            logger.error("Narration LLM returned empty/null content")
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return text or None

    except Exception:
        logger.exception("Narration script generation failed")
        return None


# ---------------------------------------------------------------------------
# Part A2 — Per-scene narration script generation (LLM)
# ---------------------------------------------------------------------------

_PER_SCENE_NARRATION_SYSTEM_PROMPT = """\
You are a friendly, clear educator narrating a math/science animation video.

You will be given:
- The original lesson content (markdown).
- Generated Manim code for each scene (each scene is a separate file with class Main, \
labeled with a comment like "# === SCENE: scene_slug ===").
- A per-scene timing breakdown with each scene's EXACT IDENTIFIER, animation duration, \
and target word count.
- Optionally, a scene plan describing the intended purpose and content of each scene.

Your task: produce a SEPARATE narration script for EACH scene.
Return a JSON array where each element has:
  - "scene": the EXACT scene identifier from the per_scene_timing section \
(e.g. "scene_1_intro", NOT "Main" or "construct")
  - "text": the narration text for that scene

CRITICAL: The "scene" key must use the EXACT identifier from <per_scene_timing>, \
NOT the class name (Main) or method name (construct) from the code. Each scene file \
uses class Main, but each has a unique identifier shown in the timing breakdown.

Example: if the timing says "- scene_2_key_concept: 25.0s", use "scene_2_key_concept" \
as the scene key, not "Main".

Guidelines:
- Each scene's word count should be close to its target (based on ~1.6 words/sec).
- It is OK if narration is slightly longer than the scene's animation — the system \
will extend the scene to fit. But aim to stay within 30% of the target.
- Be conversational and engaging — like a friendly tutor explaining to a curious student.
- Explain the CONCEPTS and INTUITION behind what is being shown — do NOT describe \
what is visually on screen. Never say "as you can see", "notice how", "watch as", \
"a circle appears", or similar screen-descriptive language.
- Each scene should be self-contained but flow naturally into the next.
- Use simple language, avoid jargon unless the content requires it.
- Return ONLY the JSON array. No surrounding text, no markdown fences.
"""


def generate_scene_narrations(
    manim_code: str,
    original_content: str,
    timings: list[SceneTiming],
    scene_plan: str | None = None,
) -> list[dict[str, str]] | None:
    """Generate per-scene narration text via LLM.

    Returns a list of dicts with ``scene`` and ``text`` keys, one per scene,
    or ``None`` on failure.
    """
    settings = get_settings()
    api_key = settings.openrouter_api_key
    if not api_key:
        logger.error("No OpenRouter API key configured — skipping narration")
        return None

    per_scene_info = "\n".join(
        f"- {t.method_name}: {t.duration_seconds}s → target ~{int(t.duration_seconds * _AGGRESSIVE_WORDS_PER_SECOND)} words"
        for t in timings
    )

    scene_plan_section = ""
    if scene_plan:
        scene_plan_section = f"<scene_plan>\n{scene_plan}\n</scene_plan>\n\n"

    user_prompt = (
        f"<content>\n{original_content}\n</content>\n\n"
        f"{scene_plan_section}"
        f"<manim_code>\n{manim_code}\n</manim_code>\n\n"
        f"<per_scene_timing>\n{per_scene_info}\n</per_scene_timing>\n\n"
        "Produce the per-scene narration JSON array now."
    )

    try:
        response = requests.post(
            settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.narration_model,
                "messages": [
                    {"role": "system", "content": _PER_SCENE_NARRATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.4,
            },
            timeout=300,
        )

        if response.status_code != 200:
            logger.error(
                "Per-scene narration LLM failed (%d): %s",
                response.status_code,
                response.text[:500],
            )
            return None

        raw = response.json()["choices"][0]["message"].get("content")
        if not raw:
            logger.error("Per-scene narration LLM returned empty/null content")
            return None
        text = raw.strip()
        # Strip markdown fences if the model wrapped the JSON
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        segments: list[dict[str, str]] = _json.loads(text)

        # Validate structure
        scene_names = {t.method_name for t in timings}
        validated: list[dict[str, str]] = []
        for seg in segments:
            if not isinstance(seg, dict) or "scene" not in seg or "text" not in seg:
                logger.warning("Skipping malformed narration segment: %s", seg)
                continue
            if seg["scene"] not in scene_names:
                logger.warning("Unknown scene name in narration: %s", seg["scene"])
                continue
            validated.append({"scene": seg["scene"], "text": seg["text"].strip()})

        # Fallback: if LLM returned wrong scene names (e.g. all "Main")
        # but the right number of segments, map by position
        if not validated and len(segments) >= len(timings):
            logger.warning(
                "Scene name validation failed — falling back to positional mapping "
                "(%d segments → %d timings)",
                len(segments),
                len(timings),
            )
            for i, timing in enumerate(timings):
                if i < len(segments):
                    seg = segments[i]
                    if isinstance(seg, dict) and "text" in seg:
                        validated.append(
                            {
                                "scene": timing.method_name,
                                "text": seg["text"].strip(),
                            }
                        )
                        logger.info(
                            "Positional map: segment %d ('%s') → %s",
                            i,
                            seg.get("scene", "?"),
                            timing.method_name,
                        )

        if not validated:
            logger.error("No valid narration segments after validation")
            return None

        logger.info(
            "Generated per-scene narration for %d/%d scenes",
            len(validated),
            len(timings),
        )
        return validated

    except _json.JSONDecodeError:
        logger.exception("Failed to parse per-scene narration JSON from LLM")
        return None
    except Exception:
        logger.exception("Per-scene narration generation failed")
        return None


def _compute_speaking_rate_with_buffer(
    text: str, target_duration: float, buffer_seconds: float
) -> float:
    word_count = len(text.split())
    if word_count == 0 or target_duration <= 0:
        return 1.0

    buffered_duration = target_duration - buffer_seconds
    if buffered_duration <= 0:
        buffered_duration = max(0.5, target_duration * 0.8)

    natural_duration = word_count / _WORDS_PER_SECOND
    ratio = natural_duration / buffered_duration
    return max(_MIN_SPEAKING_RATE, min(_MAX_SPEAKING_RATE, ratio))


# ---------------------------------------------------------------------------
# Part B — TTS synthesis
# ---------------------------------------------------------------------------


def _compute_speaking_rate(text: str, target_duration: float) -> float:
    """Compute speaking rate to fit *text* into *target_duration* seconds.

    Returns a rate clamped to [0.8, 1.3].
    """
    return _compute_speaking_rate_with_buffer(
        text, target_duration, _FULL_NARRATION_BUFFER_SECONDS
    )


async def _synthesize_scenes(
    segments: list[dict[str, str]],
    timings: list[SceneTiming],
    work_dir: Path,
    speaking_rate: float = 1.0,
) -> tuple[list[Path], list[float]] | None:
    """Synthesize TTS audio for each narration segment.

    Uses ``_split_text_for_tts()`` chunking when a scene's narration exceeds
    the split threshold, then concatenates the chunks.

    Returns ``(audio_files, audio_durations)`` — parallel lists of MP3 paths
    and their durations in seconds — or ``None`` on failure.
    """
    timing_map: dict[str, float] = {t.method_name: t.duration_seconds for t in timings}

    client = InworldTTSClient()
    audio_files: list[Path] = []
    audio_durations: list[float] = []

    try:
        for i, seg in enumerate(segments):
            scene_name = seg["scene"]
            text = seg["text"]
            duration = timing_map.get(scene_name, 5.0)

            rate = speaking_rate or _compute_speaking_rate(text, duration)
            logger.info(
                "TTS scene %s: %d words, %.1fs anim, rate=%.2f",
                scene_name,
                len(text.split()),
                duration,
                rate,
            )

            chunks = _split_text_for_tts(text)
            out_path = work_dir / f"scene_{i:03d}.mp3"

            if len(chunks) == 1:
                result = await client.synthesize(text, speaking_rate=rate)
                out_path.write_bytes(result.audio)
            else:
                logger.info(
                    "TTS scene %s: split into %d chunks: %s",
                    scene_name,
                    len(chunks),
                    [len(c) for c in chunks],
                )
                audio_parts: list[bytes] = []
                for j, chunk in enumerate(chunks):
                    result = await client.synthesize(chunk, speaking_rate=rate)
                    audio_parts.append(result.audio)
                out_path.write_bytes(b"".join(audio_parts))

            audio_files.append(out_path)
            audio_durations.append(_get_media_duration(out_path))
            logger.info(
                "TTS scene %s: audio %.2fs written to %s",
                scene_name,
                audio_durations[-1],
                out_path,
            )

        return audio_files, audio_durations

    except Exception:
        logger.exception("TTS synthesis failed")
        return None
    finally:
        await client.close()


def synthesize_scenes_with_durations(
    segments: list[dict[str, str]],
    timings: list[SceneTiming],
    work_dir: Path,
    speaking_rate: float = 1.0,
) -> tuple[list[Path], list[float]] | None:
    """Sync wrapper — returns ``(audio_files, audio_durations)`` or ``None``."""
    return asyncio.run(
        _synthesize_scenes(segments, timings, work_dir, speaking_rate=speaking_rate)
    )


async def _synthesize_full_audio(
    text: str,
    total_duration: float,
    work_dir: Path,
) -> Path | None:
    client = InworldTTSClient()

    try:
        speaking_rate = _compute_speaking_rate_with_buffer(
            text, total_duration, _FULL_NARRATION_BUFFER_SECONDS
        )
        effective_target = total_duration - _FULL_NARRATION_BUFFER_SECONDS
        logger.info(
            "TTS full narration: %d words, %.1fs total / %.1fs effective, rate=%.2f",
            len(text.split()),
            total_duration,
            effective_target,
            speaking_rate,
        )

        chunks = _split_text_for_tts(text)

        if len(chunks) == 1:
            logger.info(
                "TTS: single chunk (%d chars) — no splitting needed.",
                len(text),
            )
            result = await client.synthesize(
                text,
                speaking_rate=speaking_rate,
            )
            out_path = work_dir / "narration_full.mp3"
            out_path.write_bytes(result.audio)
            return out_path

        logger.info(
            "TTS: text is %d chars — split into %d chunks: %s",
            len(text),
            len(chunks),
            [len(c) for c in chunks],
        )

        audio_parts: list[bytes] = []
        for i, chunk in enumerate(chunks):
            logger.info(
                "TTS chunk %d/%d: %d chars, %d words",
                i + 1,
                len(chunks),
                len(chunk),
                len(chunk.split()),
            )
            result = await client.synthesize(
                chunk,
                speaking_rate=speaking_rate,
            )
            audio_parts.append(result.audio)

        out_path = work_dir / "narration_full.mp3"
        out_path.write_bytes(b"".join(audio_parts))
        logger.info(
            "TTS: all %d chunks synthesized, merged to %s",
            len(chunks),
            out_path,
        )
        return out_path

    except Exception:
        logger.exception("Full narration synthesis failed")
        return None
    finally:
        await client.close()


def synthesize_full_audio(
    text: str,
    total_duration: float,
    work_dir: Path,
) -> Path | None:
    return asyncio.run(_synthesize_full_audio(text, total_duration, work_dir))


# ---------------------------------------------------------------------------
# Part B2 — Scene extension (wait injection)
# ---------------------------------------------------------------------------

# Maximum extra hold time per scene (seconds).
_MAX_SCENE_EXTENSION: float = 25.0

# Small buffer (seconds) added so narration doesn't end at the exact cut.
_SCENE_EXTENSION_BUFFER: float = 0.5


def compute_scene_extensions(
    timings: list[SceneTiming],
    audio_durations: list[float],
    segments: list[dict[str, str]],
) -> dict[str, float]:
    """Compute how many extra seconds each scene needs to hold its last frame.

    Returns a mapping of ``scene_method_name → extra_wait_seconds``.
    Only scenes that need extension (audio > animation) are included.
    """
    extensions: dict[str, float] = {}
    for timing, audio_dur, seg in zip(timings, audio_durations, segments):
        extra = audio_dur - timing.duration_seconds + _SCENE_EXTENSION_BUFFER
        if extra <= 0:
            continue
        capped = min(extra, _MAX_SCENE_EXTENSION)
        if capped < extra:
            logger.warning(
                "Scene %s needs %.1fs extension but capped at %.1fs",
                timing.method_name,
                extra,
                _MAX_SCENE_EXTENSION,
            )
        extensions[timing.method_name] = round(capped, 2)
        logger.info(
            "Scene %s: anim=%.1fs, audio=%.1fs → wait(%.2f)",
            timing.method_name,
            timing.duration_seconds,
            audio_dur,
            capped,
        )
    return extensions


def inject_scene_waits(
    code: str,
    extensions: dict[str, float],
) -> str:
    """Insert ``self.wait(N)`` before ``self.clear_scene()`` in extended scenes.

    For the final scene (which may not have ``clear_scene()``), appends
    ``self.wait(N)`` before the closing print statement.

    Returns the modified Manim code.
    """
    if not extensions:
        return code

    from manim_video_agent.core.timing_parser import (
        _extract_method_body,
        _get_construct_calls,
    )

    scene_methods = _get_construct_calls(code)
    result = code

    for method_name, wait_secs in extensions.items():
        if method_name not in scene_methods:
            logger.warning("Cannot inject wait: method %s not found", method_name)
            continue

        is_last_scene = method_name == scene_methods[-1]
        body = _extract_method_body(result, method_name)
        if body is None:
            continue

        wait_line = f"self.wait({wait_secs})"

        if not is_last_scene:
            # Insert self.wait(N) before self.clear_scene()
            if "self.clear_scene()" in body:
                # Find the clear_scene() call within this method and insert before it
                pattern = re.compile(
                    r"(^( +))(self\.clear_scene\(\))",
                    re.MULTILINE,
                )
                new_body = pattern.sub(
                    rf"\1{wait_line}\n\1\3",
                    body,
                    count=1,
                )
                result = result.replace(body, new_body, 1)
                logger.info(
                    "Injected wait(%.2f) before clear_scene in %s",
                    wait_secs,
                    method_name,
                )
            else:
                logger.warning(
                    "No clear_scene() found in non-final scene %s", method_name
                )
        else:
            # Final scene — insert before print statement or at end of method
            print_pattern = re.compile(r'^( +)(print\(f"Total time)', re.MULTILINE)
            match = print_pattern.search(body)
            if match:
                indent = match.group(1)
                insert_line = f"{indent}{wait_line}\n"
                insert_pos = body.index(match.group(0))
                new_body = body[:insert_pos] + insert_line + body[insert_pos:]
                result = result.replace(body, new_body, 1)
                logger.info(
                    "Injected wait(%.2f) at end of final scene %s",
                    wait_secs,
                    method_name,
                )
            else:
                # Fallback: append wait at end of method body
                lines = body.rstrip().split("\n")
                if lines:
                    last_line = lines[-1]
                    indent_match = re.match(r"^(\s+)", last_line)
                    indent = indent_match.group(1) if indent_match else "        "
                    new_body = body.rstrip() + f"\n{indent}{wait_line}\n"
                    result = result.replace(body, new_body, 1)
                    logger.info(
                        "Injected wait(%.2f) as last line of %s", wait_secs, method_name
                    )

    return result


def inject_single_scene_wait(code: str, extra_seconds: float) -> str:
    """Insert ``self.wait(N)`` before the timing print in a per-scene file.

    Per-scene files end with ``print(f"Scene time: ...")`` instead of
    ``self.clear_scene()``.  This inserts the wait immediately before that
    line so the scene holds its last frame long enough for narration.

    Returns the modified code, or the original if the marker isn't found.
    """
    if extra_seconds <= 0:
        return code

    capped = min(extra_seconds, _MAX_SCENE_EXTENSION)
    pattern = re.compile(r'^( +)(print\(f"Scene time:)', re.MULTILINE)
    match = pattern.search(code)
    if not match:
        logger.warning("Cannot inject wait: no 'Scene time' print found")
        return code

    indent = match.group(1)
    insert = f"{indent}self.wait({round(capped, 2)})\n"
    pos = match.start()
    result = code[:pos] + insert + code[pos:]
    logger.info("Injected wait(%.2f) into per-scene file", capped)
    return result


# ---------------------------------------------------------------------------
# Part C — ffmpeg audio operations
# ---------------------------------------------------------------------------


def _get_media_duration(media_path: Path) -> float:
    """Probe the duration of a media file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        logger.warning("ffprobe failed for %s, estimating duration", media_path)
    return 0.0


def _build_timeline_audio(
    audio_files: list[Path],
    timings: list[SceneTiming],
    output_path: Path,
) -> bool:
    """Place each scene's audio at its exact start time on the timeline.

    Uses ffmpeg's ``adelay`` filter to offset each scene's audio to its
    correct position, then mixes them together. This prevents drift that
    occurs with simple concatenation when TTS audio length doesn't exactly
    match scene duration.

    Returns ``True`` on success.
    """
    if not audio_files:
        return False

    if len(audio_files) == 1:
        output_path.write_bytes(audio_files[0].read_bytes())
        return True

    # Compute the start time of each scene on the timeline
    scene_starts: list[float] = []
    cumulative = 0.0
    for t in timings:
        scene_starts.append(cumulative)
        cumulative += t.duration_seconds

    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, af in enumerate(audio_files):
        inputs.extend(["-i", str(af)])
        delay_ms = int(scene_starts[i] * 1000) if i < len(scene_starts) else 0
        duration = timings[i].duration_seconds if i < len(timings) else 0.0

        filter_chain = f"[{i}:a]"
        if duration > 0:
            dur = f"{duration:.3f}"
            filter_chain += f"atrim=0:{dur},asetpts=PTS-STARTPTS,apad=pad_dur={dur}"
        else:
            filter_chain += "asetpts=PTS-STARTPTS"

        if delay_ms > 0:
            filter_chain += f",adelay={delay_ms}|{delay_ms}"

        filter_chain += f"[a{i}]"
        filter_parts.append(filter_chain)

    mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_files)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(audio_files)}:duration=longest:normalize=0[out]"
    )

    filter_graph = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_graph,
        "-map",
        "[out]",
        "-ac",
        "1",
        "-ar",
        "44100",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("ffmpeg timeline build failed: %s", result.stderr[:500])
            return False
        return True
    except Exception:
        logger.exception("ffmpeg timeline build error")
        return False


def _merge_audio_video(
    video_path: str,
    audio_path: Path,
    output_path: Path,
) -> bool:
    """Merge audio track onto video using ffmpeg.

    Video is copied (no re-encoding), audio is encoded as AAC.

    Returns ``True`` on success.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("ffmpeg merge failed: %s", result.stderr[:500])
            return False
        return True
    except Exception:
        logger.exception("ffmpeg merge error")
        return False


# ---------------------------------------------------------------------------
# Part D — Top-level pipeline orchestrators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreRenderNarrationResult:
    """Data produced by the pre-render narration pipeline."""

    extended_code: str
    audio_files: list[Path]
    work_dir: Path


def run_pre_render_narration(
    manim_code: str,
    original_content: str,
    work_dir: Path,
    scene_plan: str | None = None,
) -> PreRenderNarrationResult | None:
    """Run narration pipeline stages that must happen BEFORE video render.

    Stages:
    1. Parse scene timings from Manim code.
    2. Generate per-scene narration scripts via LLM.
    3. Synthesize TTS per scene and measure audio durations.
    4. Compute scene extensions and inject waits into code.

    Returns a :class:`PreRenderNarrationResult` with the extended code and
    audio files, or ``None`` on any failure (caller should render original
    code as fallback).
    """
    try:
        timings = parse_scene_timings(manim_code)
        if not timings:
            logger.warning("No scene timings found — skipping narration")
            return None

        logger.info(
            "Pre-render narration: %d scenes, %.1fs total estimated",
            len(timings),
            total_duration(timings),
        )

        # 1. Per-scene narration generation
        segments = generate_scene_narrations(
            manim_code, original_content, timings, scene_plan=scene_plan
        )
        if not segments:
            logger.warning("Per-scene narration generation failed — skipping")
            return None

        # 2. TTS per scene → audio files + durations
        tts_result = synthesize_scenes_with_durations(segments, timings, work_dir)
        if not tts_result:
            logger.warning("Per-scene TTS synthesis failed — skipping")
            return None

        audio_files, audio_durations = tts_result
        logger.info(
            "TTS complete: %d scene audios, durations=%s",
            len(audio_files),
            [f"{d:.1f}s" for d in audio_durations],
        )

        # 3. Compute extensions + inject waits
        extensions = compute_scene_extensions(timings, audio_durations, segments)
        extended_code = inject_scene_waits(manim_code, extensions)

        if extensions:
            logger.info(
                "Injected waits into %d scene(s): %s",
                len(extensions),
                {k: f"{v:.1f}s" for k, v in extensions.items()},
            )
        else:
            logger.info(
                "No scene extensions needed — all narration fits within animations"
            )

        return PreRenderNarrationResult(
            extended_code=extended_code,
            audio_files=audio_files,
            work_dir=work_dir,
        )

    except Exception:
        logger.exception("Pre-render narration failed — will render without narration")
        return None


def run_post_render_merge(
    video_path: str,
    narration_result: PreRenderNarrationResult,
    extended_code: str,
) -> str:
    """Merge per-scene audio onto the rendered video.

    Builds a timeline audio track from the per-scene MP3s placed at the
    correct positions (using extended timings), then muxes it onto the video.

    Returns path to the narrated video, or *video_path* on failure.
    """
    try:
        extended_timings = parse_scene_timings(extended_code)
        if not extended_timings:
            logger.warning("Cannot parse extended timings — skipping audio merge")
            return video_path

        timeline_audio = narration_result.work_dir / "timeline_audio.mp3"
        if not _build_timeline_audio(
            narration_result.audio_files, extended_timings, timeline_audio
        ):
            logger.warning("Timeline audio build failed — skipping merge")
            return video_path

        video_p = Path(video_path)
        merged_path = video_p.parent / f"{video_p.stem}_narrated{video_p.suffix}"

        if not _merge_audio_video(video_path, timeline_audio, merged_path):
            logger.warning("Audio-video merge failed — returning silent video")
            return video_path

        logger.info("Narrated video created: %s", merged_path)
        return str(merged_path)

    except Exception:
        logger.exception("Post-render merge failed — returning silent video")
        return video_path


def run_narration_pipeline(
    video_path: str,
    manim_code: str,
    original_content: str,
) -> str:
    """Legacy pipeline — generates narration AFTER render (single-blob).

    Kept for backwards compatibility. New code should use
    :func:`run_pre_render_narration` + :func:`run_post_render_merge`.
    """
    try:
        timings = parse_scene_timings(manim_code)
        if not timings:
            logger.warning("No scene timings found — skipping narration")
            return video_path

        estimated_total = total_duration(timings)
        video_duration = _get_media_duration(Path(video_path))
        total = max(estimated_total, video_duration)
        logger.info(
            "Parsed %d scene(s), estimated %.1fs, video %.1fs, using %.1fs",
            len(timings),
            estimated_total,
            video_duration,
            total,
        )

        narration_text = generate_full_narration_text(
            manim_code,
            original_content,
            timings,
            total_override=total,
        )
        if not narration_text:
            logger.warning("Narration script generation failed — skipping narration")
            return video_path

        with tempfile.TemporaryDirectory(prefix="narration_") as tmpdir:
            work_dir = Path(tmpdir)

            narration_path = synthesize_full_audio(narration_text, total, work_dir)
            if not narration_path:
                logger.warning("TTS synthesis failed — skipping narration")
                return video_path

            video_p = Path(video_path)
            merged_path = video_p.parent / f"{video_p.stem}_narrated{video_p.suffix}"

            if not _merge_audio_video(video_path, narration_path, merged_path):
                logger.warning("Audio-video merge failed — skipping narration")
                return video_path

            logger.info("Narrated video created: %s", merged_path)
            return str(merged_path)

    except Exception:
        logger.exception("Narration pipeline failed — falling back to silent video")
        return video_path

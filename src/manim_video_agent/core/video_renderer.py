"""
Full video rendering with LLM-powered error retry.

Renders Manim code to MP4 via render_video.py, and on failure
sends the code + error to an LLM for automatic fixes.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

from manim_video_agent.config import get_settings

logger = logging.getLogger(__name__)

_MAX_STDERR_CHARS = 3000

RENDER_FIXER_SYSTEM_PROMPT = """\
<role>
You are a Manim code fixer. You receive Manim Python code that failed to render \
along with the error traceback. Fix the code so it renders successfully. \
Output ONLY the corrected Python code — no explanations, no markdown fences.
</role>

<rules>
- Return ONLY the fixed Python code
- Preserve ALL functionality, animations, and structure
- Keep the same class names and method names
- Fix only the issue causing the render failure
- Common fixes: missing imports, undefined variables, incorrect Manim API usage, \
  syntax errors, type errors, incompatible method arguments
- If the error is ambiguous, make the minimal change that resolves it
- NEVER remove content or animations to "fix" the error
</rules>"""


@dataclass(frozen=True)
class RenderResult:
    """Outcome of a video render attempt."""

    success: bool
    video_path: str | None
    error: str | None
    attempts: int


def render_video(
    code_file: Path,
    output_dir: Path,
    api_key: str | None = None,
    max_retries: int | None = None,
) -> RenderResult:
    """Render Manim code to MP4, retrying with LLM fixes on failure.

    Args:
        code_file: Path to the Manim Python source file.
        output_dir: Directory for rendered video output.
        api_key: OpenRouter API key (falls back to settings).
        max_retries: Max LLM fix attempts (falls back to settings).

    Returns:
        RenderResult with success status, video path, and attempt count.
    """
    settings = get_settings()
    key = api_key or settings.openrouter_api_key
    retries = max_retries if max_retries is not None else settings.render_max_retries

    output_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 2):  # 1 initial + retries
        logger.info("Render attempt %d for %s", attempt, code_file.name)

        video_path, error = _run_render(code_file, output_dir)

        if video_path is not None:
            logger.info("Render succeeded on attempt %d: %s", attempt, video_path)
            return RenderResult(
                success=True,
                video_path=video_path,
                error=None,
                attempts=attempt,
            )

        logger.warning("Render failed (attempt %d): %s", attempt, error[:200])

        # No more retries left
        if attempt > retries:
            break

        # No API key means we can't call the LLM fixer
        if not key:
            logger.warning("No API key available, skipping LLM fix")
            break

        # Try to fix with LLM
        try:
            code = code_file.read_text(encoding="utf-8")
            fixed_code = _fix_render_error(
                code=code,
                error=error or "",
                api_key=key,
                model=settings.render_fixer_model,
                base_url=settings.openrouter_base_url,
            )
            code_file.write_text(fixed_code, encoding="utf-8")
            logger.info("LLM fix applied, retrying render...")
        except Exception as exc:
            logger.error("LLM fix failed: %s", exc)
            break

    return RenderResult(
        success=False,
        video_path=None,
        error=error,
        attempts=attempt,
    )


def _run_render(code_file: Path, output_dir: Path) -> tuple[str | None, str | None]:
    """Execute render_video.py and return (video_path, error).

    Returns:
        Tuple of (video_path, None) on success or (None, error_message) on failure.
    """
    settings = get_settings()

    if settings.manim_video_agent_in_docker:
        cmd = ["python", "render_video.py", str(code_file), str(output_dir)]
    else:
        cmd = [
            "docker",
            "compose",
            "run",
            "--rm",
            "manim",
            "python",
            "render_video.py",
            str(code_file),
            str(output_dir),
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None, "Render timed out after 600 seconds"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if len(stderr) > _MAX_STDERR_CHARS:
            stderr = stderr[-_MAX_STDERR_CHARS:]
        return None, f"Exit code {result.returncode}:\n{stderr}"

    # Last stdout line is the MP4 path
    stdout_lines = result.stdout.strip().splitlines()
    if not stdout_lines:
        return None, "Render produced no output"

    video_path = stdout_lines[-1].strip()
    return video_path, None


def _fix_render_error(
    code: str,
    error: str,
    api_key: str,
    model: str,
    base_url: str,
) -> str:
    """Send code + error to LLM and return the fixed code.

    Args:
        code: Current Manim source code.
        error: Render error / traceback.
        api_key: OpenRouter API key.
        model: Model identifier.
        base_url: OpenRouter API base URL.

    Returns:
        Fixed Manim code string.

    Raises:
        RuntimeError: If the API call fails.
    """
    truncated_error = (
        error[-_MAX_STDERR_CHARS:] if len(error) > _MAX_STDERR_CHARS else error
    )

    prompt = f"""<code>
{code}
</code>

<render_error>
{truncated_error}
</render_error>

The Manim code above failed to render with the error shown. Fix the code so it \
renders successfully. Return only the corrected Python code."""

    response = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": RENDER_FIXER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Render fixer API failed: {response.status_code}: {response.text}"
        )

    raw_content = response.json()["choices"][0]["message"].get("content")
    if not raw_content:
        raise RuntimeError("Render fixer LLM returned empty/null content")
    return _extract_code(raw_content)


def _strip_think_tags(text: str) -> str:
    import re

    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_code(content: str) -> str:
    if not content:
        return ""
    content = _strip_think_tags(content)
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

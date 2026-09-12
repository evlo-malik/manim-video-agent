"""
Manim text utilities for handling Unicode symbols.

Drop-in replacement for Text() that renders Unicode math symbols
(!=, ->, >=, etc.) correctly via LaTeX.

Usage in scenes:
    from manim import *
    from manim_video_agent.manim.text_utils import latex_text

    class MyScene(Scene):
        def construct(self):
            t = latex_text("High accuracy != safety", font_size=28, color=WHITE)
            self.play(Write(t))

Auto-fix a file:
    from manim_video_agent.manim.text_utils import fix_file
    fix_file("output.py")      # rewrites the file in-place
"""

from __future__ import annotations

import re

# Maps Unicode symbols to their LaTeX equivalents.
SYMBOL_TO_LATEX: dict[str, str] = {
    "\u2260": r"\neq",
    "\u2192": r"\rightarrow",
    "\u2190": r"\leftarrow",
    "\u2194": r"\leftrightarrow",
    "\u2265": r"\geq",
    "\u2264": r"\leq",
    "\u00d7": r"\times",
    "\u00f7": r"\div",
    "\u00b1": r"\pm",
    "\u2248": r"\approx",
    "\u221e": r"\infty",
    "\u2208": r"\in",
    "\u2209": r"\notin",
    "\u2282": r"\subset",
    "\u2286": r"\subseteq",
    "\u222a": r"\cup",
    "\u2229": r"\cap",
    "\u2211": r"\sum",
    "\u220f": r"\prod",
    "\u221a": r"\sqrt{}",
    "\u2202": r"\partial",
    "\u2207": r"\nabla",
    "\u2200": r"\forall",
    "\u2203": r"\exists",
    "\u00ac": r"\neg",
    "\u2227": r"\wedge",
    "\u2228": r"\vee",
    "\u27f9": r"\Rightarrow",
    "\u27f8": r"\Leftarrow",
    "\u27fa": r"\Leftrightarrow",
    "\u00b7": r"\cdot",
    "\u00b0": r"^{\circ}",
    "\u03b1": r"\alpha",
    "\u03b2": r"\beta",
    "\u03b3": r"\gamma",
    "\u03b4": r"\delta",
    "\u03b8": r"\theta",
    "\u03bb": r"\lambda",
    "\u03c0": r"\pi",
    "\u03c3": r"\sigma",
    "\u03bc": r"\mu",
}

_SYMBOL_CHARS = set(SYMBOL_TO_LATEX.keys())

_SYMBOL_PATTERN = re.compile(
    "|".join(re.escape(s) for s in sorted(SYMBOL_TO_LATEX, key=len, reverse=True))
)


def has_unicode_symbols(text: str) -> bool:
    """Check if text contains any Unicode symbols that need LaTeX rendering."""
    return bool(_SYMBOL_PATTERN.search(text))


def _split_at_symbols(text: str) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    last_end = 0
    for match in _SYMBOL_PATTERN.finditer(text):
        if match.start() > last_end:
            segments.append((text[last_end : match.start()], False))
        segments.append((match.group(), True))
        last_end = match.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))
    return segments


def latex_text(text: str, **kwargs):
    """Drop-in replacement for manim's Text(). If the string contains
    Unicode math symbols, they get rendered via MathTex (LaTeX).
    If no symbols found, returns a plain Text().

    Example:
        latex_text("Accuracy != Safety", font_size=28, color=RED)
    """
    from manim import MathTex, Text, VGroup

    segments = _split_at_symbols(text)

    if not any(is_sym for _, is_sym in segments):
        return Text(text, **kwargs)

    if len(segments) == 1 and segments[0][1]:
        latex = SYMBOL_TO_LATEX[segments[0][0]]
        mt_kwargs = {"font_size": kwargs.get("font_size", 48)}
        if "color" in kwargs:
            mt_kwargs["color"] = kwargs["color"]
        return MathTex(latex, **mt_kwargs)

    parts = []
    mt_kwargs = {"font_size": kwargs.get("font_size", 48)}
    if "color" in kwargs:
        mt_kwargs["color"] = kwargs["color"]

    for content, is_symbol in segments:
        if not content.strip() and not is_symbol:
            continue
        if is_symbol:
            parts.append(MathTex(SYMBOL_TO_LATEX[content], **mt_kwargs))
        else:
            parts.append(Text(content.strip(), **kwargs))

    return VGroup(*parts).arrange(buff=0.15)


# ---------------------------------------------------------------------------
# Auto-fixer
# ---------------------------------------------------------------------------


def _line_has_symbol(line: str) -> bool:
    """Check if a line contains any Unicode symbol from our table."""
    return any(c in line for c in _SYMBOL_CHARS)


def fix_file(filepath: str) -> list[tuple[int, str, str]]:
    """Auto-fix a manim .py file in-place:

    - Every Text() call whose string contains a Unicode symbol
      gets replaced with latex_text()
    - Handles multi-line Text() calls
    - Adds `from manim_video_agent.manim.text_utils import latex_text` if not present

    Returns list of (line_number, old_line, new_line) changes made.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    changes: list[tuple[int, str, str]] = []
    needs_import = False

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if stripped.startswith("#"):
            continue
        if "MathTex(" in line or "latex_text(" in line:
            continue

        # Case 1: Text( on this line with a symbol on this line
        if "Text(" in line and _line_has_symbol(line):
            new_line = line.replace("Text(", "latex_text(", 1)
            if new_line != line:
                changes.append((i + 1, line.rstrip(), new_line.rstrip()))
                lines[i] = new_line
                needs_import = True
            continue

        # Case 2: symbol on this line, look back for unclosed Text(
        if _line_has_symbol(line):
            for back in range(1, min(4, i + 1)):
                prev = lines[i - back]
                if "Text(" in prev and "latex_text(" not in prev:
                    new_prev = prev.replace("Text(", "latex_text(", 1)
                    if new_prev != prev:
                        changes.append((i - back + 1, prev.rstrip(), new_prev.rstrip()))
                        lines[i - back] = new_prev
                        needs_import = True
                    break
                if prev.rstrip().endswith(")"):
                    break

    if needs_import:
        has_import = any(
            "from manim_video_agent.manim.text_utils import" in line for line in lines
        )
        if not has_import:
            insert_at = 0
            for j, line in enumerate(lines):
                if "from manim import" in line or "import manim" in line:
                    insert_at = j + 1
            import_line = "from manim_video_agent.manim.text_utils import latex_text\n"
            lines.insert(insert_at, import_line)
            changes.insert(0, (insert_at + 1, "", import_line.rstrip()))

    if changes:
        with open(filepath, "w") as f:
            f.writelines(lines)

    return changes

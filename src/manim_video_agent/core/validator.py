"""
Manim Scene Validator

Detects visual bugs in Manim code WITHOUT rendering by treating every
mobject as a bounding-box block and checking for conflicts:

- Text overflowing its parent container (RoundedRectangle, Rectangle)
- Text/objects extending beyond the frame
- Text too long for a single line
- Font sizes that are unreasonably large/small
- VGroups that exceed frame bounds after arrangement

Usage:
    python -m manim_video_agent.core.validator output.py
    python -m manim_video_agent.core.validator --errors-only output.py

    # As a library
    from manim_video_agent.core.validator import validate_source
    issues = validate_source(code_string)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Manim default frame dimensions (from docs: 8 units tall, 16:9 ratio)
FRAME_HEIGHT = 8.0
FRAME_WIDTH = FRAME_HEIGHT * 16 / 9  # ~14.222
HALF_HEIGHT = FRAME_HEIGHT / 2  # 4.0
HALF_WIDTH = FRAME_WIDTH / 2  # ~7.111


class Severity(Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    INFO = "INFO"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    message: str
    line: int
    scene_name: str
    suggestion: str = ""
    code_snippet: str = ""

    def format(self, show_code: bool = True) -> str:
        colors = {
            Severity.ERROR: "\033[91m",
            Severity.WARNING: "\033[93m",
            Severity.INFO: "\033[94m",
        }
        reset = "\033[0m"
        dim = "\033[2m"
        c = colors.get(self.severity, "")

        parts = [
            f"{c}[{self.severity.value}]{reset} "
            f"{self.scene_name}:{self.line} - {self.message}"
        ]
        if show_code and self.code_snippet:
            for sl in self.code_snippet.splitlines():
                parts.append(f"  {dim}|{reset} {sl}")
        if self.suggestion:
            parts.append(f"  {dim}-> Fix:{reset} {self.suggestion}")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.format(show_code=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snippet(source_lines: list[str], line: int, context: int = 2) -> str:
    start = max(0, line - 1 - context)
    end = min(len(source_lines), line + context)
    parts: list[str] = []
    for i in range(start, end):
        lineno = i + 1
        marker = ">>>" if lineno == line else "   "
        parts.append(f"  {marker} {lineno:4d} | {source_lines[i].rstrip()}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Unicode symbols that render poorly in manim Text()
# ---------------------------------------------------------------------------

_BAD_UNICODE_IN_TEXT: dict[str, str] = {
    "\u2260": "\\neq",
    "\u2192": "\\rightarrow",
    "\u2190": "\\leftarrow",
    "\u2265": "\\geq",
    "\u2264": "\\leq",
    "\u00d7": "\\times",
    "\u00f7": "\\div",
    "\u00b1": "\\pm",
    "\u2248": "\\approx",
    "\u221e": "\\infty",
}

# Maps problematic Unicode chars to safe ASCII replacements for Text().
UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2260": "!=",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2194": "<->",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u00d7": "x",
    "\u00f7": "/",
    "\u00b1": "+/-",
    "\u2248": "~=",
    "\u221e": "inf",
    "\u00b0": " deg",
    "\u00b7": "*",
    "\u2026": "...",
    "\u2014": " - ",
    "\u2013": "-",
    "\u00a0": " ",
    "\u2713": "[+]",
    "\u2717": "[x]",
    "\u2714": "[+]",
    "\u2718": "[x]",
}


def sanitize_text(text: str) -> str:
    """Replace Unicode symbols that render poorly in manim's Text()
    with safe ASCII equivalents.

    Usage:
        text = Text(sanitize_text("High accuracy != safety"), font_size=24)
    """
    result = text
    for char, replacement in UNICODE_REPLACEMENTS.items():
        result = result.replace(char, replacement)
    return result


def sanitize_file(filepath: str, dry_run: bool = False) -> list[tuple[int, str, str]]:
    """Scan a manim .py file and replace problematic Unicode in Text() strings.

    Returns list of (line_number, before, after) for each changed line.
    If dry_run=True, does not write changes.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    changes: list[tuple[int, str, str]] = []

    for i, line in enumerate(lines):
        has_bad_char = any(c in line for c in UNICODE_REPLACEMENTS)
        if not has_bad_char:
            continue

        stripped = line.lstrip()
        if any(stripped.startswith(prefix) for prefix in ("MathTex(", "Tex(", "#")):
            continue

        new_line = line
        for char, replacement in UNICODE_REPLACEMENTS.items():
            new_line = new_line.replace(char, replacement)

        if new_line != line:
            changes.append((i + 1, line.rstrip(), new_line.rstrip()))
            lines[i] = new_line

    if changes and not dry_run:
        with open(filepath, "w") as f:
            f.writelines(lines)

    return changes


# Approximate char width at font_size=48 in manim units (empirically measured).
_CHAR_WIDTH_PER_48 = 0.35
_CHAR_HEIGHT_PER_48 = 0.55


def _est_text_width(text: str, font_size: int) -> float:
    lines = text.split("\\n") if "\\n" in text else text.split("\n")
    longest = max(len(l) for l in lines)
    return longest * _CHAR_WIDTH_PER_48 * (font_size / 48)


def _est_text_height(text: str, font_size: int) -> float:
    lines = text.split("\\n") if "\\n" in text else text.split("\n")
    return len(lines) * _CHAR_HEIGHT_PER_48 * (font_size / 48)


# ---------------------------------------------------------------------------
# AST extraction helpers
# ---------------------------------------------------------------------------


def _get_str_arg(node: ast.Call) -> Optional[str]:
    """Get the first string argument from a Call node."""
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    if node.args and isinstance(node.args[0], ast.JoinedStr):
        return "x" * 30  # f-string placeholder
    return None


def _get_kwarg_num(node: ast.Call, name: str) -> Optional[float]:
    """Extract a numeric keyword argument value."""
    for kw in node.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            val = kw.value.value
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _get_func_name(node: ast.Call) -> Optional[str]:
    """Get the function/class name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# ---------------------------------------------------------------------------
# Block model: every relevant mobject becomes a Block with estimated bbox
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """A bounding-box block representing a mobject."""
    kind: str         # "text", "mathtex", "rect", "vgroup", etc.
    est_width: float  # estimated width in manim units
    est_height: float  # estimated height in manim units
    line: int         # source line
    label: str = ""   # human-readable label


# Numpy constants that are NOT Mobjects — calling Mobject methods on them crashes.
_NUMPY_CONSTANTS = {
    "ORIGIN", "UP", "DOWN", "LEFT", "RIGHT",
    "UL", "UR", "DL", "DR", "IN", "OUT",
}
# Mobject methods that should never be called on numpy arrays.
_MOBJECT_METHODS = {
    "shift", "scale", "move_to", "next_to", "to_edge", "to_corner",
    "align_to", "set_color", "set_opacity", "rotate", "flip",
    "stretch", "set_fill", "set_stroke", "animate", "add_updater",
    "become", "match_width", "match_height",
}

_RECT_TYPES = {"RoundedRectangle", "Rectangle", "Square"}
_TEXT_TYPES = {"Text", "MarkupText", "Paragraph"}
_MATH_TYPES = {"MathTex", "Tex"}


def _block_from_call(node: ast.Call) -> Optional[Block]:
    """Try to create a Block from an AST Call node (constructor call)."""
    name = _get_func_name(node)
    if name is None:
        return None

    if name in _RECT_TYPES:
        w = _get_kwarg_num(node, "width")
        h = _get_kwarg_num(node, "height")
        if w is not None and h is not None:
            return Block(kind="rect", est_width=w, est_height=h,
                         line=node.lineno, label=name)

    if name in _TEXT_TYPES:
        text = _get_str_arg(node)
        if text is not None:
            fs = _get_kwarg_num(node, "font_size") or 48
            return Block(kind="text", est_width=_est_text_width(text, int(fs)),
                         est_height=_est_text_height(text, int(fs)),
                         line=node.lineno, label=f'"{text[:50]}"')

    if name in _MATH_TYPES:
        text = _get_str_arg(node)
        if text is not None:
            fs = _get_kwarg_num(node, "font_size") or 48
            visible = len(text.replace("\\", "").replace("{", "").replace("}", ""))
            effective = text
            return Block(kind="mathtex",
                         est_width=visible * _CHAR_WIDTH_PER_48 * (fs / 48) * 0.7,
                         est_height=_est_text_height(effective, int(fs)),
                         line=node.lineno, label=f'"{text[:50]}"')

    return None


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class _SceneAnalyzer(ast.NodeVisitor):
    def __init__(self, scene_name: str, source_lines: list[str]):
        self.scene_name = scene_name
        self.src = source_lines
        self.issues: list[ValidationIssue] = []

    def _issue(self, sev: Severity, msg: str, line: int, fix: str = "") -> None:
        self.issues.append(ValidationIssue(
            severity=sev, message=msg, line=line,
            scene_name=self.scene_name, suggestion=fix,
            code_snippet=_snippet(self.src, line),
        ))

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_func_name(node)

        if name == "VGroup":
            self._check_vgroup_container(node)

        if name in _TEXT_TYPES:
            self._check_text_vs_frame(node)

        if name in _MATH_TYPES:
            self._check_text_vs_frame(node)

        if name in (_TEXT_TYPES | _MATH_TYPES):
            self._check_font_size(node, name)

        if name in _TEXT_TYPES:
            self._check_unicode_symbols(node)

        self._check_position(node)
        self.generic_visit(node)

    def _check_vgroup_container(self, vgroup_node: ast.Call) -> None:
        rect_block: Optional[Block] = None
        text_blocks: list[Block] = []

        for arg in vgroup_node.args:
            if not isinstance(arg, ast.Call):
                continue
            block = _block_from_call(arg)
            if block is None:
                continue
            if block.kind == "rect":
                rect_block = block
            elif block.kind in ("text", "mathtex"):
                text_blocks.append(block)

        if rect_block is None or not text_blocks:
            return

        for tb in text_blocks:
            if tb.est_width > rect_block.est_width:
                overflow_w = tb.est_width - rect_block.est_width
                self._issue(
                    Severity.ERROR,
                    f"Text overflows container: text is ~{tb.est_width:.1f}u wide "
                    f"but box is {rect_block.est_width:.1f}u wide "
                    f"(overflow: {overflow_w:.1f}u) - {tb.label}",
                    tb.line,
                    f"Widen the box to width={tb.est_width + 0.4:.1f}, "
                    f"reduce font_size, or shorten text",
                )
            elif tb.est_width > rect_block.est_width * 0.85:
                self._issue(
                    Severity.WARNING,
                    f"Text nearly fills container: ~{tb.est_width:.1f}u wide "
                    f"in {rect_block.est_width:.1f}u box - {tb.label}",
                    tb.line,
                    "Add padding or shorten text slightly",
                )

            if tb.est_height > rect_block.est_height:
                overflow_h = tb.est_height - rect_block.est_height
                self._issue(
                    Severity.ERROR,
                    f"Text overflows container vertically: ~{tb.est_height:.1f}u tall "
                    f"but box is {rect_block.est_height:.1f}u tall "
                    f"(overflow: {overflow_h:.1f}u) - {tb.label}",
                    tb.line,
                    f"Increase box height to {tb.est_height + 0.3:.1f} "
                    f"or reduce font_size",
                )

    def _check_text_vs_frame(self, node: ast.Call) -> None:
        parent = getattr(node, "_parent", None)
        if isinstance(parent, ast.Call) and _get_func_name(parent) == "VGroup":
            return

        block = _block_from_call(node)
        if block is None:
            return

        if self._has_scale_to_fit(node):
            return

        if block.est_width > FRAME_WIDTH * 0.95:
            self._issue(
                Severity.ERROR,
                f"Text likely overflows frame: ~{block.est_width:.1f}u wide "
                f"(frame={FRAME_WIDTH:.1f}u) - {block.label}",
                node.lineno,
                f"Add .scale_to_fit_width({FRAME_WIDTH - 1:.1f}) or shorten text",
            )
        elif block.est_width > FRAME_WIDTH * 0.80:
            self._issue(
                Severity.WARNING,
                f"Text may be tight in frame: ~{block.est_width:.1f}u wide "
                f"(frame={FRAME_WIDTH:.1f}u) - {block.label}",
                node.lineno,
                "Consider .scale_to_fit_width() or shorter text",
            )

        if block.est_height > FRAME_HEIGHT * 0.85:
            self._issue(
                Severity.ERROR,
                f"Text too tall for frame: ~{block.est_height:.1f}u "
                f"(frame={FRAME_HEIGHT:.1f}u)",
                node.lineno,
                "Reduce font_size or split into multiple scenes",
            )

        name = _get_func_name(node)
        if name in _TEXT_TYPES:
            text = _get_str_arg(node)
            if text:
                for text_line in (text.split("\\n") if "\\n" in text else text.split("\n")):
                    if len(text_line) > 80:
                        self._issue(
                            Severity.WARNING,
                            f"Line has {len(text_line)} chars (>80), "
                            f"likely overflows: \"{text_line[:40]}...\"",
                            node.lineno,
                            "Keep lines under ~60 characters",
                        )

    def _check_unicode_symbols(self, node: ast.Call) -> None:
        text = _get_str_arg(node)
        if text is None:
            return
        found = [ch for ch in _BAD_UNICODE_IN_TEXT if ch in text]
        if not found:
            return
        symbols = " ".join(f"'{c}'" for c in found)
        self._issue(
            Severity.ERROR,
            f"Text() contains Unicode symbols that render poorly: {symbols} "
            f"in \"{text[:50]}\"",
            node.lineno,
            "Use latex_text() from manim_video_agent.manim.text_utils instead: "
            f"latex_text(\"{text[:40]}...\", font_size=...)",
        )

    def _check_font_size(self, node: ast.Call, type_name: str) -> None:
        fs = _get_kwarg_num(node, "font_size")
        if fs is None:
            return
        if fs > 72:
            self._issue(
                Severity.WARNING,
                f"font_size={int(fs)} is very large",
                node.lineno,
                "Use <= 56 for titles, <= 36 for body text",
            )
        elif fs < 12:
            self._issue(
                Severity.WARNING,
                f"font_size={int(fs)} is too small to read",
                node.lineno,
                "Use >= 20 for readability",
            )

    def _check_position(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        method = node.func.attr
        if method not in ("move_to", "shift"):
            return
        if not node.args:
            return

        coords = self._extract_coords(node.args[0])
        if coords is None:
            return

        x, y = coords[0], coords[1]
        if abs(x) > HALF_WIDTH:
            self._issue(
                Severity.WARNING,
                f".{method}() x={x:.1f} is beyond frame (+/-{HALF_WIDTH:.1f})",
                node.lineno,
                f"Keep x between -{HALF_WIDTH:.1f} and {HALF_WIDTH:.1f}",
            )
        if abs(y) > HALF_HEIGHT:
            self._issue(
                Severity.WARNING,
                f".{method}() y={y:.1f} is beyond frame (+/-{HALF_HEIGHT:.1f})",
                node.lineno,
                f"Keep y between -{HALF_HEIGHT:.1f} and {HALF_HEIGHT:.1f}",
            )

    def _has_scale_to_fit(self, node: ast.AST) -> bool:
        parent = getattr(node, "_parent", None)
        visited: set[int] = set()
        while parent and id(parent) not in visited:
            visited.add(id(parent))
            if isinstance(parent, ast.Attribute):
                if parent.attr in ("scale_to_fit_width", "scale_to_fit_height"):
                    return True
            parent = getattr(parent, "_parent", None)
        return False

    def _extract_coords(self, node: ast.AST) -> Optional[list[float]]:
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        vals: list[float] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                vals.append(float(elt.value))
            elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                if isinstance(elt.operand, ast.Constant):
                    vals.append(-float(elt.operand.value))
            else:
                return None
        return vals if len(vals) >= 2 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _set_parents(node: ast.AST) -> None:
    for child in ast.walk(node):
        for child_child in ast.iter_child_nodes(child):
            child_child._parent = child  # type: ignore[attr-defined]


def validate_source(source: str, filename: str = "<string>") -> list[ValidationIssue]:
    """Validate manim source code for visual bugs (no rendering needed)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [ValidationIssue(
            severity=Severity.ERROR,
            message=f"Syntax error: {e.msg}",
            line=e.lineno or 0,
            scene_name=filename,
        )]

    source_lines = source.splitlines()
    _set_parents(tree)
    issues: list[ValidationIssue] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        is_scene = any(
            (isinstance(base, ast.Name) and "Scene" in base.id)
            or (isinstance(base, ast.Attribute) and "Scene" in base.attr)
            for base in node.bases
        )
        if not is_scene:
            continue

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "construct":
                analyzer = _SceneAnalyzer(node.name, source_lines)
                analyzer.visit(item)
                issues.extend(analyzer.issues)

    return issues


def validate_scene(filepath: str) -> list[ValidationIssue]:
    """Validate a manim .py file for visual bugs."""
    with open(filepath, "r") as f:
        source = f.read()
    return validate_source(source, filename=filepath)


# ---------------------------------------------------------------------------
# Runtime validation (requires manim installed - use inside Docker)
# ---------------------------------------------------------------------------


def validate_mobject_bounds(mobject, label: str = "") -> list[str]:
    """Check if a mobject is within frame bounds. Call inside construct()."""
    warnings: list[str] = []
    name = label or type(mobject).__name__
    try:
        r, l = mobject.get_right()[0], mobject.get_left()[0]
        t, b = mobject.get_top()[1], mobject.get_bottom()[1]
    except Exception:
        return warnings

    if r > HALF_WIDTH:
        warnings.append(f"{name}: right edge ({r:.2f}) exceeds frame by {r - HALF_WIDTH:.2f}u")
    if l < -HALF_WIDTH:
        warnings.append(f"{name}: left edge ({l:.2f}) exceeds frame by {-HALF_WIDTH - l:.2f}u")
    if t > HALF_HEIGHT:
        warnings.append(f"{name}: top ({t:.2f}) exceeds frame by {t - HALF_HEIGHT:.2f}u")
    if b < -HALF_HEIGHT:
        warnings.append(f"{name}: bottom ({b:.2f}) exceeds frame by {-HALF_HEIGHT - b:.2f}u")
    return warnings


def check_overlap(mob_a, mob_b) -> bool:
    """AABB overlap test between two mobjects."""
    try:
        al, ar = mob_a.get_left()[0], mob_a.get_right()[0]
        ab, at_ = mob_a.get_bottom()[1], mob_a.get_top()[1]
        bl, br = mob_b.get_left()[0], mob_b.get_right()[0]
        bb, bt = mob_b.get_bottom()[1], mob_b.get_top()[1]
    except Exception:
        return False
    return al < br and ar > bl and ab < bt and at_ > bb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from manim_video_agent.manim.text_utils import fix_file

    parser = argparse.ArgumentParser(description="Validate Manim scenes for visual bugs")
    parser.add_argument("files", nargs="+", help="Manim .py files to validate")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix errors (Unicode symbols, etc.) in-place")
    parser.add_argument("--errors-only", action="store_true",
                        help="Only show errors, not warnings")
    parser.add_argument("--no-code", action="store_true",
                        help="Hide code snippets")
    args = parser.parse_args()

    exit_code = 0
    for filepath in args.files:
        print(f"\n{'=' * 60}")
        print(f"  Validating: {filepath}")
        print(f"{'=' * 60}")

        issues = validate_scene(filepath)

        if args.errors_only:
            issues = [i for i in issues if i.severity == Severity.ERROR]

        if not issues:
            print("  No issues found.\n")
            continue

        errors = [i for i in issues if i.severity == Severity.ERROR]
        warnings = [i for i in issues if i.severity == Severity.WARNING]

        for issue in issues:
            print()
            print(issue.format(show_code=not args.no_code))

        print(f"\n  Summary: {len(errors)} error(s), {len(warnings)} warning(s)")

        if args.fix:
            changes = fix_file(filepath)
            if changes:
                print(f"\n  Auto-fixed {len(changes)} line(s):")
                for lineno, old, new in changes:
                    if old:
                        print(f"    L{lineno}: Text( -> latex_text(")
                    else:
                        print(f"    L{lineno}: + {new.strip()}")

        if errors:
            exit_code = 1

    sys.exit(exit_code)

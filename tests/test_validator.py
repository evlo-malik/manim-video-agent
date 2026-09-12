from manim_video_agent.core.validator import Severity, sanitize_text, validate_source


def test_replaces_unicode_that_plain_text_cannot_render_reliably() -> None:
    assert sanitize_text("x ≥ 2 → valid") == "x >= 2 -> valid"


def test_reports_syntax_errors() -> None:
    issues = validate_source(
        "class Main(Scene):\n    def construct(self)\n        pass"
    )

    assert any(issue.severity is Severity.ERROR for issue in issues)


def test_accepts_a_small_scene() -> None:
    code = """
from manim import *

class Main(Scene):
    def construct(self):
        title = Text("A clear title", font_size=36)
        self.play(Write(title))
        self.wait(1)
"""

    errors = [
        issue for issue in validate_source(code) if issue.severity is Severity.ERROR
    ]
    assert errors == []

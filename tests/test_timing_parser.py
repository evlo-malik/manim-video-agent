from manim_video_agent.core.timing_parser import (
    parse_scene_timings,
    parse_single_scene_timing,
    total_duration,
)


def test_extracts_construct_call_order_and_duration() -> None:
    code = """
class Main(Scene):
    def construct(self):
        self.intro()
        self.result()

    def intro(self):
        self.play(Write(title), run_time=2.5)
        self.wait(1)

    def result(self):
        self.play(FadeIn(answer))
        self.clear_scene()
"""

    timings = parse_scene_timings(code)

    assert [timing.method_name for timing in timings] == ["intro", "result"]
    assert [timing.duration_seconds for timing in timings] == [3.5, 2.0]
    assert total_duration(timings) == 5.5


def test_extracts_independent_scene_timing() -> None:
    code = """
class Main(Scene):
    def construct(self):
        self.play(Create(circle))
        self.wait(0.75)
"""

    timing = parse_single_scene_timing(code, scene_id="circle")

    assert timing is not None
    assert timing.method_name == "circle"
    assert timing.duration_seconds == 1.75


def test_returns_none_without_construct() -> None:
    assert parse_single_scene_timing("class Main: pass") is None

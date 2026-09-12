"""
Manim monkey-patch preambles for auto-scaling and frame-fitting.

These preambles are injected into generated Manim code to automatically
handle text overflow and off-screen elements at runtime.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Auto-scale move_to preamble
# ---------------------------------------------------------------------------

MOVE_TO_AUTOSCALE_PREAMBLE = '''\
# _MANIM_AUTOSCALE_PREAMBLE_V1
def _setup_autoscale_move_to():
    """Monkey-patch Mobject.move_to to auto-scale text that overflows a container."""
    from manim import (
        BackgroundRectangle,
        Circle,
        Mobject,
        Paragraph,
        Rectangle,
        RoundedRectangle,
        Square,
        SurroundingRectangle,
        Text,
        VGroup,
    )

    try:
        from manim import MarkupText
    except ImportError:
        MarkupText = None

    try:
        from manim import MathTex, Tex
    except ImportError:
        MathTex = None
        Tex = None

    _TEXT_TYPES = tuple(
        t for t in (Text, Paragraph, MarkupText, MathTex, Tex) if t is not None
    )
    _CONTAINER_TYPES = (
        Circle,
        Rectangle,
        RoundedRectangle,
        Square,
        SurroundingRectangle,
        BackgroundRectangle,
    )

    _PADDING = 0.25
    _CIRCLE_USABLE_RATIO = 0.70  # inscribed rect is ~70% of circle diameter

    def _is_text_like(mob):
        if isinstance(mob, _TEXT_TYPES):
            return True
        if isinstance(mob, VGroup) and len(mob) > 0:
            return all(_is_text_like(sub) for sub in mob)
        return False

    _original_move_to = Mobject.move_to

    def _autoscale_move_to(self, target, *args, **kwargs):
        if not _is_text_like(self) or not isinstance(target, _CONTAINER_TYPES):
            return _original_move_to(self, target, *args, **kwargs)

        _original_move_to(self, target, *args, **kwargs)

        if not hasattr(target, "width") or not hasattr(target, "height"):
            return self
        if target.width <= 0 or target.height <= 0:
            return self

        if not hasattr(self, "_autoscale_original_width"):
            self._autoscale_original_width = self.width
            self._autoscale_original_height = self.height

        is_circle = isinstance(target, Circle)

        # Available space: circles use inscribed rectangle (~70% of diameter)
        if is_circle:
            max_width = target.width * _CIRCLE_USABLE_RATIO - 2 * _PADDING
            max_height = target.height * _CIRCLE_USABLE_RATIO - 2 * _PADDING
        else:
            max_width = target.width - 2 * _PADDING
            max_height = target.height - 2 * _PADDING

        if max_width <= 0 or max_height <= 0:
            return self

        scale_x = max_width / self._autoscale_original_width if self._autoscale_original_width > 0 else 1.0
        scale_y = max_height / self._autoscale_original_height if self._autoscale_original_height > 0 else 1.0
        needs_scaling = min(scale_x, scale_y) < 1.0

        if needs_scaling:
            # Expand the parent container (once only to avoid compounding)
            if not getattr(target, "_autoscale_expanded", False):
                expand_factor = 1.3 if is_circle else 1.2
                target.scale(expand_factor)
                target._autoscale_expanded = True

            # Recalculate available space after expansion
            if is_circle:
                max_width = target.width * _CIRCLE_USABLE_RATIO - 2 * _PADDING
                max_height = target.height * _CIRCLE_USABLE_RATIO - 2 * _PADDING
            else:
                max_width = target.width - 2 * _PADDING
                max_height = target.height - 2 * _PADDING

            if max_width <= 0 or max_height <= 0:
                return self

            # Shrink text to fit within the (now expanded) container
            scale_x = max_width / self._autoscale_original_width if self._autoscale_original_width > 0 else 1.0
            scale_y = max_height / self._autoscale_original_height if self._autoscale_original_height > 0 else 1.0
            scale_factor = min(scale_x, scale_y, 1.0)

            if scale_factor < 1.0:
                target_w = self._autoscale_original_width * scale_factor
                target_h = self._autoscale_original_height * scale_factor
                current_w = self.width if self.width > 0 else 1.0
                current_h = self.height if self.height > 0 else 1.0
                rescale = min(target_w / current_w, target_h / current_h)
                self.scale(rescale)

            _original_move_to(self, target, *args, **kwargs)

        return self

    Mobject.move_to = _autoscale_move_to

_setup_autoscale_move_to()
'''


# ---------------------------------------------------------------------------
# Frame-fit preamble: keep all mobjects within visible frame
# ---------------------------------------------------------------------------

FRAME_FIT_PREAMBLE = """\
# _MANIM_FRAME_FIT_PREAMBLE_V1
def _setup_frame_fit():
    from manim import Scene, config, Mobject
    from manim.animation.animation import Animation

    try:
        from manim.animation.animation import _AnimationBuilder
    except ImportError:
        _AnimationBuilder = None

    try:
        from manim import FullScreenRectangle
    except ImportError:
        FullScreenRectangle = None

    def _should_skip(mob):
        if getattr(mob, "_disable_frame_fit", False):
            return True
        if FullScreenRectangle is not None and isinstance(mob, FullScreenRectangle):
            return True
        if getattr(mob, "is_background", False):
            return True
        return False

    def _is_off_frame(mob):
        try:
            w = mob.width
            h = mob.height
        except Exception:
            return False
        if w < 1e-6 and h < 1e-6:
            return False
        try:
            r = mob.get_right()[0]
            l = mob.get_left()[0]
            t = mob.get_top()[1]
            b = mob.get_bottom()[1]
        except Exception:
            return False
        half_w = config.frame_width / 2
        half_h = config.frame_height / 2
        margin = 0.1
        return (r > half_w + margin or l < -half_w - margin or
                t > half_h + margin or b < -half_h - margin)

    def _get_mobjects_for_arg(arg):
        mobs = []
        try:
            if isinstance(arg, Animation):
                top_mob = getattr(arg, "mobject", None)
                if top_mob is not None:
                    mobs.append(top_mob)
                sub_anims = getattr(arg, "animations", None)
                if sub_anims:
                    for sub in sub_anims:
                        mob = getattr(sub, "mobject", None)
                        if mob is not None:
                            mobs.append(mob)
            elif _AnimationBuilder is not None and isinstance(arg, _AnimationBuilder):
                target = getattr(arg.mobject, "target", None)
                if target is not None:
                    mobs.append(target)
            elif isinstance(arg, Mobject):
                mobs.append(arg)
        except Exception:
            pass
        return mobs

    def _should_keep_arg(arg):
        # Always keep remover animations (FadeOut etc.)
        if isinstance(arg, Animation) and getattr(arg, "remover", False):
            return True

        mobs = _get_mobjects_for_arg(arg)

        # No extractable mobjects — keep as safe default
        if not mobs:
            return True

        # Keep if any mobject should be skipped (backgrounds, etc.)
        if any(_should_skip(mob) for mob in mobs):
            return True

        # Keep if at least one mobject is on-frame
        return any(not _is_off_frame(mob) for mob in mobs)

    _original_play = Scene.play
    _original_add = Scene.add

    def _frame_fit_play(self, *args, **kwargs):
        filtered_args = tuple(arg for arg in args if _should_keep_arg(arg))
        if not filtered_args:
            return
        return _original_play(self, *filtered_args, **kwargs)

    def _frame_fit_add(self, *mobjects, **kwargs):
        filtered = tuple(
            mob for mob in mobjects
            if not isinstance(mob, Mobject)
               or _should_skip(mob)
               or not _is_off_frame(mob)
        )
        if not filtered:
            return
        return _original_add(self, *filtered, **kwargs)

    Scene.play = _frame_fit_play
    Scene.add = _frame_fit_add

_setup_frame_fit()
"""


# ---------------------------------------------------------------------------
# Overlap checker preamble: detect AABB collisions between mobjects
# ---------------------------------------------------------------------------

OVERLAP_CHECKER_PREAMBLE = '''\
# _MANIM_OVERLAP_CHECKER_V1
def _setup_overlap_checker():
    """Monkey-patch Scene to detect AABB overlaps between mobjects."""
    from manim import Scene, Mobject, config
    from manim import BackgroundRectangle, SurroundingRectangle

    try:
        from manim import FullScreenRectangle
    except ImportError:
        FullScreenRectangle = None

    _MIN_OVERLAP = 0.15  # units — ignore tiny edge touches
    _MIN_SIZE = 0.05     # skip point/invisible mobjects
    _reported = set()

    def _skip(mob):
        if isinstance(mob, BackgroundRectangle):
            return True
        if FullScreenRectangle and isinstance(mob, FullScreenRectangle):
            return True
        if getattr(mob, "is_background", False):
            return True
        try:
            if mob.width < _MIN_SIZE and mob.height < _MIN_SIZE:
                return True
        except Exception:
            return True
        return False

    def _aabb(mob):
        try:
            return (mob.get_left()[0], mob.get_right()[0],
                    mob.get_bottom()[1], mob.get_top()[1])
        except Exception:
            return None

    def _label(mob):
        cls = type(mob).__name__
        try:
            cx, cy = mob.get_center()[0], mob.get_center()[1]
            pos = f"@({cx:.1f},{cy:.1f})"
        except Exception:
            pos = ""
        if hasattr(mob, "text"):
            return f'{cls}("{str(mob.text)[:25]}"){pos}'
        if hasattr(mob, "tex_string"):
            return f'{cls}("{str(mob.tex_string)[:25]}"){pos}'
        n = len(getattr(mob, "submobjects", []))
        if n > 0:
            return f"{cls}({n}ch){pos}"
        return f"{cls}{pos}"

    def _is_ancestor(a, b):
        p = getattr(b, "parent", None)
        while p is not None:
            if p is a:
                return True
            p = getattr(p, "parent", None)
        return False

    def _scan(scene):
        mobs = [m for m in scene.mobjects if not _skip(m)]
        for i, a in enumerate(mobs):
            ba = _aabb(a)
            if not ba:
                continue
            for j in range(i + 1, len(mobs)):
                b = mobs[j]
                if _is_ancestor(a, b) or _is_ancestor(b, a):
                    continue
                if isinstance(a, SurroundingRectangle) or isinstance(b, SurroundingRectangle):
                    continue
                bb = _aabb(b)
                if not bb:
                    continue
                ox = min(ba[1], bb[1]) - max(ba[0], bb[0])
                oy = min(ba[3], bb[3]) - max(ba[2], bb[2])
                if ox > _MIN_OVERLAP and oy > _MIN_OVERLAP:
                    la, lb = _label(a), _label(b)
                    key = (la, lb) if la <= lb else (lb, la)
                    if key not in _reported:
                        _reported.add(key)
                        print(f"[OVERLAP_DETECTED] {la} overlaps {lb} by ({ox:.2f}x, {oy:.2f}y)")

    _orig_play = Scene.play
    _orig_add = Scene.add

    def _checked_play(self, *args, **kwargs):
        result = _orig_play(self, *args, **kwargs)
        try:
            _scan(self)
        except Exception:
            pass
        return result

    def _checked_add(self, *mobjects, **kwargs):
        result = _orig_add(self, *mobjects, **kwargs)
        try:
            _scan(self)
        except Exception:
            pass
        return result

    Scene.play = _checked_play
    Scene.add = _checked_add

_setup_overlap_checker()
'''


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------


def _find_import_block_end(lines: list[str]) -> int:
    """Find the index of the last import line in a list of code lines.

    Handles module docstrings, __future__ imports, multi-line imports,
    comments, and blank lines within the import region.
    """
    insert_after = 0
    i = 0

    # Skip leading docstring (triple-quoted)
    if i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                i += 1
            else:
                i += 1
                while i < len(lines) and quote not in lines[i]:
                    i += 1
                if i < len(lines):
                    i += 1

    # Walk through lines to find the end of the import block
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped == "":
            i += 1
            continue

        if stripped.startswith("#"):
            i += 1
            continue

        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_after = i
            i += 1
            if "(" in stripped and ")" not in stripped:
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
                if i < len(lines):
                    insert_after = i
                    i += 1
            continue

        break

    return insert_after


def inject_move_to_preamble(code: str) -> str:
    """Inject the auto-scale move_to preamble into generated manim code.

    Inserts after all imports. Idempotent via sentinel check.
    Returns a new string (never mutates input).
    """
    sentinel = "# _MANIM_AUTOSCALE_PREAMBLE_V1"
    if sentinel in code:
        return code

    lines = code.split("\n")
    insert_index = _find_import_block_end(lines) + 1

    before = "\n".join(lines[:insert_index])
    after = "\n".join(lines[insert_index:])

    return before + "\n\n" + MOVE_TO_AUTOSCALE_PREAMBLE + "\n" + after


def inject_frame_fit_preamble(code: str) -> str:
    """Inject the frame-fit preamble into generated manim code.

    Inserts after all imports. Idempotent via sentinel check.
    Returns a new string (never mutates input).
    """
    sentinel = "# _MANIM_FRAME_FIT_PREAMBLE_V1"
    if sentinel in code:
        return code

    lines = code.split("\n")
    insert_index = _find_import_block_end(lines) + 1

    before = "\n".join(lines[:insert_index])
    after = "\n".join(lines[insert_index:])

    return before + "\n\n" + FRAME_FIT_PREAMBLE + "\n" + after


def inject_overlap_checker_preamble(code: str) -> str:
    """Inject the overlap checker preamble into generated manim code.

    Inserts after all imports. Idempotent via sentinel check.
    Returns a new string (never mutates input).
    """
    sentinel = "# _MANIM_OVERLAP_CHECKER_V1"
    if sentinel in code:
        return code

    lines = code.split("\n")
    insert_index = _find_import_block_end(lines) + 1

    before = "\n".join(lines[:insert_index])
    after = "\n".join(lines[insert_index:])

    return before + "\n\n" + OVERLAP_CHECKER_PREAMBLE + "\n" + after


def inject_all_preambles(code: str) -> str:
    """Inject all runtime preambles into generated manim code.

    Chains move_to autoscale, frame-fit, and overlap checker preambles.
    Each is idempotent via its own sentinel check.
    Returns a new string (never mutates input).
    """
    result = inject_move_to_preamble(code)
    result = inject_frame_fit_preamble(result)
    return inject_overlap_checker_preamble(result)

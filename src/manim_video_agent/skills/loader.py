"""
Manim Skill Loader

Reads ManimCE best-practice rule files and injects them into LLM prompts.

Two modes:
  1. load_all()               — inject every rule file (recommended default)
  2. select_for_content(text)  — fuzzy-select only rules relevant to the input

Usage:
    from manim_video_agent.skills.loader import ManimSkillLoader

    loader = ManimSkillLoader()

    # All rules (simple, reliable)
    prompt_block = loader.format_for_prompt()

    # Selective (lighter context window)
    rules = loader.select_for_content(markdown_text)
    prompt_block = loader.format_for_prompt(rules)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from manim_video_agent import get_skills_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SKILLS_DIR = get_skills_dir("manimce-best-practices")
_DEFAULT_RULES_DIR = _DEFAULT_SKILLS_DIR / "rules"
_DEFAULT_EXAMPLES_DIR = _DEFAULT_SKILLS_DIR / "examples"
_DEFAULT_TEMPLATES_DIR = _DEFAULT_SKILLS_DIR / "templates"

# Rules that are always included in selective mode — they cover
# fundamentals needed by virtually every Manim scene.
_CORE_RULES = frozenset(
    {
        "scenes",
        "animations",
        "mobjects",
        "text",
        "positioning",
        "styling",
        "colors",
        "creation-animations",
        "timing",
    }
)

# Extended keyword map: maps search terms -> rule names.
_KEYWORD_TO_RULES: dict[str, list[str]] = {
    # graphing / axes
    "graph": ["graphing", "axes"],
    "plot": ["graphing", "axes"],
    "function": ["graphing"],
    "curve": ["graphing"],
    "axis": ["axes"],
    "axes": ["axes"],
    "coordinate": ["axes"],
    "numberplane": ["axes"],
    "numberline": ["axes"],
    "riemann": ["graphing"],
    "integral": ["graphing", "latex"],
    "derivative": ["graphing", "latex"],
    "area": ["graphing"],
    # 3d
    "3d": ["3d", "camera"],
    "threedscene": ["3d"],
    "surface": ["3d"],
    "sphere": ["3d"],
    "cube": ["3d"],
    "parametric": ["3d", "graphing"],
    # camera
    "camera": ["camera"],
    "zoom": ["camera"],
    "pan": ["camera"],
    "movingcamerascene": ["camera"],
    # latex / math
    "equation": ["latex"],
    "formula": ["latex"],
    "latex": ["latex"],
    "mathtex": ["latex"],
    "math": ["latex"],
    "tex": ["latex"],
    # transforms
    "transform": ["transform-animations"],
    "morph": ["transform-animations"],
    "replacementtransform": ["transform-animations"],
    # animation groups
    "laggedstart": ["animation-groups"],
    "animationgroup": ["animation-groups"],
    "succession": ["animation-groups"],
    "stagger": ["animation-groups"],
    "sequence": ["animation-groups"],
    # updaters
    "updater": ["updaters"],
    "valuetracker": ["updaters"],
    "dynamic": ["updaters"],
    "tracker": ["updaters"],
    "follow": ["updaters"],
    # text
    "font": ["text"],
    "typography": ["text"],
    "paragraph": ["text"],
    "markup": ["text"],
    "write": ["text-animations"],
    "typing": ["text-animations"],
    "typewriter": ["text-animations"],
    "letterbyletter": ["text-animations"],
    # shapes
    "circle": ["shapes"],
    "square": ["shapes"],
    "rectangle": ["shapes"],
    "polygon": ["shapes"],
    "triangle": ["shapes"],
    "geometry": ["shapes"],
    "shape": ["shapes"],
    # lines
    "arrow": ["lines"],
    "vector": ["lines"],
    "line": ["lines"],
    "dashedline": ["lines"],
    "brace": ["lines"],
    "connector": ["lines"],
    # grouping
    "vgroup": ["grouping"],
    "group": ["grouping"],
    "arrange": ["grouping"],
    "layout": ["grouping"],
    "grid": ["grouping", "axes"],
    # config / cli (rarely needed at generation time)
    "config": ["config"],
    "cfg": ["config"],
    "cli": ["cli"],
    "render": ["cli"],
    "quality": ["cli"],
}

# Tokenizer: split on whitespace + common punctuation, lowercased.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillRule:
    """A single parsed rule file."""

    name: str
    description: str
    tags: list[str]
    content: str  # full file content (including frontmatter)
    body: str  # content after frontmatter
    path: Path


@dataclass
class SelectionResult:
    """Result of fuzzy rule selection."""

    rules: list[SkillRule]
    scores: dict[str, float] = field(default_factory=dict)
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frontmatter parser (avoids PyYAML dependency)
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML-ish frontmatter between --- delimiters.

    Returns (metadata_dict, body_after_frontmatter).
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    header = text[3:end].strip()
    body = text[end + 3 :].lstrip("\n")

    meta: dict[str, str] = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "name":
                meta["name"] = value
            elif key == "description":
                meta["description"] = value
            elif key == "tags":
                meta["tags"] = value

    return meta, body


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ManimSkillLoader:
    """Reads ManimCE best-practice files and provides them for prompt injection."""

    def __init__(
        self,
        rules_dir: Optional[Path] = None,
        examples_dir: Optional[Path] = None,
        templates_dir: Optional[Path] = None,
    ):
        self._rules_dir = rules_dir or _DEFAULT_RULES_DIR
        self._examples_dir = examples_dir or _DEFAULT_EXAMPLES_DIR
        self._templates_dir = templates_dir or _DEFAULT_TEMPLATES_DIR
        self._rules: list[SkillRule] = []
        self._rules_by_name: dict[str, SkillRule] = {}
        self._load_rules()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        """Read and parse all rule .md files from disk."""
        if not self._rules_dir.is_dir():
            return

        for path in sorted(self._rules_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)

            name = meta.get("name", path.stem)
            description = meta.get("description", "")
            tags_raw = meta.get("tags", "")
            tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()]

            rule = SkillRule(
                name=name,
                description=description,
                tags=tags,
                content=text,
                body=body,
                path=path,
            )
            self._rules.append(rule)
            self._rules_by_name[name] = rule

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> list[SkillRule]:
        """Return all rule files."""
        return list(self._rules)

    def select_for_content(
        self,
        content: str,
        *,
        include_core: bool = True,
        min_score: float = 1.0,
    ) -> SelectionResult:
        """Select rules relevant to ``content`` via fuzzy keyword matching."""
        tokens = set(_TOKEN_RE.findall(content.lower()))
        scores: dict[str, float] = {r.name: 0.0 for r in self._rules}
        matched: dict[str, list[str]] = {r.name: [] for r in self._rules}

        # Layer 1: core rules
        if include_core:
            for name in _CORE_RULES:
                if name in scores:
                    scores[name] = float("inf")

        # Layer 2: keyword map (higher weight)
        for token in tokens:
            rule_names = _KEYWORD_TO_RULES.get(token)
            if rule_names:
                for rn in rule_names:
                    if rn in scores:
                        scores[rn] += 2.0
                        matched[rn].append(token)

        # Layer 3: frontmatter tags (lower weight)
        for rule in self._rules:
            for tag in rule.tags:
                if tag in tokens:
                    scores[rule.name] += 1.0
                    if tag not in matched[rule.name]:
                        matched[rule.name].append(tag)

        # Layer 4: substring match (fuzzy catch-all, 0.5 weight)
        for rule in self._rules:
            for tag in rule.tags:
                if len(tag) < 3:
                    continue
                for token in tokens:
                    if len(token) < 3:
                        continue
                    if tag in token or token in tag:
                        if (
                            tag not in matched[rule.name]
                            and token not in matched[rule.name]
                        ):
                            scores[rule.name] += 0.5
                            matched[rule.name].append(f"~{token}")

        # Filter
        selected = [r for r in self._rules if scores[r.name] >= min_score]

        # Stable sort: core first, then by score descending, then alphabetical
        def sort_key(r: SkillRule) -> tuple[int, float, str]:
            is_core = 0 if r.name in _CORE_RULES else 1
            return (is_core, -scores[r.name], r.name)

        selected.sort(key=sort_key)

        return SelectionResult(
            rules=selected,
            scores={k: v for k, v in scores.items() if v >= min_score},
            matched_keywords={
                k: v for k, v in matched.items() if scores.get(k, 0) >= min_score
            },
        )

    def load_examples(self) -> list[tuple[str, str]]:
        """Load example .py files. Returns list of (filename, content)."""
        if not self._examples_dir.is_dir():
            return []
        return [
            (p.name, p.read_text(encoding="utf-8"))
            for p in sorted(self._examples_dir.glob("*.py"))
        ]

    def load_templates(self) -> list[tuple[str, str]]:
        """Load template .py files. Returns list of (filename, content)."""
        if not self._templates_dir.is_dir():
            return []
        return [
            (p.name, p.read_text(encoding="utf-8"))
            for p in sorted(self._templates_dir.glob("*.py"))
        ]

    def format_for_prompt(
        self,
        rules: Optional[list[SkillRule]] = None,
        *,
        include_examples: bool = False,
        max_examples: int = 2,
    ) -> str:
        """Format selected rules into a string block for LLM system prompt injection."""
        if rules is None:
            rules = self.load_all()

        if not rules:
            return ""

        parts: list[str] = []
        parts.append("<manim_best_practices>")
        parts.append(
            "The following are ManimCE best-practice references. "
            "Use them to write correct, idiomatic Manim code.\n"
        )

        for rule in rules:
            parts.append(f"### {rule.name}")
            if rule.description:
                parts.append(f"_{rule.description}_\n")
            parts.append(rule.body.strip())
            parts.append("")  # blank line separator

        if include_examples:
            examples = self.load_examples()[:max_examples]
            if examples:
                parts.append("---")
                parts.append("## Reference Examples\n")
                for filename, content in examples:
                    parts.append(f"### {filename}")
                    parts.append(f"```python\n{content.strip()}\n```\n")

        parts.append("</manim_best_practices>")
        return "\n".join(parts)

    @property
    def rule_names(self) -> list[str]:
        """List all available rule names."""
        return [r.name for r in self._rules]

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"ManimSkillLoader({len(self._rules)} rules from {self._rules_dir})"

"""
Video Composition Planner — business logic.

Transforms vague video ideas into detailed scene-by-scene plans (scenes.md)
using the manim-composer skill references. Outputs comprehensive specifications
ready for implementation by the Manim code generation agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, Field

from manim_video_agent import get_skills_dir
from manim_video_agent.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_COMPOSER_SKILLS_DIR = get_skills_dir("manim-composer")


# ---------------------------------------------------------------------------
# Domain models (immutable dataclasses for internal use)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComposerConfig:
    api_key: str = ""
    base_url: str = ""
    research_model: str = ""
    composer_model: str = ""
    timeout: int = 0

    def __post_init__(self):
        settings = get_settings()
        if not self.api_key:
            object.__setattr__(self, "api_key", settings.openrouter_api_key)
        if not self.base_url:
            object.__setattr__(self, "base_url", settings.openrouter_base_url)
        if not self.research_model:
            object.__setattr__(self, "research_model", settings.research_model)
        if not self.composer_model:
            object.__setattr__(self, "composer_model", settings.composer_model)
        if not self.timeout:
            object.__setattr__(self, "timeout", settings.composer_timeout)


@dataclass(frozen=True)
class SkillContext:
    workflow: str
    narrative_patterns: str
    visual_techniques: str
    scene_examples: str
    scenes_template: str


@dataclass(frozen=True)
class ResearchResult:
    summary: str
    key_concepts: str
    aha_moment: str
    common_misconceptions: str
    narrative_hook: str


# ---------------------------------------------------------------------------
# API request/response schemas (Pydantic)
# ---------------------------------------------------------------------------


class VideoLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


LENGTH_LABELS = {
    VideoLength.short: "short (1-3 min)",
    VideoLength.medium: "medium (3-5 min)",
    VideoLength.long: "long (5+ min)",
}


class ComposeRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Video topic or idea")
    audience: str = Field(
        default="general — assumes high school math",
        description="Target audience description",
    )
    length: VideoLength = Field(
        default=VideoLength.medium,
        description="Target video length",
    )
    style: str = Field(
        default="3Blue1Brown — intuition-focused, visual storytelling",
        description="Visual/narration style",
    )
    focus_notes: str = Field(
        default="",
        description="Specific aspects to emphasize or skip",
    )


class ResearchResponse(BaseModel):
    summary: str
    key_concepts: str
    aha_moment: str
    common_misconceptions: str
    narrative_hook: str


class ComposeResponse(BaseModel):
    plan: str
    research: ResearchResponse
    model_used: str
    usage: Optional[dict] = None


# ---------------------------------------------------------------------------
# Skill loader (composer-specific)
# ---------------------------------------------------------------------------


class ComposerSkillLoader:
    def __init__(self, skills_dir: Path):
        self._dir = skills_dir

    def load(self) -> SkillContext:
        self.visual = """# Visual Techniques for Math Animation

Effective visualization patterns for explaining mathematical concepts.

## Core Principles

### 1. Progressive Disclosure
Never show everything at once. Build complexity gradually.

**Bad:** Show complete equation immediately
**Good:** Build equation term by term, explaining each part

```
Scene flow:
1. Show simple case: f(x) = x²
2. Add complexity: f(x) = ax²
3. Full form: f(x) = ax² + bx + c
```

### 2. Transform, Don't Replace
When possible, morph objects into new forms rather than fading out/in.

**Bad:** FadeOut(equation1), FadeIn(equation2)
**Good:** TransformMatchingTex(equation1, equation2)

This maintains visual continuity and shows the relationship between forms.

### 3. Color as Meaning
Use color consistently to encode meaning throughout the video.

**Pattern:**
- Input/given values: BLUE
- Output/results: GREEN
- Key terms being discussed: YELLOW highlight
- Errors/negatives: RED
- Neutral/supporting: WHITE/GREY

### 4. Spatial Relationships
Position encodes relationships:
- Left-to-right: transformation, time, causation
- Top-to-bottom: hierarchy, derivation
- Center: focus of attention
- Periphery: context, reference

---

## Animation Techniques

### Highlighting & Focus

**Indicate** - Brief flash to draw attention
```python
self.play(Indicate(term))
```

**Circumscribe** - Circle around important element
```python
self.play(Circumscribe(equation, color=YELLOW))
```

**FlashAround** - Dramatic attention on revelation
```python
self.play(FlashAround(result))
```

### Equation Manipulation

**Isolate terms** - Color or move specific parts
```python
equation.set_color_by_tex("x", BLUE)
```

**Step-by-step derivation** - Show each algebraic step
```python
step1 = MathTex(r"2x + 4 = 10")
step2 = MathTex(r"2x = 6")
step3 = MathTex(r"x = 3")
# Transform between steps with alignment
```

**Substitution** - Show value being plugged in
```python
# Animate the number moving into the variable's position
```

### Geometric Intuition

**Coordinate systems** - Always label axes
```python
axes = Axes(x_range=[-3, 3], y_range=[-2, 2])
labels = axes.get_axis_labels(x_label="x", y_label="f(x)")
```

**Trace paths** - Show how points move
```python
trace = TracedPath(dot.get_center, stroke_color=YELLOW)
```

**Area visualization** - For integrals, sums
```python
area = axes.get_area(graph, x_range=[a, b], color=BLUE, opacity=0.5)
```

### 3D Techniques

**Camera orbiting** - Reveal 3D structure
```python
self.play(frame.animate.reorient(60, 70), run_time=3)
```

**Projection** - Show 3D object's 2D shadow
```python
# Helps connect 3D intuition to 2D formulas
```

**Slicing** - Cut through 3D objects
```python
# Show cross-sections to understand structure
```

---

## Common Visual Metaphors

### Vectors as Arrows
- Position vectors: arrows from origin
- Addition: tip-to-tail
- Scaling: stretching/shrinking

### Functions as Machines
- Input goes in one side
- Transformation happens
- Output comes out

### Matrices as Transformations
- Show grid being transformed
- Track where basis vectors go
- Emphasize determinant as area scaling

### Derivatives as Slopes
- Tangent line touching curve
- Zoom in to show local linearity
- Animate slope changing as point moves

### Integrals as Accumulation
- Riemann sums with rectangles
- Width → 0 animation
- Area filling under curve

---

## Scene Composition

### The Golden Layout
```
┌─────────────────────────────────┐
│           TITLE/CONTEXT         │  (top edge)
├─────────────────────────────────┤
│                                 │
│      MAIN VISUALIZATION         │  (center, largest area)
│                                 │
├─────────────────────────────────┤
│    EQUATION / FORMULA           │  (bottom third)
└─────────────────────────────────┘
```

### Side-by-Side Comparison
```
┌───────────────┬───────────────┐
│   BEFORE /    │   AFTER /     │
│   CONCEPT A   │   CONCEPT B   │
└───────────────┴───────────────┘
```

### Zoomed Detail
```
┌─────────────────────────────────┐
│  ┌─────┐                        │
│  │ZOOM │ ←── magnified detail   │
│  └─────┘                        │
│         Main context            │
└─────────────────────────────────┘
```

---

## Timing Guidelines

| Action | Typical Duration |
|--------|------------------|
| Simple shape creation | 0.5-1s |
| Text/equation writing | 1-2s |
| Transformation | 1-2s |
| Camera movement | 2-3s |
| Pause for absorption | 0.5-1s |
| Complex animation | 2-4s |

### Rhythm Pattern
Fast-fast-SLOW-fast-fast-SLOW

Quick animations for setup, slow down for key insights.

---

## Color Palettes

### Classic 3b1b
- Background: #1C1C1C (dark grey)
- Primary: #58C4DD (blue)
- Secondary: #83C167 (green)
- Accent: #FFFF00 (yellow)
- Warning: #FF6666 (red)

### High Contrast
- Background: #000000
- Primary: #FFFFFF
- Accent: #FFD700

### Soft Academic
- Background: #2D2D2D
- Primary: #6ECFFF
- Secondary: #98E898
- Accent: #FFE66D"""

        self.scene = """# Scene Examples

Example scene breakdowns from 3b1b-style videos.

---

## Example 1: Explaining the Dot Product

### Scene 1: The Question
**Purpose**: Hook the viewer with the mystery

**Visual Elements**
- Two vectors a and b drawn as arrows
- The dot product formula: a · b = |a||b|cos(θ)
- Question mark animation

**Content**
Open on two vectors. Show the formula. Pose the question: "Why does multiplying components and adding them give you something related to the angle between vectors?"

**Narration Notes**
Tone: curious, slightly puzzled. Emphasize that the formula seems arbitrary.

**Technical Notes**
- Use Arrow for vectors
- MathTex for formula
- Indicate() on the cos(θ) term

---

### Scene 2: Geometric Interpretation
**Purpose**: Show projection interpretation

**Visual Elements**
- Vector a (horizontal, blue)
- Vector b (angled, green)
- Projection of b onto a (dashed line)
- Right angle marker
- Length labels

**Content**
Show that a · b equals |a| times the projection of b onto a. Animate the projection dropping down. Show this equals |a||b|cos(θ) geometrically.

**Narration Notes**
"The dot product measures how much one vector goes in the direction of another."

**Technical Notes**
- DashedLine for projection
- RightAngle mobject
- animate.rotate() for showing different angles

---

### Scene 3: Numeric Connection
**Purpose**: Connect geometry to algebra

**Visual Elements**
- Coordinate grid
- Vector a = [a₁, a₂]
- Vector b = [b₁, b₂]
- Components highlighted

**Content**
Show vectors on grid with components labeled. Demonstrate why a₁b₁ + a₂b₂ equals the geometric interpretation. Use specific numbers.

**Narration Notes**
Walk through calculation slowly. "Let's see why the algebra matches the geometry."

**Technical Notes**
- NumberPlane or Axes
- Brace for component labels
- TransformMatchingTex for equation steps

---

## Example 2: Introduction to Fourier Series

### Scene 1: The Hook
**Purpose**: Show the surprising result

**Visual Elements**
- A square wave (sharp corners)
- Sum of smooth sine waves
- Morphing animation between them

**Content**
"You can build a square wave—something with sharp corners—from perfectly smooth sine waves." Show the result first, then promise to explain how.

**Narration Notes**
Tone: wonder, slight disbelief. This should feel surprising.

**Technical Notes**
- ParametricFunction for waves
- Transform animation for the morph
- Consider showing 1, 3, 5 terms building up

---

### Scene 2: Building Blocks
**Purpose**: Introduce sine waves as basis

**Visual Elements**
- Single sine wave
- Frequency visualization (faster oscillation)
- Amplitude visualization (taller/shorter)
- Phase visualization (shifting left/right)

**Content**
Introduce the three parameters: frequency, amplitude, phase. Show each one separately, then combine.

**Narration Notes**
Go slow. "A sine wave has three knobs we can adjust..."

**Technical Notes**
- ValueTracker for animating parameters
- Updaters to make wave respond to trackers
- Labels for each parameter

---

### Scene 3: Superposition
**Purpose**: Show waves can be added

**Visual Elements**
- Two sine waves (different colors)
- Their sum (third color)
- Point-by-point addition visualization

**Content**
Show that adding waves means adding their heights at each point. Demonstrate with two specific frequencies combining.

**Narration Notes**
"Adding waves is simple—at each point, just add the heights."

**Technical Notes**
- VGroup of three function graphs
- Vertical lines showing addition at specific x values
- Animate the addition happening

---

## Example 3: Matrix as Linear Transformation

### Scene 1: Grid Transformation
**Purpose**: Visual foundation

**Visual Elements**
- 2D coordinate grid (NumberPlane)
- Basis vectors i-hat and j-hat (colored arrows)
- Grid lines transforming

**Content**
Show a grid. Highlight i-hat (1,0) and j-hat (0,1). Apply a transformation—watch the entire grid move while tracking where basis vectors land.

**Narration Notes**
"Watch what happens to the grid when we apply this transformation. Notice how every point moves."

**Technical Notes**
- NumberPlane with visible grid lines
- apply_matrix() method
- Keep basis vectors visually distinct

---

### Scene 2: Basis Vectors Determine Everything
**Purpose**: Key insight

**Visual Elements**
- Transformed i-hat and j-hat
- Arbitrary vector v as combination
- v = xi + yj visualization

**Content**
Show that knowing where i-hat and j-hat land tells you where ANY vector lands. Because v = xi + yj, the transformed v = x(new i) + y(new j).

**Narration Notes**
"Here's the key insight..." Build anticipation before the reveal.

**Technical Notes**
- Vector addition animation (tip-to-tail)
- Scaling animation for coefficients
- TransformMatchingShapes for the combination

---

## Scene Transition Patterns

### Zoom Focus
```
Full scene → Zoom into detail → Explain → Zoom out
```

### Side-by-Side Build
```
Empty left | Empty right
Add to left | Compare
Add to right | Connect them
```

### Transform Chain
```
Object A → Transform → Object B → Transform → Object C
(Maintain visual continuity throughout)
```

### Reset and Rebuild
```
Complex scene → Clear/fade most → Focus on one element → Build new complexity
```"""
        return SkillContext(
            workflow=self._read("SKILL.md"),
            narrative_patterns=self._read("references/narrative-patterns.md"),
            visual_techniques=self.visual,
            scene_examples=self.scene,
            scenes_template=self._read("templates/scenes-template.md"),
        )

    def _read(self, relative_path: str) -> str:
        path = self._dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")
        return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# OpenRouter API client (composer-specific thin wrapper)
# ---------------------------------------------------------------------------


class OpenRouterAPI:
    def __init__(self, config: ComposerConfig):
        self._config = config

    def chat(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.6,
    ) -> dict:
        selected_model = model or self._config.composer_model

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/manim-generator",
            "X-Title": "Manim Video Composer",
        }

        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }

        response = requests.post(
            self._config.base_url,
            headers=headers,
            json=payload,
            timeout=self._config.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API failed ({response.status_code}): {response.text}"
            )

        return response.json()

    def extract_content(self, response: dict) -> str:
        try:
            content = response["choices"][0]["message"].get("content")
            if not content:
                raise RuntimeError(
                    "Composer LLM returned empty/null content — model may "
                    "have exhausted output tokens"
                )
            return content
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse API response: {e}")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class PromptBuilder:
    def __init__(self, skills: SkillContext):
        self._skills = skills

    def research_system(self) -> str:
        return """\
<role>
You are a research assistant preparing material for a 3Blue1Brown-style \
math/science explainer video. Your job is to deeply understand a topic and \
identify what makes it fascinating, surprising, and teachable.
</role>

<output_format>
Respond in this exact structure (use these headers):

## Summary
[2-3 paragraph overview of the core concept, written for someone planning a video]

## Key Concepts
[Bulleted list of the essential ideas that must be covered]

## Aha Moment
[The single most important insight — the moment the viewer should think "oh wow"]

## Common Misconceptions
[What do people usually get wrong about this topic?]

## Narrative Hook
[A compelling opening question or mystery that makes viewers curious]
</output_format>

<guidelines>
- Focus on visual/geometric intuition over algebraic manipulation
- Identify what makes this topic *surprising* or *beautiful*
- Think about what a 3Blue1Brown video would emphasize
- Consider multiple representations (algebraic, geometric, physical analogy)
- Note any real-world applications that ground the abstraction
</guidelines>"""

    def research_user(self, req: ComposeRequest) -> str:
        parts = [
            f"<topic>{req.topic}</topic>",
            f"<audience>{req.audience}</audience>",
            f"<target_length>{LENGTH_LABELS[req.length]}</target_length>",
        ]
        if req.focus_notes:
            parts.append(f"<focus_notes>{req.focus_notes}</focus_notes>")

        return "\n".join(parts) + (
            "\n\nResearch this topic thoroughly. Identify the key insights, "
            "the narrative hook, and what makes it visually interesting for an "
            "educational animation."
        )

    def composer_system(self) -> str:
        return f"""\
You are a video composition planner for 3Blue1Brown-style educational \
animations. You output a scenes.md file that a code generation agent \
will implement. Focus on storytelling, not code.

{self._skills.narrative_patterns}

Write WHAT to show, not HOW to code it. Describe visuals in plain language.

Rules:
- Every scene must have: Duration, Purpose, Visual Elements, Content, Narration Notes
- Color palette must be defined and used consistently
- Total video: 2-5 minutes. Each scene: 15-60 seconds (aim for ~30 seconds).
- Progressive disclosure — build complexity gradually
- Include specific LaTeX equations where relevant
- Describe animations in natural language (fade in, morph, highlight, slide)
- Output ONLY the scenes.md content — no preamble, no commentary"""

    def scene_composer_system(self) -> str:
        """System prompt for per-scene composition with SCENE_BREAK delimiters."""
        return f"""\
You are a video composition planner for 3Blue1Brown-style educational \
animations. You design the storytelling, pacing, and visual narrative — \
a separate code generation agent handles all implementation details.

{self._skills.narrative_patterns}

Write WHAT to show, not HOW to code it. Describe visuals in plain language \
(e.g. "fade in the equation", "draw an arrow between A and B", "morph the \
circle into a square"). Never reference specific library classes or methods.

Your output MUST follow this exact structure:

## Shared Context
```json
{{
    "background_color": "#1e1e2e",
    "color_palette": {{
        "positive": "#a6e3a1",
        "negative": "#f38ba8",
        "neutral": "#89dceb",
        "emphasis": "#f9e2af"
    }},
    "topic_title": "The Topic Title",
    "total_scenes": 3
}}
```

---SCENE_BREAK---

## Scene 1: Title
**Duration:** 30-45 seconds
**Purpose:** Hook the viewer
**Visual Elements:** [what appears on screen — shapes, text, equations, graphs]
**Content:** [what the scene shows and explains step by step]
**Narration Notes:** [tone, key phrases, emotional beat]

---SCENE_BREAK---

## Scene 2: Next Scene
...

Rules:
- Each scene is SELF-CONTAINED — no references to elements from other scenes
- All scenes share the visual identity from Shared Context
- Total video: 2-5 minutes. Each scene: 15-60 seconds (aim for ~30 seconds).
- Progressive disclosure — build complexity gradually
- Include specific LaTeX equations (e.g. $E = mc^2$) where relevant
- Describe animations in natural language (fade in, slide left, highlight, morph)
- Scenes separated by exactly ---SCENE_BREAK--- on its own line
- Shared Context JSON block MUST appear before the first scene break
- Prefer 2D visuals over 3D — only suggest 3D if the concept genuinely requires it
- Keep visual designs simple: text, boxes, bullet points, arrows, and basic shapes over complex geometry
- Be mindful of rendering complexity: avoid scenes with massive numbers of animated objects
- Output ONLY the structured plan — no preamble, no commentary"""

    def composer_user(self, req: ComposeRequest, research: ResearchResult) -> str:
        return f"""\
<request>
<topic>{req.topic}</topic>
<audience>{req.audience}</audience>
<target_length>{LENGTH_LABELS[req.length]}</target_length>
<style>{req.style}</style>
{f"<focus_notes>{req.focus_notes}</focus_notes>" if req.focus_notes else ""}
</request>

<research>
## Summary
{research.summary}

## Key Concepts
{research.key_concepts}

## Aha Moment
{research.aha_moment}

## Common Misconceptions
{research.common_misconceptions}

## Narrative Hook
{research.narrative_hook}
</research>

Create a detailed scenes.md plan for this video. Follow the template exactly. \
Be specific enough that a Manim developer can implement each scene without \
guessing your intent."""


# ---------------------------------------------------------------------------
# Research parser
# ---------------------------------------------------------------------------


def parse_research(raw: str) -> ResearchResult:
    sections: dict[str, str] = {
        "summary": "",
        "key_concepts": "",
        "aha_moment": "",
        "common_misconceptions": "",
        "narrative_hook": "",
    }

    header_map = {
        "## Summary": "summary",
        "## Key Concepts": "key_concepts",
        "## Aha Moment": "aha_moment",
        "## Common Misconceptions": "common_misconceptions",
        "## Narrative Hook": "narrative_hook",
    }

    current_key: Optional[str] = None
    current_lines: list[str] = []

    for line in raw.split("\n"):
        stripped = line.strip()
        matched_key = next(
            (key for header, key in header_map.items() if stripped.startswith(header)),
            None,
        )

        if matched_key is not None:
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = matched_key
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    return ResearchResult(
        summary=sections["summary"] or raw[:500],
        key_concepts=sections["key_concepts"] or "See summary",
        aha_moment=sections["aha_moment"] or "To be determined during composition",
        common_misconceptions=sections["common_misconceptions"] or "None identified",
        narrative_hook=sections["narrative_hook"] or "To be determined",
    )


# ---------------------------------------------------------------------------
# Video composer (orchestrator)
# ---------------------------------------------------------------------------


class VideoComposer:
    def __init__(
        self,
        config: ComposerConfig,
        skills_dir: Path = _DEFAULT_COMPOSER_SKILLS_DIR,
    ):
        self._api = OpenRouterAPI(config)
        self._config = config
        self._skills = ComposerSkillLoader(skills_dir).load()
        self._prompts = PromptBuilder(self._skills)

    def compose(self, req: ComposeRequest) -> ComposeResponse:
        logger.info("Phase 1 — researching: %s", req.topic)
        research = self._research(req)
        logger.info("Phase 1 done. Hook: %s", research.narrative_hook[:80])

        logger.info("Phase 2 — composing scene plan")
        plan, usage = self._compose_plan(req, research)
        logger.info("Phase 2 done. Plan length: %d chars", len(plan))

        return ComposeResponse(
            plan=plan,
            research=ResearchResponse(
                summary=research.summary,
                key_concepts=research.key_concepts,
                aha_moment=research.aha_moment,
                common_misconceptions=research.common_misconceptions,
                narrative_hook=research.narrative_hook,
            ),
            model_used=self._config.composer_model,
            usage=usage,
        )

    def _research(self, req: ComposeRequest) -> ResearchResult:
        response = self._api.chat(
            system=self._prompts.research_system(),
            user=self._prompts.research_user(req),
            model=self._config.research_model,
            temperature=0.5,
        )
        raw = self._api.extract_content(response)
        return parse_research(raw)

    def compose_scenes(self, req: ComposeRequest) -> "CompositionResult":
        """Compose a scene plan with per-scene breakdowns for parallel generation.

        Returns a :class:`CompositionResult` with individual :class:`ScenePlan`
        items and a :class:`SharedContext` that can be injected into each
        per-scene code generation prompt.
        """
        from manim_video_agent.core.scene_plan import (
            CompositionResult,
            parse_composition,
        )

        logger.info("Phase 1 — researching: %s", req.topic)
        research = self._research(req)
        logger.info("Phase 1 done. Hook: %s", research.narrative_hook[:80])

        logger.info("Phase 2 — composing per-scene plan")
        response = self._api.chat(
            system=self._prompts.scene_composer_system(),
            user=self._prompts.composer_user(req, research),
            model=self._config.composer_model,
            temperature=0.6,
        )
        raw_plan = self._api.extract_content(response)
        logger.info("Phase 2 done. Raw plan length: %d chars", len(raw_plan))

        plans, shared_context = parse_composition(raw_plan)
        logger.info(
            "Parsed %d scene plans, shared context: %s",
            len(plans),
            shared_context.topic_title,
        )

        return CompositionResult(
            plans=plans,
            shared_context=shared_context,
            research=research,
        )

    def _compose_plan(
        self, req: ComposeRequest, research: ResearchResult
    ) -> tuple[str, Optional[dict]]:
        response = self._api.chat(
            system=self._prompts.composer_system(),
            user=self._prompts.composer_user(req, research),
            model=self._config.composer_model,
            temperature=0.6,
        )
        plan = self._api.extract_content(response)
        usage = response.get("usage")
        return plan, usage


def get_api_key() -> str:
    """Get OpenRouter API key from settings."""
    key = get_settings().openrouter_api_key
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Add it to .env or export OPENROUTER_API_KEY='your-key-here'"
        )
    return key

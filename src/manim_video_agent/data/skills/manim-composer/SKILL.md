---
name: manim-composer
description: |
  Trigger when: (1) User wants to create an educational/explainer video, (2) User has a vague concept they want visualized, (3) User mentions "3b1b style" or "explain like 3Blue1Brown", (4) User wants to plan a Manim video or animation sequence, (5) User asks to "compose" or "plan" a math/science visualization.

  Transforms vague video ideas into detailed scene-by-scene plans (scenes.md). Conducts research, asks clarifying questions about audience/scope/focus, and outputs comprehensive scene specifications ready for implementation with ManimCE or ManimGL.

  Use this BEFORE writing any Manim code. This skill plans the video; use manimce-best-practices or manimgl-best-practices for implementation.
---

## Workflow

### Phase 1: Understand the Concept

1. **Research the topic** deeply before asking questions
   - Use web search to understand the core concepts
   - Identify the key insights that make this topic interesting
   - Find the "aha moment" - what makes this click for learners
   - Note common misconceptions to address

2. **Identify the narrative hook**
   - What question does this video answer?
   - Why should the viewer care?
   - What's the surprising or counterintuitive element?

### Phase 2: Clarify with User

Ask targeted questions (not all at once - adapt based on responses):

**Audience & Scope**
- What math/science background should I assume? (e.g., "knows calculus" or "high school algebra")
- Target video length? (short: 5-10min, medium: 15-20min, long: 30min+)
- Should this be self-contained or part of a series?

**Focus & Depth**
- Any specific aspects to emphasize or skip?
- Proof-heavy or intuition-focused?
- Real-world applications to include?

**Style Preferences**
- Color scheme preferences?
- Narration style? (casual, formal, playful)
- Any specific visual metaphors you have in mind?

### Phase 3: Create scenes.md

Output a comprehensive `scenes.md` file with this structure:

```markdown
# [Video Title]

## Overview
- **Topic**: [Core concept]
- **Hook**: [Opening question/mystery]
- **Target Audience**: [Prerequisites]
- **Estimated Length**: [X minutes]
- **Key Insight**: [The "aha moment"]

## Narrative Arc
[2-3 sentences describing the journey from confusion to understanding]

---

## Scene 1: [Scene Name]
**Duration**: ~X seconds
**Purpose**: [What this scene accomplishes]

### Visual Elements
- [List of mobjects needed]
- [Animations to use]
- [Camera movements]

### Content
[Detailed description of what happens, what's shown, what's explained]

### Narration Notes
[Key points to convey, tone, pacing notes]

### Technical Notes
- [Specific Manim classes/methods to use]
- [Any tricky implementations to note]

---

## Scene 2: [Scene Name]
...

---

## Transitions & Flow
[Notes on how scenes connect, recurring visual motifs]

## Color Palette
- Primary: [color] - used for [purpose]
- Secondary: [color] - used for [purpose]
- Accent: [color] - used for [purpose]
- Background: [color]

## Mathematical Content
[List of equations, formulas, or mathematical objects that need to be rendered]

## Implementation Order
[Suggested order for implementing scenes, noting dependencies]
```

## Complexity Guidelines

### Keep It Simple — Always

When planning scenes, **default to the simplest visual that communicates the idea**. Prefer:

- Basic geometric shapes: circles, rectangles, lines, arrows, dots
- Plain text and `MathTex` labels
- Standard animations: `Write`, `FadeIn`, `FadeOut`, `Transform`, `Indicate`, `MoveToTarget`
- Simple color changes and highlights

**Avoid designing scenes that require:**
- Multi-component custom diagrams with many interacting parts
- Highly domain-specific structures that need significant custom rendering code
- Deeply nested or layered visualizations where individual pieces must be manually positioned and connected
- Animations that depend on simulating real-world system behavior rather than illustrating a concept with an analogy

**The test:** Could this scene be described in 2–3 Manim primitives? If yes, keep it that way. If a visual idea requires 10+ objects all carefully coordinated, simplify — use a text description, a labeled diagram, or an abstract representation instead.

### Prefer Abstraction Over Realism

When illustrating a complex system or process, use **abstract stand-ins**:
- A labeled box or circle instead of a detailed structure
- An arrow with a text label instead of a detailed process diagram
- A simple flow of shapes to show data or information moving
- Color and motion to suggest relationships, not detailed structural recreation

The goal is comprehension, not technical accuracy of the diagram.

---

## 3b1b Style Principles

Apply these principles when composing scenes:

### Visual Storytelling
- **Show, don't just tell** - Every concept needs a visual representation
- **Progressive revelation** - Build complexity gradually, don't show everything at once
- **Visual continuity** - Transform objects rather than replacing them when possible

### Pacing & Rhythm
- **Pause for insight** - Give viewers time to absorb key moments
- **Vary the pace** - Mix quick sequences with slower explanations
- **End scenes with resolution** - Each scene should feel complete

### Mathematical Beauty
- **Emphasize elegance** - Highlight when math is surprisingly simple or beautiful
- **Connect representations** - Show the same concept multiple ways (algebraic, geometric, intuitive)
- **Embrace abstraction gradually** - Start concrete, then generalize

### Engagement Techniques
- **Pose questions** - Make viewers curious before revealing answers
- **Acknowledge difficulty** - "This might seem confusing at first..."
- **Celebrate insight** - Make the "aha moment" feel earned

## References

- [references/narrative-patterns.md](references/narrative-patterns.md) - Common 3b1b narrative structures
- [references/visual-techniques.md](references/visual-techniques.md) - Effective visualization patterns
- [references/scene-examples.md](references/scene-examples.md) - Example scenes.md excerpts

## Templates

- [templates/scenes-template.md](templates/scenes-template.md) - Blank scenes.md template

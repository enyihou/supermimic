# SimulationStage Specification

## Overview

- **Target file:** `src/components/SimulationStage.tsx`
- **Reference:** [official simulation](https://supermimic.github.io/demo/) (not mirrored)
- **Interaction model:** time-driven simulation with pointer drag and orbit camera

## DOM and scene structure

- Fullscreen `<canvas>` behind all UI.
- Stage background is `radial-gradient(120% 90% at 50% 40%, #fbfbfd 0%, #f5f5f7 60%, #efeff2 100%)`.
- Floor is a large horizontal plane with a fading 1-unit grid and a radial shadow below the robot.
- Robot uses the locally hosted official `robot.json` and `robot.bin` geometry when available. The fallback renderer is a simple articulated G1-like rig built from grouped capsules and boxes.

## Computed stage styles

- `html, body`: height 100%, overflow hidden, color `#1d1d1f`, font `Inter`, anti-aliased.
- `canvas`: display block, cursor grab, touch-action none. Active cursor is grabbing.
- Renderer pixel ratio is clamped to 2. Camera is perspective 40 degrees and starts at `(2.8, 1.5, 2.8)` looking at the robot.
- Desktop robot is centered around 50% viewport width and roughly 56% viewport height. Mobile framing places the robot lower and closer.

## States and behavior

- Play/pause changes animation ticking and swaps a 14px pause or play SVG.
- Reset returns the step counter to zero and resets pose.
- Goal adds a translucent goal-pose rig.
- MuJoCo adds a solid actor and sets the legend body class to `sim-on`.
- Pointer drag selects a robot body, draws a thin target line, and clears the drag hint on first use.
- A worker-like local animation loop advances the active pose, increments the step counter, and updates the shadow position.

## Responsive behavior

- Desktop uses the original camera framing and 1024px shadow-quality assumption.
- Mobile uses a tighter camera and 390px scene framing.

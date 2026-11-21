# SuperMimic demo behavior bible

## Interaction model

The page is a fullscreen, fixed-height simulation. The canvas consumes the entire viewport and the UI overlays it with click-through panels. The robot is driven by a continuously ticking model loop. UI actions are direct controls and keyboard shortcuts. The canvas is draggable and orbitable.

## Controls

- World model is always active and shown as the selected legend segment.
- Goal toggles a translucent goal-pose overlay. Click or `g` toggles the `.off` state and the overlay visibility.
- MuJoCo toggles a solid real-physics actor. Click or `s` starts a pending state, then resolves to active when the worker responds. The legend gains a blue world-model dot while MuJoCo is active.
- Reset or `r` resets the simulation and step counter.
- Play/pause or Space sends resume/pause to the worker and swaps the pause/play icon.
- Policy toggles on/off or `p`, updating the switch track and worker policy flag.
- Action noise slider ranges from 0 to 0.25 in 0.01 increments. Its readout is fixed to two decimals.
- Friction slider is disabled until model boot completes, then ranges from 0 to 100 percent in integer steps.
- Motions opens a searchable motion sheet. Category headings expand/collapse their item list. Clicking a category shuffles that category. Clicking an item selects the motion and updates the dock label. Escape closes the sheet.
- On screens at or below 900px, Controls opens a sheet above the dock. Motions remains a separate sheet. Only one sheet is open at a time.
- Pointer drag on the robot sends a body target to the worker. The first drag dismisses the drag hint. Orbit camera controls remain active when the pointer is not dragging a body.

## Timed states

- A boot card is shown until the worker sends the first state. It contains a glass dense card, a monospace status label, and a 4px progress bar.
- After the first state, a headline splash appears after about 500ms for 5 seconds with `Your browser is dreaming this robot.` and the explanatory subcopy. The surrounding panels fade to 18% while the headline owns the screen.
- A sequence of hints cycles through drag hint, Goal, and MuJoCo every 5 seconds. Each hint uses a white chip and diamond pointer.

## Responsive behavior

- Desktop at 1440px uses a 52px bottom dock with reset, pause, inline controls, and a 260px motions button.
- Tablet and mobile at `max-width: 900px` use a 48px dock. Controls move into a fixed glass sheet above the dock. The motions list also becomes a fixed sheet.
- At `max-width: 600px`, the edge inset is 12px, compact legend buttons are used, the robot camera is tighter, and sheet max height grows to 70dvh in embedded mode.
- Pointer-coarse devices use 44px icon and row hit targets and 16px motion filter text.

## Visual states

- Glass panels use `rgba(255,255,255,.72)` and `backdrop-filter: saturate(180%) blur(24px)` with a faint hairline and soft lift shadow.
- Active buttons use `#1d1d1f` background and white foreground. Inactive toggles use transparent background and `#6e6e73` text.
- The floor grid is a fading shader surface. The robot shadow is a soft radial texture under the pelvis.

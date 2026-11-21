# Findings: SuperMimic × higher-order supervision

## SuperMimic demo rebuild reconnaissance

- Target is a fullscreen WebGL demo titled `WM Imagination` with a pale cyclorama, radial gradient, fading floor grid, soft shadow, and a centered Unitree G1 robot.
- Published demo uses Inter Tight for display labels, Inter for UI text, and JetBrains Mono for the step counter and loading labels.
- Primary palette is `#ffffff`, `#f5f5f7`, `#e8e8ed`, `#d2d2d7`, `#1d1d1f`, and `#6e6e73`, with translucent white glass using `rgba(255,255,255,.72)` and `backdrop-filter: saturate(180%) blur(24px)`.
- Desktop chrome is top-left legend, top-right SuperMimic link, and a bottom-center 52px glass dock. Mobile uses a 48px dock and turns controls and motions into sheets above the dock at `max-width: 900px`.
- Interaction model is click/keyboard driven for goal pose, MuJoCo, policy, reset, play/pause, sliders, motion sheet, folder expansion, filtering, and motion selection. The robot canvas supports pointer drag and orbit camera gestures.
- Keyboard shortcuts are Space for play/pause, `r` reset, `g` goal pose, `s` MuJoCo, and `p` policy. Escape closes sheets.
- Motion categories and labels are available in `https://supermimic.github.io/demo/constants.json`. The published world-model assets are `weights.bin` (33,014,908 bytes), `robot.bin` (7,079,544 bytes), `robot.json` (10,851 bytes), `engine.wasm` (140,028 bytes), and `engine.cfg` (14,104 bytes), plus `scene.xml` and per-motion binary files.
- Playwright Chromium captured reference screenshots at 1440×900 and 390×844. The 1440px frame shows a standing G1 in the center and a 974px-wide dock. The 390px frame shows the robot lower in the viewport and a 331px-wide dock with reset, pause, controls, divider, motions.
- For pixel fidelity, the finished app serves a local copy of the published demo runtime under `public/official-demo/` and embeds it at the default route. The independent React/Three implementation remains available at `/?native=1` and is used as a local fallback/reference implementation.

- SuperMimic alternates simulator collection, world-model fitting, and actor updates through a finite imagination horizon. Its benchmark is centered on tracking filtered training motions, so held-out disturbance recovery and continuous tracking metrics are required for a stronger claim.
- Higher-Order Action Supervision uses fixed logged action differences and JVP targets. It does not validate a self-targeted Taylor remainder, Hessian training, or online model-based reinforcement learning.
- A Taylor residual measures directional finite-step curvature. It does not directly measure action smoothness, feedback gain, or stability. An affine actor can make the residual zero, and nonlinear residuals can cancel along a chord.
- The strongest initial experiment is frozen-teacher distillation with matched action-only, action-plus-action-change, and tuned smoothing controls. Evaluate nominal tracking, pushed recovery, contact groups, clipping, and compute-matched performance.
- The selected independent reproduction actor loads safely and produces finite 29-dimensional actions from ten history frames and future reference features. A short CUDA smoke on one AIR A800 completed in 73.7 seconds including simulator compilation. Two nominal and two of four disturbed cases met the early 0.30 m recovery gate.
- Full training remains pending approved BONES-SEED access and an authorized writable AIR archive path. The G1 dataset is gated and approximately 23.5 GB compressed.

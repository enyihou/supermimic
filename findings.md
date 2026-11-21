# Findings: SuperMimic × higher-order supervision

## Overnight run planning audit

- The current official launcher pins SuperMimic to `5fb92297ab4fe3d86629d9f0c635334791bb0d5f` and mjlab research to `379c55e88b1dec73f93943e524a879b2ebbf45c5`.
- The intended official configuration is one GPU, 4,096 environments, 320,000 iterations, and a three-day Slurm time limit. Its actual duration is unknown until a timed benchmark runs on the target GPU.
- BONES-SEED is gated and is not present in this workspace. The current project has no completed official checkpoint, Slurm job record, or full-run logs.
- The current backup loop overwrites `latest.tgz`, suppresses archive errors, has no atomic temporary publish, and does not emit the runbook's `SUCCESS` and `job-status.txt` completion evidence. It needs hard failure handling before unattended use.
- GitHub's public `enyihou/supermimic` repository is suitable for code, pinned commit metadata, checksums, audit JSON, and sanitized graphs. It is not a safe destination for the gated dataset, credentials, or private model checkpoints.
- A private artifact root must retain the input checksum, source archive, environment metadata, logs, checkpoints, evaluation outputs, and a run manifest. A second local copy should be retained only after checking `/home` and `/public` quota headroom.
- The default website embeds the published browser runtime. A PyTorch training checkpoint cannot be substituted into that iframe without a verified export or a server-side inference service. The trained-model website milestone therefore depends on defining the inference boundary.
- The higher-order Taylor plan contains multiple phases, matched baselines, held-out recovery evaluation, and potentially multiple seeds. It should be scheduled after the official reproduction and treated as a separate experiment series rather than assumed to finish in the same overnight run.

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

## BONES-SEED access

- BONES-SEED is the Bones Studio motion-capture dataset used to supply diverse human motion trajectories for humanoid robotics. Its dataset card lists 142,220 annotated motions, including Unitree G1 MuJoCo-compatible CSV trajectories and metadata.
- The Hugging Face dataset is access-controlled. A user must sign in, accept the dataset conditions and license, and then download the archive. The current official launcher only needs the private `g1.tar.gz` archive containing the `g1/` motion tree.
- The dataset archive must stay in private storage. The public GitHub repository should contain only the download instructions, source revision, archive checksum, filtered-clip manifest, audit summaries, and graphs.
- Credential separation is feasible and preferable. The user can authenticate and download on the local SSH server, transfer only `g1.tar.gz` to gpu01, verify the pre- and post-transfer SHA-256 values, and keep the Hugging Face token off AIR. Training still reads from node-local scratch after Slurm stages the archive.
- The user's access topology is phone -> local SSH server -> AIR. The local SSH server is the download host when it has outbound HTTPS access. The AIR host has no outbound Hugging Face access.
- A live upload or network-mounted dataset during training would add an avoidable failure mode. The archive should be fully staged before the job is submitted.
- AIR verification shows `/home` and `/public` on gpu01 are local XFS filesystems suitable for staging the compressed input before submission. The personal `/data/houey` path is absent, so checkpoint backups need an approved existing project path under `/data`.

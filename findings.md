# Findings: SuperMimic × higher-order supervision

## Active reproduction evidence — 2026-11-21

- `source` starts at `1d20a8a`, matching `origin/source`, with no tracked worktree changes. Local root has about 33 GB available.
- Re-read the current AIR rules and runbook: `train` has a three-day maximum, GPU-class jobs may be cancelled for sustained utilization below 40%, and `/home` and `/public` each have a separate 200 GB per-user cap. Job data, caches and training run on allocated node-local scratch; submit only through `lab-submit` from gpu01.
- The user has authorized the owner-only `/public/houey/supermimic-archive` layout and a fully browser-based final simulation. These supersede older notes about an undecided archive location and possible server inference.
- The local Hugging Face login and gated archive access both passed without printing the credential. Download started in local tmux, pinned to `bones-studio/seed@2f59b2077b9da34dd4e43618e705c7cb962c9a66` (23,499,973,647 compressed bytes).
- AIR's gpu01 has adequate login-node quota and owner-only archive directories. A Slurm allocation on gpu02 (33617) could not see either gpu01's `/home` or `/public`. No `/data/houey` exists and the user cannot create a child of `/data`; scheduler input/output requires an AIR-approved compute-visible staging/backup path.
- Pinned mjlab expects packed motion or converted NPZ; the previous raw CSV glob cannot run. The current pins do not ship the exact Appendix E clip list. Document the actual conversion and selection and preserve a separate out-of-dataset held-out set.
- The existing mirrored reference demo's [license](https://supermimic.github.io/demo/LICENSE.html) forbids redistribution and modification. Public builds should link to the official deployment and exclude its copied runtime, model and motion assets. The official research source is separately Apache-2.0.
- The existing graphify folder contains a cache but no usable graph; source pins and source files are the authoritative inputs for this task.
- Download completed locally with exit 0 and exactly 23,499,973,647 bytes. Authenticated Hub file metadata supplied SHA-256 `52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`. Detached processes are hashing before transfer and streaming the archive inventory.
- Pinned upstream lock uses PyTorch 2.15/CUDA 13.2 and Warp 1.19/CUDA 13.x. All three train nodes report driver 560.35.03 / CUDA 12.6. The underlying mjlab pin explicitly supports its CUDA 12.6 extra; construct and separately identify the candidate runtime with Torch 2.14.1+cu126 and NVIDIA's same-version Warp 1.19 +cu12 build. Require official smoke validation before using it.
- Verified BONES archive first CSV shape matches the 120 Hz NVIDIA-documented layout: frame, XYZ translation in cm, extrinsic XYZ Euler degrees, 29 named joints in degrees. The pinned loader requires 50 Hz root pose/twist and joint arrays in its exact MJCF joint order.
- Full local inventory completed: 142,220 safe regular motion members in 124 date directories (51,527,587,825 expanded bytes). The exact local SHA-256 matched authenticated Hub metadata, and the resumable AIR transfer started.
- Pages-base desktop 1440×900 and mobile 390×844 browser tests passed progress, official outbound link, explicit missing-model state, and navigation checks. The trained worker separately passed synthetic ONNX parity and on-device physics; neither check is evidence of trained-policy quality.
- Shipped public-safe source `50db668` and Pages `b256080`. Pages reported built; both sizes passed live navigation and old mirrored artifact endpoints returned 404. Exact pinned G1 on-device simulation, seeded synthetic actor parity, pause/reset/motion/play, and corrupted-file rejection also passed locally. No test policy or dataset material was shipped.
- AIR resource-only admission test accepted the full preset. Runtime v3 installed 129 compatible packages (Python 3.13.14, Torch CUDA 12.6, MuJoCo 4.0.1, Warp 1.19); conda-pack was confirmed live. Archive-based CUDA verification remains an unrun smoke requirement.
- Runtime v3 archive publication finished and both hashes passed verification on gpu01. The exact pinned imports and expected CLI options also passed on CPU. BONES conversion passed against eight real motion samples; a separate detached workflow waits for final verified AIR transfer before staging 64 clips for the official smoke.
- Source commits `088e1f1` and `570cacb` add a private approximate Appendix E selector and detached Slurm terminal watcher. Data selection is explicitly approximate because the authors did not publish the complete list of clip IDs or all thresholds in v3.
- Rechecked AIR guide and runbook after resumption. No GPU jobs in the account queue; `/data/houey` still absent and `/data` not writable. The documented round-trip explicitly requires an owned Ceph tree. No approved alternative staging location was found.
- The relocated conda environment works on gpu01 when `conda-unpack` is invoked with its explicit Python interpreter. CPU imports confirmed PyTorch `2.14.1+cu126`, MuJoCo `4.0.1`, and Warp `1.19.0`; CUDA execution is still untested.
- No exact Appendix E clip manifest found in either pinned author repository, their releases, the arXiv v3 source archive, the author project page, or the pinned publisher tree.
- The official-smoke resource check also passed, with no allocation. The phase audits now record and cross-check the private selection digest and exact-versus-approximate cohort designation. Local tests cover both modes and verification failure paths. The Pages build now checks training/eval identity and runs browser-runtime artifact checks before any nonempty model release.
- The CoRL submission is OpenReview `psNqjklVOK`; the forum shows a verification challenge and both unauthenticated notes endpoints returned HTTP 403. Its supplementary attachments could not be verified. Do not infer from this that no attachment exists.
- The upload destination remained a private partial at 19:53 UTC. Both the deployment review and the local gate tests confirmed that the checked-in status must be regenerated from phase evidence before deploying the new gate.
- `training/stage_g1.sh` had a here-document/case-branch syntax edge that ran the remote checksum and atomic rename successfully but returned shell status 2 afterward. The destination was independently verified final with the pinned digest. The post-copy remote command is now isolated in a function and `bash -n` passes. An idempotent rerun still performs the intended full remote hash, so it may take several minutes on the 23.5 GB archive.

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
- The first SSH server has a 145 GB root filesystem with about 33 GB free and about 51 GB used under `/home/ubuntu`. It can hold the roughly 23.5 GB compressed input temporarily, but leaves limited headroom for transfer staging and should not be the checkpoint archive.
- gpu01 currently reports about 16 GB used in `/home/houey`. The server guide authorizes a 200 GB per-user working limit for `/home` and `/public`, so the compressed input and a bounded set of checkpoints fit there after a quota check. `$LOCAL_SCRATCH` remains job-temporary storage.
- Created the owner-only interim archive directory `/public/houey/supermimic-archive` on gpu01. It is empty and writable. This is suitable for staged inputs, compressed checkpoints, logs, and audit outputs within the personal working limit while a durable `/data` project path is unavailable.

## Checkpoint artifact sizing

- The official runner writes `world_model.pt`, `actor.pt`, and an optional `critic.pt` at each save interval. It may also export ONNX policy files for small motion libraries.
- Exact official sizes are not known until the first checkpoint is produced. Comparable public SuperMimic releases report approximately 69 MB for a stripped 24k checkpoint, 71 MB for a 10k policy/world-model checkpoint, and 223 MB for a 100k checkpoint.
- GitHub enforces a 100 MiB limit for a regular Git object and recommends external storage for generated binaries. A 223 MB checkpoint would require Git LFS or a release asset, and retaining hundreds of periodic checkpoints would make the repository unnecessarily large.
- The working policy is to keep every resumable checkpoint in private artifact storage, publish a manifest and checksums to GitHub, and publish only a browser export or selected release artifact after licensing and size checks.

# Project progress

Last updated: 2026-11-21

## Source push and custom-model plan

- Audited the current workspace and confirmed it is not a Git checkout, while `enyihou/supermimic` `main` is the deployed GitHub Pages artifact.
- Preparing a `source` branch containing the complete Vite app, local official runtime assets, server gate, research notes, and training launchers so the live Pages branch remains available.
- Confirmed the training launchers and current model assets are present. The browser app currently has the official runtime and an independent native route, but no second custom-checkpoint simulation route yet.
- Documented the custom-model integration boundary in `docs/research/CUSTOM_MODEL_SIMULATION.md` and added the follow-on work to `task_plan.md`.
- Pushed the complete source project to `enyihou/supermimic` on the `source` branch. The deployed Pages `main` branch remains unchanged.
- Verified `npm run build` after the documentation update and confirmed GitHub Pages reports the existing deployment as built.

## Training status verification

- Verified there is no official full SuperMimic run, completed job record, or SuperMimic checkpoint in this workspace.
- The recorded compute result is a 73.7-second frozen-teacher CUDA smoke for the independent reproduction. It did not train a new SuperMimic model.
- The official 320,000-iteration run remains a prepared launcher and is still pending BONES-SEED access and an authorized archive path.

## Overnight reproduction planning

- Audited the requested overnight plan against the AIR runbook and current launchers.
- Identified missing prerequisites and corrections. The full run needs gated dataset access, an approved private archive root, a timed throughput benchmark, resumable checkpoints, atomic backup publication, and explicit Slurm completion evidence.
- Recorded the release rule that GitHub receives source, manifests, checksums, audit reports, and sanitized graphs. Gated data, credentials, and private checkpoints remain in private storage.
- Recorded that the higher-order supervision work and browser inference export are separate milestones. They cannot be assumed to complete in the same overnight training job.

## BONES-SEED clarification

- Confirmed from the dataset card that BONES-SEED contains 142,220 annotated motions with Unitree G1 MuJoCo-compatible trajectories and requires accepting access conditions and its license before download.
- Recorded that the immediate user action is to obtain the private `g1.tar.gz` archive and provide its path plus an approved private archive destination for backups.

## Credential and staging decision

- Adopted the workflow of keeping Hugging Face credentials on the user's local machine, transferring only the downloaded archive to gpu01, verifying its checksum, and staging it into `$LOCAL_SCRATCH` for training.
- Recorded that the full archive must be present before submission so the job does not depend on a live network transfer.
- Clarified the actual topology as phone -> local SSH server -> AIR. Hugging Face access belongs on the local SSH server, while AIR receives only the downloaded archive.

## AIR transfer verification

- Verified that gpu01 `/home` and `/public` are local XFS filesystems. The archive can be transferred there, hashed, and staged to `$LOCAL_SCRATCH` by the batch job.
- Verified that `/data/houey` is absent. A private approved project path is still required for checkpoint and audit backups.

## Checkpoint storage decision

- Confirmed from the official runner that checkpoints are separate world-model, actor, and optional critic files. Comparable public runs range from roughly 69 MB to 223 MB per combined checkpoint.
- Recorded the policy of keeping periodic checkpoints in private artifact storage and publishing only manifests, checksums, graphs, and a selected browser export to GitHub.

## Storage capacity verification

- The first SSH server has approximately 33 GB free on its root filesystem. It can temporarily hold the compressed BONES-SEED archive, with limited remaining headroom.
- gpu01 has approximately 16 GB used in the user's `/home` quota and a 200 GB per-user working limit for `/home` and `/public` under the server guide. The private durable checkpoint path under `/data` still requires an approved project location.
- Created `/public/houey/supermimic-archive` on gpu01 with owner-only permissions as the interim private archive location.
- Added owner-only `inputs`, `runs`, and `manifests` subdirectories under the interim archive path for the dataset, checkpoint backups, and audit metadata.

## Overnight readiness check

- Storage layout and private archive directories are ready on gpu01.
- The overnight run is still blocked on the private BONES-SEED `g1.tar.gz` archive being transferred and checksum-verified at `/public/houey/supermimic-archive/inputs/g1.tar.gz`.
- A smoke run and throughput benchmark remain required before the full job. The full backup script still needs resumable, atomic publication and explicit completion markers.

## Goal-agent handoff readiness

- Confirmed the storage and source setup are ready for a controlled run.
- The dataset staging job was stopped before download at the user's request. No partial archive remains on the local staging directory or AIR input directory.
- The project is ready for a goal agent to continue from the preflight phase once the user explicitly starts it. The goal agent must still obtain the dataset archive, harden the backup flow, run smoke and throughput checks, then submit the full job.

## Demo rebuild

- Read clone-website and planning-with-files instructions, plus the AIR server guide.
- Verified the workspace has no frontend package and initialized the rebuild plan.
- Inspected `https://supermimic.github.io/demo/` with Paseo and local Playwright Chromium.
- Captured desktop/mobile screenshots and extracted the published DOM/CSS, motion controls, keyboard shortcuts, responsive breakpoints, and model asset URLs.
- Foundation and asset import are complete.

## Demo rebuild complete

- Added a Vite + React + Three.js foundation with `ControlChrome` and `SimulationStage` components.
- Added the fresh local implementation at `/?native=1`, including responsive sheets, keyboard shortcuts, drag target line, fallback G1 rig, and locally hosted robot geometry parsing.
- Downloaded the published model assets to `public/model/` and the complete published demo runtime plus all 36 motion binaries to `public/official-demo/`.
- Default `/` now runs the local official runtime in an iframe for exact WebGL/model fidelity while keeping all assets same-origin and offline-capable.
- Verified desktop and mobile screenshots and exercised motion filtering, Goal, Policy, responsive Controls sheet, and both default/native routes with Playwright.
- `npm run build` passes after the final native-stage warning cleanup.

## Public hosting

- Added `server.mjs` with persisted scrypt password setup, HttpOnly session cookies, and static production serving.
- Added `paseo.json` service configuration and started the production server through Paseo.
- Verified first-login setup, password login, and protected production assets through the managed local service proxy.
- Public NPM hosts currently include `portfolio.enyihou.me`, which serves an active portfolio site. No unused SuperMimic-specific hostname is configured.
- Open blocker: choose or approve the public NPM hostname before changing DNS and reverse-proxy state.

## GitHub Pages deployment

- Confirmed `https://enyihou.github.io/supermimic/` was available and `enyihou/supermimic` did not exist.
- Added a client-side first-login password gate for static hosting and configured the Pages base path.
- Created the public `enyihou/supermimic` repository, pushed the production build, and enabled GitHub Pages from `main`.
- Verified the live URL with Playwright. First login creates the password, the embedded official model loads, and the desktop interaction shell is available.
- Rebuilt the local `dist/` with the normal root base and restarted the Paseo service. Server-side password protection remains active there.

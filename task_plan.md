# Task plan: SuperMimic reproduction and demo

## Goal

Build a fresh local interactive browser simulation inspired by `https://supermimic.github.io/demo/`, with the same control model, responsive chrome, and robot presentation. Host the official robot/model assets locally where practical while keeping the implementation in this workspace independent.

Extend the project with a reproducible single-GPU run of the authors' pinned SuperMimic code, a verified private artifact archive, an evaluation report, and a second website simulation that uses the trained model when a browser inference export is available.

## Phases

- [complete] Reconnaissance and behavior capture
- [complete] Foundation and asset import
- [complete] Component implementation and page assembly
- [complete] Build verification and visual QA
- [planned] Data, storage, and single-GPU preflight
- [planned] Official smoke and throughput benchmark
- [planned] Resumable official full run
- [planned] Checkpoint audit and held-out evaluation
- [planned] Higher-order supervision experiments
- [planned] Browser export, results graphs, and trained-model simulation

## Decisions

- Use a Vite + React + TypeScript app because this repository has no existing frontend scaffold.
- Keep existing experiment files untouched.
- Use the published demo's local robot geometry/model data for fidelity, with an independent React/Three.js control layer.
- Run the official reproduction on one allocated GPU through `lab-submit` on gpu01. The current cluster limit is three days per job, so the full run must be timed and resumable.
- Keep source, manifests, checksums, audit reports, and small graphs in GitHub. Keep BONES-SEED, private checkpoints, and any credentials in an authorized private archive and local copies.
- Keep Hugging Face credentials on the user's local machine. Download the gated archive locally, transfer only the archive to gpu01, verify its checksum before submission, and let the Slurm job stage it to `$LOCAL_SCRATCH`.
- Use an explicit version manifest for every trained artifact. It must include source commits, dataset checksum, training config, checkpoint step, evaluation metrics, and artifact checksums.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Existing workspace has no `package.json` | 1 | Create a fresh Vite app in-place |
| Paseo browser host disconnected during WebGL compilation | 1 | Use local Playwright with Chromium and capture source/computed behavior there |
| Cluster runbook path unavailable on this host | 1 | No cluster job is needed for this browser build |

## Completed

- Reviewed `PLAN (3).MD` against SuperMimic v3 and Higher-Order Action Supervision v2.
- Reviewed the attached critique and retained its mathematical corrections.
- Recommended recovery-focused teacher distillation as the primary HOAS experiment, with Taylor self-consistency as an optional diagnostic.
- Implemented and verified the frozen-teacher pilot code and a short CUDA teacher smoke on one AIR A800.

## Current scope

- Keep all training code, checkpoints, raw motions, and evaluation results private.
- Use AIR local scratch for compute and an approved durable archive path for resumable outputs.
- Do not submit the full BONES-SEED run until access and storage prerequisites are available.

## Planned website extension

- After a checkpoint is trained and verified, export it into a browser-readable inference package.
- Add a versioned custom-model route alongside the published reference simulation.
- Keep the custom model behind the authenticated server deployment and expose only the UI gate on static Pages.
- Use the native Three.js stage as the first integration surface, then add comparison controls and responsive QA.

## Overnight execution gates

1. **Preflight**. Confirm the gated dataset path, its checksum and expanded size, the approved private archive root, local quota headroom, source commits, environment lock, and a unique run identifier.
2. **Smoke**. Run the existing short official job and preserve its logs, status, and output archive.
3. **Benchmark**. Measure iterations per second and checkpoint size on the target single-GPU configuration. Use the result to decide whether one three-day allocation can finish or whether a resume chain is required.
4. **Full training**. Run 320,000 iterations or the agreed target. Save checkpoints at a fixed interval and publish each completed checkpoint through a temporary path followed by an atomic rename.
5. **Audit**. Verify the Slurm terminal state, job ID, exit code, success marker, source and dataset manifests, checkpoint integrity, and a held-out rollout evaluation before declaring the run complete.
6. **Research extensions**. Run the frozen-teacher Taylor diagnostic, then matched baseline and higher-order experiments with held-out recovery evaluation. These are separate experiments and may require more than one overnight run.
7. **Website release**. Export a verified checkpoint into a browser-readable policy package or provide a server-backed inference route, add the trained-model simulation and graphs, run desktop/mobile QA, tag the release, and push only the public-safe artifacts.

## Pending

- Obtain approved BONES-SEED access.
- Obtain an authorized writable AIR archive path.
- Run a short timed training benchmark before any full-data submission.
- Run the conditional teacher-distillation pilot with matched controls and held-out recovery evaluation.
- Decide whether the trained simulation must run fully in the browser or may call an authenticated inference service.
- Define the public release policy for metrics, replays, and model weights.
- Add resumable and atomic backup logic to `training/run_full.sbatch` before submitting the full run.

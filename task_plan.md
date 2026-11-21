# Task plan: SuperMimic reproduction and demo

## Goal

Build a fresh local interactive browser simulation inspired by `https://supermimic.github.io/demo/`, with the same control model, responsive chrome, and robot presentation. Host the official robot/model assets locally where practical while keeping the implementation in this workspace independent.

Extend the project with a reproducible single-GPU run of the authors' pinned SuperMimic code, a verified private artifact archive, an evaluation report, and a second website simulation that uses the trained model when a browser inference export is available.

## Phases

- [complete] Reconnaissance and behavior capture
- [complete] Foundation and asset import
- [complete] Component implementation and page assembly
- [complete] Build verification and visual QA
- [in_progress] Data, storage, and single-GPU preflight (local input verified; transfer and smoke packing live; shared storage not yet authorized)
- [in_progress] Resumable launcher and private backup verification
- [planned] Official smoke and throughput benchmark
- [planned] Resumable official full run
- [planned] Checkpoint audit and held-out evaluation
- [separate follow-up] Higher-order supervision experiments (not required for the pinned reproduction release)
- [in_progress] Browser export, results graphs, and trained-model simulation (fixture QA and public preflight release complete; audited model pending)

## Decisions

- Use a Vite + React + TypeScript app because this repository has no existing frontend scaffold.
- Keep existing experiment files untouched.
- Link to the official published demo on its original host. The trained view uses independent pinned MuJoCo and robot assets.
- Run the official reproduction on one allocated GPU through `lab-submit` on gpu01. The current cluster limit is three days per job, so the full run must be timed and resumable.
- Keep source, manifests, checksums, audit reports, and small graphs in GitHub. Keep BONES-SEED, private checkpoints, and any credentials in an authorized private archive and local copies.
- Use the two-hop topology `phone -> local SSH server -> AIR`. Keep any Hugging Face authentication on the local SSH server only. Download the gated archive there, transfer only the archive to gpu01, verify its checksum before submission, and let the Slurm job stage it to `$LOCAL_SCRATCH`.
- Use an explicit version manifest for every trained artifact. It must include source commits, dataset checksum, training config, checkpoint step, evaluation metrics, and artifact checksums.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Existing workspace has no `package.json` | 1 | Create a fresh Vite app in-place |
| Paseo browser host disconnected during WebGL compilation | 1 | Use local Playwright with Chromium and capture source/computed behavior there |
| Cluster runbook path unavailable on this host | 1 | No cluster job is needed for this browser build |
| gpu01 `/public` and `/home` missing on Slurm gpu02 | 1 | Confirmed in Slurm job 33617; require an authorized compute-visible AIR stage/archive path. Keep local AIR paths as canonical copies. |
| Old official smoke passed raw BONES CSV glob to NPZ/packed-only loader | 1 | Implement reviewed input conversion before any GPU submission. |
| Published reference demo license prohibits mirroring | 1 | Remove copied assets from subsequent public builds; link to the official reference. |
| Upstream CUDA 13 lock vs AIR CUDA 12.6 driver | 1 | Resolve and hash an explicitly recorded cu126-compatible runtime with the same source pins; require smoke proof. |
| Environment build preflight rejected gpu01 /home quota output | 1 | Handle XFS filesystem and byte-column output on either one line or wrapped lines; retry in a fresh version directory. |
| Environment build v2 could not resolve conda-pack 0.8.2 | 1 | Use published 0.9.2 in v3. |
| Local Vite ports 4173 and 4187 already occupied | 1–2 | Used an available dedicated Pages-base port without stopping either listener. |
| Pages deployment diff check flagged whitespace embedded in bundled Three.js shaders | 1 | Keep build bytes intact, check authored files separately, and verify the generated asset allowlist and rendered build. |
| Relocation probe invoked the archive's conda-unpack shebang with no Python on PATH | 1 | Recovered with relocated bin/python explicitly; pin the interpreter in both batch launchers. |
| Pages deploy shell checker rejected parse-time extended patterns | 1 | Use a Bash regex asset allowlist. |
| OpenReview forum requires verification; notes API returns HTTP 403 | 1–2 | No supplementary manifest verified. Keep author-list/explicit-approximation gate. |
| Old public status differs from the evidence-derived generator | 1 | Regenerate from verified public phase evidence and enforce the same comparison in build/deploy. |
| Broad local Python discovery imported environment-dependent training and export tests | 1 | Use the existing CPU test environments for NumPy/MuJoCo and Torch-dependent tests; avoid installing GPU training dependencies locally. |
| The documented `gpu01` login alias differs from its OS hostname | 1 | Validate the canonical owner-only login-local XFS mount instead; reject Slurm allocations. |

## Completed

- Reviewed `PLAN (3).MD` against SuperMimic v3 and Higher-Order Action Supervision v2.
- Reviewed the attached critique and retained its mathematical corrections.
- Recommended recovery-focused teacher distillation as the primary HOAS experiment, with Taylor self-consistency as an optional diagnostic.
- Implemented and verified the frozen-teacher pilot code and a short CUDA teacher smoke on one AIR A800.

## Current scope

- Publish reproduction source and sanitized manifests, audit results, metrics, and graphs; keep raw checkpoint and motion artifacts private.
- Use AIR local scratch for compute, and the user-authorized owner-only `/public/houey/supermimic-archive` for staged input, backups, and manifests.
- Do not submit the full BONES-SEED run until access and storage prerequisites are available.

## Planned website extension

- After a checkpoint is trained and verified, export it into a browser-readable inference package.
- Add a versioned custom-model route alongside the published reference simulation.
- Run the trained model entirely in the browser. Publish only an explicitly cleared browser inference artifact; a Pages UI gate cannot restrict artifact access.
- Use the native Three.js stage as the first integration surface, then add comparison controls and responsive QA.

## Overnight execution gates

1. **Preflight**. Confirm the gated dataset path, its checksum and expanded size, the approved private archive root, local quota headroom, source commits, environment lock, and a unique run identifier.
2. **Smoke**. Run the existing short official job and preserve its logs, status, and output archive.
3. **Benchmark**. Measure iterations per second and checkpoint size on the target single-GPU configuration. Use the result to decide whether one three-day allocation can finish or whether a resume chain is required.
4. **Full training**. Run 320,000 iterations or the agreed target. Save checkpoints at a fixed interval and publish each completed checkpoint through a temporary path followed by an atomic rename.
5. **Audit**. Verify the Slurm terminal state, job ID, exit code, success marker, source and dataset manifests, checkpoint integrity, and a held-out rollout evaluation before declaring the run complete.
6. **Research extensions**. Run the frozen-teacher Taylor diagnostic, then matched baseline and higher-order experiments with held-out recovery evaluation. These are separate experiments and may require more than one overnight run.
7. **Website release**. Export a verified checkpoint into a browser-readable policy package, add the fully browser-based trained-model simulation and graphs, run desktop/mobile QA, version the release, and push only public-safe artifacts.

## Pending

- Finish the verified canonical AIR upload and CPU smoke-input pack. The gpu01 `/public/houey/supermimic-archive` archive is now verified and the 64-clip smoke package passed, but no job-visible `/data/houey` or other approved compute-visible project path has been provisioned.
- For the full cohort, obtain the authors' exact clip list or explicitly approve the documented approximation.
- Run a short timed training benchmark before any full-data submission.
- Run the conditional teacher-distillation pilot with matched controls and held-out recovery evaluation.
- Verify browser inference parity and confirm the browser export can be distributed before publishing it.
- Verify the implemented resumable/atomic backup lifecycle in the official CUDA smoke and benchmark.

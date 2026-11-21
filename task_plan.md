# Task plan: SuperMimic demo rebuild

## Goal

Build a fresh local interactive browser simulation inspired by `https://supermimic.github.io/demo/`, with the same control model, responsive chrome, and robot presentation. Host the official robot/model assets locally where practical while keeping the implementation in this workspace independent.

## Phases

- [complete] Reconnaissance and behavior capture
- [complete] Foundation and asset import
- [complete] Component implementation and page assembly
- [complete] Build verification and visual QA

## Decisions

- Use a Vite + React + TypeScript app because this repository has no existing frontend scaffold.
- Keep existing experiment files untouched.
- Use the published demo's local robot geometry/model data for fidelity, with an independent React/Three.js control layer.

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

## Pending

- Obtain approved BONES-SEED access.
- Obtain an authorized writable AIR archive path.
- Run a short timed training benchmark before any full-data submission.
- Run the conditional teacher-distillation pilot with matched controls and held-out recovery evaluation.

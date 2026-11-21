# Project progress

Last updated: 2026-11-21

## Source push and custom-model plan

- Audited the current workspace and confirmed it is not a Git checkout, while `enyihou/supermimic` `main` is the deployed GitHub Pages artifact.
- Preparing a `source` branch containing the complete Vite app, local official runtime assets, server gate, research notes, and training launchers so the live Pages branch remains available.
- Confirmed the training launchers and current model assets are present. The browser app currently has the official runtime and an independent native route, but no second custom-checkpoint simulation route yet.
- Documented the custom-model integration boundary in `docs/research/CUSTOM_MODEL_SIMULATION.md` and added the follow-on work to `task_plan.md`.
- Pushed the complete source project to `enyihou/supermimic` on the `source` branch. The deployed Pages `main` branch remains unchanged.
- Verified `npm run build` after the documentation update and confirmed GitHub Pages reports the existing deployment as built.

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

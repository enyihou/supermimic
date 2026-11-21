# Custom SuperMimic simulation plan

The current website has two useful pieces for the planned trained-model demo.

- The default route embeds the locally hosted published SuperMimic runtime at `public/official-demo/`.
- The `/?native=1` route is an independent React and Three.js stage. It already loads the local robot mesh from `public/model/robot.json` and `public/model/robot.bin`.
- The `training/` launchers prepare the pinned SuperMimic repositories, run the official training entry point, and archive private checkpoints.

The browser-facing custom simulation still needs a model export boundary. The training checkpoints are private PyTorch artifacts and are not automatically equivalent to the published browser `weights.bin` format. A trained run therefore needs an explicit export step that records the checkpoint revision, observation/action schema, robot asset revision, and a browser-readable inference package.

## Intended website shape

1. Keep the published reference simulation as the `Official` view.
2. Add a separate `Trained model` view or route that selects a versioned model manifest such as `/models/custom-v1/manifest.json`.
3. Serve the custom model package with the same authenticated server path as the app. A static Pages deployment can gate the UI, but it cannot keep the model files private once their URLs are known.
4. Use the native Three.js stage as the integration surface for the first custom view. It can load a versioned robot mesh and motion/reference data without changing the official iframe.
5. Add an evaluation panel that reports the model version, checkpoint step, motion clip, and whether the displayed result is the reference or trained policy.

## Work still required after training

- Define and implement the checkpoint-to-browser export format.
- Export one trained checkpoint and verify finite actions against the training-side observation schema.
- Add a model registry and route selection in `src/App.tsx`.
- Load the exported policy in a worker or WebAssembly runtime so rendering stays responsive.
- Add a side-by-side or toggle comparison against the official runtime, then test desktop and mobile behavior.

The current source branch is ready to receive that export and integration work. No trained checkpoint is present in this repository.

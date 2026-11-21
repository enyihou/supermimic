# Trained-model browser release

The trained simulation uses an independent MuJoCo WASM runtime and an actor-only ONNX export. The official reference stays at the [authors’ site](https://supermimic.github.io/demo/). The [official web demo license](https://supermimic.github.io/demo/LICENSE.html) is separate from the Apache-2.0 training repository and does not grant rights to mirror, modify, or redistribute the demo’s engine, weights, binary motions, or prepared robot assets. Do not use those assets for this release.

## Pinned model contract

The `SuperMimic-GenCoord-G1` launcher uses the pinned default `actor_obs=rep_state`, one frame of history, and the motion task. Its input is the 68-value `[height, body gravity, body linear velocity, body angular velocity, joint positions, joint velocities]` representation. Its 640-value goal contains ten `[reference joint positions, reference joint velocities]` frames followed by ten relative root rotations (the first two matrix columns, flattened by row); reference frames are five control steps apart. The 29 raw actor outputs map to PD position targets through per-joint `raw * scale + offset`. Joint order, action scales, offsets, timestep, and four substeps come from the pinned mjlab environment and saved model. No learned dynamics are used in this view: MuJoCo advances every state locally from the exported actor’s actions.

The pinned project’s built-in ONNX export is for a **proprioception** actor and embeds motion arrays in the motion graph. Neither that graph nor the published demo’s `weights.bin` is a substitute for this contract. The independent export uses only the motion goal encoder, state/goal normalizers, and action decoder from the audited checkpoint. Synthetic reference commands are generated in-browser and are not part of the held-out evaluation.

## Publication boundary

The [BONES-SEED license](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md) distinguishes model results from raw data, restricts redistribution of raw or recoverable data, and requires model attribution. Training data includes Motion Data by [Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. Confirm release authority under the accepted license before publishing the export. Do not publish `actor.pt`, whole train checkpoints, motion arrays, raw holdout reports, or an export that includes them. A Pages UI gate cannot make assets on Pages private.

The release manifest is part of the model package, not a replacement for the Slurm, checkpoint, and held-out audit. It records model ID, source revisions, dataset SHA-256, checkpoint step, aggregate holdout metrics, and asset hashes. Browser inference must fail visibly on an absent, unsupported, corrupted, or parity-failing package; it must not show the official policy under a trained-model label.

## Export after the full-run audit

Use the pinned Python environment and CPU. The input here is the audited private checkpoint directory, **not** a git checkout or public folder. Fill in checkpoint hashes and the reproduction commit from the run evidence, not the export machine’s current HEAD:

```sh
python training/export_browser_policy.py "$PRIVATE_CHECKPOINT" "$PRIVATE_EXPORT_DIR/$MODEL_ID" \
  --model-id "$MODEL_ID" --checkpoint-step 320000 \
  --expected-actor-sha256 "$ACTOR_SHA256" --expected-world-model-sha256 "$WORLD_MODEL_SHA256" \
  --dataset-sha256 "$DATASET_SHA256" --reproduction-commit "$TRAINING_SOURCE_COMMIT" \
  --metrics "$PUBLIC_HOLDOUT_METRICS_JSON"
```

The export contains `actor.onnx`, `scene.mjbpack`, `manifest.json`, and license/attribution files. It exercises the motion encoder against the pinned PyTorch implementation on synthetic observations. The published manifest carries two synthetic action parity vectors; the browser checks these and both artifact hashes before enabling simulation. `scene.mjbpack` is a gzip container for the independently sourced pinned mjlab G1 model and Unitree geometry, with notices alongside the release. Its extension keeps web hosts from automatically decoding it before hash verification. The worker decompresses it locally and obtains actual render meshes and both the pose and controller state from that same scene. The physics and ONNX runtime do not send inputs to a server.

Verify the release in a private browser preview before publishing. Review the public bundle for only the allowlisted output and for license conditions. Publish it as `models/$MODEL_ID/` under the Vite base path, with the trained route pointing to its versioned `manifest.json`. Record browser configuration and evidence for desktop and mobile. The final policy must track the browser demo commands acceptably; test this on the actual trained weights, not just the synthetic fixture.

Run `npm run verify:release` (also invoked by `npm run build`) before
publishing the registry. An active release requires the checksum-verified
dataset preflight and the attached held-out audit in
`public/reproduction/full-audit.json`. The gate checks the release metadata,
exact training step, public artifact list and hashes, executable action parity,
and scene compatibility. It does not replace the private export or Slurm audit
records and does not authorize distribution of the package by itself.

After reviewing the public-only source diff and pushing the exact source
revision, publish Pages with `bash scripts/deploy-pages.sh "$SOURCE_COMMIT"`.
It builds in a temporary directory, rechecks the published model assets and
the progress report against the audited phase evidence, rejects unexpected
assets and symlinks, and pushes a normal fast-forward commit to the separate
Pages branch. The production build requires Python 3 to verify the progress
report (`npm run verify:status`).

For a local test independent of any training run, generate a temporary random actor ONNX against the pinned Python source with `python training/test_browser_export.py --fixture-dir "$FIXTURE"`, start Vite, and run `node scripts/test-browser-runtime.mjs http://127.0.0.1:5173/ "$FIXTURE"`. This tests inference parity, actual local MuJoCo stepping, mesh transfer, reset, and failure on altered hashes in Chromium. It is not a trained-model evaluation.

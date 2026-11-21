# Official SuperMimic reproduction

The launcher uses [SuperMimic at `5fb92297`](https://github.com/HybridRobotics/supermimic/tree/5fb92297ab4fe3d86629d9f0c635334791bb0d5f)
and [mjlab research at `379c55e8`](https://github.com/HybridRobotics/mjlab_research/tree/379c55e88b1dec73f93943e524a879b2ebbf45c5).
Full recipe: `SuperMimic-GenCoord-G1`, one GPU, 4,096 environments, exactly
320,000 completed iterations. The reviewed AIR overlay adds lifecycle and
recovery code to the pinned runner. A resumed allocation restores the model,
optimizer, replay, curriculum, and RNG state; simulator episodes restart at an
explicit transition boundary rather than claiming bitwise physics continuity.

## Preflight and packages

1. Read the current AIR server guide and runbook from gpu01. Check quotas,
   resource availability and that the Slurm nodes can read the shared approved
   inputs and write the shared private runs root. gpu01's `/public` and `/home`
   are login-local, not job-visible. Keep the canonical input/backups on the
   approved private gpu01 paths and use an AIR-approved compute-visible project
   path to stage compressed inputs and receive compressed outputs. Do not use
   shared storage as the live training environment or motion library.
2. On the **local SSH server**, check the gated login with
   `hf auth whoami >/dev/null` and download only `g1.tar.gz` from
   [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed),
   revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`, under the
   accepted license. Use `bash training/stage_g1.sh LOCAL_ARCHIVE SHA256 BYTES`
   to hash the source, transfer only the archive to gpu01, recheck the hash, and
   atomically publish it. Never authenticate Hugging Face or sync its cache on
   AIR. `training/inspect_bones.py` checks safe members, CSV layout, and expanded
   size without extracting the archive.
3. On the local SSH server, run `bash training/prepare_official.sh CHECKOUT_ROOT`
   and `bash training/package_official_code.sh CHECKOUT_ROOT CODE_TAR_GZ`.
   The latter archives the Git trees at the pins and verifies that the reviewed
   overlay applies before publishing `code.tar.gz`. Transfer code separately
   from the data. On gpu01 build the relocatable environment with
   `bash training/build_air_environment.sh NEW_BUILD_DIRECTORY training/requirements-air-cu126.lock`.
   This uses PyTorch 2.14.1+cu126 and Warp 1.19.0+cu12 for AIR's R560 drivers,
   not the upstream CUDA 13 lock. The source pins are unchanged. Validate CUDA,
   MuJoCo Warp, and compilation in the allocated smoke before relying on the
   runtime. Record both environment and code archive hashes.
4. Convert verified BONES input on gpu01 using `training/pack_bones.py`. Pass
   the G1 XML from the pinned mjlab source
   (`src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`). Use
   `--sha256 "$BONES_SHA256" --smoke-clips 64` for a bounded smoke input,
   and `--sha256 "$BONES_SHA256" --keep-list "$PRIVATE_CLIP_LIST"` for the
   reviewed full selection. The pinned loader does not accept BONES CSVs.
   The converter resamples 120 Hz CSV to 50 Hz, converts units, and writes the
   packed root pose/twist and joint arrays. Keep the list and packed motions
   private. The pins do not include the paper's exact 129,514-clip list; audit
   the actual filter, clip count and selection hash and do not claim exact
   Appendix E parity without evidence.
5. On gpu01, package each packed-motion directory at the archive root
   (`tar -C "$PACKED_DIR" -czf "$MOTIONS_TGZ" .`) and compute SHA-256. Stage
   `environment.tar.gz`, `code.tar.gz` and `SHA256SUMS` together in a new
   private version (the checksum file lists exactly the two archive basenames).
   Follow the runbook to publish verified versions to the approved shared
   project path. Confirm scratch and archive quotas using actual sizes.

Training data includes Motion Data by [Bones Studio](https://bones.studio/).
Use of the underlying dataset is subject to the BONES Motion Capture Dataset
License Agreement.

## Submit in order

On gpu01, use `lab-submit`, with absolute job-visible paths and precreated
private scheduler log directories. Set `CODE` to the directory of the reviewed
batch scripts; `DRIVER` is the absolute job-visible copy of `run_full.sbatch`.
The run directories inside `$RUNS` are owner-only. Audit each job before moving
to the next one.

```sh
lab-submit --kind gpu -- --parsable --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
  "$CODE/run_smoke.sbatch" "$DRIVER" "$PACKAGE" "$SMOKE_MOTIONS" "$SMOKE_SHA256" \
  "$BONES_SHA256" "$RUNS" "$SMOKE_RUN_ID"

lab-submit --kind gpu -- --parsable --job-name=supermimic-benchmark --time=02:00:00 \
  --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
  "$CODE/run_full.sbatch" "$PACKAGE" "$FULL_MOTIONS" "$FULL_SHA256" \
  "$BONES_SHA256" "$RUNS" "$BENCH_RUN_ID" benchmark fresh

lab-submit --kind gpu -- --parsable --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
  "$CODE/run_full.sbatch" "$PACKAGE" "$FULL_MOTIONS" "$FULL_SHA256" \
  "$BONES_SHA256" "$RUNS" "$FULL_RUN_ID" full fresh
```

The benchmark uses the full-scale motion package and 4,096 environments.
Use measured steady-state throughput, publication cost, and checkpoint size
to plan wall time and storage. The full job preserves a rolling latest
checkpoint every 1,000 completed iterations and a milestone every 32,000,
including the final state. If an allocation stops early, capture its terminal
Slurm record, verify the latest checkpoint, and resubmit the same full recipe,
package and run ID with `full resume`. Do not resume a live allocation.

Monitor `squeue`, a concise training log tail, and published checkpoint
manifests. Save the terminal Slurm record promptly; controller records age out.
Success requires a matching `COMPLETED`/`0:0` record, verified archive hashes,
the exact target in the latest checkpoint, and the attempt's `SUCCESS` and
`job-status.txt`. See [held-out evaluation and audits](HELD_OUT_EVAL.md) and
[browser release](../docs/research/CUSTOM_MODEL_SIMULATION.md). Keep checkpoint
archives, data, credentials and raw logs private.

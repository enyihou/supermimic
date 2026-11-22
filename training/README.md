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
4. On gpu01, use `training/prepare_air_smoke.sh` for the bounded 64-clip smoke.
   It verifies both input hashes, confirms the pinned loader can read the
   packed arrays on CPU, and atomically publishes a private compressed smoke
   package. Use the CPU batch job below for the full pool. Both paths use the
   pinned mjlab G1 scene and `pack_bones.py` to resample 120 Hz CSVs to 50 Hz
   and write the root pose/twist and joint arrays the pinned loader expects.
   Keep the packed motions and clip list private. The pins do not ship the
   authors' exact 129,514-clip list; audit the actual clip count, filter and
   selection hash. Use the [approximate selector](BONES_SELECTION.md) only
   with explicit approval, and do not claim exact Appendix E parity.
5. Stage `environment.tar.gz`, `code.tar.gz` and `SHA256SUMS` together in a new
   private version with `training/package_air_inputs.sh` (the checksum file
   lists exactly the two archive basenames). The code archive includes
   `run.sh` as its entry point. Reuse this exact package for the benchmark,
   full run, and evaluation after a successful smoke.
   Follow the runbook to publish verified versions to the approved shared
   project path. Confirm scratch and archive quotas using actual sizes.

For the full pool, run conversion on allocated scratch using a CPU-only
allocation. Pass a normalized, privately reviewed authors' allowlist and its
SHA-256 as the final two arguments. The allowlist must contain one exact
archive member (`g1/csv/DATE/MOTION.csv`) per line, with no header or duplicate:

```sh
env -u BASH_ENV -u ENV lab-submit --kind cpu -- --parsable --chdir="$SHARED_PROJECT" \
  --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
  "$CODE/run_prepare_motions.sbatch" "$PACKAGE" "$SHARED_BONES_ARCHIVE" "$BONES_SHA256" \
  "$PRIVATE_MOTIONS_ROOT" "$COHORT_BUILD_ID" authors-manifest "$PRIVATE_CLIP_LIST" "$PRIVATE_CLIP_LIST_SHA256"
```

If explicitly approved to use the approximation, replace the last three
arguments with `--accept-approximate "$PINNED_SONIC_SOURCE" "$PINNED_SONIC_SOURCE_SHA256"`.
Verify the Slurm terminal status, `SUCCESS`, `job-status.txt`, preparation
metadata and archive checksum before passing the same packed input to the
benchmark and full run. Neither the allowlist nor the packed motions are public.
For the approved approximate path, the versioned gpu01 operations helpers
can detach the input stage, Slurm submission and terminal watcher, then run
`finish_air_cohort.sh` with the shared project, saved cohort ID, Slurm job ID
and a fresh benchmark run ID. It audits the actual zero-GPU completion and
packed provenance, verifies a private canonical backup, and only then submits
the benchmark against that exact motion digest. Its submission intent and job
ID must be reconciled before any resubmission. Only the allowlisted output of
`audit_air_cohort.py` is suitable for `public/reproduction/cohort-audit.json`.

Training data includes Motion Data by [Bones Studio](https://bones.studio/).
Use of the underlying dataset is subject to the BONES Motion Capture Dataset
License Agreement.

## Submit in order

On gpu01, use `lab-submit`, with absolute job-visible paths and precreated
private scheduler log directories. Set `SHARED_PROJECT` to the approved,
job-visible working directory and `CODE` to the directory of the reviewed
batch scripts; `DRIVER` is the absolute job-visible copy of `run_full.sbatch`.
The run directories inside `$RUNS` are owner-only. Audit each job before moving
to the next one.

```sh
env -u BASH_ENV -u ENV lab-submit --kind gpu -- --parsable --chdir="$SHARED_PROJECT" \
  --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
  "$CODE/run_smoke.sbatch" "$DRIVER" "$PACKAGE" "$SMOKE_MOTIONS" "$SMOKE_SHA256" \
  "$BONES_SHA256" "$RUNS" "$SMOKE_RUN_ID"

env -u BASH_ENV -u ENV lab-submit --kind gpu -- --parsable --chdir="$SHARED_PROJECT" \
  --job-name=supermimic-benchmark --time=02:00:00 \
  --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
  "$CODE/run_full.sbatch" "$PACKAGE" "$FULL_MOTIONS" "$FULL_SHA256" \
  "$BONES_SHA256" "$RUNS" "$BENCH_RUN_ID" benchmark fresh

env -u BASH_ENV -u ENV lab-submit --kind gpu -- --parsable --chdir="$SHARED_PROJECT" \
  --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
  "$CODE/run_full.sbatch" "$PACKAGE" "$FULL_MOTIONS" "$FULL_SHA256" \
  "$BONES_SHA256" "$RUNS" "$FULL_RUN_ID" full fresh
```

The benchmark uses the full-scale motion package and 4,096 environments.
Use measured steady-state throughput, publication cost, and checkpoint size
to plan wall time and storage. `plan_air_full.py` validates the benchmark
against its published preparation and smoke audits and estimates wall time,
allocation count and private storage from the measured windows. Compare the
estimated reserves with current `gpu01` quotas, confirm the GPU type and
partition limit, and pass the planned `--time` at submission. The full job preserves a rolling latest
checkpoint every 1,000 completed iterations and a milestone every 32,000,
including the final state. For a planned walltime handoff, capture the terminal
Slurm record and copy the stopped attempt, its latest checkpoint, and any new
milestones to the independent private archive. From the login host, check:

```sh
python3 -B training/backup_air_attempt.py "$RUNS" "$FULL_RUN_ID" "$OLD_JOB_ID" --step "$RESUME_STEP"
python3 -B training/verify_air_resume.py "$RUNS" "$FULL_RUN_ID" "$OLD_JOB_ID" \
  --slurm-record "$OLD_SLURM_JSON" --benchmark "$BENCHMARK_AUDIT"
```

The verifier requires the intentional `FAILED/124:0` batch exit, matching
stage-out and checkpoint evidence, a step that advanced in the old allocation,
and matching canonical copies. Before submitting the same recipe, package and
run ID with `full resume`, recheck Slurm for another pending or running attempt.
If the gate fails (including an unplanned timeout or a stale latest pointer),
reconcile the evidence manually; do not submit based on queue disappearance.

For each full attempt, start the login-side worker immediately after
submission with its job-specific private state directory (`job-id` and
`inputs.txt`, as below):

```sh
bash training/finish_air_smoke.sh "$SHARED_PROJECT" "$FULL_RUN_ID" "$JOB_ID" \
  "$PRIVATE_STATE" full "$BENCHMARK_AUDIT"
```

Run it detached with a private log. Success backs up all ten milestones and
produces `full-audit.json` only after binding the completed run to the audited
benchmark. A planned wall-time stop backs up the retained milestones and
handoff checkpoint, then invokes the resume verifier. The worker does not
submit a new allocation. Keep each attempt's state and terminal record.

Monitor `squeue`, a concise training log tail, and published checkpoint
manifests. Save the terminal Slurm record promptly; controller records age out.
`python3 training/watch_slurm.py "$JOB_ID" "$PRIVATE_SLURM_JSON"` can be run
detached on gpu01 to capture the exact job's terminal controller state
before it expires. This uses detailed `scontrol` output so Slurm 21.08's actual
GPU allocation is recorded, not inferred from its GPU request. An observation
timeout does not authorize a restart. For the smoke, `finish_air_smoke.sh`
can run detached with the job-visible project, run ID, job ID, and an owner-only
login-local state directory containing `job-id` and `inputs.txt` (the submitted
`dataset_sha256`, `motion_sha256`, `code_sha256`, `environment_sha256` and
`reproduction_commit` as `key=value`). It captures terminal evidence, backs up
the attempt, and verifies a successful smoke before public review. If the
worker stops, restart it with the same arguments and existing state; do not
submit again to replace a worker.
Success requires a matching `COMPLETED`/`0:0` record, verified archive hashes,
the exact target in the latest checkpoint, and the attempt's `SUCCESS` and
`job-status.txt`. See [held-out evaluation and audits](HELD_OUT_EVAL.md) and
[browser release](../docs/research/CUSTOM_MODEL_SIMULATION.md). Keep checkpoint
archives, data, credentials and raw logs private.

After audit or interruption, copy explicit checkpoint and attempt publications
to the independent private gpu01 archive using [the backup helper](AIR_BACKUPS.md).
Keep the terminal Slurm records under the approved private manifests root.

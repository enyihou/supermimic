# Held-out evaluation and audited phase summaries

The [SuperMimic benchmark](https://supermimic.github.io/) measures the paper's
filtered **training** motions. This separate generalization check uses all 40
G1-retargeted [Unitree LAFAN1 recordings](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
from a fixed revision of Ubisoft's independent motion-capture corpus. It is
not a reproduction of the paper's motion counts or scores and should not be
compared as one. Training must use only the audited BONES input, without
warm-starting from other weights. The test proves corpus-level holdout, not
that similar motions never appear in BONES.

Fetch the pinned suite on the **local** SSH server. It needs no credentials.
Every byte is checked against the pinned Git tree in `lafan1_g1_eval_manifest.tsv`.
Keep raw and converted LAFAN1 motions out of GitHub and the website. The
dataset card identifies noncommercial/no-derivatives terms for the underlying
LAFAN1 recordings; the archive is an internal evaluation input only.

```sh
python3 training/evaluate_heldout.py fetch --output "$LOCAL_PRIVATE/lafan1-g1-v1.tar.gz"
sha256sum "$LOCAL_PRIVATE/lafan1-g1-v1.tar.gz"
```

Transfer the archive to private AIR input storage, recheck the digest, and
make a verified copy available to the allocation under the current AIR
runbook. On gpu01, submit the evaluation with the **same** verified code and
environment package as the final checkpoint. Include the evaluation scripts
and suite manifest when building `code.tar.gz`. All environment, model, suite,
and logs are staged on allocated scratch; no Hub or W&B is used by the job.
Wait for the full training audit and independent private backup. Use the
immutable step-320,000 checkpoint directory (not `latest`) and the
`checkpoint_archive_sha256` from that audit. The batch script verifies the
staged archive against the audit's digest before opening the holdout.

```sh
env -u BASH_ENV -u ENV lab-submit --kind gpu -- --parsable --chdir="$SHARED_PROJECT" \
  --output="$PRIVATE_LOGS/%j.out" --error="$PRIVATE_LOGS/%j.err" \
  training/run_eval.sbatch "$PACKAGE_DIR" "$CHECKPOINT_DIR" "$LAFAN1_ARCHIVE" \
  "$LAFAN1_SHA256" "$BONES_SHA256" "$PRIVATE_EVALUATIONS_ROOT" "$CHECKPOINT_SHA256"
```

The simulator uses the pinned MuJoCo Warp environment and checkpoint actor.
Each environment starts on a different clip's first frame. Noise, reset
jitter, pushes, CoM randomization, and encoder bias are disabled; foot
tangential friction is 1.0. A clip passes only if it reaches its last frame
without a training failure. Normalized return is the collected reward divided
by the original clip length times the positive per-step reward ceiling, treating
steps after failure as zero. Pose errors are measured before each simulated
control step and averaged per clip, then across clips. Results are aggregated;
no trajectory poses, clip logs, or weights are needed on GitHub.

Start the terminal watcher immediately after submission, detached on gpu01
with a private log. Record the exact job ID. Queue disappearance is not a
terminal record:

```sh
nohup python3 -B -u training/watch_slurm.py "$JOB_ID" "$PRIVATE_RECORD" \
  --interval 10 </dev/null >>"$PRIVATE_WATCHER_LOG" 2>&1 &
```

Run the audit against the independently verified package and input hashes:

```sh
python3 training/audit_official.py verify --run "$PRIVATE_RUN" --phase "$PHASE" \
  --job-id "$JOB_ID" --slurm-record "$PRIVATE_RECORD" \
  --dataset-sha256 "$BONES_SHA256" --motion-sha256 "$PACKED_MOTIONS_SHA256" \
  --code-sha256 "$CODE_SHA256" --environment-sha256 "$ENVIRONMENT_SHA256" \
  --reproduction-commit "$REPRODUCTION_COMMIT" \
  --num-envs "$NUM_ENVS" --iterations "$TARGET_ITERATIONS" --output "$PUBLIC_PHASE_AUDIT"
```

After the full training and held-out evaluation have both finished, bind the
held-out output to the checkpoint already verified in the training audit:

```sh
python3 training/audit_official.py attach-evaluation \
  --training-audit "$PUBLIC_PHASE_AUDIT" --evaluation "$PRIVATE_EVALUATIONS_ROOT/$EVAL_JOB_ID" \
  --job-id "$EVAL_JOB_ID" --slurm-record "$PRIVATE_EVAL_RECORD" \
  --suite-sha256 "$LAFAN1_SHA256" --output "$PUBLIC_RELEASE_AUDIT"
```

The phase and release JSONs include only public-safe allowlisted data. The
private attempt outputs and checkpoints remain on AIR. A successful training
phase alone does not attest to held-out quality or a runnable browser export.

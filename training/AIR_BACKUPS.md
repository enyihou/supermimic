# Canonical AIR attempt backups

Use this after the current AIR guide and runbook have been read, on the `gpu01` login host, with an approved owner-only job-visible runs root. A Slurm allocation writes to that root; the login host maintains the independent owner-only canonical copy under `/public/$USER/supermimic-archive/runs`. This script makes no attempt to provision private storage or change the job-owned/shared source.

```sh
python3 -B training/backup_air_attempt.py "$JOB_VISIBLE_RUNS_ROOT" "$RUN_ID" "$JOB_ID" \
  --step 32000 --step 64000
```

Omit `--step` to copy only the published attempt. Its terminal manifest can represent a successful, interrupted, or failed attempt. The backup operation checks the published exit code and `SUCCESS` marker; it does not query Slurm or equate a successful copy with a successful training run. Keep the terminal controller evidence and run `audit_official.py` separately against the source run, including its verified rolling checkpoint pointer, when auditing a phase. The canonical backup intentionally does not maintain that pointer.

Checkpoint steps are explicit directories, never the mutable `latest` symlink. Select present milestones whenever possible. If a rolling checkpoint is deleted before its files can be opened, the command fails cleanly; reselect a still-available step. When a step's files have already been opened, the copy requires their original sizes and SHA-256 checksums even if the source removes the directory. No archive is unpacked. Selected checkpoint jobs may differ from the attempt's job ID after a verified resume, but must belong to the same run and recipe.

Each attempt or checkpoint publishes independently after complete size/hash readback, at `runs/$RUN_ID/attempts/$JOB_ID` or `runs/$RUN_ID/checkpoints/step-NNNNNNNNN`. The script refuses to clobber an existing directory and treats a matching canonical copy as already verified. An interruption may leave earlier verified units published; rerun with the same arguments once the source is available. The lock serializes this helper and quota is checked before copies against disk free space, the XFS hard quota when present, and the 200 GB user cap. Reserve space for other simultaneous writers before launching. Do not remove canonical data to make this command pass without separate approval.

Only private files are written. Credentials, raw motions, and checkpoints must not be published. If the process is killed before cleanup, a hidden `.backup-*` directory is not a published unit; inspect it privately, then clean up only the confirmed stale partial after the process has exited.

Local tests, without connecting to AIR:

```sh
python3 -B -m unittest discover -s training -p test_backup_air_attempt.py
```

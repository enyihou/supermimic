"""Verify the private handoff after a planned walltime stop on AIR gpu01."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import tarfile
from contextlib import ExitStack
from pathlib import Path

from air_launch import RECIPES, TASK
from audit_official import SLURM_SOURCE, parse_fields, read_tar, require, verify_checkpoint, verify_hashes
from backup_air_attempt import (
    ATTEMPT_NAMES, JOB_ID, RUN_ID, canonical_destination, existing_matches, open_directory, private_file,
    snapshot,
)
from publish_phase_summary import read_json, verify_phase


STEP = re.compile(r"step-[0-9]{9}")
IDENTITY_FIELDS = (
    "source_commits", "dataset_sha256", "motion_archive_sha256", "motion_manifest",
    "code_sha256", "environment_sha256", "reproduction_commit",
)


def verify_resume(runs: Path, canonical_runs: Path, run_id: str, job: str, record: dict, benchmark: dict) -> dict:
    require(RUN_ID.fullmatch(run_id) is not None and JOB_ID.fullmatch(job) is not None, "Invalid run or job ID")
    stage = verify_phase(benchmark, "benchmark")
    require(stage["finished_job"] != job, "A benchmark job cannot be a full-training attempt")
    require(type(record.get("schema_version")) is int and record["schema_version"] == 1
            and record.get("source") == SLURM_SOURCE
            and record.get("job_id") == job and type(record.get("allocated_gpus")) is int
            and record["allocated_gpus"] == 1 and record.get("state") == "FAILED"
            and record.get("exit_code") == "124:0",
            "One-GPU Slurm exit from the planned walltime warning is not verified")
    envs, target, save, milestone = RECIPES["full"]
    expected = {key: benchmark[key] for key in IDENTITY_FIELDS}
    expected.update(run_id=run_id, mode="full", task_id=TASK, num_envs=envs, max_iterations=target,
                    save_interval=save, milestone_interval=milestone, seed=42,
                    overrides=["--num-envs", str(envs), "--env.seed", "42", "--runner.max-iterations", str(target),
                               "--runner.save-interval", str(save)])
    run = runs / run_id
    attempts = run / "attempts"
    checkpoints = run / "checkpoints"
    attempt = attempts / job

    require(runs.resolve(strict=True) == runs and canonical_runs.resolve(strict=True) == canonical_runs
            and not runs.samefile(canonical_runs), "The shared and canonical roots must be separate real directories")
    with ExitStack() as stack:
        source_root = stack.enter_context(open_directory(runs))
        canonical_root = stack.enter_context(open_directory(canonical_runs))
        source = stack.enter_context(open_directory(run_id, source_root))
        canonical = stack.enter_context(open_directory(run_id, canonical_root))
        # Same advisory lock as run_full.sbatch. Do not read an actively changing pointer.
        fd = os.open("lock", os.O_NOFOLLOW | os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=source)
        stack.callback(os.close, fd)
        require(private_file(os.fstat(fd)), "Run lock must be a private regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A training allocation still owns the run lock") from exc

        old = snapshot(stack, source, run_id, "attempts", job)
        require(old.members.keys() == ATTEMPT_NAMES, "A completed attempt cannot authorize resume")
        require(not any(os.path.lexists(path / "SUCCESS") for path in attempts.iterdir()
                        if JOB_ID.fullmatch(path.name)), "The run already has a successful attempt")
        with open_directory("attempts", canonical) as dst_attempts:
            require(existing_matches(dst_attempts, old), "Missing verified canonical attempt")
        verify_hashes(attempt, {"outputs.tar.gz", "job-status.txt"})
        status_text = (attempt / "job-status.txt").read_text(encoding="ascii")
        status = parse_fields(status_text)
        last = status["last_published_iteration"]
        require(status["reason"] == "walltime_warning" and status["exit_code"] == "124"
                and status["mode"] == "full" and last.isdecimal() and 0 < int(last) < target,
                "Published attempt is not an intentional full-training stop with a partial checkpoint")
        step = int(last)
        name = f"step-{step:09d}"
        require(os.readlink(checkpoints / "latest") == name
                and max((p.name for p in checkpoints.iterdir() if STEP.fullmatch(p.name)), default="") == name,
                "The stopped attempt does not identify the latest checkpoint; reconcile before resuming")
        saved = snapshot(stack, source, run_id, "checkpoints", name)
        with open_directory("checkpoints", canonical) as dst_checkpoints:
            require(existing_matches(dst_checkpoints, saved), "Missing verified canonical checkpoint")
        meta, config, _, digest = verify_checkpoint(checkpoints / name)
        required = {"job-status.txt", "run-identity.json", "environment.json", "result.json"}
        _, outputs, _ = read_tar(attempt / "outputs.tar.gz", required)
        require(required <= outputs.keys() and outputs["job-status.txt"] == status_text.encode(),
                "Attempt archive is missing control evidence or does not match the published job status")
        identity = json.loads(outputs["run-identity.json"])
        require(all(meta.get(key) == config.get(key) == identity.get(key) == value
                    for key, value in expected.items()), "Checkpoint identity does not match the benchmark and full recipe")
        env = json.loads(outputs["environment.json"])
        require(env.get("job_id") == job and type(env.get("num_cuda_devices")) is int
                and env["num_cuda_devices"] == 1 and env.get("cuda_available") is True,
                "Training process did not verify its single CUDA device")
        result = json.loads(outputs["result.json"])
        resumed = result.get("resumed_from_iterations")
        require(result.get("stopped") is True and result.get("job_id") == meta.get("job_id") == job
                and result.get("checkpoint_job_id") == job and result.get("checkpoint_sha256") == digest
                and result.get("completed_iterations") == step and result.get("target_iterations") == target
                and type(resumed) is int and 0 <= resumed < step and meta.get("resumed_from_iterations") == resumed,
                "The previous allocation did not stop with a verified advancement of its checkpoint")
        return {"run_id": run_id, "previous_job_id": job, "checkpoint_step": step, "checkpoint_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("job_id")
    parser.add_argument("--slurm-record", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = verify_resume(args.runs_root, canonical_destination(), args.run_id, args.job_id,
                                read_json(args.slurm_record), read_json(args.benchmark))
    except (OSError, EOFError, ValueError, KeyError, AttributeError, TypeError, RecursionError,
            subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, f"Cannot verify planned AIR resume: {error}\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

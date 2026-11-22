#!/usr/bin/env python3
"""Verify the private archive and emit only allowlisted public evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from audit_training_metrics import METRICS_MEMBER, learning_curves, parse_metrics


SOURCE_COMMITS = {
    "supermimic": "5fb92297ab4fe3d86629d9f0c635334791bb0d5f",
    "mjlab_research": "379c55e88b1dec73f93943e524a879b2ebbf45c5",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
JOB_ID = re.compile(r"[1-9][0-9]*")
CHECKPOINT_FILES = {"actor.pt", "world_model.pt", "training_state.pt", "run-config.json"}
OPTIONAL_CHECKPOINT_FILES = {"critic.pt"}
MAX_SMALL_FILE = 4 * 1024**2
SLURM_SOURCE = "scontrol -dd -o show job"
LEGACY_SLURM_SOURCE = "scontrol show job -o"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_json(path: Path, value: dict) -> None:
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    # The gpu01 login host has Python 3.10; hashlib.file_digest arrived in 3.11.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fields(data: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in data.splitlines():
        key, sep, value = line.partition("=")
        if sep and key and key not in fields:
            fields[key] = value
        else:
            raise ValueError("invalid or duplicate job status field")
    return fields


def checksums(data: str) -> dict[str, str]:
    entries = {}
    for line in data.splitlines():
        pair = re.fullmatch(r"([0-9a-f]{64}) [ *]([A-Za-z0-9._/-]+)", line)
        require(pair is not None, "invalid checksum manifest line")
        digest, name = pair.groups()
        parts = PurePosixPath(name).parts
        require(name == PurePosixPath(name).as_posix() and parts[0] != "/"
                and ".." not in parts and name != ".", "unsafe checksum path")
        require(name not in entries, "duplicate checksum entry")
        entries[name] = digest
    require(bool(entries), "checksum manifest is empty")
    return entries


def verify_hashes(root: Path, required: set[str]) -> dict[str, str]:
    entries = checksums((root / "SHA256SUMS").read_text(encoding="utf-8"))
    require(required <= entries.keys(), "outer checksum manifest omits required files")
    for name, digest in entries.items():
        path = root / name
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root.resolve()), "unsafe archive artifact")
        require(sha256_file(path) == digest, f"outer checksum mismatch for {name}")
    return entries


def read_tar(path: Path, capture: set[str], *, metrics: dict | None = None
             ) -> tuple[dict[str, str], dict[str, bytes], dict[str, int]]:
    """Read (not extract) the compressed artifacts without loading models into RAM."""
    hashes: dict[str, str] = {}
    contents: dict[str, bytes] = {}
    sizes: dict[str, int] = {}
    seen: set[str] = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            parts = PurePosixPath(name).parts
            if member.isdir() and name in {".", ""}:
                continue
            require(bool(parts) and parts[0] != "/" and ".." not in parts
                    and name == PurePosixPath(name).as_posix(), "unsafe tar member path")
            require(name not in seen, "duplicate tar member")
            seen.add(name)
            if member.isdir():
                continue
            require(member.isfile() and not member.issparse(), "non-regular archive member")
            sizes[name] = member.size
            stream = archive.extractfile(member)
            assert stream is not None
            if metrics is not None and name == METRICS_MEMBER:
                parsed = parse_metrics(stream, member.size)
                metrics.update(parsed)
                hashes[name] = parsed["sha256"]
            elif name in capture:
                require(member.size <= MAX_SMALL_FILE, "control file exceeds size limit")
                data = stream.read()
                require(len(data) == member.size, "truncated archive member")
                contents[name] = data
                hashes[name] = hashlib.sha256(data).hexdigest()
            else:
                hasher = hashlib.sha256()
                size = 0
                while chunk := stream.read(1024**2):
                    hasher.update(chunk)
                    size += len(chunk)
                require(size == member.size, "truncated archive member")
                hashes[name] = hasher.hexdigest()
    return hashes, contents, sizes


def verify_checkpoint(path: Path) -> tuple[dict, dict, dict[str, str], str]:
    outer = verify_hashes(path, {"checkpoint.tar.gz", "checkpoint.json"})
    hashes, contents, sizes = read_tar(path / "checkpoint.tar.gz", {"run-config.json", "SHA256SUMS"})
    required = CHECKPOINT_FILES | {"SHA256SUMS"}
    require(required <= hashes.keys() <= required | OPTIONAL_CHECKPOINT_FILES,
            "checkpoint is missing required files or contains unexpected files")
    internal = checksums(contents["SHA256SUMS"].decode())
    require(internal.keys() == hashes.keys() - {"SHA256SUMS"},
            "inner checksum manifest does not cover all checkpoint contents")
    require(all(hashes[name] == digest for name, digest in internal.items()),
            "inner checkpoint checksum mismatch")
    require(all(sizes[name] > 0 for name in hashes),
            "empty required checkpoint member")
    manifest = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    require(manifest.get("checkpoint_schema_version") == 1, "unsupported checkpoint schema")
    require(manifest.get("archive") == {"sha256": outer["checkpoint.tar.gz"],
                                        "bytes": (path / "checkpoint.tar.gz").stat().st_size},
            "published archive is not the archive specified by the checkpoint manifest")
    require(manifest.get("files") == {key: {"sha256": digest, "bytes": sizes[key]}
                                      for key, digest in internal.items()},
            "published checkpoint manifest does not match inner payload")
    return manifest, json.loads(contents["run-config.json"]), internal, outer["checkpoint.tar.gz"]


def normalize_slurm_record(job_id: str, record: str) -> dict:
    """Reduce one live or privately saved detailed controller record to audit fields."""
    require(JOB_ID.fullmatch(job_id) is not None, "expected a numeric Slurm job ID")
    record = record.strip()
    require(record.startswith("JobId=") and len(record.splitlines()) == 1,
            "expected exactly one Slurm controller record")
    pairs = re.findall(r"(?:^|\s)(JobId|JobState|ExitCode|AllocTRES|NodeList|JOB_GRES)=([^\s]+)", record)
    fields = dict(pairs)
    require(len(fields) == len(pairs), "duplicate field in Slurm controller record")
    require(fields.get("JobId") == job_id, "Slurm controller job ID mismatch")
    state, exit_code = fields.get("JobState", ""), fields.get("ExitCode", "")
    require(re.fullmatch(r"[A-Z_][A-Z_0-9]*\+?", state) is not None
            and re.fullmatch(r"[0-9]+:[0-9]+", exit_code) is not None,
            "missing or invalid Slurm state or exit code")
    no_allocation = {"", "(null)", "None", "N/A", "(none)"}
    allocated = fields.get("AllocTRES", "")
    gpu = 0
    if allocated not in no_allocation:
        counts = {}
        for part in allocated.split(","):
            name, sep, count = part.partition("=")
            if name == "gres/gpu" or name.startswith("gres/gpu:"):
                require(sep and count.isdecimal() and name not in counts, "invalid allocated GPU TRES")
                counts[name] = int(count)
        gpu = counts["gres/gpu"] if "gres/gpu" in counts else sum(counts.values())
    elif state.rstrip("+") != "PENDING" and fields.get("NodeList", "") not in no_allocation:
        # Older Slurm emits JOB_GRES from the actual job resources in -dd
        # output. Its terminal per-node GRES can be empty after teardown.
        # TRES can mean requested resources; TresPerNode is always a request.
        for part in fields.get("JOB_GRES", "").split(","):
            if part == "gpu" or part.startswith("gpu:"):
                count = re.fullmatch(r"gpu:(?:[A-Za-z0-9_.-]+:)?([0-9]+)", part)
                require(count is not None, "invalid allocated GPU GRES")
                gpu += int(count[1])
    # Capture evidence from authoritative controller and discard user, paths and node names.
    return {"schema_version": 1, "source": SLURM_SOURCE, "job_id": job_id,
            "state": state, "exit_code": exit_code,
            "allocated_gpus": gpu}


def capture_slurm(job_id: str) -> dict:
    require(JOB_ID.fullmatch(job_id) is not None, "expected a numeric Slurm job ID")
    record = subprocess.run(["scontrol", "-dd", "-o", "show", "job", job_id],
                            capture_output=True, text=True, check=True, timeout=20).stdout
    return normalize_slurm_record(job_id, record)


def verify_success(attempt: Path, job_id: str, slurm_record: Path,
                   extra: set[str], *, metrics: dict | None = None
                   ) -> tuple[dict[str, bytes], dict[str, str]]:
    require(JOB_ID.fullmatch(job_id) is not None, "expected a numeric Slurm job ID")
    slurm = json.loads(slurm_record.read_text(encoding="utf-8"))
    require(slurm.get("source") in (SLURM_SOURCE, LEGACY_SLURM_SOURCE) and slurm.get("job_id") == job_id
            and slurm.get("allocated_gpus") == 1 and slurm.get("state") == "COMPLETED"
            and slurm.get("exit_code") == "0:0", "one-GPU Slurm completion not verified")
    require((attempt / "SUCCESS").is_file(), "private attempt lacks SUCCESS marker")
    attempt_hashes = verify_hashes(attempt, {"outputs.tar.gz", "job-status.txt"})
    status_text = (attempt / "job-status.txt").read_text(encoding="utf-8")
    status = parse_fields(status_text)
    require(status.get("job_id") == job_id and status.get("exit_code") == "0"
            and status.get("reason") == "completed",
            "private job status does not match the successful Slurm allocation")
    _, outputs, _ = read_tar(attempt / "outputs.tar.gz", {"job-status.txt", *extra}, metrics=metrics)
    require(outputs.get("job-status.txt") == status_text.encode(), "archive job status mismatch")
    require(extra <= outputs.keys(), "attempt archive is missing required output")
    return outputs, attempt_hashes


def verify_run(args: argparse.Namespace) -> dict:
    require(SHA256.fullmatch(args.dataset_sha256) is not None, "invalid expected dataset digest")
    require(SHA256.fullmatch(args.motion_sha256) is not None, "invalid expected packed motion digest")
    require(SHA256.fullmatch(args.code_sha256) is not None, "invalid expected code package digest")
    require(SHA256.fullmatch(args.environment_sha256) is not None, "invalid expected environment digest")
    require(re.fullmatch(r"[0-9a-f]{40}", args.reproduction_commit) is not None,
            "invalid expected reproduction source commit")
    attempt = args.run / "attempts" / args.job_id
    expected_outputs = {"result.json", "run-identity.json", "environment.json"}
    if args.phase == "benchmark":
        expected_outputs.add("checkpoints.jsonl")
    parsed_metrics = {} if args.phase == "full" else None
    outputs, attempt_hashes = verify_success(
        attempt, args.job_id, args.slurm_record, expected_outputs, metrics=parsed_metrics,
    )
    identity = json.loads(outputs["run-identity.json"])
    result = json.loads(outputs["result.json"])
    environment = json.loads(outputs["environment.json"])
    require(environment.get("num_cuda_devices") == 1 and environment.get("cuda_available") is True
            and environment.get("job_id") == args.job_id,
            "training process did not verify one CUDA device")
    motion = identity.get("motion_manifest")
    require(isinstance(motion, dict) and all(
        isinstance(motion.get(key), str) and SHA256.fullmatch(motion[key])
        for key in ("sha256", "clipset_sha256")
    ), "invalid packed motion provenance")
    cohort = {key: motion.get(key) for key in ("sha256", "clipset_sha256", "clips", "frames",
                                               "input_fps", "fps", "selection")}
    require(type(cohort["clips"]) is int and cohort["clips"] > 0
            and type(cohort["frames"]) is int and cohort["frames"] >= 2 * cohort["clips"]
            and cohort["input_fps"] == 120 and cohort["fps"] == 50,
            "invalid packed motion counts or sample rates")
    if args.phase == "smoke":
        require(cohort["clips"] == 64 and cohort["selection"] == "archive-order-smoke",
                "wrong smoke motion selection")
    else:
        selection_id = motion.get("selection_id")
        approximate = motion.get("approximate")
        allowlist_sha = motion.get("allowlist_sha256")
        require(cohort["selection"] == "explicit-allowlist"
                and selection_id in {"supermimic-v3-authors-manifest", "supermimic-v3-appendix-e-approx-v1"}
                and approximate is (selection_id == "supermimic-v3-appendix-e-approx-v1")
                and (approximate or cohort["clips"] == 129_514)
                and isinstance(allowlist_sha, str) and SHA256.fullmatch(allowlist_sha),
                "invalid full motion selection")
        cohort.update(selection_id=selection_id, approximate=approximate, allowlist_sha256=allowlist_sha)
    checkpoint = args.run / "checkpoints" / f"step-{args.iterations:09d}"
    meta, config, hashes, archive_sha = verify_checkpoint(checkpoint)
    expected = {"source_commits": SOURCE_COMMITS, "reproduction_commit": args.reproduction_commit,
                "dataset_sha256": args.dataset_sha256,
                "motion_archive_sha256": args.motion_sha256, "code_sha256": args.code_sha256,
                "motion_manifest": cohort,
                "environment_sha256": args.environment_sha256, "mode": args.phase,
                "num_envs": args.num_envs, "max_iterations": args.iterations,
                "task_id": "SuperMimic-GenCoord-G1", "run_id": args.run.name}
    require(all(meta.get(key) == value == identity.get(key) == config.get(key)
                for key, value in expected.items()), "checkpoint and attempt identity mismatch")
    require(meta.get("completed_iterations") == args.iterations and result.get("stopped") is False
            and result.get("completed_iterations") == result.get("target_iterations") == args.iterations,
            "run did not complete the exact iteration target")
    require(result.get("job_id") == args.job_id and result.get("checkpoint_job_id") == meta.get("job_id")
            and result.get("checkpoint_sha256") == archive_sha, "final result does not bind checkpoint")
    require(meta.get("job_id") == args.job_id or result.get("resumed_from_iterations") == args.iterations,
            "checkpoint job ID mismatch")
    require(config.get("runner_cfg", {}).get("max_iterations") == args.iterations
            and config.get("env_cfg", {}).get("scene", {}).get("num_envs") == args.num_envs,
            "stored runner config does not match the expected iteration target")
    require(config.get("motion_clips") == cohort["clips"]
            and config.get("motion_frames") == cohort["frames"] and config.get("motion_fps") == cohort["fps"],
            "run's loaded motion arrays differ from the packed motion manifest")
    require((args.run / "checkpoints" / "latest").readlink() == Path(checkpoint.name),
            "rolling checkpoint alias does not point at the verified final step")
    elapsed = result.get("loop_elapsed_seconds")
    resumed_from = result.get("resumed_from_iterations")
    require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0
            and type(resumed_from) is int and 0 <= resumed_from <= args.iterations,
            "training throughput inputs are missing or invalid")
    if args.phase == "full":
        require(args.num_envs == 4096 and args.iterations == 320000,
                "full reproduction requires 4096 environments and 320000 iterations")
    retained = []
    if args.phase == "full":
        milestone = config.get("milestone_interval")
        require(milestone == 32000, "unexpected full-run milestone interval")
        for step in range(milestone, args.iterations, milestone):
            kept = args.run / "checkpoints" / f"step-{step:09d}"
            saved, _, _, digest = verify_checkpoint(kept)
            require(saved.get("completed_iterations") == step and saved.get("milestone") is True
                    and all(saved.get(key) == value for key, value in expected.items()),
                    "retained milestone does not match the verified run")
            retained.append({"step": step, "archive_sha256": digest,
                             "archive_bytes": saved["archive"]["bytes"]})
    retained.append({"step": args.iterations, "archive_sha256": archive_sha,
                     "archive_bytes": meta["archive"]["bytes"]})

    audit = {
        "schema_version": 1,
        "phase": args.phase,
        "source_commits": SOURCE_COMMITS,
        "reproduction_commit": args.reproduction_commit,
        "dataset_sha256": args.dataset_sha256,
        "motion_archive_sha256": args.motion_sha256,
        "code_sha256": args.code_sha256,
        "environment_sha256": args.environment_sha256,
        "allocations": [{"job_id": args.job_id, "state": "COMPLETED", "exit_code": "0:0",
                         "allocated_gpus": 1}],
        "num_envs": args.num_envs,
        "checkpoint_step": args.iterations,
        "checkpoint_job_id": meta["job_id"],
        "checkpoint_archive_sha256": archive_sha,
        "checkpoint_archive_bytes": meta["archive"]["bytes"],
        "checkpoint_artifact_sha256": hashes,
        "retained_checkpoints": retained,
        "motions": {"clips": config["motion_clips"], "frames": config["motion_frames"],
                    "fps": config["motion_fps"]},
        "motion_manifest": cohort,
        "outputs_archive_sha256": attempt_hashes["outputs.tar.gz"],
        "verified_checks": ["slurm_terminal_state", "job_status", "success_marker",
                            "source_pins", "dataset_checksum", "packed_motion_checksum",
                            "target_step", "checkpoint_archive_integrity", "milestone_retention"],
        "held_out_evaluation_verified": False,
    }
    if args.phase == "benchmark":
        require(resumed_from == 0, "head-to-head benchmark must start without a checkpoint")
        rows = [json.loads(line) for line in outputs["checkpoints.jsonl"].splitlines()]
        require(len(rows) >= 3 and rows[-1].get("archive", {}).get("sha256") == archive_sha,
                "benchmark does not have enough samples or does not bind the final checkpoint")
        samples = []
        previous = 0
        for row in rows:
            step = row["completed_iterations"]
            gap = row["checkpoint_interval_iterations"]
            compute = row["interval_before_checkpoint_seconds"]
            publish = row["checkpoint_write_seconds"] + row["archive_stageout_seconds"]
            size = row["archive"]["bytes"]
            require(type(step) is int and type(gap) is int and step - previous == gap > 0
                    and type(size) is int and size > 0
                    and math.isfinite(compute) and compute > 0 and math.isfinite(publish) and publish > 0
                    and row.get("job_id") == args.job_id,
                    "benchmark checkpoint samples are invalid")
            samples.append({"step": step, "iterations": gap, "compute_seconds": compute,
                            "checkpoint_seconds": publish, "checkpoint_archive_bytes": size})
            previous = step
        require(previous == args.iterations, "benchmark sample series does not cover target step")
        steady = samples[1:]  # first window can include graph compilation and warmup
        steady_speed = (sum(row["iterations"] for row in steady) /
                        sum(row["compute_seconds"] for row in steady))
        average_checkpoint = sum(row["checkpoint_seconds"] for row in steady) / len(steady)
        audit["benchmark"] = {
            "loop_elapsed_seconds": elapsed,
            "end_to_end_iterations_per_second": args.iterations / elapsed,
            "steady_iterations_per_second": steady_speed,
            "steady_environment_steps_per_second": args.num_envs * steady_speed,
            "mean_steady_checkpoint_seconds": average_checkpoint,
            "estimated_320000_iteration_seconds_at_1000_step_saves": (
                320000 / steady_speed + 320 * average_checkpoint),
            "checkpoint_samples": samples,
        }
    elif args.phase == "full":
        log_interval = config["runner_cfg"].get("log_interval")
        audit["learning_curves"] = learning_curves(
            parsed_metrics or None, resumed_from, args.iterations, log_interval, elapsed,
        )
    return audit


def verify_full_benchmark_binding(full: dict, benchmark: dict) -> None:
    """Gate the finished full audit on the independently audited benchmark inputs."""
    from publish_phase_summary import series_benchmark, verify_phase

    previous = verify_phase(benchmark, "benchmark")
    series_benchmark(benchmark, previous)
    current = verify_phase(full, "full")
    require(previous["finished_job"] not in (current["finished_job"], full["checkpoint_job_id"]),
            "Full training must be a separate allocation from the benchmark")
    keys = ("source_commits", "dataset_sha256", "motion_archive_sha256", "motion_manifest",
            "code_sha256", "environment_sha256", "reproduction_commit")
    require(all(full[key] == benchmark[key] for key in keys),
            "Full training does not match the verified benchmark's inputs and provenance")


def attach_evaluation(args: argparse.Namespace) -> dict:
    from evaluate_heldout import MANIFEST, REPOSITORY, REVISION, SEED, SUITE_ID, suite_manifest

    audit = json.loads(args.training_audit.read_text(encoding="utf-8"))
    require(audit.get("phase") == "full" and audit.get("checkpoint_step") == 320000
            and audit.get("source_commits") == SOURCE_COMMITS,
            "held-out evaluation must attach to a verified full run")
    outputs, hashes = verify_success(args.evaluation, args.job_id, args.slurm_record, {"metrics.json"})
    outcome = json.loads(outputs["metrics.json"])
    require(outcome.get("schema_version") == 1 and outcome.get("protocol") == SUITE_ID
            and outcome.get("evaluation_job_id") == args.job_id
            and outcome.get("training_job_id") == audit.get("checkpoint_job_id")
            and outcome.get("checkpoint_step") == audit.get("checkpoint_step")
            and outcome.get("checkpoint_archive_sha256") == audit.get("checkpoint_archive_sha256")
            and outcome.get("source_commits") == SOURCE_COMMITS
            and outcome.get("training_dataset_sha256") == audit.get("dataset_sha256"),
            "held-out evidence does not match the audited full checkpoint")
    count = len(suite_manifest())
    require(SHA256.fullmatch(args.suite_sha256) is not None
            and args.suite_sha256 != audit.get("dataset_sha256"), "invalid held-out archive SHA-256")
    suite = {"source": REPOSITORY, "revision": REVISION, "archive_sha256": args.suite_sha256,
             "clip_manifest_sha256": sha256_file(MANIFEST), "clips": count}
    require(outcome.get("suite") == suite, "evaluation did not use the complete pinned suite")
    cfg = outcome["rollout_config"]
    require(cfg.get("nominal") is True and cfg.get("control_hz") == 50 and cfg.get("seed") == SEED
            and cfg.get("foot_tangential_friction") == 1.0 and cfg.get("max_step_reward") == 0.1
            and cfg.get("return_normalization") == "zero unsimulated tail, per-clip reward ceiling",
            "unexpected held-out rollout protocol")
    metrics = outcome["metrics"]
    total, complete, failed = (metrics[key] for key in ("evaluated_clips", "completed_clips", "failed_clips"))
    actual_steps, reference_steps = (metrics[key] for key in ("evaluated_steps", "reference_steps"))
    require(all(type(value) is int for value in (total, complete, failed, actual_steps, reference_steps))
            and total == complete + failed == count and 0 <= complete <= count
            and metrics.get("completion_rate") == complete / count
            and count <= actual_steps <= reference_steps
            and (complete != count or actual_steps == reference_steps),
            "held-out metrics counts are inconsistent")
    value_keys = ("mean_normalized_return", "mean_root_position_error_m",
                  "mean_heading_aligned_body_position_error_m")
    for key in value_keys:
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value)
                and (value <= 1.0 + 1e-5 if key == "mean_normalized_return" else value >= 0),
                "held-out metric not finite or out of range")
    allowed_metrics = ("evaluated_clips", "completed_clips", "failed_clips", "completion_rate",
                       "evaluated_steps", "reference_steps", *value_keys)
    audit["held_out_evaluation_verified"] = True
    audit["verified_checks"] = list(dict.fromkeys([*audit["verified_checks"], "held_out_evaluation"]))
    audit["held_out_evaluation"] = {
        "job_id": args.job_id, "state": "COMPLETED", "exit_code": "0:0", "allocated_gpus": 1,
        "suite": suite, "metrics": {key: metrics[key] for key in allowed_metrics},
        "metrics_sha256": hashlib.sha256(outputs["metrics.json"]).hexdigest(),
        "outputs_archive_sha256": hashes["outputs.tar.gz"],
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture-slurm", help="save controller evidence on gpu01")
    capture.add_argument("job_id")
    capture.add_argument("--controller-record", type=Path,
                         help="use privately saved stdout of scontrol -dd -o show job instead of querying Slurm")
    capture.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify checkpoint and successful Slurm attempt")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--slurm-record", type=Path, required=True)
    verify.add_argument("--phase", choices=["smoke", "benchmark", "full"], required=True)
    verify.add_argument("--dataset-sha256", required=True)
    verify.add_argument("--motion-sha256", required=True)
    verify.add_argument("--code-sha256", required=True)
    verify.add_argument("--environment-sha256", required=True)
    verify.add_argument("--reproduction-commit", required=True)
    verify.add_argument("--num-envs", type=int, required=True)
    verify.add_argument("--iterations", type=int, required=True)
    verify.add_argument("--benchmark-audit", type=Path,
                        help="for full training, also verify that inputs match the successful benchmark")
    verify.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("attach-evaluation", help="bind completed held-out evaluation to full audit")
    evaluation.add_argument("--training-audit", type=Path, required=True)
    evaluation.add_argument("--evaluation", type=Path, required=True)
    evaluation.add_argument("--slurm-record", type=Path, required=True)
    evaluation.add_argument("--suite-sha256", required=True)
    evaluation.add_argument("--job-id", required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "capture-slurm":
            result = (capture_slurm(args.job_id) if args.controller_record is None else
                      normalize_slurm_record(args.job_id, args.controller_record.read_text(encoding="utf-8")))
        elif args.command == "verify":
            if args.benchmark_audit is not None:
                require(args.phase == "full" and args.output.resolve() != args.benchmark_audit.resolve(),
                        "Benchmark evidence can only gate full training and must not be replaced")
            result = verify_run(args)
            if args.benchmark_audit is not None:
                from publish_phase_summary import read_json

                verify_full_benchmark_binding(result, read_json(args.benchmark_audit))
        else:
            result = attach_evaluation(args)
        atomic_json(args.output, result)
    except (ValueError, KeyError, OSError, tarfile.TarError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    main()

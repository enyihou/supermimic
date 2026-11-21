"""AIR checkpoint overlay for the pinned SuperMimic trainer.

Training always uses node-local files. Only closed, hashed checkpoint archives
reach the private run directory. The simulator itself is deliberately reset on
resume; the replay data and learning state survive, with an episode boundary
inserted so no training window joins the two allocations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import tarfile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_sums(path: Path, files: dict[str, str]) -> None:
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())))


def checkpoint_name(step: int) -> str:
    return f"step-{step:09d}"


def latest_checkpoint(root: Path, *, verify_archive: bool) -> tuple[Path, dict]:
    path = root / os.readlink(root / "latest")
    if path.parent != root or not re.fullmatch(r"step-[0-9]{9}", path.name):
        raise RuntimeError("Latest checkpoint target is outside the run")
    return path, read_checkpoint(path, verify_archive=verify_archive)


def read_checkpoint(path: Path, *, verify_archive: bool) -> dict:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"Latest checkpoint is not a published checkpoint directory: {path}")
    for name in ("checkpoint.json", "checkpoint.tar.gz", "SHA256SUMS"):
        if not (path / name).is_file() or (path / name).is_symlink():
            raise RuntimeError(f"Missing or non-regular published checkpoint file: {name}")
    manifest = json.loads((path / "checkpoint.json").read_text())
    if manifest["checkpoint_schema_version"] != 1:
        raise RuntimeError("Unsupported checkpoint schema")
    if checkpoint_name(manifest["completed_iterations"]) != path.name:
        raise RuntimeError("Latest checkpoint directory disagrees with the manifest step")
    archive = path / "checkpoint.tar.gz"
    expected = {
        "checkpoint.tar.gz": manifest["archive"]["sha256"],
        "checkpoint.json": sha256(path / "checkpoint.json"),
    }
    if (path / "SHA256SUMS").read_text() != "".join(
        f"{digest}  {name}\n" for name, digest in sorted(expected.items())
    ):
        raise RuntimeError("Published checkpoint checksum manifest mismatch")
    if archive.stat().st_size != manifest["archive"]["bytes"]:
        raise RuntimeError("Published checkpoint archive size mismatch")
    if verify_archive and sha256(archive) != expected["checkpoint.tar.gz"]:
        raise RuntimeError("Published checkpoint archive hash mismatch")
    return manifest


def unpack_checkpoint(path: Path, manifest: dict, work: Path) -> None:
    """Copy once to scratch, verify, then extract exactly the expected files."""
    archive = work / "restore.tar.gz"
    scratch = work / "restore"
    required = {"world_model.pt", "actor.pt", "training_state.pt", "run-config.json"}
    if set(manifest["files"]) not in (required, required | {"critic.pt"}):
        raise RuntimeError("Unexpected checkpoint payload file manifest")
    scratch.mkdir()
    shutil.copyfile(path / "checkpoint.tar.gz", archive)
    if sha256(archive) != manifest["archive"]["sha256"]:
        raise RuntimeError("Checkpoint changed in transit to scratch")
    expected = {*manifest["files"], "SHA256SUMS"}
    seen: set[str] = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise RuntimeError(f"Unexpected checkpoint archive member {member.name!r}")
            seen.add(member.name)
            if member.name != "SHA256SUMS" and member.size != manifest["files"][member.name]["bytes"]:
                raise RuntimeError(f"Checkpoint member-size mismatch: {member.name}")
            if member.name == "SHA256SUMS" and member.size > 4096:
                raise RuntimeError("Unreasonable inner checkpoint checksum manifest size")
            with bundle.extractfile(member) as source, (scratch / member.name).open("xb") as target:
                shutil.copyfileobj(source, target)
    if seen != expected:
        raise RuntimeError("Checkpoint archive is incomplete")
    sums = {}
    for name, meta in manifest["files"].items():
        artifact = scratch / name
        sums[name] = sha256(artifact)
        if sums[name] != meta["sha256"] or artifact.stat().st_size != meta["bytes"]:
            raise RuntimeError(f"Invalid checkpoint payload: {name}")
    if (scratch / "SHA256SUMS").read_text() != "".join(
        f"{digest}  {name}\n" for name, digest in sorted(sums.items())
    ):
        raise RuntimeError("Inner checkpoint checksum manifest mismatch")
    archive.unlink()


def publish_checkpoint(
    local: Path, root: Path, identity: dict, step: int, job_id: str, details: dict
) -> dict:
    """Publish a verified immutable step, then atomically move the latest pointer."""
    started = time.monotonic()
    archive = local.parent / "checkpoint.tar.gz"
    files = {
        artifact.name: {"bytes": artifact.stat().st_size, "sha256": sha256(artifact)}
        for artifact in sorted(local.iterdir())
    }
    write_sums(local / "SHA256SUMS", {name: meta["sha256"] for name, meta in files.items()})
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for name in sorted([*files, "SHA256SUMS"]):
            bundle.add(local / name, arcname=name, recursive=False)
    archive_hash = sha256(archive)
    archive_size = archive.stat().st_size

    root.mkdir(exist_ok=True)
    name = checkpoint_name(step)
    destination = root / name
    partial = Path(tempfile.mkdtemp(prefix=f".{name}.{job_id}.", dir=root))
    pointer = root / f".latest.{job_id}"
    try:
        if destination.exists():
            # A crash may have completed the directory rename but not the final
            # pointer update. Never overwrite an existing immutable checkpoint.
            raise RuntimeError(f"Checkpoint step already exists: {destination}; inspect latest before recovery")
        stored = partial / "checkpoint.tar.gz"
        shutil.copyfile(archive, stored)
        sync_path(stored)
        if sha256(stored) != archive_hash:
            raise RuntimeError("Stage-out checksum mismatch")
        manifest = {
            **identity,
            **details,
            "checkpoint_schema_version": 1,
            "completed_iterations": step,
            "job_id": job_id,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "archive": {"sha256": archive_hash, "bytes": archive_size},
            "files": files,
            "archive_stageout_seconds": time.monotonic() - started,
        }
        write_json(partial / "checkpoint.json", manifest)
        write_sums(
            partial / "SHA256SUMS",
            {"checkpoint.tar.gz": archive_hash, "checkpoint.json": sha256(partial / "checkpoint.json")},
        )
        for path in (partial / "checkpoint.json", partial / "SHA256SUMS", partial):
            sync_path(path)
        partial.rename(destination)
        sync_path(root)
        pointer.symlink_to(name)
        pointer.replace(root / "latest")
        sync_path(root)
    finally:
        pointer.unlink(missing_ok=True)
        if partial.exists():
            shutil.rmtree(partial)
        archive.unlink(missing_ok=True)
    return manifest


def find_resume_checkpoint(root: Path, identity: dict) -> tuple[Path, dict]:
    """Recover a committed directory even if the last pointer update was interrupted."""
    greatest = max(
        (path for path in root.iterdir() if re.fullmatch(r"step-[0-9]{9}", path.name)),
        default=None,
    )
    if greatest is None:
        raise RuntimeError("No published checkpoint to resume")
    link = root / "latest"
    if not link.is_symlink() or os.readlink(link) != greatest.name:
        manifest = read_checkpoint(greatest, verify_archive=True)
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Newer published checkpoint does not match this run")
        with tempfile.TemporaryDirectory(prefix=".verify.", dir=root) as temp:
            pointer = Path(temp) / "latest"
            pointer.symlink_to(greatest.name)
            pointer.replace(link)
            sync_path(root)
    path, manifest = latest_checkpoint(root, verify_archive=False)
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Resume configuration, code or dataset differs from the checkpoint")
    return path, manifest


def _optimizer_groups(runner) -> dict:
    policy = runner.policy_learner
    return {
        "world_model": runner.world_model_learner.optimizers,
        "actor": policy.actor_optimizers,
        "critic": getattr(policy, "critic_optimizers", []),
    }


def _load_optimizers(runner, states: dict) -> None:
    import torch

    for name, optimizers in _optimizer_groups(runner).items():
        for optimizer, state in zip(optimizers, states[name], strict=True):
            # load_state_dict moves per-parameter state, but not param-group
            # hyperparameters. Fused capturable Adam needs its LR on device.
            for current, restored in zip(optimizer.param_groups, state["param_groups"], strict=True):
                for key, value in current.items():
                    if torch.is_tensor(value) and torch.is_tensor(restored.get(key)):
                        restored[key] = restored[key].to(device=value.device, dtype=value.dtype)
            optimizer.load_state_dict(state)


class CheckpointManager:
    def __init__(self, runner) -> None:
        self.runner = runner
        self.root = Path(os.environ["SM_RUN_DIR"]) / "checkpoints"
        self.output = Path(os.environ["SM_OUTPUT_DIR"])
        self.work = Path(os.environ["SM_WORK_DIR"])
        self.job_id = os.environ["SLURM_JOB_ID"]
        self.identity = json.loads((self.output / "run-identity.json").read_text())
        self.resume = os.environ["SM_RESUME"] == "resume"
        self.milestone = self.identity["milestone_interval"]
        self.max_iterations = runner.cfg.max_iterations
        self.latest = None
        self.start_step = 0
        self.loop_started = 0.0
        self.last_published = 0.0
        self.stop_requested = False
        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            signal.signal(sig, self._request_stop)
        cfg = {
            **self.identity,
            "task_id": runner.task_id,
            "env_cfg": asdict(runner.env.cfg),
            "runner_cfg": asdict(runner.cfg),
            "motion_clips": len(runner.motion.motion_names),
            "motion_frames": runner.motion.time_step_total,
            "motion_fps": runner.motion.fps,
            "resume_semantics": "learning state retained; new simulator episodes via a marked replay boundary",
        }
        # Environment config includes types and callables. Its complete private
        # repr is for audit; resume identity compares exact inputs and CLI.
        (self.output / "run-config.json").write_text(
            json.dumps(cfg, default=repr, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    def _request_stop(self, *_args) -> None:
        self.stop_requested = True

    def restore(self, ratio) -> int:
        if not self.resume:
            return 0
        import numpy as np
        import torch

        path, self.latest = find_resume_checkpoint(self.root, self.identity)
        unpack_checkpoint(path, self.latest, self.work)
        folder = self.work / "restore"
        runner = self.runner
        def load(name: str):
            return torch.load(folder / name, weights_only=True, mmap=True, map_location="cpu")

        runner._wm.load_bundle(load("world_model.pt"))
        runner.policy_learner.actor.load_state_dict(load("actor.pt"))
        critic = getattr(runner.policy_learner, "critic", None)
        if critic is not None:
            critic.load_state_dict(load("critic.pt"))
        state = load("training_state.pt")
        if state["completed_iterations"] != self.latest["completed_iterations"]:
            raise RuntimeError("Checkpoint payload step differs from manifest")
        _load_optimizers(runner, state["optimizers"])
        ratio._prev = state["ratio_prev"]
        replay = runner.replay_buffer
        replay._step, replay.num_transitions = state["replay_step"], state["replay_count"]
        if not 0 <= replay._step < replay.buffer_size or not 0 <= replay.num_transitions <= replay.buffer_size:
            raise RuntimeError("Invalid replay cursor")
        if set(replay._storage.keys()) != set(state["replay"]):
            raise RuntimeError("Replay field schema changed")
        for key, tensor in state["replay"].items():
            replay._storage[key].copy_(tensor)
        if replay.num_transitions:
            replay._storage["truncated"][:, (replay._step - 1) % replay.buffer_size] = True
        sampler = runner._sampler
        if sampler is not None:
            for name in ("value", "count", "effort"):
                getattr(sampler, name).copy_(state["sampler"][name])
            runner.motion_cmd.set_sampling_distribution(state["sampler"]["probs"].to(runner.device))

        random.setstate(state["python_rng"])
        algo, keys, pos, has_gauss, cached_gaussian = state["numpy_rng"]
        np.random.set_state((algo, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian))
        torch.set_rng_state(state["torch_rng"])
        if torch.device(runner.device).type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], runner.device)

        self.start_step = state["completed_iterations"]
        if not 0 < self.start_step <= self.max_iterations:
            raise RuntimeError("Restored completed iteration count is out of range")
        shutil.rmtree(folder)
        print(f"[AIR] Restored {self.start_step} completed iterations from job {self.latest['job_id']}", flush=True)
        return self.start_step

    def begin(self) -> None:
        self.loop_started = self.last_published = time.monotonic()

    def log_metrics(self, completed: int, row: dict) -> None:
        metrics = {
            key: float(value) if math.isfinite(value) else None
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "completed_iterations": completed,
                        "loop_elapsed_seconds": time.monotonic() - self.loop_started,
                        "metrics": metrics,
                    },
                    allow_nan=False,
                ) + "\n"
            )

    def save(self, completed: int, ratio) -> None:
        import numpy as np
        import torch

        runner = self.runner
        started = time.monotonic()
        scratch = self.work / "saving"
        scratch.mkdir()
        try:
            torch.save(runner._wm.bundle(), scratch / "world_model.pt")
            torch.save(runner.policy_learner.actor.state_dict(), scratch / "actor.pt")
            critic = getattr(runner.policy_learner, "critic", None)
            if critic is not None:
                torch.save(critic.state_dict(), scratch / "critic.pt")
            algo, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
            replay = runner.replay_buffer
            sampler = runner._sampler
            torch.save(
                {
                    "completed_iterations": completed,
                    "ratio_prev": ratio._prev,
                    "optimizers": {
                        name: [optimizer.state_dict() for optimizer in optimizers]
                        for name, optimizers in _optimizer_groups(runner).items()
                    },
                    "replay_step": replay._step,
                    "replay_count": replay.num_transitions,
                    "replay": dict(replay._storage.items()),
                    "sampler": {
                        "value": sampler.value,
                        "count": sampler.count,
                        "effort": sampler.effort,
                        "probs": runner.motion_cmd._external_probs,
                    } if sampler is not None else None,
                    "python_rng": random.getstate(),
                    "numpy_rng": (algo, keys.tolist(), pos, has_gauss, cached_gaussian),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state(runner.device)
                        if torch.device(runner.device).type == "cuda" else None
                    ),
                },
                scratch / "training_state.pt",
            )
            shutil.copyfile(self.output / "run-config.json", scratch / "run-config.json")
            previous = self.latest
            self.latest = publish_checkpoint(
                scratch, self.root, self.identity, completed, self.job_id,
                {
                    "parent_checkpoint": {
                        "job_id": previous["job_id"],
                        "completed_iterations": previous["completed_iterations"],
                        "archive_sha256": previous["archive"]["sha256"],
                    } if previous is not None else None,
                    "checkpoint_write_seconds": time.monotonic() - started,
                    "interval_before_checkpoint_seconds": started - self.last_published,
                    "checkpoint_interval_iterations": completed - (
                        previous["completed_iterations"] if previous is not None else self.start_step
                    ),
                    "resumed_from_iterations": self.start_step,
                    "loop_elapsed_seconds": started - self.loop_started,
                    "milestone": completed % self.milestone == 0 or completed == self.max_iterations,
                },
            )
            self.last_published = time.monotonic()
            with (self.output / "checkpoints.jsonl").open("a") as stream:
                stream.write(json.dumps(self.latest, sort_keys=True, allow_nan=False) + "\n")
            print(
                f"[AIR] Published checkpoint {completed}/{self.max_iterations} "
                f"({self.latest['archive']['bytes']} bytes, "
                f"{self.latest['checkpoint_write_seconds'] + self.latest['archive_stageout_seconds']:.1f}s)",
                flush=True,
            )
            for candidate in self.root.iterdir():
                if not re.fullmatch(r"step-[0-9]{9}", candidate.name) or candidate.is_symlink():
                    continue
                iteration = int(candidate.name.removeprefix("step-"))
                if iteration < completed and iteration % self.milestone and iteration != self.max_iterations:
                    shutil.rmtree(candidate)
        finally:
            shutil.rmtree(scratch)

    def finish(self, completed: int) -> None:
        write_json(
            self.output / "result.json",
            {
                "job_id": self.job_id,
                "completed_iterations": completed,
                "target_iterations": self.max_iterations,
                "stopped": self.stop_requested and completed < self.max_iterations,
                "resumed_from_iterations": self.start_step,
                "checkpoint_job_id": self.latest["job_id"] if self.latest else None,
                "checkpoint_sha256": self.latest["archive"]["sha256"] if self.latest else None,
                "loop_elapsed_seconds": time.monotonic() - self.loop_started,
            },
        )

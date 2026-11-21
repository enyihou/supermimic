"""Local backup tests: no production AIR paths, XFS commands, or GPU work."""

import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backup_air_attempt import backup, canonical_destination, check_quota, fail_if_present_rename, parse_quota_kib


def private_file(path, value):
    path.write_bytes(value.encode() if isinstance(value, str) else value)
    path.chmod(0o600)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sums(folder, *names):
    private_file(folder / "SHA256SUMS", "".join(
        f"{sha((folder / name).read_bytes())}  {name}\n" for name in names))


def archive(name, contents):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(contents)
        tar.addfile(info, io.BytesIO(contents))
    return buffer.getvalue()


class CanonicalBackup(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.source, self.destination = root / "shared", root / "canonical"
        self.src_run = self.source / "my-run"
        self.dst_run = self.destination / "my-run"
        for path in (self.source, self.destination, self.src_run, self.src_run / "attempts", self.src_run / "checkpoints"):
            path.mkdir(mode=0o700)
        self.quota_calls = []
        self.attempt = self.write_attempt("123")

    def quota(self, target, size):
        self.assertEqual(target, self.destination)
        self.quota_calls.append(size)

    def write_attempt(self, job, *, success=False):
        attempt = self.src_run / "attempts" / job
        attempt.mkdir(mode=0o700)
        code, reason = (0, "completed") if success else (124, "walltime_warning")
        status = (f"job_id={job}\nexit_code={code}\nreason={reason}\nrun_id=my-run\nmode=full\n"
                  "last_published_iteration=32000\n")
        private_file(attempt / "job-status.txt", status)
        private_file(attempt / "outputs.tar.gz", archive("job-status.txt", status.encode()))
        sums(attempt, "outputs.tar.gz", "job-status.txt")
        if success:
            private_file(attempt / "SUCCESS", "Verified target checkpoint and job outputs.\n")
        return attempt

    def write_checkpoint(self, step):
        folder = self.src_run / "checkpoints" / f"step-{step:09d}"
        folder.mkdir(mode=0o700)
        private_file(folder / "checkpoint.tar.gz", archive("checkpoint-fixture", b"opaque training state"))
        data = (folder / "checkpoint.tar.gz").read_bytes()
        private_file(folder / "checkpoint.json", json.dumps({
            "checkpoint_schema_version": 1, "job_id": "123", "completed_iterations": step,
            "run_id": "my-run", "mode": "full", "archive": {"bytes": len(data), "sha256": sha(data)},
        }))
        sums(folder, "checkpoint.json", "checkpoint.tar.gz")
        return folder

    def go(self, *steps, job="123", quota=None):
        return backup(self.source, self.destination, "my-run", job, list(steps), quota or self.quota)

    def verify_copy(self, source, destination):
        entries = sorted(path.name for path in source.iterdir())
        self.assertEqual(entries, sorted(path.name for path in destination.iterdir()))
        self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
        for name in entries:
            self.assertEqual((source / name).read_bytes(), (destination / name).read_bytes())
            self.assertEqual((destination / name).stat().st_mode & 0o777, 0o600)

    def test_failed_attempt_and_explicit_milestones_copy_idempotently(self):
        checkpoint = self.write_checkpoint(32_000)
        next_checkpoint = self.write_checkpoint(64_000)
        (checkpoint.parent / "latest").symlink_to(next_checkpoint.name)
        self.assertEqual(self.go(32_000, 64_000), [
            ("attempts/123", True), ("checkpoints/step-000032000", True), ("checkpoints/step-000064000", True),
        ])
        self.verify_copy(self.attempt, self.dst_run / "attempts" / "123")
        self.verify_copy(checkpoint, self.dst_run / "checkpoints" / checkpoint.name)
        self.verify_copy(next_checkpoint, self.dst_run / "checkpoints" / next_checkpoint.name)
        self.assertFalse((self.dst_run / "checkpoints" / "latest").exists())
        self.assertEqual(self.go(32_000, 64_000), [
            ("attempts/123", False), ("checkpoints/step-000032000", False), ("checkpoints/step-000064000", False),
        ])

    def test_success_is_preserved_and_contradictory_marker_rejected(self):
        completed = self.write_attempt("124", success=True)
        self.assertEqual(self.go(job="124"), [("attempts/124", True)])
        self.verify_copy(completed, self.dst_run / "attempts" / "124")
        private_file(self.attempt / "SUCCESS", "Verified target checkpoint and job outputs.\n")
        with self.assertRaisesRegex(ValueError, "SUCCESS marker disagrees"):
            self.go()
        self.assertFalse((self.dst_run / "attempts" / "123").exists())

    def test_checksum_mismatch_never_publishes_a_partial_dir(self):
        original = (self.attempt / "outputs.tar.gz").read_bytes()
        private_file(self.attempt / "outputs.tar.gz", original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "Source checksum mismatch: outputs"):
            self.go()
        self.assertEqual(list((self.dst_run / "attempts").iterdir()), [])

    def test_extra_names_symlinks_and_missing_steps_fail_before_backup(self):
        extra = self.attempt / "private-extras"
        private_file(extra, b"not allowed")
        with self.assertRaisesRegex(ValueError, "Unexpected backup publication file"):
            self.go()
        extra.unlink()
        with self.assertRaises(FileNotFoundError):
            self.go(42)
        member = self.attempt / "outputs.tar.gz"
        member.unlink()
        member.symlink_to("job-status.txt")
        with self.assertRaises(OSError):
            self.go()
        self.assertFalse(self.dst_run.exists())

    def test_conflicting_destination_is_never_replaced(self):
        self.go()
        original = self.dst_run / "attempts" / "123" / "outputs.tar.gz"
        old = original.read_bytes()
        changed = old[:-1] + bytes([old[-1] ^ 1])
        private_file(original, changed)
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.go()
        self.assertEqual(original.read_bytes(), changed)

    def test_closed_source_step_can_finish_copy_during_pruning(self):
        checkpoint = self.write_checkpoint(12_000)
        original = (checkpoint / "checkpoint.tar.gz").read_bytes()
        def prune_after_snapshot(target, size):
            self.quota(target, size)
            if checkpoint.exists():
                for artifact in checkpoint.iterdir():
                    artifact.unlink()
                checkpoint.rmdir()
        self.go(12_000, quota=prune_after_snapshot)
        self.assertEqual((self.dst_run / "checkpoints" / checkpoint.name / "checkpoint.tar.gz").read_bytes(), original)

    def test_capacity_failure_leaves_no_published_unit(self):
        self.write_checkpoint(32_000)
        def reject_bytes(_, size):
            if size:
                raise ValueError("test quota limit")
        with self.assertRaisesRegex(ValueError, "test quota limit"):
            self.go(32_000, quota=reject_bytes)
        self.assertFalse((self.dst_run / "attempts" / "123").exists())
        self.assertFalse((self.dst_run / "checkpoints" / "step-000032000").exists())

    def test_noclobber_rename_refuses_empty_destination(self):
        (self.destination / "before").mkdir()
        (self.destination / "existing").mkdir()
        fd = os.open(self.destination, os.O_DIRECTORY | os.O_RDONLY)
        try:
            with self.assertRaises(FileExistsError):
                fail_if_present_rename(fd, "before", "existing")
            self.assertTrue((self.destination / "before").is_dir())
            self.assertTrue((self.destination / "existing").is_dir())
        finally:
            os.close(fd)

    def test_quota_wrapping_ambiguity_and_insufficiency(self):
        self.assertEqual(parse_quota_kib("Filesystem Blocks Quota Limit\n/dev/sda1 13 20 40 00\n"), (13, 40))
        self.assertEqual(parse_quota_kib("/dev/mapper/wider-device\n   13 20 40 00\n"), (13, 40))
        for malformed in ("", "/dev/sda1 12 unknown\n", "/dev/a 1 2 3\n/dev/b 1 2 3\n"):
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                parse_quota_kib(malformed)
        with patch("backup_air_attempt.subprocess.check_output", return_value="/dev/xfs 13 20 40 00\n"):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                check_quota(self.destination, 1)
        with patch("backup_air_attempt.Path.is_file", return_value=True), patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
            with self.assertRaisesRegex(ValueError, "gpu01 login host"):
                canonical_destination()


if __name__ == "__main__":
    unittest.main()

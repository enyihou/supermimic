"""Test the login-side resume gate with small, private synthetic publications."""

import fcntl
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from air_checkpoint import publish_checkpoint, write_sums
from audit_official import SLURM_SOURCE, sha256_file
from backup_air_attempt import backup
from test_publish_phase_summary import phase
from verify_air_resume import IDENTITY_FIELDS, verify_resume


class VerifyAirResumeTest(unittest.TestCase):
    job = "103"
    run_id = "test-full"
    step = 32211

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs, self.canonical = (self.root / name for name in ("shared", "canonical"))
        self.run = self.runs / self.run_id
        self.checkpoints = self.run / "checkpoints"
        self.attempts = self.run / "attempts"
        self.attempt = self.attempts / self.job
        for directory in (self.runs, self.canonical, self.run, self.checkpoints, self.attempts, self.attempt):
            directory.mkdir(mode=0o700)
        self.lock = self.run / "lock"
        self.lock.touch(mode=0o600)
        self.benchmark = phase("benchmark")
        self.slurm = {"schema_version": 1, "source": SLURM_SOURCE, "job_id": self.job, "allocated_gpus": 1,
                      "state": "FAILED", "exit_code": "124:0"}
        self.identity = {key: self.benchmark[key] for key in IDENTITY_FIELDS}
        self.identity.update(run_id=self.run_id, mode="full", task_id="SuperMimic-GenCoord-G1",
                             num_envs=4096, max_iterations=320000, save_interval=1000,
                             milestone_interval=32000, seed=42,
                             overrides=["--num-envs", "4096", "--env.seed", "42", "--runner.max-iterations",
                                        "320000", "--runner.save-interval", "1000"])
        local = self.root / "checkpoint"
        local.mkdir(mode=0o700)
        for name in ("world_model.pt", "actor.pt", "training_state.pt"):
            (local / name).write_bytes(name.encode())
        (local / "run-config.json").write_text(json.dumps(self.identity))
        meta = publish_checkpoint(local, self.checkpoints, self.identity, self.step, self.job,
                                  {"resumed_from_iterations": 0})
        self.digest = meta["archive"]["sha256"]
        self.result = {"stopped": True, "job_id": self.job, "checkpoint_job_id": self.job,
                       "checkpoint_sha256": self.digest, "resumed_from_iterations": 0,
                       "completed_iterations": self.step, "target_iterations": 320000}
        for artifact in (self.checkpoints / f"step-{self.step:09d}").iterdir():
            artifact.chmod(0o600)

    def publish_attempt_and_backup(self, *, reason="walltime_warning", copy_checkpoint=True):
        status = self.attempt / "job-status.txt"
        status.write_text(f"job_id={self.job}\nexit_code=124\nreason={reason}\nrun_id={self.run_id}\n"
                          f"mode=full\nlast_published_iteration={self.step}\n")
        outputs = self.attempt / "outputs.tar.gz"
        with tarfile.open(outputs, "w:gz") as archive:
            archive.add(status, arcname="./job-status.txt")
            for name, data in {"run-identity.json": self.identity, "result.json": self.result,
                               "environment.json": {"job_id": self.job, "num_cuda_devices": 1,
                                                    "cuda_available": True}}.items():
                path = self.root / name
                path.write_text(json.dumps(data))
                archive.add(path, arcname=f"./{name}")
        write_sums(self.attempt / "SHA256SUMS", {path.name: sha256_file(path) for path in (status, outputs)})
        for artifact in self.attempt.iterdir():
            artifact.chmod(0o600)
        backup(self.runs, self.canonical, self.run_id, self.job, [self.step] if copy_checkpoint else [],
               lambda *_: None)

    def verify(self):
        return verify_resume(self.runs, self.canonical, self.run_id, self.job, self.slurm, self.benchmark)

    def test_verified_planned_stop_can_resume(self):
        self.publish_attempt_and_backup()
        self.assertEqual(self.verify(), {"run_id": self.run_id, "previous_job_id": self.job,
                                        "checkpoint_step": self.step, "checkpoint_sha256": self.digest})

    def test_watcher_or_controller_loss_does_not_authorize_resume(self):
        self.publish_attempt_and_backup()
        for fields in ({"state": "RUNNING"}, {"state": "TIMEOUT"}, {"exit_code": "1:0"}, {"job_id": "999"},
                       {"allocated_gpus": True}, {"allocated_gpus": 0}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "planned walltime"):
                verify_resume(self.runs, self.canonical, self.run_id, self.job, self.slurm | fields, self.benchmark)

    def test_live_lock_refuses_resume(self):
        self.publish_attempt_and_backup()
        with self.lock.open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            with self.assertRaisesRegex(ValueError, "still owns the run lock"):
                self.verify()

    def test_missing_or_corrupt_backup_refuses_resume(self):
        self.publish_attempt_and_backup(copy_checkpoint=False)
        with self.assertRaises(FileNotFoundError):
            self.verify()
        backup(self.runs, self.canonical, self.run_id, self.job, [self.step], lambda *_: None)
        canonical = self.canonical / self.run_id / "checkpoints" / f"step-{self.step:09d}" / "checkpoint.tar.gz"
        original = canonical.read_bytes()
        canonical.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, "differs"):
            self.verify()

    def test_newer_unreconciled_checkpoint_refuses_resume(self):
        self.publish_attempt_and_backup()
        (self.checkpoints / f"step-{self.step + 1:09d}").mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError, "latest checkpoint"):
            self.verify()

    def test_mismatched_approved_input_refuses_resume(self):
        self.publish_attempt_and_backup()
        self.benchmark["motion_archive_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            self.verify()

    def test_no_training_advancement_refuses_resume(self):
        self.result["resumed_from_iterations"] = self.step
        self.publish_attempt_and_backup()
        with self.assertRaisesRegex(ValueError, "verified advancement"):
            self.verify()

    def test_unplanned_failure_refuses_resume(self):
        self.publish_attempt_and_backup(reason="training_error")
        with self.assertRaisesRegex(ValueError, "intentional full-training stop"):
            self.verify()


if __name__ == "__main__":
    unittest.main()

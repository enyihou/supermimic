from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from air_checkpoint import publish_checkpoint, write_sums
from audit_official import SOURCE_COMMITS, attach_evaluation, sha256_file, verify_run
from evaluate_heldout import MANIFEST, REPOSITORY, REVISION, SEED, SUITE_ID


class PhaseAuditTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run = self.root / "official-test"
        (self.run / "checkpoints").mkdir(parents=True)
        self.identity = {
            "source_commits": SOURCE_COMMITS,
            "reproduction_commit": "e" * 40,
            "dataset_sha256": "a" * 64,
            "motion_archive_sha256": "b" * 64,
            "code_sha256": "c" * 64,
            "environment_sha256": "d" * 64,
            "mode": "smoke", "num_envs": 256, "max_iterations": 200,
            "task_id": "SuperMimic-GenCoord-G1", "run_id": self.run.name,
        }
        self.args = argparse.Namespace(
            run=self.run, job_id="1234", slurm_record=self.root / "train-slurm.json",
            phase="smoke", num_envs=256, iterations=200,
            dataset_sha256="a" * 64, motion_sha256="b" * 64,
            code_sha256="c" * 64, environment_sha256="d" * 64,
            reproduction_commit="e" * 40,
        )

    @staticmethod
    def write_json(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def checkpoint(self, step, job="1234", milestone=True):
        local = self.root / f"writing-{step}"
        local.mkdir()
        for name in ("actor", "world_model", "critic", "training_state"):
            (local / f"{name}.pt").write_bytes(f"{name} {step}".encode())
        motion = {
            "sha256": "1" * 64, "clipset_sha256": "2" * 64,
            "clips": 64, "frames": 1000, "input_fps": 120, "fps": 50,
            "selection": "archive-order-smoke",
        } if self.args.phase == "smoke" else {
            "sha256": "1" * 64, "clipset_sha256": "2" * 64,
            "clips": 129_514, "frames": 47_000_000, "input_fps": 120, "fps": 50,
            "selection": "explicit-allowlist", "selection_id": "supermimic-v3-authors-manifest",
            "approximate": False, "allowlist_sha256": "3" * 64,
        }
        self.identity["motion_manifest"] = motion
        self.write_json(local / "run-config.json", {
            **self.identity, "runner_cfg": {"max_iterations": self.args.iterations, "log_interval": 10},
            "env_cfg": {"scene": {"num_envs": self.args.num_envs}, "private": "/do-not-publish"},
            "motion_clips": motion["clips"], "motion_frames": motion["frames"], "motion_fps": 50,
            "milestone_interval": 32000,
        })
        return publish_checkpoint(local, self.run / "checkpoints", self.identity, step, job,
                                  {"milestone": milestone})

    def attempt(self, path, job, slurm_record, payload):
        path.mkdir(parents=True)
        status = f"job_id={job}\nexit_code=0\nreason=completed\n"
        (path / "job-status.txt").write_text(status, encoding="utf-8")
        (path / "SUCCESS").touch()
        mkdir = self.root / "summaries"
        mkdir.mkdir(exist_ok=True)
        with tarfile.open(path / "outputs.tar.gz", "w:gz") as archive:
            archive.add(path / "job-status.txt", arcname="./job-status.txt")
            for name, data in payload.items():
                item = mkdir / name
                if name.endswith(".jsonl"):
                    item.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
                else:
                    self.write_json(item, data)
                archive.add(item, arcname=name)
        write_sums(path / "SHA256SUMS", {name: sha256_file(path / name)
                                        for name in ("outputs.tar.gz", "job-status.txt")})
        self.write_json(slurm_record, {
            "source": "scontrol show job -o", "job_id": job, "allocated_gpus": 1,
            "state": "COMPLETED", "exit_code": "0:0",
        })

    def train_attempt(self, job_final="1234", job_checkpoint="1234", *, resumed=None, metrics=None):
        step = self.args.iterations
        meta = self.checkpoint(step, job_checkpoint)
        path = self.run / "attempts" / job_final
        payload = {
            "run-identity.json": self.identity,
            "result.json": {
                "completed_iterations": step, "target_iterations": step, "stopped": False,
                "job_id": job_final, "checkpoint_job_id": job_checkpoint,
                "checkpoint_sha256": meta["archive"]["sha256"],
                "resumed_from_iterations": (step if job_final != job_checkpoint else 0) if resumed is None else resumed,
                "loop_elapsed_seconds": 5.0,
            },
            "environment.json": {"num_cuda_devices": 1, "cuda_available": True, "job_id": job_final},
        }
        if self.args.phase == "benchmark":
            payload["checkpoints.jsonl"] = [
                {"completed_iterations": step_sample, "checkpoint_interval_iterations": 200,
                 "interval_before_checkpoint_seconds": 60 if step_sample == 200 else 25.0,
                 "checkpoint_write_seconds": 3.0, "archive_stageout_seconds": 2.0,
                 "job_id": job_final, "archive": meta["archive"]}
                for step_sample in range(200, step + 1, 200)
            ]
        if metrics is not None:
            payload["metrics.jsonl"] = metrics
        self.attempt(path, job_final, self.args.slurm_record, payload)
        return path, meta

    def test_smoke_verified_but_heldout_not_verified(self):
        self.train_attempt()
        data = verify_run(self.args)
        self.assertEqual(data["phase"], "smoke")
        self.assertFalse(data["held_out_evaluation_verified"])
        self.assertNotIn("do-not-publish", json.dumps(data))

    def test_rejects_unverified_slurm_state_and_checkpoint_tampering(self):
        path, _ = self.train_attempt()
        slurm = json.loads(self.args.slurm_record.read_text())
        self.write_json(self.args.slurm_record, {**slurm, "state": "CANCELLED"})
        with self.assertRaisesRegex(ValueError, "Slurm completion"):
            verify_run(self.args)
        self.write_json(self.args.slurm_record, slurm)
        checkpoint = self.run / "checkpoints" / "step-000000200" / "checkpoint.tar.gz"
        with checkpoint.open("ab") as stream:
            stream.write(b"damage")
        with self.assertRaisesRegex(ValueError, "outer checksum mismatch"):
            verify_run(self.args)

    def test_explicit_no_iteration_recovery_can_finish(self):
        self.train_attempt(job_checkpoint="1233")
        data = verify_run(self.args)
        self.assertEqual(data["checkpoint_job_id"], "1233")
        self.assertEqual(data["allocations"][0]["job_id"], "1234")

    def test_benchmark_excludes_first_window_from_steady_throughput(self):
        self.args.phase = self.identity["mode"] = "benchmark"
        self.args.iterations = self.identity["max_iterations"] = 1200
        self.args.num_envs = self.identity["num_envs"] = 4096
        self.train_attempt()
        result = verify_run(self.args)["benchmark"]
        self.assertEqual(result["steady_iterations_per_second"], 8.0)
        self.assertEqual(result["mean_steady_checkpoint_seconds"], 5.0)
        self.assertEqual(len(result["checkpoint_samples"]), 6)

    def test_full_and_heldout_evidence_binds_to_same_checkpoint(self):
        self.args.phase = self.identity["mode"] = "full"
        self.args.iterations = self.identity["max_iterations"] = 320000
        self.args.num_envs = self.identity["num_envs"] = 4096
        for step in range(32000, 320000, 32000):
            self.checkpoint(step)
        _, meta = self.train_attempt(resumed=319980, metrics=[
            {"completed_iterations": 319981, "loop_elapsed_seconds": 1.0,
             "metrics": {"Train/mean_reward": 0.7, "hidden": "/do-not-publish"}},
            {"completed_iterations": 319991, "loop_elapsed_seconds": 3.0,
             "metrics": {"Train/mean_reward": 0.82, "Loss/state": 0.1}},
        ])
        verified = verify_run(self.args)
        self.assertEqual(len(verified["retained_checkpoints"]), 10)
        self.assertEqual(verified["learning_curves"]["coverage"], {
            "scope": "final-allocation-only", "resumed_from_iterations": 319980, "log_interval": 10,
            "first_logged_step": 319981, "last_logged_step": 319991, "logged_rows": 2,
        })
        self.assertEqual(verified["learning_curves"]["series"][0]["points"], [
            {"step": 319981, "value": 0.7}, {"step": 319991, "value": 0.82},
        ])
        args = argparse.Namespace(training_audit=self.root / "training-audit.json",
                                  evaluation=self.root / "9876", job_id="9876",
                                  slurm_record=self.root / "eval-slurm.json", suite_sha256="f" * 64)
        self.write_json(args.training_audit, verified)
        metrics = {
            "schema_version": 1, "protocol": SUITE_ID, "evaluation_job_id": "9876",
            "training_job_id": "1234", "checkpoint_step": 320000, "source_commits": SOURCE_COMMITS,
            "checkpoint_archive_sha256": meta["archive"]["sha256"],
            "training_dataset_sha256": self.identity["dataset_sha256"],
            "suite": {"source": REPOSITORY, "revision": REVISION, "archive_sha256": args.suite_sha256,
                      "clip_manifest_sha256": sha256_file(MANIFEST), "clips": 40},
            "rollout_config": {"nominal": True, "control_hz": 50, "seed": SEED,
                               "foot_tangential_friction": 1.0, "max_step_reward": 0.1,
                               "return_normalization": "zero unsimulated tail, per-clip reward ceiling"},
            "metrics": {"evaluated_clips": 40, "completed_clips": 39, "failed_clips": 1,
                        "completion_rate": 39 / 40, "evaluated_steps": 339, "reference_steps": 350,
                        "mean_normalized_return": 0.7, "mean_root_position_error_m": 0.15,
                        "mean_heading_aligned_body_position_error_m": 0.12,
                        "sensitive_metric_from_extra_field": "/do-not-publish"},
        }
        self.attempt(args.evaluation, args.job_id, args.slurm_record, {"metrics.json": metrics})
        published = attach_evaluation(args)
        self.assertTrue(published["held_out_evaluation_verified"])
        self.assertEqual(published["learning_curves"], verified["learning_curves"])
        self.assertNotIn("do-not-publish", json.dumps(published))
        self.write_json(args.training_audit, {**verified, "checkpoint_archive_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "does not match the audited full checkpoint"):
            attach_evaluation(args)


if __name__ == "__main__":
    unittest.main()

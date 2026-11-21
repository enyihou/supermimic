import importlib.util
import os
import random
import signal
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from air_checkpoint import (
    CheckpointManager,
    checkpoint_name,
    find_resume_checkpoint,
    latest_checkpoint,
    publish_checkpoint,
    unpack_checkpoint,
    write_json,
)
from air_launch import RECIPES, verify_final


class CheckpointContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.run_dir = self.work / "run"
        self.run_dir.mkdir()
        self.root = self.run_dir / "checkpoints"
        self.identity = {
            "mode": "full", "dataset_sha256": "4" * 64, "motion_archive_sha256": "2" * 64,
            "num_envs": 4096, "max_iterations": 320000,
        }
        self.job = "16001"

    def publish(self, step, job_id=None):
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            local = Path(tmp) / "payload"
            local.mkdir()
            for name in ("actor.pt", "world_model.pt", "critic.pt", "training_state.pt", "run-config.json"):
                (local / name).write_text(f"data {step} {name}\n")
            return publish_checkpoint(local, self.root, self.identity, step, job_id or self.job, {})

    def test_publish_and_restore_all_payloads(self):
        published = self.publish(1000)
        path, restored = latest_checkpoint(self.root, verify_archive=True)
        self.assertEqual(path.name, checkpoint_name(1000))
        self.assertEqual(published, restored)
        unpack_checkpoint(path, restored, self.work)
        self.assertEqual((self.work / "restore/actor.pt").read_text(), "data 1000 actor.pt\n")
        self.assertFalse((self.work / "restore.tar.gz").exists())

    def test_failed_publication_keeps_previous_step(self):
        self.publish(1000)
        with patch("air_checkpoint.sync_path", side_effect=OSError("simulated storage failure")):
            with self.assertRaises(OSError):
                self.publish(2000)
        self.assertEqual(latest_checkpoint(self.root, verify_archive=True)[1]["completed_iterations"], 1000)
        self.assertEqual({p.name for p in self.root.iterdir()}, {"latest", checkpoint_name(1000)})

    def test_recover_rename_before_pointer_switch(self):
        self.publish(1000)
        self.publish(2000)
        link = self.root / "latest"
        link.unlink()
        link.symlink_to(checkpoint_name(1000))
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            find_resume_checkpoint(self.root, {**self.identity, "num_envs": 256})
        self.assertEqual(link.readlink().name, checkpoint_name(1000))
        _, resumed = find_resume_checkpoint(self.root, self.identity)
        self.assertEqual(resumed["completed_iterations"], 2000)
        self.assertEqual(link.readlink().name, checkpoint_name(2000))

    def test_reject_corruption_and_wrong_resume_identity(self):
        self.publish(1000)
        with self.assertRaisesRegex(RuntimeError, "differs"):
            find_resume_checkpoint(self.root, {**self.identity, "dataset_sha256": "3" * 64})
        path = self.root / checkpoint_name(1000) / "checkpoint.tar.gz"
        payload = path.read_bytes()
        path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            latest_checkpoint(self.root, verify_archive=True)

    def test_success_gate_links_exact_job_step_and_checkpoint_hash(self):
        output = self.work / "output"
        output.mkdir()
        write_json(output / "run-identity.json", self.identity)
        result = {
            "job_id": self.job,
            "completed_iterations": 320000,
            "target_iterations": 320000,
            "stopped": False,
            "resumed_from_iterations": 300000,
            "checkpoint_job_id": self.job,
        }
        self.publish(319000)
        write_json(output / "result.json", {**result, "checkpoint_sha256": "missing"})
        with self.assertRaisesRegex(RuntimeError, "exact target"):
            verify_final(self.run_dir, output, self.job)
        final = self.publish(320000)
        result["checkpoint_sha256"] = final["archive"]["sha256"]
        write_json(output / "result.json", result)
        verify_final(self.run_dir, output, self.job)
        with self.assertRaisesRegex(RuntimeError, "exact target"):
            verify_final(self.run_dir, output, "16002")

    def test_walltime_recovery_requires_explicit_final_checkpoint_restore(self):
        output = self.work / "output"
        output.mkdir()
        write_json(output / "run-identity.json", self.identity)
        final = self.publish(320000)
        result = {
            "job_id": "16002",
            "completed_iterations": 320000,
            "target_iterations": 320000,
            "stopped": False,
            "resumed_from_iterations": 319000,
            "checkpoint_job_id": self.job,
            "checkpoint_sha256": final["archive"]["sha256"],
        }
        write_json(output / "result.json", result)
        with self.assertRaisesRegex(RuntimeError, "different job ID"):
            verify_final(self.run_dir, output, "16002")
        result["resumed_from_iterations"] = 320000
        write_json(output / "result.json", result)
        verify_final(self.run_dir, output, "16002")

    def test_full_recipe_and_benchmark_covers_full_replay(self):
        self.assertEqual(RECIPES["full"], (4096, 320000, 1000, 32000))
        self.assertEqual(RECIPES["benchmark"][0], RECIPES["full"][0])
        self.assertGreaterEqual(RECIPES["benchmark"][1], 1000)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
        "optional CPU tensor serialization test requires torch and numpy",
    )
    def test_learning_state_round_trip_and_fresh_episode_boundary(self):
        import numpy as np
        import torch

        @dataclass
        class Cfg:
            max_iterations: int = 320000

        def build_runner():
            def model():
                weights = torch.nn.Linear(3, 2)
                optimizer = torch.optim.Adam(weights.parameters(), lr=torch.tensor(0.001))
                weights(torch.ones(1, 3)).square().sum().backward()
                optimizer.step()
                return weights, optimizer

            dynamics, dynamics_optimizer = model()
            actor, actor_optimizer = model()
            critic, critic_optimizer = model()
            sampler = SimpleNamespace(value=torch.arange(4.0), count=torch.ones(4), effort=torch.zeros(4))
            motion_cmd = SimpleNamespace(_external_probs=torch.full((4,), 0.25))
            motion_cmd.set_sampling_distribution = lambda probs: motion_cmd._external_probs.copy_(probs)
            replay = SimpleNamespace(buffer_size=3, _step=2, num_transitions=2, _storage={
                "state": torch.arange(18.0).view(2, 3, 3),
                "truncated": torch.zeros(2, 3, dtype=torch.bool),
            })
            return SimpleNamespace(
                cfg=Cfg(), device="cpu", task_id="SuperMimic-GenCoord-G1", env=SimpleNamespace(cfg=Cfg()),
                motion=SimpleNamespace(motion_names=["fixture"], time_step_total=80, fps=50.0),
                _wm=SimpleNamespace(bundle=dynamics.state_dict, load_bundle=dynamics.load_state_dict),
                world_model_learner=SimpleNamespace(optimizers=[dynamics_optimizer]),
                policy_learner=SimpleNamespace(
                    actor=actor, actor_optimizers=[actor_optimizer],
                    critic=critic, critic_optimizers=[critic_optimizer],
                ),
                _sampler=sampler, motion_cmd=motion_cmd, replay_buffer=replay,
            )

        output, scratch = self.work / "output", self.work / "work"
        output.mkdir()
        scratch.mkdir()
        identity = {**self.identity, "milestone_interval": 1000}
        write_json(output / "run-identity.json", identity)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
        fixtures = {
            "SM_OUTPUT_DIR": str(output), "SM_WORK_DIR": str(scratch),
            "SM_RUN_DIR": str(self.run_dir), "SLURM_JOB_ID": self.job,
        }
        trained = build_runner()
        trained._sampler.value += 10
        trained.motion_cmd._external_probs.copy_(torch.tensor([0.15, 0.2, 0.35, 0.3]))
        ratio = SimpleNamespace(_prev=999.5)
        with patch.dict(os.environ, {**fixtures, "SM_RESUME": "fresh"}):
            initial = CheckpointManager(trained)
            initial.begin()
            initial.save(1000, ratio)
        expected_rng = (random.random(), np.random.rand(), torch.rand(1).item())
        restarted = build_runner()
        restarted.replay_buffer._storage["state"].zero_()
        ratio = SimpleNamespace(_prev=None)
        with patch.dict(os.environ, {**fixtures, "SM_RESUME": "resume", "SLURM_JOB_ID": "16002"}):
            resumed = CheckpointManager(restarted)
            self.assertEqual(resumed.restore(ratio), 1000)
        self.assertEqual(ratio._prev, 999.5)
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expected_rng)
        self.assertTrue(torch.equal(trained._sampler.value, restarted._sampler.value))
        self.assertTrue(torch.equal(trained.motion_cmd._external_probs, restarted.motion_cmd._external_probs))
        for attr in ("actor", "critic"):
            old, new = getattr(trained.policy_learner, attr), getattr(restarted.policy_learner, attr)
            self.assertTrue(torch.equal(old.weight, new.weight))
            old_opt = getattr(trained.policy_learner, f"{attr}_optimizers")[0]
            new_opt = getattr(restarted.policy_learner, f"{attr}_optimizers")[0]
            self.assertTrue(torch.equal(old_opt.state[old.weight]["exp_avg"], new_opt.state[new.weight]["exp_avg"]))
        self.assertTrue(torch.equal(trained._wm.bundle()["weight"], restarted._wm.bundle()["weight"]))
        replay = restarted.replay_buffer
        self.assertEqual((replay._step, replay.num_transitions), (2, 2))
        self.assertTrue(torch.equal(trained.replay_buffer._storage["state"], replay._storage["state"]))
        self.assertEqual(replay._storage["truncated"].tolist(), [[False, True, False]] * 2)


if __name__ == "__main__":
    unittest.main()

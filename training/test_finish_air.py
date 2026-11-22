"""Exercise login-side finishing decisions without a scheduler or large files."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class FinishAirTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.state = self.root / "private-state"
        self.state.mkdir(mode=0o700)
        self.shared = self.root / "shared"
        self.job = "33900"
        self.run = "official-full-test"
        self.attempt = self.shared / "runs" / self.run / "attempts" / self.job
        self.attempt.mkdir(parents=True)
        (self.state / "job-id").write_text(self.job + "\n")
        (self.state / "inputs.txt").write_text("dataset_sha256=dataset\nmotion_sha256=motion\n"
                                               "code_sha256=code\nenvironment_sha256=env\n"
                                               "reproduction_commit=commit\n")
        self.record = self.state / (self.job + ".slurm.json")
        self.benchmark = self.root / "benchmark-audit.json"
        self.benchmark.write_text("{}")
        self.worker = self.root / "finish_air_smoke.sh"
        shutil.copyfile(Path(__file__).with_name(self.worker.name), self.worker)
        for file in ("backup_air_attempt.py", "audit_official.py", "verify_air_resume.py"):
            (self.root / file).write_text("import json, sys\nfrom pathlib import Path\n"
                                          "print(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]))\n")

    def finish(self, state, exit_code, *args):
        self.record.write_text(json.dumps({"job_id": self.job, "state": state, "exit_code": exit_code}))
        result = subprocess.run(["bash", str(self.worker), str(self.shared), self.run, self.job,
                                 str(self.state), *args], text=True, capture_output=True, timeout=10)
        calls = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("[")]
        return result, calls

    def test_success_backs_up_every_full_milestone_and_binds_audit(self):
        result, (backup, audit) = self.finish("COMPLETED", "0:0", "full", str(self.benchmark))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(backup[:4], ["backup_air_attempt.py", str(self.shared / "runs"), self.run, self.job])
        self.assertEqual(backup[4::2], ["--step"] * 10)
        self.assertEqual(list(map(int, backup[5::2])), list(range(32000, 320001, 32000)))
        self.assertEqual(audit[-2:], ["--benchmark-audit", str(self.benchmark)])
        self.assertIn("320000", audit)
        self.assertIn(str(self.state / "full-audit.json"), audit)

    def test_planned_stop_checks_backed_up_checkpoint_before_reporting_ready(self):
        for last, keep in ((64000, [32000, 64000]), (65201, [32000, 64000, 65201])):
            with self.subTest(last=last):
                (self.attempt / "job-status.txt").write_text(f"last_published_iteration={last}\n")
                result, (backup, resume) = self.finish("FAILED", "124:0", "full", str(self.benchmark))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(map(int, backup[5::2])), keep)
                self.assertEqual(resume, ["verify_air_resume.py", str(self.shared / "runs"), self.run, self.job,
                                          "--slurm-record", str(self.record), "--benchmark", str(self.benchmark)])
                self.assertIn("no job submitted", result.stdout)

    def test_unplanned_timeout_backs_up_attempt_but_never_authorizes_resume(self):
        result, (backup,) = self.finish("TIMEOUT", "0:0", "full", str(self.benchmark))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(backup, ["backup_air_attempt.py", str(self.shared / "runs"), self.run, self.job])
        self.assertIn("TIMEOUT", result.stderr)

    def test_successful_benchmark_behavior_is_unchanged(self):
        result, (backup, audit) = self.finish("COMPLETED", "0:0", "benchmark")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(backup[4:], ["--step", "600", "--step", "1200"])
        self.assertNotIn("--benchmark-audit", audit)


if __name__ == "__main__":
    unittest.main()

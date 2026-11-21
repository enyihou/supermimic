from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from watch_slurm import watch


class SlurmWatcherTest(unittest.TestCase):
    def test_temporary_loss_is_not_a_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            final = {"state": "COMPLETED", "job_id": "1234", "exit_code": "0:0"}
            replies = [subprocess.CalledProcessError(1, "scontrol"), {"state": "RUNNING"}, final]
            with (
                patch("watch_slurm.capture_slurm", side_effect=replies) as poll,
                patch("watch_slurm.time.sleep", return_value=None),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(watch("1234", path, 5, 30), final)
            self.assertEqual(json.loads(path.read_text()), final)
            self.assertEqual(poll.call_count, 3)
            self.assertEqual(path.stat().st_mode & 0o077, 0)

    def test_shared_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o755)
            with self.assertRaisesRegex(ValueError, "owner-only"):
                watch("1234", Path(directory) / "result.json", 5, 30)


if __name__ == "__main__":
    unittest.main()

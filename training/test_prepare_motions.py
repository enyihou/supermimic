"""Exercise the preparation batch script locally without a Slurm allocation."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

from air_launch import APPROXIMATE_SELECTION, EXACT_SELECTION, PINS, REQUIRED_MOTIONS


# A small simulated packer checks the real script's argv and emits loader-
# compatible arrays. This tests batch staging, checksums and publication,
# not numerical CSV conversion (covered separately) or scheduler behavior.
PACKER = """\
import argparse
import hashlib
import json
from pathlib import Path
import stat
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("raw", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--g1-xml", type=Path, required=True)
p.add_argument("--sha256", required=True)
p.add_argument("--workers", type=int, required=True)
p.add_argument("--keep-list", type=Path, required=True)
p.add_argument("--keep-list-sha256", required=True)
p.add_argument("--selection-id", required=True)
a = p.parse_args()
assert a.g1_xml.is_file() and a.workers == 3
assert stat.S_IMODE(a.raw.stat().st_mode) == 0o400
assert stat.S_IMODE(a.keep_list.stat().st_mode) == 0o400
assert hashlib.sha256(a.raw.read_bytes()).hexdigest() == a.sha256
assert hashlib.sha256(a.keep_list.read_bytes()).hexdigest() == a.keep_list_sha256
assert a.selection_id == "supermimic-v3-authors-manifest"
a.out.mkdir()
manifest = {
    "dataset_sha256": a.sha256, "clipset_sha256": "b" * 64,
    "clips": 129514, "frames": 259028, "input_fps": 120, "fps": 50,
    "selection": "explicit-allowlist", "selection_id": a.selection_id,
    "allowlist_sha256": a.keep_list_sha256, "approximate": False,
}
(a.out / "motion-manifest.json").write_text(json.dumps(manifest))
(a.out / "motion_names.json").write_text(json.dumps(["a"] * manifest["clips"]))
(a.out / "joint_names.json").write_text("[]")
np.save(a.out / "motion_ends.npy", np.arange(2, manifest["frames"] + 1, 2, dtype=np.int64))
np.save(a.out / "fps.npy", np.array(50, dtype=np.int64))
for field, width in (
    ("joint_pos", 29), ("joint_vel", 29), ("root_pos_w", 3), ("root_quat_w", 4),
    ("root_lin_vel_w", 3), ("root_ang_vel_w", 3),
):
    np.lib.format.open_memmap(a.out / (field + ".npy"), mode="w+", dtype="<f4", shape=(manifest["frames"], width))
"""


def small_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(importlib.util.find_spec("numpy"), "batch fixture requires numpy")
class PrepareMotionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.publish = self.root / "private"
        self.publish.mkdir(mode=0o700)
        self.publish.chmod(0o2700)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.package = self.root / "package"
        self.package.mkdir()
        self.bones = self.root / "g1.tar.gz"
        self.bones.write_bytes(b"private motion fixture")
        self.selection = self.root / "keep.txt"
        self.selection.write_text("g1/csv/day/private-fixture.csv\n")
        self.job = "12345"
        self.build_id = "prepared-fixture"

        # Replace only the cluster's fixed path and physical-space reading
        # to exercise the actual batch body on a laptop-sized fixture.
        here = Path(__file__).resolve().parent
        script = (here / "run_prepare_motions.sbatch").read_text()
        script = script.replace('"/scratch/jobs/$SLURM_JOB_ID"', f'"{self.scratch}"')
        script = script.replace(
            """free_bytes() { df -B1 --output=avail -- "$1" | awk 'NR == 2 { print $1 }'; }""",
            """free_bytes() { printf '%s\\n' 200000000000; }""",
        )
        self.script = self.root / "run_prepare_motions.sbatch"
        self.script.write_text(script)
        fixture_env = self.root / "environment"
        bin_dir = fixture_env / "bin"
        bin_dir.mkdir(parents=True)
        interpreter = bin_dir / "python"
        interpreter.write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
        interpreter.chmod(0o700)
        activate = bin_dir / "activate"
        activate.write_text("# fixture\n")
        # Deliberately invalid for direct execution: the batch must run this
        # through its staged Python, not an inherited/home Python.
        unpack = bin_dir / "conda-unpack"
        unpack.write_text("#!/never-present/python\n")
        unpack.chmod(0o700)
        with tarfile.open(self.package / "environment.tar.gz", "w:gz") as archive:
            archive.add(bin_dir, arcname="bin")

        code = self.root / "code"
        training = code / "training"
        training.mkdir(parents=True)
        for item in ("air_launch.py", "air_checkpoint.py"):
            shutil.copyfile(here / item, training / item)
        (training / "pack_bones.py").write_text(PACKER)
        selector_source = (here / "select_bones_clips.py").read_text()
        pinned_line = next(line for line in selector_source.splitlines() if line.startswith("SONIC_SOURCE_SHA256 = "))
        self.pinned_sonic_sha = pinned_line.split('"')[1]
        (training / "select_bones_clips.py").write_text("raise RuntimeError('selector must not run')\n")
        (code / "PINS.txt").write_text("".join(f"{repo}={sha}\n" for repo, sha in PINS.items()))
        (code / "REPRODUCTION_COMMIT.txt").write_text("c" * 40 + "\n")
        xml = code / "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
        xml.parent.mkdir(parents=True)
        xml.write_text("<fixture />\n")
        with tarfile.open(self.package / "code.tar.gz", "w:gz") as archive:
            for entry in code.iterdir():
                archive.add(entry, arcname=entry.name)
        (self.package / "SHA256SUMS").write_text("".join(
            f"{small_sha(self.package / name)}  {name}\n" for name in ("environment.tar.gz", "code.tar.gz")
        ))

    def run_fixture(self, *, mode="authors-manifest", selection_sha=None, gpu_count="0"):
        environment = {
            **os.environ,
            "LOCAL_SCRATCH": str(self.scratch),
            "SLURM_JOB_ID": self.job,
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_JOB_GPUS": "",
            "SLURM_GPUS_ON_NODE": gpu_count,
            "SLURM_CPUS_PER_TASK": "4",
        }
        return subprocess.run(
            [
                "bash", self.script, self.package, self.bones, small_sha(self.bones), self.publish,
                self.build_id, mode, self.selection, selection_sha or small_sha(self.selection),
            ],
            env=environment, text=True, capture_output=True, timeout=30,
        )

    def test_publishes_only_immutable_verified_private_artifacts(self):
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        dest = self.publish / self.build_id
        self.assertEqual({p.name for p in dest.iterdir()}, {
            "motions.tar.gz", "motion-manifest.json", "preparation.json", "job-status.txt", "SHA256SUMS", "SUCCESS",
        })
        self.assertEqual(list(self.publish.iterdir()), [dest])
        self.assertEqual(dest.stat().st_mode & 0o077, 0)
        for line in (dest / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split()
            self.assertEqual(small_sha(dest / name), digest)
        preparation = json.loads((dest / "preparation.json").read_text())
        self.assertEqual(preparation["job_id"], self.job)
        self.assertEqual(preparation["dataset_sha256"], small_sha(self.bones))
        self.assertEqual(preparation["motion_manifest"]["selection_id"], EXACT_SELECTION)
        self.assertIsNone(preparation["approximate_selector"])
        self.assertIn("exit_code=0", (dest / "job-status.txt").read_text().splitlines())
        with tarfile.open(dest / "motions.tar.gz") as archive:
            self.assertEqual({member.name.removeprefix("./") for member in archive if member.isfile()}, REQUIRED_MOTIONS)
        retry = self.run_fixture()
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("already exists", retry.stderr)
        self.assertEqual(small_sha(dest / "motions.tar.gz"), (dest / "SHA256SUMS").read_text().split()[0])

    def test_bad_selection_fails_without_publishing_an_archive(self):
        result = self.run_fixture(selection_sha="0" * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Selection input SHA-256 mismatch", result.stderr)
        failed = self.publish / f".{self.build_id}.job-{self.job}.failed.txt"
        self.assertEqual(list(self.publish.iterdir()), [failed])
        self.assertIn("phase=stage_inputs", failed.read_text().splitlines())

    def test_rejects_gpu_and_unselected_or_unpinned_approximation(self):
        for mode, digest, gpus, message in [
            ("authors-manifest", None, "1", "must request zero GPUs"),
            (APPROXIMATE_SELECTION, None, "0", "explicitly approved"),
            ("--accept-approximate", None, "0", "pinned SONIC source SHA-256"),
            ("--accept-approximate", self.pinned_sonic_sha, "0", "Selection input SHA-256 mismatch"),
        ]:
            with self.subTest(mode=mode, digest=digest, gpus=gpus):
                result = self.run_fixture(mode=mode, selection_sha=digest, gpu_count=gpus)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse((self.publish / self.build_id).exists())


if __name__ == "__main__":
    unittest.main()

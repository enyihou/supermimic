from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inspect_bones import BASE_COLUMNS
from select_bones_clips import (
    G1Feasibility, excludes_by_name, feasible_reason, scan_motions, select,
    sha256, sonic_keywords,
)


class DecisionTest(unittest.TestCase):
    def test_keyword_rules_filter_objects_not_floor_acrobatics(self):
        path = lambda label: PurePosixPath("g1/csv/20250101") / (label + ".csv")
        words = ("chair", "ladder", "stair")
        for label in ("STEP_on_STAiRS", "walk_box_01", "sit", "Jump-Off-50cm", "Look Over Fence"):
            self.assertTrue(excludes_by_name(path(label), words))
        for label in ("handstand", "floor_cartwheel", "safety_roll", "transition", "boxing"):
            self.assertFalse(excludes_by_name(path(label), words))

    def test_name_source_is_hash_checked_and_never_executed(self):
        src = b'''
raise RuntimeError("not meant to execute")
argparser.add_argument("--filter-keywords", default=[
    "230m", "acrobatics_", "cartwheel", "handstand", "handstand_ff_",
    "jump_to_shoulder_roll", "kozak_dance", "monkey_jump", "safety_roll", "CHAIR",
])
'''
        with patch("select_bones_clips.SONIC_SOURCE_SHA256", hashlib.sha256(src).hexdigest()):
            self.assertEqual(sonic_keywords(src), ("chair",))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                sonic_keywords(src + b" ")

    def test_clip_thresholds_use_consecutive_frames_and_strict_boundaries(self):
        nz = np.zeros(50)
        nf = np.zeros(50, dtype=bool)
        for size in (29, 30):
            z = nz.copy()
            z[:size] = 0.21
            self.assertEqual(feasible_reason(z, nf), "elevated_support" if size == 30 else None)
        for size in (24, 25):
            flags = nf.copy()
            flags[-size:] = True
            self.assertEqual(feasible_reason(nz, flags), "floating_feet" if size == 25 else None)
        for value, count, fails in ((-0.12, 1, False), (-0.121, 1, True), (-0.09, 40, False), (-0.09, 41, True)):
            z = nz.copy()
            z[:count] = value
            self.assertEqual(feasible_reason(z, nf), "ground_penetration" if fails else None)
        self.assertIsNone(feasible_reason(np.full(50, 0.20), nf))
        flags = nf.copy()
        flags[0:15] = flags[20:35] = True
        self.assertIsNone(feasible_reason(nz, flags))


class ArchiveTest(unittest.TestCase):
    @staticmethod
    def archive(entries):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            for name, data in entries:
                entry = tarfile.TarInfo(name)
                if data is None:
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = "walk.csv"
                    archive.addfile(entry)
                else:
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
        return payload.getvalue()

    def test_rejects_links_traversal_duplicates_and_truncated_gzip(self):
        from collections import Counter

        name = "g1/csv/20250101/walk.csv"
        for entries in (
            [(name, None)],
            [("g1/csv/20250101/../../walk.csv", b"test")],
            [(name, b"test"), (name, b"test")],
        ):
            with self.assertRaises(ValueError):
                list(scan_motions(io.BytesIO(self.archive(entries)), (), Counter()))
        broken = self.archive([(name, b"test")])[:-4]
        with self.assertRaises(EOFError):
            list(scan_motions(io.BytesIO(broken), (), Counter()))


@unittest.skipUnless(os.getenv("BONES_TEST_MJLAB_G1_XML") and os.getenv("BONES_TEST_SONIC_SOURCE"),
                     "set pinned G1 XML and SONIC source paths for integration")
class GeometryAndSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml = Path(os.environ["BONES_TEST_MJLAB_G1_XML"])
        cls.pin = Path(os.environ["BONES_TEST_SONIC_SOURCE"])
        cls.model = G1Feasibility(cls.xml)

    @staticmethod
    def pose(root_z, *, speed=0.0):
        count = 32
        n0 = np.zeros((count, 3))
        return {
            "root_pos_w": np.tile([0, 0, root_z], (count, 1)),
            "root_quat_w": np.tile([1., 0, 0, 0], (count, 1)),
            "root_lin_vel_w": np.tile([speed, 0, 0], (count, 1)),
            "root_ang_vel_w": n0,
            "joint_pos": np.zeros((count, 29)),
            "joint_vel": np.zeros((count, 29)),
        }

    def test_exact_capsule_distance_not_ankle_z_and_stationary_gate(self):
        z = self.model.update(np.array([0, 0, 0.793]), np.array([1., 0, 0, 0]), np.zeros(29))
        distance_left, distance_right = self.model.distances()
        self.assertGreater(z - distance_left, 0.03)
        self.assertAlmostEqual(distance_left, distance_right, places=8)
        self.assertIsNone(self.model.classify(self.pose(0.793)))
        self.assertEqual(self.model.classify(self.pose(0.823)), "floating_feet")
        self.assertIsNone(self.model.classify(self.pose(0.823, speed=0.31)))
        self.assertEqual(self.model.classify(self.pose(1.10)), "elevated_support")
        self.assertEqual(self.model.classify(self.pose(0.40)), "ground_penetration")

    def csv(self, height=79.3):
        columns = (*BASE_COLUMNS, *(name + "_dof" for name in self.model.joint_names))
        data = [",".join(columns)]
        for j in range(78):
            data.append(f"{j},0,0,{height}," + ",".join("0" for _ in range(32)))
        return ("\n".join(data) + "\n").encode()

    def test_ordered_allowlist_compatible_with_packer_and_private_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test.tar.gz"
            # Preserve original member order, not guessed filename or category order.
            keep_z = "g1/csv/20250101/z_walk.csv"
            keep_a = "g1/csv/20250101/a_cartwheel.csv"
            entries = (
                (keep_z, self.csv()),
                ("g1/csv/20250101/step_off_box_0.csv", self.csv()),
                ("g1/csv/20250101/unknown_floor.csv", self.csv(83)),
                (keep_a, self.csv()),
            )
            source.write_bytes(ArchiveTest.archive(entries))
            output = root / "selection"
            args = argparse.Namespace(
                g1_xml=self.xml, sonic_source=self.pin, archive=source,
                output=output, sha256=sha256(source), workers=2,
            )
            with patch("select_bones_clips.EXPECTED_CLIPS", len(entries)):
                result = select(args)
            self.assertEqual(result["counts"]["after_names"], 3)
            self.assertEqual(result["counts"]["feasibility_exclusions"]["floating_feet"], 1)
            self.assertEqual((output / "keep.txt").read_text().splitlines(), [keep_z, keep_a])
            self.assertEqual(result["allowlist_sha256"], sha256(output / "keep.txt"))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((output / "keep.txt").stat().st_mode), 0o600)
            self.assertTrue(result["approximate"])
            self.assertNotIn(keep_a, (output / "summary.json").read_text())
            malformed = root / "malformed.tar.gz"
            malformed.write_bytes(ArchiveTest.archive([(keep_a, b"header\nPRIVATE INVALID VALUES")]))
            args.output, args.archive, args.sha256 = root / "abort", malformed, sha256(malformed)
            with patch("select_bones_clips.EXPECTED_CLIPS", 1):
                with self.assertRaisesRegex(ValueError, "^Motion failed conversion"):
                    select(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()

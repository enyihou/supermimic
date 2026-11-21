import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from inspect_bones import BASE_COLUMNS, inspect
from pack_bones import FIELDS, TARGET_FPS, convert, pack


JOINTS = [f"test_{idx}_joint" for idx in range(29)]


def example_csv():
    header = ",".join([*BASE_COLUMNS, *(f"{joint}_dof" for joint in JOINTS)])
    # 120 input frames/s; 100 cm/s x translation and 60 deg/s world yaw.
    rows = [",".join(map(str, [idx, idx * 100 / 120, 0, 80, 0, 0, idx / 2, *range(29)]))
            for idx in range(13)]
    return ("\n".join([header, *rows]) + "\n").encode()


class BonesPackTest(unittest.TestCase):
    def test_units_quaternion_and_joint_reordering(self):
        arrays = convert(example_csv(), JOINTS[::-1])
        self.assertEqual(arrays["root_pos_w"].shape, (6, 3))
        np.testing.assert_allclose(arrays["root_pos_w"][:, 0], np.arange(6) / TARGET_FPS)
        np.testing.assert_allclose(arrays["root_pos_w"][:, 2], .8)
        np.testing.assert_allclose(arrays["root_lin_vel_w"][:, 0], 1)
        np.testing.assert_allclose(arrays["root_ang_vel_w"][:, 2], np.pi / 3)
        np.testing.assert_allclose(arrays["joint_pos"][0], np.deg2rad(np.arange(29)[::-1]))
        np.testing.assert_allclose(arrays["root_quat_w"][0], [1, 0, 0, 0])

    def test_inventory_and_packed_npy_are_reopenable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "g1.tar.gz"
            rows = example_csv()
            with tarfile.open(source, "w:gz") as archive:
                for name in ("a.csv", "b.csv"):
                    member = tarfile.TarInfo(f"g1/csv/day/{name}")
                    member.size = len(rows)
                    archive.addfile(member, io.BytesIO(rows))
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(inspect(source, 2)["csv_expanded_bytes"], 2 * len(rows))
            output = root / "packed"
            got = pack(source, output, JOINTS, source_sha=expected, keep=None, smoke_clips=2, workers=1)
            self.assertEqual(got["frames"], 12)
            np.testing.assert_array_equal(np.load(output / "motion_ends.npy"), [6, 12])
            for field, width in FIELDS.items():
                self.assertEqual(np.load(output / f"{field}.npy", mmap_mode="r").shape, (12, width))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                pack(source, root / "bad", JOINTS, source_sha="0" * 64, keep=None, smoke_clips=2, workers=1)
            self.assertFalse((root / "bad").exists())

    def test_inventory_rejects_links(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "g1.tar.gz"
            with tarfile.open(source, "w:gz") as archive:
                member = tarfile.TarInfo("g1/csv/day/bad.csv")
                member.type = tarfile.SYMTYPE
                member.linkname = "../../etc/passwd"
                archive.addfile(member)
            with self.assertRaisesRegex(ValueError, "Unsafe member"):
                inspect(source, 0)


if __name__ == "__main__":
    unittest.main()

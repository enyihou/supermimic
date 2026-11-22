from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from air_launch import APPROXIMATE_SELECTION, read_motion_manifest
from audit_air_cohort import (
    ARCHIVE, CODE_SHA, DATASET_SHA, ENV_SHA, HASHSUMS, RAW_CLIPS, REASONS, REVISION, SONIC_SHA, SUCCESS,
    backup, verify,
)
from audit_official import SLURM_SOURCE, SOURCE_COMMITS, sha256_file


class ApproximateCohortAuditTest(unittest.TestCase):
    job = "33802"

    def setUp(self):
        self.addCleanup(os.umask, os.umask(0o077))
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.cohort = self.root / "approved-approximate"
        self.cohort.mkdir()
        self.record = self.root / "terminal.json"
        self.slurm = {"schema_version": 1, "source": SLURM_SOURCE, "job_id": self.job,
                      "state": "COMPLETED", "exit_code": "0:0", "allocated_gpus": 0}
        self.record.write_text(json.dumps(self.slurm))
        self.manifest = {
            "dataset_sha256": DATASET_SHA, "selection_id": APPROXIMATE_SELECTION, "approximate": True,
            "allowlist_sha256": "a" * 64, "clipset_sha256": "b" * 64, "clips": 129_000, "frames": 47_500_000,
            "input_fps": 120, "fps": 50, "selection": "explicit-allowlist",
        }
        (self.cohort / ARCHIVE).write_bytes(gzip.compress(b"fixture, not a real motion library"))
        (self.cohort / "motion-manifest.json").write_text(json.dumps(self.manifest))
        (self.cohort / "job-status.txt").write_text(
            f"job_id={self.job}\nexit_code=0\nphase=completed\ncohort_build_id={self.cohort.name}\n")
        (self.cohort / "SUCCESS").write_bytes(SUCCESS)
        self.preparation = {
            "schema_version": 1, "job_id": self.job, "cohort_build_id": self.cohort.name,
            "dataset_sha256": DATASET_SHA, "selection_input_sha256": SONIC_SHA,
            "code_sha256": CODE_SHA, "environment_sha256": ENV_SHA, "reproduction_commit": REVISION,
            "motion_manifest": read_motion_manifest(self.cohort, "full", DATASET_SHA),
            "approximate_selector": {
                "approximate": True, "equivalent_to_authors_manifest": False, "id": APPROXIMATE_SELECTION,
                "dataset_sha256": DATASET_SHA, "allowlist_sha256": self.manifest["allowlist_sha256"],
                "name_rule": {"source_sha256": SONIC_SHA}, "robot": {"mjlab_commit": SOURCE_COMMITS["mjlab_research"]},
                "counts": {"raw": RAW_CLIPS, "after_names": 129_003, "selected": 129_000,
                           "feasibility_exclusions": dict.fromkeys(REASONS, 1)},
            },
        }
        self.publish_fixture_sums()

    def publish_fixture_sums(self):
        (self.cohort / "preparation.json").write_text(json.dumps(self.preparation))
        (self.cohort / "SHA256SUMS").write_text("".join(
            f"{sha256_file(self.cohort / name)}  {name}\n" for name in sorted(HASHSUMS)))

    def check(self):
        return verify(self.cohort, self.cohort.name, self.job, self.record)

    def test_public_result_and_private_backup(self):
        evidence, sums = self.check()
        self.assertEqual(evidence["counts"]["selected"], 129_000)
        self.assertIs(evidence["motion_manifest"]["approximate"], True)
        self.assertNotIn("canonical_backup", evidence["verified_checks"])
        self.assertNotIn("fixture", json.dumps(evidence))
        archive = self.root / "backup"
        archive.mkdir()
        with patch("audit_air_cohort.check_quota") as quota:
            backup(self.cohort, archive, sums)
            backup(self.cohort, archive, sums)
        quota.assert_called_once()
        (archive / self.cohort.name / "SUCCESS").write_text("not the verified marker\n")
        with self.assertRaisesRegex(ValueError, "success marker"):
            backup(self.cohort, archive, sums)

    def test_zero_gpu_terminal_evidence_is_mandatory(self):
        self.record.write_text(json.dumps({**self.slurm, "allocated_gpus": 1}))
        with self.assertRaisesRegex(ValueError, "Zero-GPU"):
            self.check()

    def test_bad_archive_and_selection_provenance_rejected(self):
        archived = (self.cohort / ARCHIVE).read_bytes()
        (self.cohort / ARCHIVE).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.check()
        (self.cohort / ARCHIVE).write_bytes(archived)
        self.preparation["approximate_selector"]["equivalent_to_authors_manifest"] = True
        self.publish_fixture_sums()
        with self.assertRaisesRegex(ValueError, "approved approximate recipe"):
            self.check()


if __name__ == "__main__":
    unittest.main()

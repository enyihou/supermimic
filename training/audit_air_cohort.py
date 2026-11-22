"""Verify the approved approximate AIR cohort and back it up without extracting it."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from air_launch import APPROXIMATE_SELECTION, read_motion_manifest
from audit_official import JOB_ID, SLURM_SOURCE, SOURCE_COMMITS, atomic_json, parse_fields, require, verify_hashes
from backup_air_attempt import RUN_ID, canonical_destination, check_quota, private_directory, private_file


DATASET_SHA = "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"
SONIC_SHA = "b6b50d441a4876a382eb3ff33ba940c487d1647230b5ecacaa37cfcf521258da"
CODE_SHA = "5698e6d903d27b3cd895c081cfe8a0a3069bd01d5a95c1e7c7e60c0118627f19"
ENV_SHA = "9e8e488d41a6aaed3af99c70372f42164e65505ea2cebd8d01e5090e81cf2172"
REVISION = "f17e90729f959bcd9f5e4c4cd428cab5dd307143"
ARCHIVE = "motions.tar.gz"
HASHSUMS = {ARCHIVE, "motion-manifest.json", "preparation.json", "job-status.txt"}
FILES = HASHSUMS | {"SHA256SUMS", "SUCCESS"}
REASONS = {"elevated_support", "ground_penetration", "floating_feet"}
RAW_CLIPS = 142_220
SUCCESS = b"Packed motions and checksums verified. Check Slurm terminal state separately.\n"


def publication_files(path: Path) -> dict:
    require(path.resolve(strict=True) == path and private_directory(path.lstat()),
            "Cohort directory is not canonical and owner-only")
    require({item.name for item in path.iterdir()} == FILES, "Unexpected cohort publication files")
    for member in FILES:
        info = (path / member).lstat()
        limit = 60_000_000_000 if member == ARCHIVE else 64 * 1024
        require(private_file(info) and 0 < info.st_size <= limit, f"Unexpected cohort artifact: {member}")
    sums = verify_hashes(path, HASHSUMS)
    require(sums.keys() == HASHSUMS, "Cohort checksum manifest does not match the expected file set")
    require((path / "SUCCESS").read_bytes() == SUCCESS, "Cohort lacks the expected success marker")
    return sums


def verify(path: Path, cohort: str, job: str, record: Path) -> tuple[dict, dict]:
    require(RUN_ID.fullmatch(cohort) and JOB_ID.fullmatch(job) and path.name == cohort,
            "Cohort and job identifiers do not match")
    slurm = json.loads(record.read_bytes())
    require(slurm == {"schema_version": 1, "source": SLURM_SOURCE, "job_id": job,
                      "state": "COMPLETED", "exit_code": "0:0", "allocated_gpus": 0},
            "Zero-GPU successful Slurm completion not verified")
    sums = publication_files(path)
    status = parse_fields((path / "job-status.txt").read_text())
    require(status == {"job_id": job, "exit_code": "0", "phase": "completed", "cohort_build_id": cohort},
            "Published status does not match successful Slurm job")
    preparation = json.loads((path / "preparation.json").read_bytes())
    identity = {"schema_version": 1, "job_id": job, "cohort_build_id": cohort,
                "dataset_sha256": DATASET_SHA, "selection_input_sha256": SONIC_SHA,
                "code_sha256": CODE_SHA, "environment_sha256": ENV_SHA, "reproduction_commit": REVISION}
    require(preparation.keys() == identity.keys() | {"motion_manifest", "approximate_selector"}
            and all(preparation.get(key) == value for key, value in identity.items()),
            "Cohort preparation does not match submitted recipe and input hashes")
    manifest = read_motion_manifest(path, "full", DATASET_SHA)
    require(manifest == preparation["motion_manifest"] and manifest["sha256"] == sums["motion-manifest.json"]
            and manifest["approximate"] is True and manifest["selection_id"] == APPROXIMATE_SELECTION,
            "Approximate motion manifest does not match the published preparation")
    selector = preparation["approximate_selector"]
    require(selector["approximate"] is True and selector["equivalent_to_authors_manifest"] is False
            and selector["id"] == APPROXIMATE_SELECTION and selector["dataset_sha256"] == DATASET_SHA
            and selector["allowlist_sha256"] == manifest["allowlist_sha256"]
            and selector["name_rule"]["source_sha256"] == SONIC_SHA
            and selector["robot"]["mjlab_commit"] == SOURCE_COMMITS["mjlab_research"],
            "Selector summary does not match the approved approximate recipe")
    counts = selector["counts"]
    raw, named, selected = (counts[key] for key in ("raw", "after_names", "selected"))
    reasons = counts["feasibility_exclusions"]
    require(all(type(value) is int for value in (raw, named, selected))
            and 64 < selected == manifest["clips"] <= named <= raw == RAW_CLIPS
            and reasons.keys() == REASONS and all(type(value) is int and value >= 0 for value in reasons.values())
            and named - selected == sum(reasons.values()), "Inconsistent full-corpus filter counts")
    with (path / ARCHIVE).open("rb") as stream:
        require(stream.read(2) == b"\x1f\x8b", "Packed archive is not gzip")
    evidence = {"schema_version": 1, "phase": "cohort", "source_commits": SOURCE_COMMITS,
            "reproduction_commit": REVISION, "dataset_sha256": DATASET_SHA, "code_sha256": CODE_SHA,
            "environment_sha256": ENV_SHA, "sonic_source_sha256": SONIC_SHA,
            "allocation": {key: slurm[key] for key in ("job_id", "state", "exit_code", "allocated_gpus")},
            "motion_archive_sha256": sums[ARCHIVE], "motion_archive_bytes": (path / ARCHIVE).stat().st_size,
            "motion_manifest": manifest, "counts": {"raw": raw, "after_names": named, "selected": selected},
            "verified_checks": ["slurm_terminal_state", "job_status", "success_marker", "source_pins",
                                "dataset_checksum", "packed_motion_checksum", "approximate_provenance"]}
    return evidence, sums


def backup(path: Path, root: Path, expected: dict) -> None:
    require(root.resolve(strict=True) == root and private_directory(root.lstat()),
            "Canonical input root is not owner-only")
    copied = root / path.name
    if not copied.exists() and not copied.is_symlink():
        check_quota(root, sum((path / member).stat().st_size for member in FILES))
        with tempfile.TemporaryDirectory(prefix=".cohort-backup-", dir=root) as staging:
            temp = Path(staging) / path.name
            temp.mkdir(mode=0o700)
            for member in sorted(FILES):
                shutil.copyfile(path / member, temp / member)
                with (temp / member).open("rb") as stream:
                    os.fsync(stream.fileno())
            require(publication_files(temp) == expected, "Canonical cohort copy did not pass checksum readback")
            os.rename(temp, copied)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    else:
        require(publication_files(copied) == expected, "Existing canonical cohort copy does not match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("job_id")
    parser.add_argument("--slurm-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        root = canonical_destination().parent
        require(args.output.parent.resolve(strict=True) == args.output.parent
                and args.output.parent == root / "manifests" / args.cohort.name
                and private_directory(args.output.parent.lstat()), "Audit needs the canonical private state directory")
        evidence, sums = verify(args.cohort, args.cohort.name, args.job_id, args.slurm_record)
        backup(args.cohort, root / "inputs", sums)
        evidence["verified_checks"].append("canonical_backup")
        atomic_json(args.output, evidence)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Cohort audit failed: {error}\n")


if __name__ == "__main__":
    main()

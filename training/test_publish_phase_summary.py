"""Synthetic evidence tests. These fixtures are not publication artifacts."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from publish_phase_summary import (
    ARCHIVE_BYTES, BENCHMARK_STEPS, CHECKS, DATASET_ID, DATASET_REVISION, DATASET_SHA256,
    EXPANDED_BYTES, MANIFEST, MIB, MILESTONE_STEPS, RAW_CLIPS, RECIPE, REPOSITORY, REVISION,
    SOURCE_COMMITS, generate, read_json, sha256_file,
)


FIXTURE_DATE = "2026-11-21"
PRIVATE_MARKER = "/test/secret/DO-NOT-PUBLISH"


def preflight(*, air=True):
    return {
        "schema_version": 1, "phase": "dataset_preflight", "updated_utc": FIXTURE_DATE,
        "source_commits": {**SOURCE_COMMITS, "private_notes": PRIVATE_MARKER},
        "status": PRIVATE_MARKER,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "archive": "g1.tar.gz",
                    "sha256": DATASET_SHA256, "archive_bytes": ARCHIVE_BYTES, "csv_files": RAW_CLIPS,
                    "csv_expanded_bytes": EXPANDED_BYTES, "local_sha256_verified": True,
                    "local_archive_members_verified": True, "air_sha256_verified": air,
                    "metadata": {"private_root": PRIVATE_MARKER}},
    }


def alloc(job_id):
    return {"job_id": job_id, "state": "COMPLETED", "exit_code": "0:0", "allocated_gpus": 1,
            "node": PRIVATE_MARKER}


def phase(name, *, eval=False, approximate=True):
    environments, step = RECIPE[name]
    job_id = {"smoke": "101", "benchmark": "102", "full": "103"}[name]
    motions = 64 if name == "smoke" else (129530 if approximate else 129514)
    points = MILESTONE_STEPS if name == "full" else [step]
    retained = [{"step": p, "archive_sha256": f"{p:064x}", "archive_bytes": MIB + 100 * j,
                 "private_archive_path": PRIVATE_MARKER} for j, p in enumerate(points)]
    result = {
        "schema_version": 1, "phase": name, "source_commits": SOURCE_COMMITS,
        "reproduction_commit": "1" * 40, "dataset_sha256": DATASET_SHA256,
        "motion_archive_sha256": ("b" if name == "smoke" else "c") * 64,
        "environment_sha256": "d" * 64, "code_sha256": "e" * 64,
        "allocations": [alloc(job_id)], "num_envs": environments, "checkpoint_step": step,
        "checkpoint_job_id": job_id, "checkpoint_archive_sha256": retained[-1]["archive_sha256"],
        "checkpoint_archive_bytes": retained[-1]["archive_bytes"],
        "checkpoint_artifact_sha256": {
            key: "f" * 64 for key in ("actor.pt", "world_model.pt", "training_state.pt", "run-config.json")
        },
        "retained_checkpoints": retained, "motions": {"clips": motions, "frames": motions * 50, "fps": 50.0,
                                                      "filenames": [PRIVATE_MARKER]},
        "motion_manifest": {"clips": motions, "frames": motions * 50, "fps": 50, "input_fps": 120,
                            "sha256": "3" * 64, "clipset_sha256": "4" * 64,
                            "selection": "archive-order-smoke" if name == "smoke" else "explicit-allowlist",
                            "private_dir": PRIVATE_MARKER},
        "outputs_archive_sha256": "9" * 64, "verified_checks": sorted(CHECKS),
        "held_out_evaluation_verified": eval,
    }
    if name != "smoke":
        result["motion_manifest"].update({
            "selection_id": ("supermimic-v3-appendix-e-approx-v1" if approximate else "supermimic-v3-authors-manifest"),
            "approximate": approximate, "allowlist_sha256": "5" * 64,
        })
    if name == "benchmark":
        samples = [{"step": p, "iterations": 200, "compute_seconds": 30 if p == 200 else 20,
                    "checkpoint_seconds": 1.75, "checkpoint_archive_bytes": MIB, "raw_log": PRIVATE_MARKER}
                   for p in BENCHMARK_STEPS]
        elapsed = 140.5
        result["benchmark"] = {
            "loop_elapsed_seconds": elapsed, "end_to_end_iterations_per_second": step / elapsed,
            "steady_iterations_per_second": 10.0, "steady_environment_steps_per_second": 40960,
            "mean_steady_checkpoint_seconds": 1.75,
            "estimated_320000_iteration_seconds_at_1000_step_saves": 32560,
            "checkpoint_samples": samples,
        }
    if eval:
        result["verified_checks"].append("held_out_evaluation")
        result["held_out_evaluation"] = {
            **alloc("104"), "metrics_sha256": "7" * 64, "outputs_archive_sha256": "8" * 64,
            "suite": {"source": REPOSITORY, "revision": REVISION, "clips": 40, "archive_sha256": "2" * 64,
                      "clip_manifest_sha256": sha256_file(MANIFEST), "private_files": [PRIVATE_MARKER]},
            "metrics": {"evaluated_clips": 40, "completed_clips": 34, "failed_clips": 6,
                        "completion_rate": 0.85, "evaluated_steps": 335, "reference_steps": 500,
                        "mean_normalized_return": 0.64, "mean_root_position_error_m": 0.14,
                        "mean_heading_aligned_body_position_error_m": 0.16, "raw": PRIVATE_MARKER},
        }
    return result


def stages(result):
    return {item["id"]: item["state"] for item in result["phases"]}


class PublicPhaseSummaryTest(unittest.TestCase):
    def test_preflight_only_and_dataset_must_be_independently_checked(self):
        prior = preflight(air=False)
        pending = generate(prior, FIXTURE_DATE)
        self.assertEqual(stages(pending)["data"], "in-progress")
        self.assertEqual(stages(pending)["smoke"], "blocked")
        self.assertEqual(pending["series"], [])
        self.assertEqual(pending["training"]["completedIterations"], 0)
        with self.assertRaisesRegex(ValueError, "independent AIR dataset checksum"):
            generate(prior, FIXTURE_DATE, smoke=phase("smoke"))
        prior["dataset"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "independently checked BONES revision"):
            generate(prior, FIXTURE_DATE)

    def test_benchmark_samples_do_not_count_as_full_training(self):
        reported = generate(preflight(), FIXTURE_DATE, smoke=phase("smoke"), benchmark=phase("benchmark"))
        self.assertEqual(stages(reported)["smoke"], "verified")
        self.assertEqual(stages(reported)["benchmark"], "verified")
        self.assertEqual(stages(reported)["train"], "waiting")
        self.assertEqual(reported["training"]["completedIterations"], 0)
        graph = {item["id"]: item for item in reported["series"]}
        self.assertEqual(len(graph), 4)
        self.assertEqual([pt["step"] for pt in graph["benchmark-iteration-rate"]["points"]], BENCHMARK_STEPS)
        self.assertEqual(graph["benchmark-iteration-rate"]["points"][1]["value"], 10.0)
        self.assertEqual(graph["benchmark-archive-size"]["points"][-1]["value"], 1.0)

    def test_full_and_heldout_are_separate_and_nested_metadata_never_leaks(self):
        args = {"smoke": phase("smoke"), "benchmark": phase("benchmark"), "full": phase("full")}
        trained = generate(preflight(), FIXTURE_DATE, **args)
        self.assertEqual(trained["training"]["completedIterations"], 320000)
        self.assertEqual(stages(trained)["train"], "verified")
        self.assertEqual(stages(trained)["evaluation"], "waiting")
        self.assertFalse(any(item["id"].startswith("heldout-") for item in trained["series"]))
        args["full"] = phase("full", eval=True)
        heldout = generate(preflight(), FIXTURE_DATE, **args)
        self.assertEqual(stages(heldout)["evaluation"], "verified")
        self.assertEqual(stages(heldout)["release"], "waiting")
        self.assertIn("approximate Appendix E selection", heldout["summary"])
        self.assertEqual(len(heldout["series"]), 9)
        self.assertNotIn(PRIVATE_MARKER, json.dumps(heldout))
        self.assertEqual(heldout, generate(preflight(), FIXTURE_DATE, **args))
        self.assertEqual(heldout["series"][-4]["points"], [{"step": 320000, "value": 85.0}])

    def test_author_manifest_label_requires_exact_cohort_marker_and_count(self):
        args = {"smoke": phase("smoke"), "benchmark": phase("benchmark", approximate=False),
                "full": phase("full", approximate=False)}
        report = generate(preflight(), FIXTURE_DATE, **args)
        self.assertIn("authors’ Appendix E clip manifest", report["summary"])
        self.assertNotIn("reproduced the paper", report["summary"])
        args["full"]["motion_manifest"]["approximate"] = True
        with self.assertRaisesRegex(ValueError, "Inconsistent Appendix E"):
            generate(preflight(), FIXTURE_DATE, **args)

    def test_refuses_missing_unsupported_or_conflicting_audits(self):
        good = {"smoke": phase("smoke"), "benchmark": phase("benchmark"), "full": phase("full", eval=True)}
        with self.assertRaisesRegex(ValueError, "in sequence"):
            generate(preflight(), FIXTURE_DATE, full=good["full"])
        for route, key, wrong in (
            (("smoke",), "num_envs", 4096),
            (("smoke",), "phase", "full"),
            (("benchmark", "benchmark"), "steady_iterations_per_second", 99),
            (("full",), "motion_archive_sha256", "6" * 64),
            (("full", "motion_manifest"), "sha256", "6" * 64),
            (("full", "motion_manifest"), "clipset_sha256", "6" * 64),
            (("full", "motion_manifest"), "allowlist_sha256", "6" * 64),
            (("full", "held_out_evaluation", "metrics"), "completion_rate", 0.90),
            (("full", "held_out_evaluation", "metrics"), "mean_root_position_error_m", float("nan")),
        ):
            bad = copy.deepcopy(good)
            obj = bad
            for part in route:
                obj = obj[part]
            obj[key] = wrong
            with self.subTest(route=route, key=key), self.assertRaises(ValueError):
                generate(preflight(), FIXTURE_DATE, **bad)
        bad = copy.deepcopy(good)
        del bad["full"]["held_out_evaluation"]["metrics"]["mean_normalized_return"]
        with self.assertRaisesRegex(ValueError, "held-out return"):
            generate(preflight(), FIXTURE_DATE, **bad)

    def test_cli_is_atomic_on_invalid_input_and_deterministic_on_valid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "preflight.json", root / "status.json"
            source.write_text(json.dumps(preflight(air=False)))
            script = Path(__file__).with_name("publish_phase_summary.py")
            cmd = [sys.executable, str(script), "--preflight", str(source), "--date", FIXTURE_DATE,
                   "--output", str(output)]
            subprocess.run(cmd, check=True)
            public_bytes = output.read_bytes()
            subprocess.run(cmd, check=True)
            self.assertEqual(output.read_bytes(), public_bytes)
            source.write_text('{"invalid": NaN}')
            rejected = subprocess.run(cmd, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(output.read_bytes(), public_bytes)
            source.write_text('{"key": 1, "key": 2}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_json(source)
            source.write_text('{"nested": {"value": 1e999}}')
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                read_json(source)


if __name__ == "__main__":
    unittest.main()

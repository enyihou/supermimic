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
    EXPANDED_BYTES, MANIFEST, MAX_JSON_BYTES, MIB, MILESTONE_STEPS, RAW_CLIPS, RECIPE, REPOSITORY, REVISION,
    SOURCE_COMMITS, generate, read_json, sha256_file,
)


FIXTURE_DATE = "2026-11-21"
PRIVATE_MARKER = "/test/secret/DO-NOT-PUBLISH"


def preflight(*, air=True):
    return {
        "schema_version": 1, "phase": "dataset_preflight", "updated_utc": FIXTURE_DATE,
        "source_commits": dict(SOURCE_COMMITS),
        "status": "local_and_air_verified" if air else "local_verified_air_transfer_pending",
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "archive": "g1.tar.gz",
                    "sha256": DATASET_SHA256, "archive_bytes": ARCHIVE_BYTES, "csv_files": RAW_CLIPS,
                    "csv_expanded_bytes": EXPANDED_BYTES, "local_sha256_verified": True,
                    "local_archive_members_verified": True, "air_sha256_verified": air},
    }


def alloc(job_id):
    return {"job_id": job_id, "state": "COMPLETED", "exit_code": "0:0", "allocated_gpus": 1}


def phase(name, *, eval=False, approximate=True):
    environments, step = RECIPE[name]
    job_id = {"smoke": "101", "benchmark": "102", "full": "103"}[name]
    motions = 64 if name == "smoke" else (129530 if approximate else 129514)
    points = MILESTONE_STEPS if name == "full" else [step]
    retained = [{"step": p, "archive_sha256": f"{p:064x}", "archive_bytes": MIB + 100 * j}
                for j, p in enumerate(points)]
    result = {
        "schema_version": 1, "phase": name, "source_commits": dict(SOURCE_COMMITS),
        "reproduction_commit": "1" * 40, "dataset_sha256": DATASET_SHA256,
        "motion_archive_sha256": ("b" if name == "smoke" else "c") * 64,
        "environment_sha256": "d" * 64, "code_sha256": "e" * 64,
        "allocations": [alloc(job_id)], "num_envs": environments, "checkpoint_step": step,
        "checkpoint_job_id": job_id, "checkpoint_archive_sha256": retained[-1]["archive_sha256"],
        "checkpoint_archive_bytes": retained[-1]["archive_bytes"],
        "checkpoint_artifact_sha256": {
            key: "f" * 64 for key in ("actor.pt", "world_model.pt", "training_state.pt", "run-config.json")
        },
        "retained_checkpoints": retained, "motions": {"clips": motions, "frames": motions * 50, "fps": 50},
        "motion_manifest": {"clips": motions, "frames": motions * 50, "fps": 50, "input_fps": 120,
                            "sha256": "3" * 64, "clipset_sha256": "4" * 64,
                            "selection": "archive-order-smoke" if name == "smoke" else "explicit-allowlist"},
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
                    "checkpoint_seconds": 1.75, "checkpoint_archive_bytes": MIB}
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
                      "clip_manifest_sha256": sha256_file(MANIFEST)},
            "metrics": {"evaluated_clips": 40, "completed_clips": 34, "failed_clips": 6,
                        "completion_rate": 0.85, "evaluated_steps": 335, "reference_steps": 500,
                        "mean_normalized_return": 0.64, "mean_root_position_error_m": 0.14,
                        "mean_heading_aligned_body_position_error_m": 0.16},
        }
    return result


def curves():
    return {
        "schema_version": 1, "metrics_jsonl_sha256": "5" * 64,
        "coverage": {"scope": "final-allocation-only", "resumed_from_iterations": 319980,
                     "log_interval": 10, "first_logged_step": 319981, "last_logged_step": 319991, "logged_rows": 2},
        "series": [{"id": "train-episode-reward", "observation_count": 2,
                    "points": [{"step": 319981, "value": 0.75}, {"step": 319991, "value": 0.83}]}],
    }


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

    def test_full_and_heldout_are_separate(self):
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

    def test_only_verified_final_allocation_measurements_are_graphed(self):
        args = {"smoke": phase("smoke"), "benchmark": phase("benchmark"),
                "full": phase("full", eval=True)}
        args["full"]["learning_curves"] = curves()
        report = generate(preflight(), FIXTURE_DATE, **args)
        chart = next(item for item in report["series"] if item["id"] == "train-episode-reward")
        self.assertEqual(chart["points"], [{"step": 319981, "value": 0.75}, {"step": 319991, "value": 0.83}])
        detail = next(row["detail"] for row in report["phases"] if row["id"] == "train")
        self.assertIn("Earlier allocations are not graphed (resumed at 319,980).", detail)
        args["full"]["learning_curves"]["series"][0]["points"][0]["step"] = 319971
        with self.assertRaisesRegex(ValueError, "curve point"):
            generate(preflight(), FIXTURE_DATE, **args)

    def test_rejects_private_fields_in_public_evidence_at_all_depths(self):
        good = {"preflight": preflight(), "smoke": phase("smoke"), "benchmark": phase("benchmark"),
                "full": phase("full", eval=True)}
        good["full"]["learning_curves"] = curves()
        generate(updated_at=FIXTURE_DATE, **good)
        for route in (
            ("preflight",), ("preflight", "source_commits"), ("preflight", "dataset"),
            ("smoke",), ("smoke", "source_commits"), ("smoke", "allocations", 0), ("smoke", "motions"),
            ("smoke", "motion_manifest"), ("smoke", "retained_checkpoints", 0),
            ("smoke", "checkpoint_artifact_sha256"),
            ("benchmark",), ("benchmark", "benchmark"), ("benchmark", "benchmark", "checkpoint_samples", 0),
            ("full",), ("full", "held_out_evaluation"), ("full", "held_out_evaluation", "suite"),
            ("full", "held_out_evaluation", "metrics"), ("full", "learning_curves"),
            ("full", "learning_curves", "coverage"), ("full", "learning_curves", "series", 0),
            ("full", "learning_curves", "series", 0, "points", 0),
        ):
            bad = copy.deepcopy(good)
            obj = bad
            for part in route:
                obj = obj[part]
            obj["unexpected_private_field"] = PRIVATE_MARKER
            with self.subTest(route=route):
                with self.assertRaisesRegex(ValueError, "Unexpected or missing fields") as error:
                    generate(updated_at=FIXTURE_DATE, **bad)
                self.assertNotIn(PRIVATE_MARKER, str(error.exception))
        bad = copy.deepcopy(good)
        bad["preflight"]["status"] = PRIVATE_MARKER
        with self.assertRaisesRegex(ValueError, "preflight status"):
            generate(updated_at=FIXTURE_DATE, **bad)
        bad = copy.deepcopy(good)
        bad["smoke"]["verified_checks"].append(PRIVATE_MARKER)
        with self.assertRaisesRegex(ValueError, "Unexpected or missing audit checks"):
            generate(updated_at=FIXTURE_DATE, **bad)
        bad = copy.deepcopy(good)
        bad["smoke"]["checkpoint_artifact_sha256"]["critic.pt"] = "a" * 64
        self.assertEqual(generate(updated_at=FIXTURE_DATE, **bad), generate(updated_at=FIXTURE_DATE, **good))
        bad["smoke"]["checkpoint_artifact_sha256"]["critic.pt"] = PRIVATE_MARKER
        with self.assertRaisesRegex(ValueError, "artifact critic.pt"):
            generate(updated_at=FIXTURE_DATE, **bad)

    def test_registry_release_requires_full_and_attached_evaluation(self):
        model = {"schema_version": 1, "active": "g1-trained-v1"}
        inactive = {"schema_version": 1, "active": None}
        self.assertEqual(generate(preflight(), FIXTURE_DATE, registry=inactive), generate(preflight(), FIXTURE_DATE))
        with self.assertRaisesRegex(ValueError, "requires an audited full run"):
            generate(preflight(), FIXTURE_DATE, registry=model)
        args = {"smoke": phase("smoke"), "benchmark": phase("benchmark"), "full": phase("full")}
        with self.assertRaisesRegex(ValueError, "requires verified held-out evaluation"):
            generate(preflight(), FIXTURE_DATE, **args, registry=model)
        args["full"] = phase("full", eval=True)
        report = generate(preflight(), FIXTURE_DATE, **args, registry=model)
        self.assertEqual(stages(report)["release"], "verified")
        self.assertIn(model["active"], report["summary"])
        with self.assertRaisesRegex(ValueError, "Invalid active"):
            generate(preflight(), FIXTURE_DATE, **args, registry={**model, "active": "g1-fixture-v1"})

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
        with self.assertRaisesRegex(ValueError, "held-out metrics"):
            generate(preflight(), FIXTURE_DATE, **bad)

    def test_cli_is_atomic_on_invalid_input_and_deterministic_on_valid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, registry, output = root / "preflight.json", root / "releases.json", root / "status.json"
            source.write_text(json.dumps(preflight(air=False)))
            registry.write_text(json.dumps({"schema_version": 1, "active": None}))
            script = Path(__file__).with_name("publish_phase_summary.py")
            cmd = [sys.executable, str(script), "--preflight", str(source), "--date", FIXTURE_DATE,
                   "--registry", str(registry), "--output", str(output)]
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
            source.write_text(" " * (MAX_JSON_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "size bounds"):
                read_json(source)
            link = root / "linked.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular file"):
                read_json(link)


if __name__ == "__main__":
    unittest.main()

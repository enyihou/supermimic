"""Build a public status report from verified phase audits. Does not deploy it."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path

from audit_official import SOURCE_COMMITS, atomic_json, sha256_file
from evaluate_heldout import MANIFEST, REPOSITORY, REVISION


DATASET_ID = "bones-studio/seed"
DATASET_REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
DATASET_SHA256 = "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"
ARCHIVE_BYTES = 23499973647
EXPANDED_BYTES = 51527587825
RAW_CLIPS = 142220
EVAL_CLIPS = 40
MIB = 2**20
RECIPE = {"smoke": (256, 200), "benchmark": (4096, 1200), "full": (4096, 320000)}
SELECTION_IDS = {"supermimic-v3-authors-manifest": False, "supermimic-v3-appendix-e-approx-v1": True}
MILESTONE_STEPS = list(range(32000, 320001, 32000))
BENCHMARK_STEPS = list(range(200, 1201, 200))
CHECKS = frozenset({
    "slurm_terminal_state", "job_status", "success_marker", "source_pins", "dataset_checksum",
    "packed_motion_checksum", "target_step", "checkpoint_archive_integrity", "milestone_retention",
})
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
JOB = re.compile(r"[1-9][0-9]*")
LIMIT = 2**53 - 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value, label) -> dict:
    require(isinstance(value, dict), f"Expected object for {label}")
    return value


def integer(value, label, low=0, high=LIMIT) -> int:
    require(type(value) is int and low <= value <= high, f"Invalid integer for {label}")
    return value


def real(value, label, low=-LIMIT, high=LIMIT, *, positive=False) -> float:
    require(type(value) in (int, float) and low <= value <= high and math.isfinite(value)
            and (not positive or value > 0), f"Invalid numeric value for {label}")
    return float(value)


def fingerprint(value, label, pattern=HEX64) -> str:
    require(isinstance(value, str) and pattern.fullmatch(value) is not None, f"Invalid fingerprint for {label}")
    return value


def exact_fields(obj: dict, expected: dict, label: str) -> None:
    require(all(type(obj.get(key)) is type(value) and obj[key] == value
                for key, value in expected.items()), f"Inconsistent evidence for {label}")


def iso_day(text: str) -> date:
    require(isinstance(text, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) is not None,
            "Expected an ISO date (YYYY-MM-DD)")
    return date.fromisoformat(text)


def read_json(path: Path) -> dict:
    def forbidden_constant(_):
        raise ValueError("Non-finite JSON constant")

    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), "Non-finite JSON number")
        return value

    def unique_keys(pairs):
        obj = dict(pairs)
        require(len(obj) == len(pairs), "Duplicate JSON object field")
        return obj

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=forbidden_constant,
                          parse_float=finite_float, object_pairs_hook=unique_keys)
    return mapping(value, "input JSON")


def verify_preflight(preflight: dict, report_date: date) -> bool:
    exact_fields(preflight, {"schema_version": 1, "phase": "dataset_preflight"}, "preflight")
    exact_fields(mapping(preflight.get("source_commits"), "preflight pins"), SOURCE_COMMITS, "preflight source pins")
    require(iso_day(preflight.get("updated_utc")) <= report_date, "Report date predates preflight evidence")
    dataset = mapping(preflight.get("dataset"), "preflight dataset")
    exact_fields(dataset, {
        "id": DATASET_ID, "revision": DATASET_REVISION, "archive": "g1.tar.gz",
        "sha256": DATASET_SHA256, "archive_bytes": ARCHIVE_BYTES, "csv_files": RAW_CLIPS,
        "csv_expanded_bytes": EXPANDED_BYTES, "local_sha256_verified": True,
        "local_archive_members_verified": True,
    }, "independently checked BONES revision")
    require(type(dataset.get("air_sha256_verified")) is bool, "Missing AIR archive verification state")
    return dataset["air_sha256_verified"]


def terminal(row, label: str) -> str:
    obj = mapping(row, label)
    exact_fields(obj, {"state": "COMPLETED", "exit_code": "0:0", "allocated_gpus": 1}, label)
    return fingerprint(obj.get("job_id"), f"{label} job ID", JOB)


def verify_phase(audit: dict, phase: str) -> dict:
    envs, step = RECIPE[phase]
    exact_fields(audit, {
        "schema_version": 1, "phase": phase, "num_envs": envs, "checkpoint_step": step,
        "dataset_sha256": DATASET_SHA256,
    }, f"{phase} recipe")
    exact_fields(mapping(audit.get("source_commits"), f"{phase} pins"), SOURCE_COMMITS, f"{phase} source pins")
    for label in ("motion_archive_sha256", "code_sha256", "environment_sha256",
                  "checkpoint_archive_sha256", "outputs_archive_sha256"):
        fingerprint(audit.get(label), f"{phase} {label}")
    fingerprint(audit.get("reproduction_commit"), f"{phase} reproduction revision", HEX40)
    fingerprint(audit.get("checkpoint_job_id"), f"{phase} checkpoint job ID", JOB)
    checkpoint_bytes = integer(audit.get("checkpoint_archive_bytes"), f"{phase} archive size", low=1)
    artifacts = mapping(audit.get("checkpoint_artifact_sha256"), f"{phase} artifacts")
    for name in ("actor.pt", "world_model.pt", "training_state.pt", "run-config.json"):
        fingerprint(artifacts.get(name), f"{phase} artifact {name}")
    passed = audit.get("verified_checks")
    require(isinstance(passed, list) and all(isinstance(item, str) for item in passed)
            and CHECKS <= set(passed), f"Missing successful audit checks for {phase}")
    allocations = audit.get("allocations")
    require(isinstance(allocations, list) and len(allocations) == 1, f"Missing final allocation record for {phase}")
    finished_job = terminal(allocations[0], f"{phase} final allocation")

    motions = mapping(audit.get("motions"), f"{phase} motions")
    clips = integer(motions.get("clips"), f"{phase} motion count", low=1, high=RAW_CLIPS)
    frames = integer(motions.get("frames"), f"{phase} motion frames", low=2 * clips)
    require(real(motions.get("fps"), f"{phase} frame rate") == 50.0, f"Unexpected motion frame rate for {phase}")
    if phase == "smoke":
        require(clips == 64, "Smoke must use the 64-clip subset")
    else:
        require(clips > 64, f"{phase} requires the full reviewed selection, not a smoke subset")
    manifest = mapping(audit.get("motion_manifest"), f"{phase} motion provenance")
    cohort = {"clips": clips, "frames": frames, "input_fps": 120, "fps": 50,
              "selection": "archive-order-smoke" if phase == "smoke" else "explicit-allowlist"}
    exact_fields(manifest, cohort, f"{phase} motion provenance")
    for key in ("sha256", "clipset_sha256"):
        cohort[key] = fingerprint(manifest.get(key), f"{phase} motion {key}")
    if phase != "smoke":
        selection = manifest.get("selection_id")
        require(isinstance(selection, str) and selection in SELECTION_IDS, f"Unknown cohort for {phase}")
        approximate = SELECTION_IDS[selection]
        require(manifest.get("approximate") is approximate and (approximate or clips == 129_514),
                f"Inconsistent Appendix E provenance for {phase}")
        cohort.update(selection_id=selection, approximate=approximate,
                      allowlist_sha256=fingerprint(manifest.get("allowlist_sha256"), f"{phase} allowlist"))
    else:
        require(not {"selection_id", "approximate", "allowlist_sha256"} & manifest.keys(),
                "Smoke must not claim authors' or approximate full cohort")
    retained = audit.get("retained_checkpoints")
    required_steps = MILESTONE_STEPS if phase == "full" else [step]
    require(isinstance(retained, list) and len(retained) == len(required_steps), f"Unexpected retained steps for {phase}")
    for row, expected_step in zip(retained, required_steps, strict=True):
        mapping(row, f"{phase} retained checkpoint")
        exact_fields(row, {"step": expected_step}, f"{phase} retained checkpoint step")
        integer(row.get("archive_bytes"), f"{phase} retained checkpoint size", low=1)
        fingerprint(row.get("archive_sha256"), f"{phase} retained checkpoint")
    exact_fields(retained[-1], {"archive_sha256": audit["checkpoint_archive_sha256"],
                                "archive_bytes": checkpoint_bytes}, f"{phase} target checkpoint")
    held_out = audit.get("held_out_evaluation_verified")
    require(type(held_out) is bool, f"Missing evaluation flag for {phase}")
    if not held_out or phase != "full":
        require(held_out is False and "held_out_evaluation" not in audit and "held_out_evaluation" not in passed,
                f"Unexpected held-out evaluation state for {phase}")
    if phase != "benchmark":
        require("benchmark" not in audit, f"Unexpected benchmark data for {phase}")
    return {"step": step, "envs": envs, "clips": clips, "frames": frames, "archive_bytes": checkpoint_bytes,
            "finished_job": finished_job, "retained": retained, "cohort": cohort}


def make_series(id_: str, label: str, unit: str, points) -> dict:
    return {"id": id_, "label": label, "unit": unit,
            "points": [{"step": step, "value": value} for step, value in points]}


def series_benchmark(audit: dict, stage: dict) -> tuple[list[dict], float]:
    obj = mapping(audit.get("benchmark"), "benchmark metrics")
    samples = obj.get("checkpoint_samples")
    require(isinstance(samples, list) and len(samples) == len(BENCHMARK_STEPS), "Missing benchmark checkpoint windows")
    computes, publishes, sizes = [], [], []
    for row, step in zip(samples, BENCHMARK_STEPS, strict=True):
        mapping(row, "benchmark checkpoint sample")
        exact_fields(row, {"step": step, "iterations": 200}, "benchmark sample extent")
        computes.append(real(row.get("compute_seconds"), "benchmark sample compute time", positive=True))
        publishes.append(real(row.get("checkpoint_seconds"), "benchmark sample checkpoint time", positive=True))
        sizes.append(integer(row.get("checkpoint_archive_bytes"), "benchmark sample archive size", low=1))
    require(sizes[-1] == stage["archive_bytes"], "Benchmark final sampled checkpoint size differs from audit")
    elapsed = real(obj.get("loop_elapsed_seconds"), "benchmark end-to-end loop time", positive=True)
    total = sum(computes) + sum(publishes)
    require(elapsed + max(0.01, elapsed * 1e-6) >= total, "Benchmark elapsed time is shorter than its sample durations")
    steady_speed = (len(computes) - 1) * 200 / sum(computes[1:])
    ckpt_avg = sum(publishes[1:]) / (len(publishes) - 1)
    for key, expected in {
        "end_to_end_iterations_per_second": stage["step"] / elapsed,
        "steady_iterations_per_second": steady_speed,
        "steady_environment_steps_per_second": stage["envs"] * steady_speed,
        "mean_steady_checkpoint_seconds": ckpt_avg,
        "estimated_320000_iteration_seconds_at_1000_step_saves": 320000 / steady_speed + 320 * ckpt_avg,
    }.items():
        value = real(obj.get(key), key, positive=True)
        require(math.isclose(expected, value, rel_tol=1e-8, abs_tol=1e-8), f"Inconsistent benchmark metric: {key}")
    rates = [real(200 / value, "benchmark interval iteration rate", positive=True) for value in computes]
    return [
        make_series("benchmark-iteration-rate", "Benchmark interval throughput", "iterations/s", zip(BENCHMARK_STEPS, rates)),
        make_series("benchmark-archive-size", "Benchmark checkpoint size", "MiB", zip(BENCHMARK_STEPS, (v / MIB for v in sizes))),
        make_series("benchmark-checkpoint-time", "Benchmark checkpoint write and backup time", "s", zip(BENCHMARK_STEPS, publishes)),
    ], steady_speed


def series_evaluation(audit: dict, claimed_jobs: set[str]) -> tuple[list[dict], int]:
    require("held_out_evaluation" in audit["verified_checks"], "Held-out verification check is missing")
    held_out = mapping(audit.get("held_out_evaluation"), "held-out result")
    eval_job = terminal(held_out, "held-out allocation")
    require(eval_job not in claimed_jobs, "Evaluation must have its own allocation")
    fingerprint(held_out.get("metrics_sha256"), "held-out metrics")
    fingerprint(held_out.get("outputs_archive_sha256"), "held-out outputs")
    suite = mapping(held_out.get("suite"), "held-out suite")
    exact_fields(suite, {
        "source": REPOSITORY, "revision": REVISION, "clips": EVAL_CLIPS,
        "clip_manifest_sha256": sha256_file(MANIFEST),
    }, "independent held-out suite")
    require(fingerprint(suite.get("archive_sha256"), "held-out archive") != DATASET_SHA256, "Holdout shares training dataset")
    values = mapping(held_out.get("metrics"), "held-out metrics")
    exact_fields(values, {"evaluated_clips": EVAL_CLIPS}, "held-out evaluated clip count")
    completed = integer(values.get("completed_clips"), "completed held-out clips", high=EVAL_CLIPS)
    failed = integer(values.get("failed_clips"), "failed held-out clips", high=EVAL_CLIPS)
    steps = integer(values.get("evaluated_steps"), "held-out evaluated steps", low=EVAL_CLIPS)
    reference = integer(values.get("reference_steps"), "held-out reference steps", low=steps)
    rate = real(values.get("completion_rate"), "held-out completion rate", low=0, high=1)
    require(completed + failed == EVAL_CLIPS and rate == completed / EVAL_CLIPS
            and (failed > 0 or reference == steps), "Held-out counts and rate are inconsistent")
    step = audit["checkpoint_step"]
    rows = (
        ("heldout-completion", "Held-out clip completion", "%", 100 * rate),
        ("heldout-return", "Held-out normalized return", "", real(values.get("mean_normalized_return"), "held-out return", high=1.00001)),
        ("heldout-root-error", "Held-out mean root position error", "m", real(values.get("mean_root_position_error_m"), "held-out root error", low=0)),
        ("heldout-aligned-body-error", "Held-out mean heading-aligned body position error", "m",
         real(values.get("mean_heading_aligned_body_position_error_m"), "held-out aligned body error", low=0)),
    )
    return [make_series(id_, label, unit, [(step, value)]) for id_, label, unit, value in rows], completed


def cohort_description(stage: dict) -> str:
    if stage["cohort"]["approximate"]:
        return f"{stage['clips']:,} BONES-SEED clips selected by our approximate Appendix E filter"
    return f"{stage['clips']:,} BONES-SEED clips selected by the authors’ Appendix E manifest"


def generate(preflight: dict, updated_at: str, *, smoke=None, benchmark=None, full=None) -> dict:
    """Regenerate the complete status; no private fields or old graph data pass through."""
    air_verified = verify_preflight(preflight, iso_day(updated_at))
    require(air_verified or all(row is None for row in (smoke, benchmark, full)),
            "Training audits require the independent AIR dataset checksum")
    require((smoke is not None or benchmark is None) and (benchmark is not None or full is None),
            "Phase audits must be supplied in sequence: smoke, benchmark, full")
    series: list[dict] = []
    summary = ("The local archive and source pins are verified. Training has not started.")
    label = {
        "source": ("Pinned sources and launcher", "verified", "Both upstream source revisions are pinned and verified."),
        "data": ("Dataset preflight", "in-progress",
                 "The local archive hash and inventory passed for 142,220 G1 CSVs. The AIR checksum is pending."),
        "smoke": ("Official smoke", "waiting",
                  "The 64-clip smoke job requires approved private storage accessible from its allocation."),
        "benchmark": ("Throughput and checkpoint size", "waiting",
                      "Measure iteration rate and checkpoint size at the planned training scale."),
        "train": ("Single-GPU training", "waiting", "Full training follows a verified smoke test and benchmark."),
        "evaluation": ("Held-out evaluation", "waiting",
                       "No trained checkpoint has been evaluated on the separate LAFAN1 G1 suite."),
        "release": ("Browser model release", "waiting", "Release requires an audited checkpoint and verified browser export."),
    }
    if air_verified:
        label["data"] = (label["data"][0], "verified",
                         "The local and AIR archive SHA-256 checks match. The local inventory verified 142,220 G1 CSVs.")
        summary = "The BONES-SEED G1 archive is checksum-verified on AIR. No training phase is verified yet."
    else:
        label["smoke"] = (label["smoke"][0], "blocked", "The smoke job requires a verified AIR dataset copy.")
    jobs = set()
    step = 0
    if smoke is not None:
        audit = verify_phase(smoke, "smoke")
        jobs.update((audit["finished_job"], smoke["checkpoint_job_id"]))
        label["smoke"] = (label["smoke"][0], "verified",
                          "The one-GPU job completed 200 iterations with 256 environments and 64 clips. Its checkpoint verified.")
        series.append(make_series("smoke-archive-size", "Smoke checkpoint size", "MiB",
                                  [(audit["step"], audit["archive_bytes"] / MIB)]))
        summary = "The official smoke test and its checkpoint passed audit. Full training has not been verified."
    if benchmark is not None:
        audit = verify_phase(benchmark, "benchmark")
        benchmark_stage = audit
        current_jobs = {audit["finished_job"], benchmark["checkpoint_job_id"]}
        require(jobs.isdisjoint(current_jobs), "Benchmark and smoke must be separate allocations")
        jobs.update(current_jobs)
        points, rate = series_benchmark(benchmark, audit)
        series.extend(points)
        label["benchmark"] = (label["benchmark"][0], "verified",
                              f"The separate 1,200-iteration benchmark measured {rate:.3g} steady iterations/s "
                              f"across 4,096 environments, excluding its first window. Input: {cohort_description(audit)}.")
        summary = "The official smoke test and throughput benchmark passed audit. Benchmark iterations are not full training."
    if full is not None:
        audit = verify_phase(full, "full")
        current_jobs = {audit["finished_job"], full["checkpoint_job_id"]}
        require(jobs.isdisjoint(current_jobs), "Full training must use separate allocations")
        jobs.update(current_jobs)
        require(full["motion_archive_sha256"] == benchmark["motion_archive_sha256"]
                and full["environment_sha256"] == benchmark["environment_sha256"],
                "Full training does not match the benchmark's packed dataset and environment")
        require(audit["cohort"] == benchmark_stage["cohort"],
                "Full training motion provenance differs from the benchmark")
        step = audit["step"]
        label["train"] = (label["train"][0], "verified",
                          f"Completed 320,000 iterations with 4,096 environments on one GPU. "
                          f"Ten retained checkpoints verified. Input: {cohort_description(audit)}."
                          + (" This is not the authors’ exact training cohort." if audit["cohort"]["approximate"] else ""))
        series.append(make_series("train-archive-size", "Full-run milestone checkpoint size", "MiB",
                                  ((row["step"], row["archive_bytes"] / MIB) for row in audit["retained"])))
        method = (" with an approximate Appendix E selection" if audit["cohort"]["approximate"]
                  else " with the authors’ Appendix E clip manifest")
        summary = f"The pinned implementation completed the full single-GPU training target{method}. Held-out evaluation is pending."
        if full["held_out_evaluation_verified"]:
            points, completed = series_evaluation(full, jobs)
            series.extend(points)
            label["evaluation"] = (label["evaluation"][0], "verified",
                                   f"The final checkpoint was evaluated on 40 separate LAFAN1 G1 clips: "
                                   f"{completed} reached the end without a training termination. This is not the paper’s training-set benchmark.")
            label["release"] = (label["release"][0], "waiting", "The audited checkpoint still needs a verified browser release.")
            summary = (f"Full single-GPU training{method} and independent held-out evaluation passed their audits. "
                       "Browser release is pending.")
    return {
        "schemaVersion": 1,
        "updatedAt": updated_at,
        "summary": summary,
        "training": {"numEnvironments": RECIPE["full"][0], "completedIterations": step,
                     "targetIterations": RECIPE["full"][1]},
        "phases": [{"id": id_, "label": name, "state": state, "detail": detail}
                   for id_, (name, state, detail) in label.items()],
        "series": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path, help="Independent local/AIR dataset preflight JSON")
    parser.add_argument("--date", required=True, help="Explicit status date YYYY-MM-DD")
    parser.add_argument("--smoke", type=Path, help="Verified smoke audit JSON")
    parser.add_argument("--benchmark", type=Path, help="Verified benchmark audit JSON")
    parser.add_argument("--full", type=Path, help="Verified full audit JSON, optionally with attached held-out evidence")
    parser.add_argument("--output", required=True, type=Path, help="Status JSON to write and review separately from deployment")
    args = parser.parse_args()
    try:
        paths = [path for path in (args.preflight, args.smoke, args.benchmark, args.full) if path is not None]
        require(all(path.resolve() != args.output.resolve() for path in paths),
                "Output must not replace an audit or preflight input")
        inputs = {phase: read_json(path) if (path := getattr(args, phase)) is not None else None
                  for phase in RECIPE}
        report = generate(read_json(args.preflight), args.date, **inputs)
        atomic_json(args.output, report)
    except (OSError, ValueError, OverflowError, ZeroDivisionError) as exc:
        parser.exit(1, f"Phase summary rejected: {exc}\n")


if __name__ == "__main__":
    main()

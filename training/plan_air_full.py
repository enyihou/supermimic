"""Size the full Slurm run from a verified 4,096-environment benchmark."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from publish_phase_summary import (
    MILESTONE_STEPS, RECIPE, atomic_json, read_json, require, series_benchmark, verify_phase, verify_prepared_cohort,
)


MAX_SECONDS = 3 * 86400  # train partition at the time of this implementation; recheck the runbook
SIGNAL_SECONDS = 900
FULL_SAVE_INTERVAL = 1000
TERMINATION_SAVE_SECONDS = 600  # run_full.sbatch's grace for a verified checkpoint


def plan(benchmark: dict, cohort: dict, smoke: dict) -> dict:
    verify_phase(smoke, "smoke")
    verify_prepared_cohort(cohort, smoke)
    stage = verify_phase(benchmark, "benchmark")
    _, rate = series_benchmark(benchmark, stage)
    require(all(cohort[key] == benchmark[key] for key in ("motion_archive_sha256", "motion_manifest",
                "code_sha256", "environment_sha256", "reproduction_commit")),
            "Benchmark is not bound to the approved preparation and package")
    target = RECIPE["full"][1]
    rows = benchmark["benchmark"]["checkpoint_samples"]
    steady = rows[1:]
    slowest = max(row["compute_seconds"] / row["iterations"] for row in steady)
    longest_save = max(row["checkpoint_seconds"] for row in steady)
    largest = max(row["checkpoint_archive_bytes"] for row in rows)
    require(longest_save * 1.25 + 30 < TERMINATION_SAVE_SECONDS,
            "Measured checkpoint publication time needs a longer signal/save window before launch")

    # Initial setup and compilation will repeat on a new allocation. The first
    # benchmark window includes its warmup; add 10 minutes for archive stage-in.
    startup = max(1800, math.ceil(rows[0]["compute_seconds"] + rows[0]["checkpoint_seconds"] + 600))
    # Worst steady benchmark window, 25% headroom, and explicit periodic saves.
    run_seconds = math.ceil(1.25 * (target * slowest + (target // FULL_SAVE_INTERVAL) * longest_save))
    usable = MAX_SECONDS - startup - SIGNAL_SECONDS
    require(usable > 0, "Benchmark startup is incompatible with an AIR allocation")
    allocations = math.ceil(run_seconds / usable)
    required = startup + run_seconds + SIGNAL_SECONDS
    # Do not request a three-day reservation when one shorter job is enough.
    walltime = MAX_SECONDS if allocations > 1 else min(MAX_SECONDS, math.ceil(required / 3600) * 3600)
    current_checkpoint_bytes = math.ceil(largest * 1.25)
    keep_count = len(MILESTONE_STEPS)
    # Ceph: milestones + rolling latest + space for simultaneous stage-out and
    # a prior unpruned copy. Canonical: milestones + saved interruption points.
    return {
        "schema_version": 1, "phase": "full_allocation_plan", "benchmark_job_id": stage["finished_job"],
        "approximate": stage["cohort"]["approximate"], "selection_id": stage["cohort"]["selection_id"],
        "motion_archive_sha256": benchmark["motion_archive_sha256"],
        "gpus": 1, "num_envs": stage["envs"], "target_iterations": target,
        "steady_iterations_per_second": rate, "slowest_steady_seconds_per_iteration": slowest,
        "longest_steady_checkpoint_seconds": longest_save, "largest_measured_checkpoint_bytes": largest,
        "startup_seconds_per_allocation": startup, "signal_lead_seconds": SIGNAL_SECONDS,
        "estimated_steady_work_seconds_with_headroom": run_seconds, "time_limit_seconds": walltime,
        "estimated_allocations": allocations, "save_interval": FULL_SAVE_INTERVAL,
        "milestone_interval": MILESTONE_STEPS[0],
        "estimated_additional_ceph_bytes": (keep_count + 3) * current_checkpoint_bytes + allocations * (1 << 30),
        "estimated_additional_canonical_bytes": (keep_count + allocations) * current_checkpoint_bytes + allocations * (1 << 30),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = (args.benchmark, args.cohort, args.smoke)
        require(all(path.resolve() != args.output.resolve() for path in paths), "Plan must not replace phase evidence")
        atomic_json(args.output, plan(*(read_json(path) for path in paths)))
    except (OSError, ValueError, OverflowError, RecursionError) as error:
        parser.exit(1, f"Could not size full allocation: {error}\n")


if __name__ == "__main__":
    main()

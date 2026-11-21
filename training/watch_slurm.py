"""Save private, authoritative terminal Slurm evidence on gpu01."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from audit_official import JOB_ID, atomic_json, capture_slurm


TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "SPECIAL_EXIT", "TIMEOUT",
}


def watch(job_id: str, output: Path, interval: int, maximum: int) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir() or (output.parent.stat().st_mode & 0o077):
        raise ValueError("Terminal record needs an existing owner-only destination")
    deadline = time.monotonic() + maximum
    last = None
    while time.monotonic() < deadline:
        try:
            record = capture_slurm(job_id)
        except (OSError, ValueError, subprocess.SubprocessError):
            # The controller can be transiently unavailable or the record
            # missing. Neither means that the job ended, and neither justifies
            # a competing submission. Keep polling this exact job ID.
            state = "controller record unavailable"
        else:
            state = record["state"]
            if state.rstrip("+") in TERMINAL:
                atomic_json(output, record)
                return record
        if state != last:
            print(f"{job_id}: {state}", file=sys.stderr, flush=True)
            last = state
        time.sleep(interval)
    raise TimeoutError(f"No terminal record observed for {job_id}; check it explicitly before resuming")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--maximum", type=int, default=7 * 86400)
    args = parser.parse_args()
    if not JOB_ID.fullmatch(args.job_id) or not 5 <= args.interval <= 60 or args.maximum < args.interval:
        parser.error("Expected a numeric job ID, 5–60 second interval, and a larger observation budget")
    result = watch(args.job_id, args.output, args.interval, args.maximum)
    print(json.dumps(result, sort_keys=True))

"""Verify and project the pinned runner's allocation-local metrics into public curves."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from typing import BinaryIO


METRICS_MEMBER = "metrics.jsonl"
MAX_METRICS_BYTES = 256 * 1024**2
MAX_ROW_BYTES = 8192
MAX_POINTS = 128
SAFE_NUMBER = 2**53 - 1
SHA256 = re.compile(r"[0-9a-f]{64}")
SCOPE = "final-allocation-only"
METRICS = {
    "train-episode-reward": ("Train/mean_reward", "Training mean episode reward", "reward"),
    "train-episode-length": ("Train/mean_episode_length", "Training mean episode length", "steps"),
    "train-world-model-loss": ("Loss/state", "World model training loss", ""),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cadence(resumed: int, target: int, interval: int) -> range:
    require(type(resumed) is int and type(target) is int and 0 <= resumed <= target <= 320000
            and type(interval) is int and 1 <= interval <= 320000, "invalid training metrics cadence")
    # Pinned runner logs when zero-based step % interval == 0, but records step+1.
    first = ((resumed + interval - 1) // interval) * interval + 1
    return range(first, target + 1, interval)


def parse_row(line: bytes) -> dict:
    def no_duplicate(pairs):
        obj = dict(pairs)
        require(len(obj) == len(pairs), "duplicate training metrics field")
        return obj

    def no_nonfinite(_):
        raise ValueError("training metrics contain a non-finite JSON constant")

    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), "training metrics contain a non-finite JSON number")
        return value

    try:
        obj = json.loads(line, object_pairs_hook=no_duplicate, parse_constant=no_nonfinite,
                         parse_float=finite_float)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        # Some JSON decoder errors would otherwise reveal bytes from private logs.
        raise ValueError("invalid training metrics JSON row") from exc
    require(isinstance(obj, dict), "training metrics row must be an object")
    return obj


def numeric(value, metric: str) -> bool:
    if not (type(value) in (int, float) and -SAFE_NUMBER <= value <= SAFE_NUMBER and math.isfinite(value)):
        return False
    if metric == "train-episode-length":
        return value > 0
    if metric == "train-world-model-loss":
        return value >= 0
    return True


def downsample(rows: list[tuple[int, float]]) -> list[dict]:
    """Select uniformly in observation index, preserving first and last actual values."""
    n = len(rows)
    indices = range(n) if n <= MAX_POINTS else (i * (n - 1) // (MAX_POINTS - 1) for i in range(MAX_POINTS))
    return [{"step": rows[i][0], "value": rows[i][1]} for i in indices]


def parse_metrics(stream: BinaryIO, length: int) -> dict:
    """Stream one size-bounded tar member; keep only numeric allowlisted rows."""
    require(type(length) is int and 0 < length <= MAX_METRICS_BYTES,
            "training metrics archive member exceeds size bounds")
    digest = hashlib.sha256()
    count = 0
    size = 0
    first = last = previous_delta = previous_elapsed = None
    selected: dict[str, list[tuple[int, float]]] = {key: [] for key in METRICS}
    while line := stream.readline(MAX_ROW_BYTES + 1):
        require(len(line) <= MAX_ROW_BYTES and line.endswith(b"\n"), "training metrics row exceeds bounds or is truncated")
        digest.update(line)
        size += len(line)
        count += 1
        require(size <= length and count <= 320000, "training metrics exceed supported row bounds")
        obj = parse_row(line)
        step = obj.get("completed_iterations")
        require(type(step) is int and 1 <= step <= 320000 and (last is None or step > last),
                "training metrics steps must be unique, positive, and strictly increasing")
        if last is not None:
            delta = step - last
            require(previous_delta is None or delta == previous_delta, "training metrics steps have inconsistent cadence")
            previous_delta = delta
        if first is None:
            first = step
        last = step
        elapsed = obj.get("loop_elapsed_seconds")
        require(type(elapsed) in (int, float) and 0 <= elapsed <= SAFE_NUMBER and math.isfinite(elapsed)
                and (previous_elapsed is None or elapsed >= previous_elapsed), "invalid training metric elapsed time")
        previous_elapsed = elapsed
        metrics = obj.get("metrics")
        require(isinstance(metrics, dict), "training metrics row lacks an object payload")
        for id_, (key, _, _) in METRICS.items():
            value = metrics.get(key)
            if value is not None:  # Pinned writer maps non-finite observations to null.
                require(numeric(value, id_), "invalid value for allowlisted training metric")
                selected[id_].append((step, float(value)))
    require(size == length and count > 0, "empty or truncated training metrics archive member")
    return {
        "sha256": digest.hexdigest(),
        "rows": count,
        "first": first,
        "last": last,
        "delta": previous_delta,
        "elapsed": previous_elapsed,
        "series": [{"id": id_, "observation_count": len(rows), "points": downsample(rows)}
                   for id_, rows in selected.items() if rows],
    }


def learning_curves(parsed: dict | None, resumed: int, target: int, interval: int, elapsed: float) -> dict:
    expected = cadence(resumed, target, interval)
    if not expected:
        require(parsed is None, "training metrics exist although no iterations were logged")
    else:
        require(isinstance(parsed, dict), "attempt archive is missing expected training metrics")
        require(parsed["rows"] == len(expected) and parsed["first"] == expected[0]
                and parsed["last"] == expected[-1] and (len(expected) == 1 or parsed["delta"] == interval)
                and parsed["elapsed"] <= elapsed, "training metrics do not cover the expected allocation-local cadence")
    return {
        "schema_version": 1,
        "coverage": {
            "scope": SCOPE,
            "resumed_from_iterations": resumed,
            "log_interval": interval,
            "first_logged_step": parsed["first"] if parsed else None,
            "last_logged_step": parsed["last"] if parsed else None,
            "logged_rows": len(expected),
        },
        "metrics_jsonl_sha256": parsed["sha256"] if parsed else None,
        "series": parsed["series"] if parsed else [],
    }


def public_series(input_curves, target: int) -> tuple[list[dict], dict]:
    """Validate the public audit shape again; construct new chart objects, never forward raw fields."""
    require(isinstance(input_curves, dict) and type(input_curves.get("schema_version")) is int
            and input_curves["schema_version"] == 1, "unsupported training curves schema")
    coverage = input_curves.get("coverage")
    require(isinstance(coverage, dict) and coverage.get("scope") == SCOPE, "invalid training curves coverage")
    expected = cadence(coverage.get("resumed_from_iterations"), target, coverage.get("log_interval"))
    fixed = {"first_logged_step": expected[0] if expected else None,
             "last_logged_step": expected[-1] if expected else None, "logged_rows": len(expected)}
    require(all(key in coverage and type(coverage[key]) is type(value) and coverage[key] == value
                for key, value in fixed.items()), "training curves coverage disagrees with the logged cadence")
    digest = input_curves.get("metrics_jsonl_sha256")
    require((isinstance(digest, str) and SHA256.fullmatch(digest)) if expected
            else ("metrics_jsonl_sha256" in input_curves and digest is None), "training curves lack a valid log digest")
    series = input_curves.get("series")
    require(isinstance(series, list) and 0 <= len(series) <= (len(METRICS) if expected else 0),
            "invalid training curves series list")
    chart = []
    ordered_ids = list(METRICS)
    previous_series = -1
    for entry in series:
        require(isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"] in METRICS,
                "unexpected training curve ID")
        id_ = entry["id"]
        index = ordered_ids.index(id_)
        count = entry.get("observation_count")
        points = entry.get("points")
        require(index > previous_series and type(count) is int and 1 <= count <= len(expected)
                and isinstance(points, list) and len(points) == min(count, MAX_POINTS), "invalid training curve observations")
        previous_series = index
        last_step = coverage["resumed_from_iterations"]
        data = []
        for row in points:
            require(isinstance(row, dict) and type(row.get("step")) is int and row["step"] in expected
                    and row["step"] > last_step and numeric(row.get("value"), id_), "invalid training curve point")
            last_step = row["step"]
            data.append({"step": last_step, "value": row["value"]})
        _, label, unit = METRICS[id_]
        chart.append({"id": id_, "label": label, "unit": unit, "points": data})
    return chart, {**fixed, "resumed_from_iterations": coverage["resumed_from_iterations"]}

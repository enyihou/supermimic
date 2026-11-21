"""Synthetic logger files for metrics audit validation; no real run artifacts."""

import hashlib
import io
import json
import unittest

from audit_training_metrics import (
    MAX_METRICS_BYTES, MAX_POINTS, MAX_ROW_BYTES, cadence, learning_curves, parse_metrics, public_series,
)


def record(step, value=0.6, **metrics):
    return {"completed_iterations": step, "loop_elapsed_seconds": step / 10,
            "metrics": {"Train/mean_reward": value, **metrics}}


def raw(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def parse(data):
    return parse_metrics(io.BytesIO(data), len(data))


class TrainingMetricsTest(unittest.TestCase):
    def test_zero_based_logger_predicate_uses_one_based_completed_steps(self):
        self.assertEqual(list(cadence(0, 30, 10)), [1, 11, 21])
        self.assertEqual(list(cadence(1, 30, 10)), [11, 21])
        self.assertEqual(list(cadence(9, 30, 10)), [11, 21])
        self.assertEqual(list(cadence(10, 30, 10)), [11, 21])
        self.assertEqual(list(cadence(21, 30, 10)), [])
        self.assertEqual(list(cadence(30, 30, 10)), [])

    def test_remaining_allocation_is_never_padded_with_pre_resume_points(self):
        data = raw([record(step, other="/secret/run") for step in [319981, 319991]])
        result = learning_curves(parse(data), 319976, 320000, 10, 40000.0)
        self.assertEqual(result["coverage"]["resumed_from_iterations"], 319976)
        self.assertEqual(result["coverage"]["first_logged_step"], 319981)
        self.assertEqual(result["coverage"]["last_logged_step"], 319991)
        self.assertEqual(result["coverage"]["logged_rows"], 2)
        self.assertEqual(result["metrics_jsonl_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(result["series"][0]["points"], [
            {"step": 319981, "value": 0.6}, {"step": 319991, "value": 0.6},
        ])
        self.assertNotIn("/secret", json.dumps(result))
        with self.assertRaisesRegex(ValueError, "cadence"):
            learning_curves(parse(data), 319960, 320000, 10, 40000.0)

    def test_no_rows_is_allowed_only_when_no_log_slot_was_due(self):
        result = learning_curves(None, 320000, 320000, 10, 3.0)
        self.assertEqual(result["coverage"]["logged_rows"], 0)
        self.assertIsNone(result["metrics_jsonl_sha256"])
        self.assertEqual(result["series"], [])
        self.assertEqual(public_series(result, 320000)[0], [])
        with self.assertRaisesRegex(ValueError, "missing expected"):
            learning_curves(None, 319990, 320000, 10, 3.0)

    def test_downsampling_keeps_actual_endpoints_and_ignores_nulls(self):
        rows = [record(step, float(step), **{"Loss/state": 0.5 if step % 20 == 11 else None})
                for step in cadence(0, 2000, 10)]
        result = learning_curves(parse(raw(rows)), 0, 2000, 10, 220)
        reward, loss = result["series"]
        self.assertEqual(reward["observation_count"], 200)
        self.assertEqual(len(reward["points"]), MAX_POINTS)
        self.assertEqual(reward["points"][0], {"step": 1, "value": 1})
        self.assertEqual(reward["points"][-1], {"step": 1991, "value": 1991})
        self.assertTrue(all(point["step"] == point["value"] for point in reward["points"]))
        self.assertEqual(loss["observation_count"], 100)
        self.assertEqual(loss["points"][0], {"step": 11, "value": 0.5})
        chart, coverage = public_series(result, 2000)
        self.assertEqual(coverage["logged_rows"], 200)
        self.assertEqual(chart[1]["id"], "train-world-model-loss")
        self.assertEqual(chart[1]["points"], loss["points"])

    def test_reject_malformed_unbounded_nonfinite_and_out_of_order_rows(self):
        vectors = [
            (raw([record(11), record(11)]), "strictly increasing"),
            (raw([record(1), record(11), record(15)]), "inconsistent cadence"),
            (raw([record(1, float("inf"))]), "non-finite"),
            (raw([record(1, True)]), "allowlisted"),
            (raw([record(1, **{"Loss/state": -1})]), "allowlisted"),
            (raw([record(1, **{"Train/mean_episode_length": 0})]), "allowlisted"),
            (raw([record(1, "private-value")]), "allowlisted"),
            (b" " * (MAX_ROW_BYTES + 1) + b"\n", "row exceeds"),
            (raw([record(1)])[:-1], "truncated"),
            (b"{}\n", "strictly increasing"),
            (b'{"duplicate":0,"duplicate":1}\n', "duplicate"),
        ]
        for data, message in vectors:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse(data)
        with self.assertRaisesRegex(ValueError, "size bounds"):
            parse_metrics(io.BytesIO(), MAX_METRICS_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "truncated"):
            parse_metrics(io.BytesIO(raw([record(1)])), 10000)

    def test_status_projection_rejects_out_of_cadence_or_fabricated_rows(self):
        result = learning_curves(parse(raw([record(1), record(11)])), 0, 20, 10, 2)
        result["series"][0]["points"][1]["step"] = 12
        with self.assertRaisesRegex(ValueError, "curve point"):
            public_series(result, 20)
        result["series"][0]["points"][1]["step"] = 11
        result["coverage"]["logged_rows"] = 3
        with self.assertRaisesRegex(ValueError, "cadence"):
            public_series(result, 20)


if __name__ == "__main__":
    unittest.main()

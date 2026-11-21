"""Exercise actor export with synthetic weights and no dataset or checkpoint."""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from export_browser_policy import build_robot_scene, new_pinned_actor, write_actor, write_scene


def fixture(output: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    actor = new_pinned_actor()
    with torch.no_grad():
        for normalizer in (actor.obs_normalizer, actor.goal_normalizers["motion"]):
            normalizer._mean.normal_(0, 0.1)
            normalizer._std.uniform_(0.2, 2.0)
    parity, error = write_actor(actor, output / "actor.onnx")
    (output / "parity.json").write_text(json.dumps({"parity": parity, "export_max_abs_error": error}))
    print(f"Synthetic pinned actor PyTorch/ONNX parity passed (max abs error {error:.3g})")


def robot_fixture(output: Path) -> None:
    # Check the pinned G1 scene compiler and native MJB writer with a mocked
    # world bundle, not a checkpoint or any other learned state.
    _, robot = build_robot_scene()
    with tempfile.TemporaryDirectory(prefix="browser-robot-test-") as temp:
        private = Path(temp)
        torch.save({"dynamics": {key: torch.tensor(robot[key]) for key in ("action_scale", "action_offset")}}, private / "world_model.pt")
        actual = write_scene(private, output / "scene.mjbpack")
        assert all(actual[key] == value for key, value in robot.items())
    (output / "robot.json").write_text(json.dumps(actual))
    print(f"Pinned G1 scene round-tripped: {len(actual['bodies'])} robot bodies")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--fixture-dir", type=Path, help="Write an ephemeral ONNX fixture, never a real trained release")
    args.add_argument("--g1-scene", action="store_true", help="Also exercise the full pinned mjlab G1 scene compiler")
    opts = args.parse_args()
    if opts.fixture_dir:
        opts.fixture_dir.mkdir(exist_ok=False, parents=True)
        fixture(opts.fixture_dir)
        if opts.g1_scene:
            robot_fixture(opts.fixture_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="browser-export-test-") as temp:
            fixture(Path(temp))
            if opts.g1_scene:
                robot_fixture(Path(temp))

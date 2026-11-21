#!/usr/bin/env python3
"""Export an audited GenCoord-G1 actor and its independently sourced MJLab scene.

Run in the pinned SuperMimic environment. No motion files, replay buffers,
optimizer state, world-model tensors or raw checkpoints enter the output.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn


PINS = {
    "supermimic": "5fb92297ab4fe3d86629d9f0c635334791bb0d5f",
    "mjlab_research": "379c55e88b1dec73f93943e524a879b2ebbf45c5",
}
FORMAT = "supermimic-gencoord-browser-v1"
TASK = "SuperMimic-GenCoord-G1"
MUJOCO_VERSION = "4.0.1"
STATE_DIM, GOAL_DIM, ACTION_DIM = 68, 640, 29
GOAL_FRAMES, GOAL_STRIDE = 10, 5


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_pin(module_path: str, expected: str) -> Path:
    src = Path(module_path).resolve().parent
    git = ["git", "-C", str(src)]
    root = Path(subprocess.check_output([*git, "rev-parse", "--show-toplevel"], text=True).strip())
    rev = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
    if rev != expected or dirty:
        raise ValueError(f"{root.name}: expected a clean checkout at {expected}")
    return root


class MotionActor(nn.Module):
    """The deterministic forward path of the pinned MultiEncoderAE motion slot."""

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.state_normalizer = actor.obs_normalizer
        self.goal_normalizer = actor.goal_normalizers["motion"]
        self.encoder = actor.encoders["motion"]
        self.decoder = actor.decoder

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.encoder(self.goal_normalizer(goal)))
        return self.decoder(torch.cat((z, self.state_normalizer(state)), dim=-1))


def new_pinned_actor() -> nn.Module:
    from supermimic.models import MultiEncoderAE

    # Shapes and slot mix from the pinned default GenCoord training config.
    # The inactive velocity branch is loaded strictly but is not exported.
    return MultiEncoderAE(
        obs_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        task_specs=[
            SimpleNamespace(name="motion", goal_dim=GOAL_DIM),
            SimpleNamespace(name="velocity", goal_dim=3),
        ],
        slot_mix={"motion": 1.0, "velocity": 0.0},
        latent_dim=32,
        hidden_dims=(2048, 1024, 512, 256),
        latent_bc_weight=0.1,
        history_length=1,
    ).cpu().eval()


def write_actor(actor: nn.Module, path: Path) -> tuple[list[dict], float]:
    from tensordict import TensorDict

    wrapped = MotionActor(actor).cpu().eval()
    inputs = (torch.zeros(1, STATE_DIM), torch.zeros(1, GOAL_DIM))
    torch.onnx.export(
        wrapped,
        inputs,
        str(path),
        export_params=True,
        opset_version=18,
        input_names=["state", "goal"],
        output_names=["action"],
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    reference = ReferenceEvaluator(graph)
    # Synthetic inputs only. Exercise all features and keep one load-time
    # vector that also checks the actual browser ONNX execution provider.
    rng = torch.Generator().manual_seed(412013)
    samples, max_error = [], 0.0
    with torch.inference_mode():
        for scale in (0.0, 0.1, 1.0, 3.0):
            state = torch.randn((1, STATE_DIM), generator=rng) * scale
            goal = torch.randn((1, GOAL_DIM), generator=rng) * scale
            expected = actor(TensorDict({"history": state[:, None, :], "goal_motion": goal}, [1])).numpy()
            actual, = reference.run(["action"], {"state": state.numpy(), "goal": goal.numpy()})
            if actual.shape != (1, ACTION_DIM) or not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise ValueError("Unexpected or non-finite policy output")
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            if scale in (0.1, 1.0):
                samples.append({"state": state[0].tolist(), "goal": goal[0].tolist(), "action": expected[0].tolist()})
    return samples, max_error


def build_robot_scene() -> tuple[object, dict]:
    import mujoco
    from mjlab.scene import Scene
    from mjlab.utils.lab_api.string import resolve_matching_names_values
    from supermimic.tasks.env_cfgs import gencoord_env_cfg

    if mujoco.mj_versionString() != MUJOCO_VERSION:
        raise ValueError(f"Expected native MuJoCo {MUJOCO_VERSION}")
    cfg = gencoord_env_cfg()
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    robot = scene["robot"]
    _, action_names = robot.find_joints_by_actuator_names(cfg.actions["joint_pos"].actuator_names)
    if tuple(action_names) != robot.joint_names or len(action_names) != ACTION_DIM:
        raise ValueError("Unexpected training joint/action order")

    key = model.key("init_state")
    qpos = np.asarray(key.qpos, dtype=np.float64)
    joints = []
    for name in robot.joint_names:
        joint_id = model.joint(f"robot/{name}").id
        actuator_ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if actuator_ids.size != 1 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Unexpected actuator setup for {name}")
        joints.append({"name": f"robot/{name}", "actuator": model.actuator(int(actuator_ids[0])).name})

    action_cfg = cfg.actions["joint_pos"]
    indices, _, values = resolve_matching_names_values(action_cfg.scale, action_names)
    expected_scale = np.ones(ACTION_DIM)
    expected_scale[indices] = values
    q_indices = [model.jnt_qposadr[model.joint(joint["name"]).id] for joint in joints]
    if not action_cfg.use_default_offset or action_cfg.clip is not None:
        raise ValueError("Unexpected training action transform")

    body_ids = [model.body(f"robot/{name}").id for name in robot.body_names]
    body_order = {body_id: i for i, body_id in enumerate(body_ids)}
    bodies = [
        {"name": model.body(body_id).name, "parent": body_order.get(int(model.body_parentid[body_id]), -1)}
        for body_id in body_ids
    ]
    root_body = model.body("robot/pelvis").name
    free_joint = model.joint(int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).item())).name
    return model, {
        "engine_version": MUJOCO_VERSION,
        "timestep": model.opt.timestep,
        "substeps": cfg.decimation,
        "keyframe": key.name,
        "root_body": root_body,
        "free_joint": free_joint,
        "joints": joints,
        "bodies": bodies,
        "action_scale": expected_scale.astype(np.float32).tolist(),
        "action_offset": qpos[q_indices].astype(np.float32).tolist(),
    }


def write_scene(checkpoint_dir: Path, path: Path) -> dict:
    import mujoco

    model, robot = build_robot_scene()
    dynamics = torch.load(checkpoint_dir / "world_model.pt", map_location="cpu", weights_only=True)["dynamics"]
    for key in ("action_scale", "action_offset"):
        actual = np.asarray(dynamics[key], dtype=np.float32).reshape(-1)
        np.testing.assert_allclose(actual, robot[key], rtol=1e-6, atol=1e-7)
        robot[key] = actual.tolist()  # Use the values actually saved by training.
    with tempfile.TemporaryDirectory(prefix=".mjb-", dir=path.parent) as temporary:
        binary = Path(temporary) / "scene.mjb"
        mujoco.mj_saveModel(model, str(binary), None)
        # Round trip with the matching native engine; JS checks the names/dims again.
        restored = mujoco.MjModel.from_binary_path(str(binary))
        if (restored.nq, restored.nv, restored.nu) != (36, 35, ACTION_DIM) or restored.nmesh < 1:
            raise ValueError("Unexpected exported MuJoCo model dimensions or geometry")
        robot["scene_uncompressed_bytes"] = binary.stat().st_size
        with binary.open("rb") as source, path.open("wb") as compressed:
            with gzip.GzipFile(filename="", fileobj=compressed, mode="wb", compresslevel=3, mtime=0) as output:
                shutil.copyfileobj(source, output)
    return robot


def digest_arg(value: str, length: int) -> str:
    if not re.fullmatch(rf"[a-f0-9]{{{length}}}", value):
        raise argparse.ArgumentTypeError(f"Expected a lowercase {length}-digit hash")
    return value


def read_public_metrics(path: Path) -> dict[str, float]:
    metrics = json.loads(path.read_text())
    if not isinstance(metrics, dict) or not metrics or len(metrics) > 100:
        raise ValueError("Provide a nonempty, flat public aggregate metrics object")
    if any(
        not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
        or type(value) not in (int, float)
        or not math.isfinite(value)
        for name, value in metrics.items()
    ):
        raise ValueError("Metrics must have lowercase identifiers and finite numeric values")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="Private directory containing actor.pt and world_model.pt")
    parser.add_argument("output_dir", type=Path, help="New version directory, initially outside the public tree")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--expected-actor-sha256", type=lambda s: digest_arg(s, 64), required=True)
    parser.add_argument("--expected-world-model-sha256", type=lambda s: digest_arg(s, 64), required=True)
    parser.add_argument("--dataset-sha256", type=lambda s: digest_arg(s, 64), required=True)
    parser.add_argument("--reproduction-commit", type=lambda s: digest_arg(s, 40), required=True)
    parser.add_argument("--metrics", type=Path, required=True, help="Audited public, flat aggregate numeric JSON")
    args = parser.parse_args()
    if args.checkpoint_step < 1 or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.model_id):
        parser.error("Expected a positive checkpoint step and a lowercase hyphenated model ID")
    if args.output_dir.exists() or args.output_dir.name != args.model_id:
        parser.error("Output must be a new directory whose name equals the model ID")

    import mjlab
    import supermimic

    mjlab_root = check_pin(mjlab.__file__, PINS["mjlab_research"])
    check_pin(supermimic.__file__, PINS["supermimic"])
    actor_path = args.checkpoint_dir / "actor.pt"
    if sha256(actor_path) != args.expected_actor_sha256 or sha256(args.checkpoint_dir / "world_model.pt") != args.expected_world_model_sha256:
        raise ValueError("Private checkpoint does not match the audited hashes")
    metrics = read_public_metrics(args.metrics)
    torch.set_num_threads(1)
    actor = new_pinned_actor()
    actor.load_state_dict(torch.load(actor_path, map_location="cpu", weights_only=True), strict=True)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".browser-export-", dir=args.output_dir.parent) as workdir:
        release = Path(workdir) / args.model_id
        release.mkdir(mode=0o700)
        samples, max_error = write_actor(actor, release / "actor.onnx")
        robot = write_scene(args.checkpoint_dir, release / "scene.mjbpack")
        shutil.copyfile(mjlab_root / "LICENSE", release / "mjlab-LICENSE.txt")
        shutil.copyfile(Path(__file__).parent / "licenses" / "Unitree-BSD-3-Clause.txt", release / "Unitree-LICENSE.txt")
        (release / "ATTRIBUTION.md").write_text(
            "# SuperMimic browser policy\n\n"
            "Training data includes Motion Data by [Bones Studio](https://bones.studio/). "
            "Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.\n\n"
            "The scene is compiled from the pinned [mjlab source](https://github.com/HybridRobotics/mjlab_research), "
            "licensed under Apache-2.0; see mjlab-LICENSE.txt. The model contains [Unitree G1 robot geometry]"
            "(https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/"
            "unitree_robots/g1), BSD-3-Clause; see Unitree-LICENSE.txt.\n"
        )
        artifacts = {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(release.iterdir())
        }
        manifest = {
            "format": FORMAT,
            "model_id": args.model_id,
            "task": TASK,
            "dataset_sha256": args.dataset_sha256,
            "checkpoint_step": args.checkpoint_step,
            "source_commits": {**PINS, "reproduction": args.reproduction_commit},
            "checkpoint_actor_sha256": args.expected_actor_sha256,
            "checkpoint_world_model_sha256": args.expected_world_model_sha256,
            "metrics": metrics,
            "artifacts": artifacts,
            "policy": {
                "state_dim": STATE_DIM, "goal_dim": GOAL_DIM, "action_dim": ACTION_DIM,
                "goal_frames": GOAL_FRAMES, "goal_stride": GOAL_STRIDE,
                "parity": samples, "export_max_abs_error": max_error,
            },
            "robot": robot,
        }
        (release / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        release.rename(args.output_dir)
    print(f"Exported {args.model_id}: actor parity max absolute error {max_error:.3g}")


if __name__ == "__main__":
    main()

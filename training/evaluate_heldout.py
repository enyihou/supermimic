#!/usr/bin/env python3
"""Stage and score the predeclared off-training-corpus G1 evaluation suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import re
import statistics
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from audit_official import SOURCE_COMMITS, atomic_json, require, sha256_file, verify_checkpoint


SUITE_ID = "lafan1-unitree-g1-v1"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
REPOSITORY = "lvhaidong/LAFAN1_Retargeting_Dataset"
MANIFEST = Path(__file__).with_name("lafan1_g1_eval_manifest.tsv")
SEED = 1789
INPUT_FPS = 30
OUTPUT_FPS = 50
JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def suite_manifest() -> dict[str, tuple[int, str]]:
    result = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, size, blob = line.split("\t")
        require(re.fullmatch(r"[a-zA-Z0-9_]+\.csv", name) is not None
                and re.fullmatch(r"[0-9a-f]{40}", blob) is not None and name not in result,
                "invalid pinned clip manifest")
        result[name] = int(size), blob
    require(len(result) == 40, "the pinned suite must contain all 40 G1 motions")
    return result


def verify_clip(data: bytes, size: int, blob: str) -> None:
    require(len(data) == size, "clip byte count does not match pinned tree")
    git_object = hashlib.sha1(f"blob {len(data)}\0".encode())
    git_object.update(data)
    require(git_object.hexdigest() == blob, "clip hash does not match pinned Git blob")


def fetch_suite(output: Path) -> None:
    """Download from local machine without any credentials and pack deterministically."""
    require(not output.exists(), "refusing to replace an existing evaluation archive")
    manifest = suite_manifest()

    def fetch(name: str) -> tuple[str, bytes]:
        size, blob = manifest[name]
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/g1/{name}"
        with urlopen(url, timeout=45) as response:
            data = response.read(size + 1)
        verify_clip(data, size, blob)
        return name, data

    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            with (gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped,
                  tarfile.open(fileobj=zipped, mode="w|", format=tarfile.USTAR_FORMAT) as archive,
                  concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool):
                for name, data in pool.map(fetch, manifest):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    member.mode = 0o600
                    archive.addfile(member, io.BytesIO(data))
            import os
            stream.flush()
            os.fsync(stream.fileno())
            # Publish without replacing a competing fetch.
            output.hardlink_to(temp)
        finally:
            temp.unlink(missing_ok=True)
    print(f"{sha256_file(output)}  {output.name}")


def prepare_clips(archive_path: Path, target: Path) -> list[Path]:
    """Authenticate every CSV in the suite and write temporary mjlab NPZ files."""
    import numpy as np

    manifest = suite_manifest()
    files: dict[str, Path] = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = member.name
            require(name in manifest and name not in files and member.isfile(),
                    "unexpected evaluation archive member")
            size, blob = manifest[name]
            require(member.size == size, "evaluation archive member size mismatch")
            stream = archive.extractfile(member)
            assert stream is not None
            data = stream.read()
            verify_clip(data, size, blob)
            values = np.loadtxt(io.BytesIO(data), delimiter=",", dtype=np.float64)
            path = target / (Path(name).stem + ".npz")
            np.savez(path, fps=[OUTPUT_FPS], joint_names=JOINTS, **resample(values))
            files[name] = path
    require(files.keys() == manifest.keys(), "evaluation archive is missing pinned motions")
    return [files[name] for name in manifest]


def resample(values):
    import numpy as np

    require(values.ndim == 2 and values.shape[0] >= 3 and values.shape[1] == 7 + len(JOINTS)
            and np.isfinite(values).all(), "invalid Unitree G1 LAFAN1 pose data")
    position = values[:, :3]
    quats = values[:, [6, 3, 4, 5]]  # Unitree's xyzw CSV -> mjlab's wxyz
    joints = values[:, 7:]
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    require(np.allclose(norms, 1.0, atol=5e-3), "non-unit LAFAN1 source quaternion")
    quats = quats / norms

    frame_count = (len(values) - 1) * OUTPUT_FPS // INPUT_FPS + 1
    output_indices = np.arange(frame_count) * INPUT_FPS
    lo = output_indices // OUTPUT_FPS
    hi = np.minimum(lo + 1, len(values) - 1)
    weight = (output_indices % OUTPUT_FPS)[:, None] / OUTPUT_FPS
    pos = position[lo] * (1 - weight) + position[hi] * weight
    joint = joints[lo] * (1 - weight) + joints[hi] * weight

    a, b = quats[lo], quats[hi]
    dot = (a * b).sum(axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    blend = np.where(sine > 1e-7, np.sin(weight * angle) / np.maximum(sine, 1e-7), weight)
    rem = np.where(sine > 1e-7, np.sin((1 - weight) * angle) / np.maximum(sine, 1e-7), 1 - weight)
    quat = a * rem + b * blend
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

    # q_next * conjugate(q_prev) gives a centered world-frame rotation delta.
    next_w, next_xyz = quat[2:, :1], quat[2:, 1:]
    prev_w, neg_prev_xyz = quat[:-2, :1], -quat[:-2, 1:]
    rel_w = next_w * prev_w - (next_xyz * neg_prev_xyz).sum(axis=-1, keepdims=True)
    rel_v = next_w * neg_prev_xyz + prev_w * next_xyz + np.cross(next_xyz, neg_prev_xyz)
    rel_v = np.where(rel_w < 0, -rel_v, rel_v)
    magnitude = np.linalg.norm(rel_v, axis=-1, keepdims=True)
    delta = rel_v * (2 * np.arctan2(magnitude, np.abs(rel_w)) / np.maximum(magnitude, 1e-12))
    omega = np.pad(delta * (OUTPUT_FPS / 2), ((1, 1), (0, 0)), mode="edge")
    arrays = dict(root_pos_w=pos, root_quat_w=quat,
                  root_lin_vel_w=np.gradient(pos, 1.0 / OUTPUT_FPS, axis=0),
                  root_ang_vel_w=omega, joint_pos=joint,
                  joint_vel=np.gradient(joint, 1.0 / OUTPUT_FPS, axis=0))
    return {name: array.astype(np.float32) for name, array in arrays.items()}


def deterministic_env(task_id, clips):
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = len(clips)
    cfg.seed = SEED
    cfg.auto_reset = False
    for group in cfg.observations.values():
        group.enable_corruption = False
    motion = cfg.commands["motion"]
    motion.motion_file = [str(path) for path in clips]
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0, 0)
    motion.sampling_mode = "start"
    # Keep all three context terms so the checkpoint's conditioning schema is
    # unchanged, while pinning their applied values to the nominal simulator.
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["base_com"].params["ranges"] = {axis: (0.0, 0.0) for axis in range(3)}
    cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    return cfg


def rollout(clips, checkpoint, config, device):
    import torch
    import supermimic.tasks  # noqa: F401 (register official tasks)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from supermimic.representations import extract_actor_goal
    from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
    from supermimic.runners.supermimic_runner import build_actor, build_world_model
    from supermimic.tasks.task_spec import build_actor_obs_td

    if device == "cuda:0":
        require(torch.cuda.device_count() == 1, "evaluation requires one visible allocated GPU")
    task_id = config.get("task_id", "SuperMimic-GenCoord-G1")
    require(task_id == "SuperMimic-GenCoord-G1", "unexpected task for G1 held-out evaluation")
    cfg = load_runner_cfg(config)
    check_representation_matches(task_id, cfg.representation)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=deterministic_env(task_id, clips), device=device))
    try:
        wm = build_world_model(cfg, env, device)
        wm.load_bundle(torch.load(checkpoint / "world_model.pt", weights_only=True, map_location=device))
        actor = build_actor(cfg, wm, device)
        actor.set_slot_mix({"motion": 1.0})
        actor.load_state_dict(torch.load(checkpoint / "actor.pt", weights_only=True, map_location=device))
        actor.eval()
        rep = wm.representation
        base = env.unwrapped
        motion = base.command_manager.get_term("motion")
        require(abs(motion.motion.fps * base.step_dt - 1.0) < 1e-7, "reference and control rate differ")
        require(motion.motion.motion_names == [path.stem for path in clips], "loaded evaluation suite differs")
        starts = torch.cat([motion.motion.motion_ends.new_zeros(1), motion.motion.motion_ends[:-1]])
        lengths = motion.motion.motion_ends - starts
        with torch.inference_mode():
            env.reset()
            indices = torch.arange(len(clips), device=device)
            for index, frame in enumerate(starts.tolist()):
                motion.reset_to_frame(indices[index:index + 1], frame)
            base.scene.write_data_to_sim()
            base.sim.forward()
            base.sim.sense()
            motion.update_relative_body_poses()
            require(torch.equal(motion.time_steps, starts), "rollout did not initialize at clip starts")

            active = torch.ones_like(lengths, dtype=torch.bool)
            successes = torch.zeros_like(active)
            steps = torch.zeros_like(lengths)
            accum = torch.zeros((len(clips), 3), dtype=torch.float64, device=device)
            context = wm.sim_context.read(base)
            history = None
            # Score the official pre-step reference/termination semantics. Inactive
            # environments keep the batched allocator shape but never enter metrics.
            for _ in range(int(lengths.max().item())):
                obs = env.get_observations()
                sensor = rep.compute_actor_sensor(rep.represent(obs["full_state"]), context,
                                                  sim_actor_obs=obs["actor"], noisy=False)
                if history is None:
                    history = sensor.unsqueeze(1).repeat(1, cfg.history_length, 1)
                else:
                    history = torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
                action = actor(build_actor_obs_td({"motion": extract_actor_goal(obs["goal"])}, history))
                require(bool(torch.isfinite(action).all()), "actor produced non-finite actions")
                root_error = (motion.robot_anchor_pos_w - motion.anchor_pos_w).norm(dim=-1)
                body_error = (motion.robot_body_pos_w - motion.body_pos_relative_w).norm(dim=-1).mean(dim=-1)
                _, rewards, dones, _ = env.step(action)
                done = dones.bool()
                manager = base.termination_manager
                clip_end = manager.get_term("motion_end")
                require(not bool((active & done & ~clip_end & ~manager.terminated).any()),
                        "an active evaluation clip timed out before its final frame")
                step_values = torch.stack([rewards, root_error, body_error], dim=-1)
                require(bool(torch.isfinite(step_values[active]).all()), "evaluation emitted non-finite metrics")
                accum += torch.where(active[:, None], step_values, 0.0)
                steps += active
                successes |= active & clip_end & ~manager.terminated
                active &= ~done
                if not bool(active.any()):
                    break
                reset_ids = done.nonzero(as_tuple=True)[0]
                if len(reset_ids):
                    base.reset(env_ids=reset_ids)
            require(not bool(active.any()), "some clips did not reach termination or their final frame")
            require(bool(torch.all(~successes | (steps == lengths))), "completed clip length mismatch")
            max_step_reward = base.step_dt * sum(term.weight for term in env.cfg.rewards.values()
                                                 if term.weight > 0.0)
            per_clip = {
                "completed": successes.tolist(),
                "steps": steps.tolist(),
                "reference_steps": lengths.tolist(),
                "normalized_return": (accum[:, 0] / (lengths * max_step_reward)).tolist(),
                "mean_root_position_error_m": (accum[:, 1] / steps).tolist(),
                "mean_heading_aligned_body_position_error_m": (accum[:, 2] / steps).tolist(),
            }
        return per_clip, float(max_step_reward)
    finally:
        env.close()


def evaluate(args) -> dict:
    import os

    meta, config, _, archive_sha = verify_checkpoint(args.checkpoint)
    require(meta.get("source_commits") == SOURCE_COMMITS, "checkpoint pins do not match")
    require(meta.get("dataset_sha256") == config.get("dataset_sha256") == args.training_dataset_sha256,
            "checkpoint training dataset identity does not match inspected BONES input")
    runner = config.get("runner_cfg", {})
    require("wandb_run_path" in runner and runner["wandb_run_path"] is None,
            "cannot establish holdout for a model warm-started from a different run")
    for key, expected in (("code_sha256", args.code_sha256), ("environment_sha256", args.environment_sha256)):
        require(meta.get(key) == config.get(key) == expected,
                "evaluation must use the training job's verified package")
    suite_sha = sha256_file(args.suite)
    require(suite_sha == args.suite_sha256 and suite_sha != args.training_dataset_sha256,
            "staged evaluation suite digest mismatch")
    # Data and model reads from here on are local to the Slurm allocation.
    with tempfile.TemporaryDirectory(dir=args.scratch, prefix="held-out-") as tmp:
        local = Path(tmp)
        clips = prepare_clips(args.suite, local)
        models = {"actor.pt", "world_model.pt"}
        with tarfile.open(args.checkpoint / "checkpoint.tar.gz", "r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name in models:
                    import shutil
                    stream = archive.extractfile(member)
                    assert stream is not None
                    with (local / name).open("xb") as output:
                        shutil.copyfileobj(stream, output)
        per_clip, max_step_reward = rollout(clips, local, config, args.device)
    count = len(per_clip["completed"])
    completed = sum(per_clip["completed"])
    return {
        "schema_version": 1,
        "protocol": SUITE_ID,
        "evaluation_job_id": os.environ.get("SLURM_JOB_ID"),
        "training_job_id": str(meta["job_id"]),
        "checkpoint_step": meta["completed_iterations"],
        "checkpoint_archive_sha256": archive_sha,
        "source_commits": SOURCE_COMMITS,
        "training_dataset_sha256": args.training_dataset_sha256,
        "suite": {"source": REPOSITORY, "revision": REVISION, "archive_sha256": suite_sha,
                  "clip_manifest_sha256": sha256_file(MANIFEST), "clips": count},
        "rollout_config": {"nominal": True, "control_hz": OUTPUT_FPS, "seed": SEED,
                           "foot_tangential_friction": 1.0, "max_step_reward": max_step_reward,
                           "return_normalization": "zero unsimulated tail, per-clip reward ceiling"},
        "metrics": {
            "evaluated_clips": count,
            "completed_clips": completed,
            "failed_clips": count - completed,
            "completion_rate": completed / count,
            "evaluated_steps": sum(per_clip["steps"]),
            "reference_steps": sum(per_clip["reference_steps"]),
            "mean_normalized_return": statistics.mean(per_clip["normalized_return"]),
            "mean_root_position_error_m": statistics.mean(per_clip["mean_root_position_error_m"]),
            "mean_heading_aligned_body_position_error_m": statistics.mean(
                per_clip["mean_heading_aligned_body_position_error_m"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="assemble pinned LAFAN1 input archive locally")
    fetch.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="score a verified checkpoint against the pinned suite")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--training-dataset-sha256", required=True)
    run.add_argument("--code-sha256", required=True)
    run.add_argument("--environment-sha256", required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--suite-sha256", required=True)
    run.add_argument("--scratch", type=Path, required=True)
    run.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch_suite(args.output)
        else:
            atomic_json(args.output, evaluate(args))
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()

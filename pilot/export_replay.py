"""Export two pinned, matched-action comparisons for a private browser replay."""

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from experiment import POST, PRE, SOURCE, action_labels, write_report


ACTOR_SHA = "22a255c1ae64efc4296244b5040a3444b756296daa35cb57a71710b3d03952ca"
WORLD_SHA = "4811ef0877e806ceb8c1d51c902c972662bf36253b618d4f727d9d175a12fa40"
CLIPS = ("walk1_subject1", "run1_subject2")
TRACKS = ("world", "physics", "reference")
HORIZON = 32
ERROR_METRICS = ("rootPositionCm", "rootOrientationDeg", "jointRmseDeg")


def verify(path, digest):
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
            raise ValueError(f"Unexpected digest: {path.name}")


def pose_errors(predicted, actual):
    """Batched [cm, shortest rotation degrees, hinge-angle RMS degrees]."""
    predicted, actual = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    q, p = predicted[..., 3:7], actual[..., 3:7]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    p = np.where((q * p).sum(-1, keepdims=True) < 0, -p, p)
    orientation = 4 * np.arctan2(np.linalg.norm(q - p, axis=-1), np.linalg.norm(q + p, axis=-1))
    return np.stack((
        100 * np.linalg.norm(predicted[..., :3] - actual[..., :3], axis=-1),
        np.rad2deg(orientation),
        np.rad2deg(np.sqrt(np.mean((predicted[..., 7:] - actual[..., 7:]) ** 2, axis=-1))),
    ), axis=-1)


@torch.no_grad()
def record(env, actor, world, *, clips=CLIPS, kicks=(0, .45, -.45) * 2,
           selected=(0, 2), phase=.48, include_errors=False):
    device = env.device
    # Preserve the six-environment layout, clip phase and kick protocol of the smoke.
    motion = torch.tensor(
        [env.library.names.index(clip) for clip in clips], device=device
    ).repeat_interleave(len(kicks) // len(clips))
    start = ((env.library.lengths[motion] - (PRE + POST + 11)) * phase).long()
    inputs, _ = env.reset(options={"motion_id": motion, "frame_id": start})
    kick = torch.tensor(kicks, dtype=torch.float32, device=device)
    pushed = (kick != 0).nonzero().flatten()
    selected = list(selected)
    starts = [0, PRE, *range(PRE + HORIZON, PRE + POST, HORIZON)]
    physics, imagined, reference, errors = [], [], [], []

    def goal(frame):
        ref = env.library.get_frame(motion, frame)
        return torch.cat((ref["body_pos_w"][:, 0], ref["body_quat_w"][:, 0], ref["joint_pos"]), -1)

    pose_size = 7 + env.layout.num_joints
    physical = env.read_state().clone()
    predicted = physical.clone()
    physics.append(physical[selected, :pose_size])
    imagined.append(predicted[selected, :pose_size])
    reference.append(goal(start)[selected])
    if include_errors:
        forecasts, holds, ages = [predicted[selected, :pose_size]], [physical[selected, :pose_size]], [0]
        anchor_time = 0
        anchor_pose = holds[0]
    for t in range(PRE + POST):
        _, history, features = action_labels(inputs, actor)
        action = actor.forward_features(history, features)
        if t == PRE:
            velocity = physical[pushed, pose_size:pose_size + 6].clone()
            velocity[:, 1] += kick[pushed]
            env.robot.write_root_link_velocity_to_sim(velocity, env_ids=pushed)
        inputs, _, terminated, truncated, info = env.step(action)
        if bool((terminated | truncated).any()):
            raise RuntimeError(f"Unexpected reset at step {t}; refusing to export mixed episodes")
        tx = info["transition"]
        if t in starts:
            predicted = physical.clone()
            if t == PRE:
                # The kick is an external impulse, not a learned input. Start
                # both rollouts with the post-kick velocity, before stepping.
                predicted[pushed, pose_size:pose_size + 6] = velocity
            imagined[-1] = predicted[selected, :pose_size]
            if include_errors:
                anchor_time = t
                anchor_pose = physical[selected, :pose_size]
        torch.testing.assert_close(tx["action"], action if env.foundation_config.action_clip is None
                                   else action.clamp(-env.foundation_config.action_clip,
                                                     env.foundation_config.action_clip))
        predicted = world(predicted, tx["action"], tx["context"])
        if not bool(predicted.isfinite().all()):
            raise RuntimeError(f"Nonfinite world-model prediction at step {t}")
        physical = tx["next_state"]
        physics.append(physical[selected, :pose_size])
        imagined.append(predicted[selected, :pose_size])
        if include_errors:
            # Preserve the incoming forecast even when the displayed pose will be
            # overwritten by the next resync. Freeze the same starting pose as baseline.
            forecasts.append(imagined[-1])
            holds.append(anchor_pose)
            ages.append(t + 1 - anchor_time)
        reference.append(goal(start + t + 1)[selected])
        errors.append(info["metrics"]["root_position_error"][selected].clamp(max=1))
    tail = torch.stack(errors[-10:]).mean(0).cpu().tolist()
    poses = torch.stack(tuple(torch.stack(s) for s in (imagined, physics, reference)), dim=1)
    poses = poses.permute(2, 1, 0, 3).cpu().numpy()
    result = (poses, tail, starts, start[selected].cpu().tolist())
    if include_errors:
        metric_rows = []
        for track in (forecasts, holds):
            raw = torch.stack(track, dim=1).cpu().numpy()
            aligned = pose_errors(raw, poses[:, 1])
            if not np.isfinite(aligned).all():
                raise RuntimeError("Nonfinite pose errors")
            metric_rows.append(aligned.transpose(0, 2, 1).tolist())
        return (*result, [
            {"forecastSteps": ages, **{
                key: {"model": model, "hold": hold}
                for key, model, hold in zip(ERROR_METRICS, model_metrics, hold_metrics, strict=True)
            }} for model_metrics, hold_metrics in zip(*metric_rows, strict=True)
        ])
    return result


def export_geometry(model, joint_names, poses, output):
    """Use native FK: the browser draws only compiled mesh data and rigid poses."""
    data = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected exactly one floating root")
    root = int(model.jnt_qposadr[free[0]])
    joints = []
    for name in joint_names:
        candidates = [j for j in range(model.njnt) if model.joint(j).name.split("/")[-1] == name]
        if len(candidates) != 1:
            raise ValueError(f"Joint mapping not unambiguous: {name}")
        joints.append(int(model.jnt_qposadr[candidates[0]]))

    geoms = np.flatnonzero(
        (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (model.geom_group == 1)
        & (model.geom_bodyid > 0)
    )
    if not len(geoms):
        raise ValueError("No G1 visual meshes found")
    meshes = {}
    descriptors = []
    with (output / "meshes.bin").open("wb") as asset:
        for geom in geoms:
            mesh = int(model.geom_dataid[geom])
            if mesh not in meshes:
                va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                start = asset.tell()
                asset.write(model.mesh_vert[va:va+vn].astype("<f4", copy=False).tobytes())
                middle = asset.tell()
                asset.write(model.mesh_face[fa:fa+fn].astype("<u4", copy=False).tobytes())
                meshes[mesh] = {"verticesAt": start, "vertexCount": vn, "indicesAt": middle,
                                "indexCount": fn * 3}
            descriptors.append(meshes[mesh] | {"rgb": model.geom_rgba[geom, :3].tolist()})

    results = np.empty((*poses.shape[:-1], len(geoms), 7), dtype="<f4")
    for index in np.ndindex(poses.shape[:-1]):
        p = poses[index]
        data.qpos[root:root + 7] = p[:7]
        data.qpos[joints] = p[7:]
        mujoco.mj_kinematics(model, data)
        results[index][:, :3] = data.geom_xpos[geoms]
        for j, geom in enumerate(geoms):
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.geom_xmat[geom])
            results[index][j, 3:] = quaternion[[1, 2, 3, 0]]
    if not np.isfinite(results).all():
        raise RuntimeError("Nonfinite geometry transform")
    (output / "poses.bin").write_bytes(results.tobytes())
    return descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "teacher", "world", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    verify(args.teacher, ACTOR_SHA)
    verify(args.world, WORLD_SHA)
    if torch.cuda.device_count() != 1 or args.output.exists():
        parser.error("Expose exactly one authorized GPU and choose a new output directory")

    from src.tasks.supermimic.config.task_cfg import FoundationConfig
    from src.tasks.supermimic.env import SuperMimicEnv
    from src.tasks.supermimic.rl.inference import load_actor, load_world_model, validate_environment

    device = "cuda:0"
    actor, actor_saved, _ = load_actor(args.teacher, args.manifest, device)
    world, world_saved = load_world_model(args.world, args.manifest, device)
    iteration = actor_saved["iteration"]
    if world_saved.get("iteration") != iteration or actor_saved["simulation"] != world_saved["simulation"]:
        raise ValueError("Checkpoint iteration or simulation fingerprint does not match")
    cfg = actor_saved["config"]
    env = SuperMimicEnv(FoundationConfig(
        manifest=args.manifest, clips=CLIPS, num_envs=6, seed=17,
        episode_length_s=100.0, action_clip=cfg.get("action_clip"),
        relative_orientation=cfg.get("relative_orientation", False),
        motion_anchor_ori_b=cfg.get("motion_anchor_ori_b", False),
    ), device=device)
    try:
        for saved in (actor_saved, world_saved):
            validate_environment(env, saved)
        poses, tail, starts, frames = record(env, actor, world)
        args.output.mkdir()
        geoms = export_geometry(env.sim.mj_model, env.robot.joint_names, poses, args.output)
        labels = ("Nominal walk", "Lateral kick")
        scenarios = [
            {"label": label, "clip": CLIPS[0], "startFrame": frame, "tailErrorM": error,
             "recovered": error <= 0.30, "fell": False, "resyncFrames": starts,
             "kickFrame": None if i == 0 else PRE,
             "targetXY": poses[i, 2, :, :2].tolist()}
            for i, (label, frame, error) in enumerate(zip(labels, frames, tail, strict=True))
        ]
        write_report(args.output / "manifest.json", {
            "schema": 1, "publication": "private", "sourceRevision": SOURCE,
            "actorSha256": ACTOR_SHA, "worldSha256": WORLD_SHA,
            "checkpointIteration": iteration, "frames": poses.shape[2],
            "dt": env.step_dt, "tracks": list(TRACKS), "horizon": HORIZON,
            "geoms": geoms, "scenarios": scenarios,
        })
        print(f"{iteration=}, {len(geoms)=}, {poses.shape=}, {tail=}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

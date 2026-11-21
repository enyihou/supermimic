"""Generate a motion from robot geometry and equations, then record it on AIR."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from experiment import POST, PRE, SOURCE, write_report
from export_replay import ACTOR_SHA, HORIZON, TRACKS, WORLD_SHA, export_geometry, record, verify


CLIP = "analytic_weight_shift"
KICK = -.35


def generate_reference(xml, directory):
    from scripts.data.convert_lafan_to_npz import CSV_JOINT_NAMES, load_robot, motion_to_arrays

    model, joints, bodies, _ = load_robot(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for side, sign in (("left", 1), ("right", -1)):
        pose = {
            "hip_pitch": -.25, "hip_roll": .025 * sign, "knee": .50,
            "ankle_pitch": -.25, "ankle_roll": -.025 * sign,
            "shoulder_pitch": .1, "shoulder_roll": .20 * sign, "elbow": .6,
        }
        for joint, angle in pose.items():
            data.joint(f"{side}_{joint}_joint").qpos[:] = angle
    mujoco.mj_kinematics(model, data)
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    contact = np.flatnonzero(np.isin(model.geom_bodyid, feet)
                             & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
    if len(contact) != 8:
        raise RuntimeError("Expected eight sole contact spheres")
    data.qpos[2] -= np.min(data.geom_xpos[contact, 2] - model.geom_size[contact, 0])
    mujoco.mj_kinematics(model, data)
    home = data.qpos.copy()
    target_pos = data.xpos[feet].copy()
    target_rot = data.xquat[feet].copy()
    legs = []
    for side in ("left", "right"):
        names = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        ids = [model.joint(f"{side}_{name}_joint").id for name in names]
        legs.append((model.jnt_qposadr[ids], model.jnt_dofadr[ids], model.jnt_range[ids]))
    jac_p, jac_r = np.empty((3, model.nv)), np.empty((3, model.nv))
    angular = np.empty(3)
    order = [model.joint(name).qposadr[0] for name in CSV_JOINT_NAMES]
    frames = PRE + POST + 20
    motion = np.empty((frames, 36))
    for i in range(frames):
        t = i / 50
        # Begin still, shift weight to one foot, then shift to the other.
        u = 2 * np.pi * t / 2.6
        data.qpos[:3] = home[:3] + [.025 * (1 - np.cos(u)),
                                     .060 * np.sin(u) * (1 - np.cos(u)),
                                     -.033 * (1 - np.cos(u))]
        for j, (body, (qadr, vadr, limits)) in enumerate(zip(feet, legs, strict=True)):
            for _ in range(9):
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                mujoco.mj_jacBody(model, data, jac_p, jac_r, body)
                mujoco.mju_subQuat(angular, target_rot[j], data.xquat[body])
                world_error = data.xmat[body].reshape(3, 3) @ angular
                error = np.r_[target_pos[j] - data.xpos[body], world_error]
                jac = np.r_[jac_p[:, vadr], jac_r[:, vadr]]
                delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
                data.qpos[qadr] = np.clip(data.qpos[qadr] + delta, *limits.T)
        mujoco.mj_kinematics(model, data)
        if np.linalg.norm(data.xpos[feet] - target_pos, axis=-1).max() > .002:
            raise RuntimeError(f"Analytic stance IK did not converge for frame {i}")
        motion[i] = np.r_[data.qpos[:7], data.qpos[order]]
    directory.mkdir()
    np.savez_compressed(directory / f"{CLIP}.npz", **motion_to_arrays(model, joints, bodies, motion, 50.0))
    manifest = directory / "manifest.json"
    write_report(manifest, {"schema_version": 1, "fps": 50, "quaternion_order": "wxyz",
                            "clips": [{"file": f"{CLIP}.npz", "frames": frames}]})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "world", "output", "motion-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    verify(args.teacher, ACTOR_SHA)
    verify(args.world, WORLD_SHA)
    if torch.cuda.device_count() != 1 or args.output.exists() or args.motion_dir.exists():
        parser.error("Expose exactly one authorized GPU and choose new output paths")

    from src.tasks.supermimic.config.robot_cfg import G1_XML
    from src.tasks.supermimic.config.task_cfg import FoundationConfig
    from src.tasks.supermimic.env import SuperMimicEnv
    from src.tasks.supermimic.rl.inference import load_actor, load_world_model, validate_environment

    manifest = generate_reference(G1_XML, args.motion_dir)
    device = "cuda:0"
    actor, actor_saved, _ = load_actor(args.teacher, manifest, device)
    world, world_saved = load_world_model(args.world, manifest, device)
    iteration = actor_saved["iteration"]
    if world_saved.get("iteration") != iteration or actor_saved["simulation"] != world_saved["simulation"]:
        raise ValueError("Checkpoint iteration or simulation fingerprint does not match")
    cfg = actor_saved["config"]
    env = SuperMimicEnv(FoundationConfig(
        manifest=manifest, clips=(CLIP,), num_envs=2, seed=17, episode_length_s=100.0,
        action_clip=cfg.get("action_clip"), relative_orientation=cfg.get("relative_orientation", False),
        motion_anchor_ori_b=cfg.get("motion_anchor_ori_b", False),
    ), device=device)
    try:
        for saved in (actor_saved, world_saved):
            validate_environment(env, saved)
        poses, tail, starts, frames, errors = record(env, actor, world, clips=(CLIP,), kicks=(0, KICK),
                                                    selected=(0, 1), phase=0, include_errors=True)
        args.output.mkdir()
        geoms = export_geometry(env.sim.mj_model, env.robot.joint_names, poses, args.output)
        scenarios = [
            {"label": label, "clip": CLIP, "startFrame": frame, "tailErrorM": error,
             "recovered": error <= .30, "fell": False, "resyncFrames": starts,
             "kickFrame": PRE if i else None, "kickMps": KICK if i else 0,
             "errors": errors[i],
             "targetXY": poses[i, 2, :, :2].tolist()}
            for i, (label, frame, error) in enumerate(zip(("Weight shift", "Lateral kick"), frames, tail, strict=True))
        ]
        write_report(args.output / "manifest.json", {
            "schema": 1, "publication": "public", "referenceSource": "original analytic stance IK",
            "sourceRevision": SOURCE, "checkpointIteration": iteration, "frames": poses.shape[2],
            "dt": env.step_dt, "tracks": list(TRACKS), "horizon": HORIZON,
            "geoms": geoms, "scenarios": scenarios,
        })
        print(f"{iteration=}, {len(geoms)=}, {poses.shape=}, {tail=}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

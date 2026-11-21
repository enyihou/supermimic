"""Run and verify one allocation of the pinned SuperMimic recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from air_checkpoint import latest_checkpoint, sha256, write_json


PINS = {
    "supermimic": "5fb92297ab4fe3d86629d9f0c635334791bb0d5f",
    "mjlab_research": "379c55e88b1dec73f93943e524a879b2ebbf45c5",
}
TASK = "SuperMimic-GenCoord-G1"
RECIPES = {
    "full": (4096, 320000, 1000, 32000),
    "benchmark": (4096, 1200, 200, 600),
    "smoke": (256, 200, 50, 100),
}
REQUIRED_MOTIONS = {
    "fps.npy", "joint_names.json", "motion_names.json", "motion_ends.npy",
    "joint_pos.npy", "joint_vel.npy", "root_pos_w.npy", "root_quat_w.npy",
    "root_lin_vel_w.npy", "root_ang_vel_w.npy", "motion-manifest.json",
}
EXACT_SELECTION = "supermimic-v3-authors-manifest"
APPROXIMATE_SELECTION = "supermimic-v3-appendix-e-approx-v1"


def read_motion_manifest(motion: Path, mode: str, bones_sha256: str) -> dict:
    """Bind a verified packed archive to its underlying BONES revision and use."""
    path = motion / "motion-manifest.json"
    contents = path.read_bytes()
    manifest = json.loads(contents)
    if not isinstance(manifest, dict) or manifest.get("dataset_sha256") != bones_sha256:
        raise RuntimeError("Packed motion manifest does not bind to the submitted BONES SHA-256")
    clipset = manifest.get("clipset_sha256")
    clips = manifest.get("clips")
    frames = manifest.get("frames")
    input_fps = manifest.get("input_fps")
    fps = manifest.get("fps")
    selection = manifest.get("selection")
    if not isinstance(clipset, str) or not re.fullmatch(r"[0-9a-f]{64}", clipset):
        raise RuntimeError("Packed motion manifest is missing a valid selection SHA-256")
    if (
        type(clips) is not int or clips <= 0 or type(frames) is not int
        or frames < 2 * clips or input_fps != 120 or fps != 50
    ):
        raise RuntimeError("Packed motion manifest has invalid clip/frame counts or sample rates")
    cohort = {}
    if mode == "smoke":
        if selection != "archive-order-smoke" or clips != 64:
            raise RuntimeError("The official smoke requires the verified 64-clip smoke package")
    elif mode in ("benchmark", "full"):
        if selection != "explicit-allowlist":
            raise RuntimeError("Benchmark and full run require the reviewed full motion selection")
        selection_id = manifest.get("selection_id")
        allowlist = manifest.get("allowlist_sha256")
        approximate = manifest.get("approximate")
        if (
            selection_id not in {EXACT_SELECTION, APPROXIMATE_SELECTION}
            or not isinstance(allowlist, str) or not re.fullmatch(r"[0-9a-f]{64}", allowlist)
            or approximate is not (selection_id == APPROXIMATE_SELECTION)
            or (selection_id == EXACT_SELECTION and clips != 129_514)
        ):
            raise RuntimeError("Full motion set requires valid exact or approximate cohort provenance")
        cohort = {"selection_id": selection_id, "allowlist_sha256": allowlist, "approximate": approximate}
    else:
        raise ValueError(f"Unknown training mode: {mode}")
    return {
        **cohort,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "clipset_sha256": clipset,
        "clips": clips,
        "frames": frames,
        "input_fps": input_fps,
        "fps": fps,
        "selection": selection,
    }


def verify_motion_arrays(motion: Path, manifest: dict) -> None:
    """Inspect only metadata and array shapes before loading the full set."""
    import numpy as np

    names = json.loads((motion / "motion_names.json").read_text())
    ends = np.load(motion / "motion_ends.npy", mmap_mode="r", allow_pickle=False)
    fps = np.load(motion / "fps.npy", mmap_mode="r", allow_pickle=False)
    clips = manifest["clips"]
    frames = manifest["frames"]
    if (
        not isinstance(names, list) or len(names) != clips
        or ends.shape != (clips,) or ends.dtype.kind not in "iu" or ends[-1] != frames
        or fps.size != 1 or fps.item() != manifest["fps"]
    ):
        raise RuntimeError("Packed motion arrays disagree with the clip/frame/rate manifest")
    for field, width in (
        ("joint_pos", 29), ("joint_vel", 29), ("root_pos_w", 3), ("root_quat_w", 4),
        ("root_lin_vel_w", 3), ("root_ang_vel_w", 3),
    ):
        values = np.load(motion / f"{field}.npy", mmap_mode="r", allow_pickle=False)
        if values.shape != (frames, width) or values.dtype != np.dtype("<f4"):
            raise RuntimeError(f"Unexpected packed motion array shape or dtype: {field}")


def launch(mode: str, motion: Path, code: Path, bones_sha256: str, motion_sha256: str) -> None:
    import torch

    output = Path(os.environ["SM_OUTPUT_DIR"])
    if not all((motion / name).is_file() and not (motion / name).is_symlink() for name in REQUIRED_MOTIONS):
        raise RuntimeError("Packed motion archive does not contain the expected mjlab arrays")
    motion_manifest = read_motion_manifest(motion, mode, bones_sha256)
    verify_motion_arrays(motion, motion_manifest)
    if (code / "PINS.txt").read_text().splitlines() != [f"{key}={pin}" for key, pin in PINS.items()]:
        raise RuntimeError("The source bundle does not record the two expected pinned revisions")
    revision = (code / "REPRODUCTION_COMMIT.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Invalid reproduction source revision")

    num_envs, iterations, save_interval, milestone = RECIPES[mode]
    seed = 42
    overrides = [
        "--num-envs", str(num_envs),
        "--env.seed", str(seed),
        "--runner.max-iterations", str(iterations),
        "--runner.save-interval", str(save_interval),
    ]
    identity = {
        "run_id": os.environ["SM_RUN_ID"],
        "task_id": TASK,
        "mode": mode,
        "source_commits": PINS,
        "reproduction_commit": revision,
        "dataset_sha256": bones_sha256,
        "motion_archive_sha256": motion_sha256,
        "motion_manifest": motion_manifest,
        "code_sha256": os.environ["SM_CODE_SHA256"],
        "environment_sha256": os.environ["SM_ENV_SHA256"],
        "runner_overlay_sha256": sha256(code / "training/air_runner.patch"),
        "checkpoint_overlay_sha256": sha256(code / "training/air_checkpoint.py"),
        "num_envs": num_envs,
        "max_iterations": iterations,
        "save_interval": save_interval,
        "milestone_interval": milestone,
        "seed": seed,
        "overrides": overrides,
    }
    write_json(output / "run-identity.json", identity)
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    cuda_available = torch.cuda.is_available()
    num_cuda_devices = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0) if cuda_available and num_cuda_devices == 1 else None
    write_json(
        output / "environment.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "num_cuda_devices": num_cuda_devices,
            "cuda_available": cuda_available,
            "gpu": props.name if props else None,
            "gpu_total_memory_bytes": props.total_memory if props else None,
            "nvidia_driver_versions": sorted(set(nvidia_smi.stdout.splitlines())),
        },
    )
    if props is None:
        raise RuntimeError("Official reproduction requires exactly one usable Slurm-visible CUDA device")
    trainer = code / "supermimic/src/supermimic/scripts/train.py"
    os.execv(
        sys.executable,
        [
            sys.executable, "-u", str(trainer), TASK, *overrides,
            "--motion-file", str(motion),
            "--runner.run-name", os.environ["SM_RUN_ID"],
        ],
    )


def verify_final(run: Path, output: Path, job_id: str) -> None:
    result = json.loads((output / "result.json").read_text())
    identity = json.loads((output / "run-identity.json").read_text())
    _, checkpoint = latest_checkpoint(run / "checkpoints", verify_archive=True)
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Final checkpoint has a different verified run identity")
    target = identity["max_iterations"]
    if result["job_id"] != job_id or result["stopped"] or not (
        result["completed_iterations"] == result["target_iterations"] == checkpoint["completed_iterations"] == target
    ):
        raise RuntimeError("Trainer did not record successful completion of the exact target")
    if (
        result["checkpoint_sha256"] != checkpoint["archive"]["sha256"]
        or result["checkpoint_job_id"] != checkpoint["job_id"]
    ):
        raise RuntimeError("Final status does not bind to the published checkpoint")
    if checkpoint["job_id"] != job_id and result["resumed_from_iterations"] != target:
        raise RuntimeError("A different job ID is allowed only when recovering a previously finished checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("mode", choices=RECIPES)
    train.add_argument("motion", type=Path)
    train.add_argument("code", type=Path)
    train.add_argument("bones_sha256")
    train.add_argument("motion_sha256")
    verify = commands.add_parser("verify-final")
    verify.add_argument("run", type=Path)
    verify.add_argument("output", type=Path)
    verify.add_argument("job_id")
    args = parser.parse_args()
    if args.command == "train":
        launch(args.mode, args.motion, args.code, args.bones_sha256, args.motion_sha256)
    else:
        verify_final(args.run, args.output, args.job_id)


if __name__ == "__main__":
    main()

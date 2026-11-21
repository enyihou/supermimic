"""Convert private BONES-SEED G1 CSVs to the pinned mjlab packed motion format."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from inspect_bones import parse_header


INPUT_FPS = 120
TARGET_FPS = 50
APPROXIMATION_ID = "supermimic-v3-appendix-e-approx-v1"
EXACT_SELECTION_ID = "supermimic-v3-authors-manifest"
FIELDS = {
    "root_pos_w": 3,
    "root_quat_w": 4,
    "root_lin_vel_w": 3,
    "root_ang_vel_w": 3,
    "joint_pos": 29,
    "joint_vel": 29,
}


def convert(payload: bytes, target_joints: list[str]) -> dict[str, np.ndarray]:
    header, sep, _ = payload.partition(b"\n")
    if not sep:
        raise ValueError("Motion CSV has no frames")
    csv_joints = parse_header(header)
    if set(csv_joints) != set(target_joints):
        raise ValueError("Motion/robot joint sets differ")
    data = np.loadtxt(io.BytesIO(payload), delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 36 or not np.isfinite(data).all():
        raise ValueError("Expected 36 finite numeric columns")
    if not (np.diff(data[:, 0]) == 1).all():
        raise ValueError("BONES-SEED frame numbers are not contiguous")
    n = ((len(data) - 1) * TARGET_FPS) // INPUT_FPS + 1
    if n < 2:
        raise ValueError("Motion is shorter than two control frames")
    sample = np.arange(n) * (INPUT_FPS / TARGET_FPS)
    idx = sample.astype(np.int64)
    after = np.minimum(idx + 1, len(data) - 1)
    blend = (sample - idx)[:, None]
    root = (data[idx, 1:4] * (1 - blend) + data[after, 1:4] * blend) / 100
    joints = np.deg2rad(data[:, [7 + csv_joints.index(joint) for joint in target_joints]])
    joints = joints[idx] * (1 - blend) + joints[after] * blend
    rotations = Slerp(np.arange(len(data)), Rotation.from_euler("xyz", data[:, 4:7], degrees=True))(sample)
    if n == 2:
        omega = np.repeat((rotations[1] * rotations[0].inv()).as_rotvec()[None] * TARGET_FPS, 2, axis=0)
    else:
        omega = (rotations[2:] * rotations[:-2].inv()).as_rotvec() * (TARGET_FPS / 2)
        omega = np.concatenate((omega[:1], omega, omega[-1:]), axis=0)
    fields = {
        "root_pos_w": root,
        "root_quat_w": rotations.as_quat()[:, [3, 0, 1, 2]],
        "root_lin_vel_w": np.gradient(root, 1 / TARGET_FPS, axis=0),
        "root_ang_vel_w": omega,
        "joint_pos": joints,
        "joint_vel": np.gradient(joints, 1 / TARGET_FPS, axis=0),
    }
    return {key: np.asarray(value, dtype="<f4", order="C") for key, value in fields.items()}


def robot_joints(xml: Path) -> list[str]:
    root = ElementTree.parse(xml).getroot()
    joints = [joint.attrib["name"] for joint in root.findall("worldbody//body/joint")
              if joint.get("type", "hinge") != "free"]
    if len(joints) != 29 or len(set(joints)) != 29:
        raise ValueError("Expected the pinned 29-joint G1's canonical MJCF")
    return joints


def verified_allowlist(path: Path, expected_sha: str, kind: str) -> tuple[set[str], dict]:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise ValueError("Allowlist SHA-256 mismatch")
    entries = source.decode("utf-8").splitlines()
    keep = set(entries)
    names_valid = all(re.fullmatch(r"g1/csv/[^/\\\x00]+/[^/\\\x00]+\.csv", item)
                      and ".." not in item.split("/") for item in keep)
    if not keep or len(keep) != len(entries) or not names_valid:
        raise ValueError("Invalid or repeated allowlist entries")
    if kind == EXACT_SELECTION_ID and len(keep) != 129_514:
        raise ValueError("The authors' manifest must contain 129514 clip entries")
    if kind not in {APPROXIMATION_ID, EXACT_SELECTION_ID}:
        raise ValueError("Unsupported full motion cohort")
    return keep, {"selection_id": kind, "allowlist_sha256": expected_sha,
                  "approximate": kind == APPROXIMATION_ID}


def pack(archive: Path, dest: Path, joints: list[str], *, source_sha: str,
         keep: set[str] | None, smoke_clips: int | None, workers: int,
         cohort: dict | None = None) -> dict:
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    if (keep is None) == (smoke_clips is None) or (keep is not None) != (cohort is not None):
        raise ValueError("Select either a bounded smoke or a verified private full cohort")
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != source_sha:
            raise ValueError("Source archive SHA-256 mismatch")
        source.seek(0)
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".motion-pack-") as work:
            work = Path(work)
            out = work / "packed"
            out.mkdir(mode=0o700)
            ends: list[int] = []
            names: list[str] = []
            got: set[str] = set()
            clipset = hashlib.sha256()
            frames = 0
            with (
                tarfile.open(fileobj=source, mode="r|gz", bufsize=1 << 20) as tar,
                ExitStack() as streams,
                ProcessPoolExecutor(max_workers=workers) as pool,
            ):
                sinks = {field: streams.enter_context((work / field).open("wb")) for field in FIELDS}

                def inputs():
                    for member in tar:
                        if not member.isfile() or not member.name.endswith(".csv"):
                            continue
                        if keep is not None and member.name not in keep:
                            continue
                        if member.name in got or member.size > 64 * 1024 ** 2:
                            raise ValueError("Duplicate or oversized motion CSV")
                        file = tar.extractfile(member)
                        if file is None:
                            raise ValueError("Unable to read motion CSV")
                        with file:
                            got.add(member.name)
                            yield member.name, file.read()
                        if smoke_clips is not None and len(got) >= smoke_clips:
                            return

                items = inputs()
                while batch := list(itertools.islice(items, 2 * workers)):
                    results = pool.map(convert, (value for _, value in batch), itertools.repeat(joints))
                    for (name, _), arrays in zip(batch, results, strict=True):
                        for field, sink in sinks.items():
                            sink.write(arrays[field].tobytes())
                        frames += arrays["joint_pos"].shape[0]
                        ends.append(frames)
                        names.append(name.removeprefix("g1/csv/").removesuffix(".csv"))
                        clipset.update(name.encode("utf-8") + b"\0")

            if not ends or (keep is not None and got != keep) or (
                smoke_clips is not None and len(ends) != smoke_clips
            ):
                raise ValueError("Source archive did not contain the complete selected motion set")

            for field, width in FIELDS.items():
                raw = work / field
                with raw.open("rb") as buf, (out / f"{field}.npy").open("wb") as npy:
                    np.lib.format.write_array_header_2_0(npy, {
                        "descr": "<f4", "fortran_order": False, "shape": (frames, width),
                    })
                    shutil.copyfileobj(buf, npy, length=8 << 20)
                raw.unlink()
            np.save(out / "fps.npy", np.array(TARGET_FPS, dtype=np.int64), allow_pickle=False)
            np.save(out / "motion_ends.npy", np.array(ends, dtype=np.int64), allow_pickle=False)
            for name, values in [("joint_names", joints), ("motion_names", names)]:
                (out / f"{name}.json").write_text(json.dumps(values) + "\n")
            manifest = {
                **(cohort or {}),
                "dataset_sha256": source_sha,
                "clipset_sha256": clipset.hexdigest(),
                "clips": len(ends), "frames": frames,
                "input_fps": INPUT_FPS, "fps": TARGET_FPS,
                "selection": "archive-order-smoke" if smoke_clips is not None else "explicit-allowlist",
            }
            (out / "motion-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            out.rename(dest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=4)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--smoke-clips", type=int, help="Small deterministic archive-order sample")
    selection.add_argument("--keep-list", type=Path, help="Private allowlist of exact tar member names")
    parser.add_argument("--keep-list-sha256", help="Independently reviewed private allowlist SHA-256")
    parser.add_argument("--selection-id", choices=[APPROXIMATION_ID, EXACT_SELECTION_ID])
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.smoke_clips is not None and not 1 <= args.smoke_clips <= 128:
        parser.error("workers must be 1–32; smoke clips must be 1–128")
    keep = cohort = None
    if args.keep_list:
        if not args.selection_id or not re.fullmatch(r"[0-9a-f]{64}", args.keep_list_sha256 or ""):
            parser.error("a full keep list requires --selection-id and --keep-list-sha256")
        keep, cohort = verified_allowlist(args.keep_list, args.keep_list_sha256, args.selection_id)
    elif args.selection_id or args.keep_list_sha256:
        parser.error("full cohort options are not for the bounded smoke")
    args.output_directory.parent.mkdir(parents=True, exist_ok=True)
    result = pack(
        args.archive, args.output_directory, robot_joints(args.g1_xml),
        source_sha=args.sha256, keep=keep, smoke_clips=args.smoke_clips, workers=args.workers,
        cohort=cohort,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Produce a private, explicitly approximate SuperMimic v3 Appendix E allowlist."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile

import numpy as np

from inspect_bones import EXPECTED_CLIPS, issparse
from pack_bones import TARGET_FPS, convert, robot_joints


PAPER = "https://arxiv.org/pdf/2610.28480v3"
SONIC_COMMIT = "1e1ec4f5f27e6c3e03048f693bd7c60cce04afda"
SONIC_SOURCE_SHA256 = "b6b50d441a4876a382eb3ff33ba940c487d1647230b5ecacaa37cfcf521258da"
SONIC_SOURCE_URL = (
    "https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/"
    f"{SONIC_COMMIT}/gear_sonic/data_process/filter_and_copy_bones_data.py"
)
MJLAB_COMMIT = "379c55e88b1dec73f93943e524a879b2ebbf45c5"
G1_XML_SHA256 = "d6026290f4264198a0e8064f94bca7f20f7d3c72e9f6b48c019531360a9a8395"
OMIT_SONIC = frozenset({
    "230m", "acrobatics_", "cartwheel", "handstand", "handstand_ff_",
    "jump_to_shoulder_roll", "kozak_dance", "monkey_jump", "safety_roll",
})
PAPER_TERMS = ("box", "sit", "jump_off_50cm", "look_over_fence")
PAPER_NAME_FILTER = re.compile(
    r"(?<![a-z0-9])(?:box|sit|jump[ _-]+off[ _-]+50cm|look[ _-]+over[ _-]+fence)(?![a-z0-9])",
)
MAX_CLIP_BYTES = 64 * 1024**2


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sonic_keywords(source: bytes) -> tuple[str, ...]:
    if hashlib.sha256(source).hexdigest() != SONIC_SOURCE_SHA256:
        raise ValueError("SONIC source SHA-256 mismatch")
    matches = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument" and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--filter-keywords"
        ):
            for kw in node.keywords:
                if kw.arg == "default":
                    matches.append(ast.literal_eval(kw.value))
    if len(matches) != 1 or not isinstance(matches[0], list) or not all(
        isinstance(value, str) and value for value in matches[0]
    ):
        raise ValueError("Expected one literal keyword list in the pinned SONIC source")
    defaults = [value.casefold() for value in matches[0]]
    if len(defaults) != len(set(defaults)) or not OMIT_SONIC <= set(defaults):
        raise ValueError("SONIC keyword vocabulary does not match the reviewed adaptation")
    return tuple(value for value in defaults if value not in OMIT_SONIC)


def excludes_by_name(member: PurePosixPath, keywords: tuple[str, ...]) -> bool:
    # SONIC checks immediate parent and filename. In the raw archive the
    # immediate parent is a recording date, not a motion category.
    label = f"{member.parent.name}/{member.stem}".casefold()
    return any(word in label for word in keywords) or PAPER_NAME_FILTER.search(label) is not None


@dataclass(frozen=True)
class Gates:
    expanded_floor_m: float = 0.20
    elevated_frames: int = 30
    deep_penetration_m: float = -0.12
    repeated_penetration_m: float = -0.08
    repeated_penetration_fraction: float = 0.80
    foot_clearance_m: float = 0.02
    floating_frames: int = 25
    standing_pelvis_min_m: float = 0.70
    upright_max_tilt_degrees: float = 20.0
    stationary_root_max_m_s: float = 0.15
    stationary_root_max_rad_s: float = 0.40


# These data thresholds are versioned as part of the protocol, not CLI tuning knobs.
GATES = Gates()
APPROXIMATION_ID = "supermimic-v3-appendix-e-approx-v1"


def sustained(mask: np.ndarray, frames: int) -> bool:
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return bool(np.any(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1) >= frames))


def feasible_reason(zmin: np.ndarray, floating: np.ndarray, gates: Gates = GATES) -> str | None:
    if (
        zmin.ndim != 1 or len(zmin) < 2 or not np.isfinite(zmin).all()
        or zmin.shape != floating.shape or floating.dtype != np.dtype(bool)
    ):
        raise ValueError("Invalid feasibility feature arrays")
    if sustained(zmin > gates.expanded_floor_m, gates.elevated_frames):
        return "elevated_support"
    if (
        (zmin < gates.deep_penetration_m).any()
        or np.count_nonzero(zmin < gates.repeated_penetration_m) > len(zmin) *
        gates.repeated_penetration_fraction
    ):
        return "ground_penetration"
    if sustained(floating, gates.floating_frames):
        return "floating_feet"
    return None


def standing_stationary(arrays: dict, gates: Gates = GATES) -> np.ndarray:
    q = arrays["root_quat_w"].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return (
        (arrays["root_pos_w"][:, 2] >= gates.standing_pelvis_min_m)
        & (up_z >= math.cos(math.radians(gates.upright_max_tilt_degrees)))
        & (np.linalg.norm(arrays["root_lin_vel_w"], axis=1) <= gates.stationary_root_max_m_s)
        & (np.linalg.norm(arrays["root_ang_vel_w"], axis=1) <= gates.stationary_root_max_rad_s)
    )


class G1Feasibility:
    """FK in the pinned MuJoCo model; no renderer or dynamic stepping."""

    def __init__(self, g1_xml: Path):
        import mujoco

        if sha256(g1_xml) != G1_XML_SHA256:
            raise ValueError("G1 XML SHA-256 mismatch")
        self.mj = mujoco
        self.joint_names = robot_joints(g1_xml)
        spec = mujoco.MjSpec.from_file(str(g1_xml))
        spec.worldbody.add_geom(
            name="appendix_e_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.1),
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.plane = self.model.geom("appendix_e_floor").id
        self.feet = tuple(
            tuple(self.model.geom(f"{side}_foot{number}_collision").id for number in range(1, 8))
            for side in ("left", "right")
        )
        for foot in self.feet:
            if any(self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE
                   or not self.model.geom_contype[geom] for geom in foot):
                raise ValueError("Unexpected G1 foot collision geometry")
        base = self.model.joint("floating_base_joint").id
        self.base_q = int(self.model.jnt_qposadr[base])
        hinge_ids = [self.model.joint(name).id for name in self.joint_names]
        if (
            self.model.jnt_type[base] != mujoco.mjtJoint.mjJNT_FREE
            or self.model.njnt != 30 or self.model.nq != 36
            or any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in hinge_ids)
            or self.model.jnt_bodyid[base] != self.model.body("pelvis").id
        ):
            raise ValueError("Unexpected compiled G1 joint layouts")
        self.joint_q = self.model.jnt_qposadr[hinge_ids]

    def update(self, root: np.ndarray, quat: np.ndarray, joints: np.ndarray) -> float:
        self.data.qpos[self.base_q:self.base_q + 3] = root
        self.data.qpos[self.base_q + 3:self.base_q + 7] = quat
        self.data.qpos[self.joint_q] = joints
        self.mj.mj_kinematics(self.model, self.data)
        return float(self.data.xpos[1:, 2].min())

    def distances(self) -> tuple[float, float]:
        edge_values = [
            self.mj.mj_geomDistance(self.model, self.data, self.plane, geom, 1.0, None)
            for foot in self.feet for geom in foot
        ]
        if not all(math.isfinite(value) for value in edge_values):
            raise ValueError("MuJoCo emitted invalid foot distance")
        return min(edge_values[:7]), min(edge_values[7:])

    def classify(self, arrays: dict) -> str | None:
        n = len(arrays["joint_pos"])
        if n < 2 or not all(np.isfinite(data).all() and len(data) == n for data in arrays.values()):
            raise ValueError("Invalid converted motion")
        if not np.allclose(np.linalg.norm(arrays["root_quat_w"], axis=1), 1.0, atol=1e-6):
            raise ValueError("Invalid converted orientation")
        zmin = np.empty(n)
        floating = np.zeros(n, dtype=bool)
        candidates = standing_stationary(arrays)
        # An isolated qualifying frame cannot satisfy the duration gate.
        check_distances = sustained(candidates, GATES.floating_frames)
        for i, (root, quat, joints) in enumerate(zip(
            arrays["root_pos_w"], arrays["root_quat_w"], arrays["joint_pos"], strict=True,
        )):
            zmin[i] = self.update(root, quat, joints)
            if check_distances and candidates[i]:
                floating[i] = min(self.distances()) > GATES.foot_clearance_m
        return feasible_reason(zmin, floating)


_FILTER: G1Feasibility | None = None


def worker_start(g1_xml: Path) -> None:
    global _FILTER
    _FILTER = G1Feasibility(g1_xml)


def classify_csv(payload: bytes) -> str | None:
    if _FILTER is None:
        raise RuntimeError("No initialized G1 feasibility model")
    try:
        return _FILTER.classify(convert(payload, _FILTER.joint_names))
    except (ValueError, TypeError, OverflowError, FloatingPointError, RuntimeError) as exc:
        # Some CSV parser errors contain values from the gated archive.
        raise ValueError(f"Motion failed conversion or geometry validation ({type(exc).__name__})") from None


def scan_motions(source, keywords: tuple[str, ...], counts: Counter):
    seen: set[str] = set()
    with gzip.GzipFile(fileobj=source, mode="rb") as gzip_stream:
        with tarfile.open(fileobj=gzip_stream, mode="r|", bufsize=1 << 20) as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                if (
                    path.as_posix() != member.name or path.is_absolute() or ".." in path.parts
                    or "\\" in member.name or any(ord(char) < 32 for char in member.name)
                    or member.name in seen or issparse(member)
                ):
                    raise ValueError("Unsafe or duplicate BONES-SEED archive member")
                seen.add(member.name)
                if member.isdir():
                    if path.parts[:1] != ("g1",) or len(path.parts) > 3 or (
                        len(path.parts) > 1 and path.parts[:2] != ("g1", "csv")
                    ):
                        raise ValueError("Unexpected BONES-SEED archive directory")
                    continue
                if (
                    not member.isfile() or path.parts[:2] != ("g1", "csv")
                    or len(path.parts) != 4 or path.suffix != ".csv"
                    or not 1 <= member.size <= MAX_CLIP_BYTES
                ):
                    raise ValueError("Unexpected BONES-SEED archive file")
                counts["raw_clips"] += 1
                if excludes_by_name(path, keywords):
                    counts["name_rejected"] += 1
                    continue
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError("Could not read BONES-SEED member")
                with stream:
                    payload = stream.read()
                if len(payload) != member.size:
                    raise ValueError("Truncated BONES-SEED member")
                yield member.name, payload
        while gzip_stream.read(1 << 20):
            pass  # validate the gzip trailer beyond end-of-tar padding


def publish(output: Path, keep: list[str], summary: dict) -> dict:
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".motion-selection-") as temporary:
        stage = Path(temporary) / "selection"
        stage.mkdir(mode=0o700)
        for name, value in (
            ("keep.txt", "\n".join(keep) + "\n"),
            ("summary.json", json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"),
        ):
            with (stage / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(value)
                os.fchmod(stream.fileno(), 0o600)
                stream.flush()
                os.fsync(stream.fileno())
        checksums = {name: sha256(stage / name) for name in ("keep.txt", "summary.json")}
        with (stage / "SHA256SUMS").open("x", encoding="ascii") as stream:
            stream.write("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
            os.fchmod(stream.fileno(), 0o600)
            stream.flush()
            os.fsync(stream.fileno())
        for directory in (stage, output.parent):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                if directory == output.parent:
                    if output.exists() or output.is_symlink():
                        raise FileExistsError("Selection output already exists")
                    stage.rename(output)
                os.fsync(fd)
            finally:
                os.close(fd)
    return checksums


def select(args: argparse.Namespace) -> dict:
    import mujoco

    output = args.output.parent.resolve(strict=True) / args.output.name
    root = Path(__file__).resolve().parent.parent
    parent = output.parent.stat()
    if (
        output.is_relative_to(root) or output.exists() or output.is_symlink()
        or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise ValueError("Expected a new output directory under an owner-only parent outside the repo")
    if sha256(args.g1_xml) != G1_XML_SHA256 or TARGET_FPS != 50:
        raise ValueError("Pinned G1 XML or converter control rate does not match Appendix E selection")
    with args.sonic_source.open("rb") as stream:
        keywords = sonic_keywords(stream.read(65_537))
    counts: Counter = Counter()
    keep = []
    with args.archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != args.sha256:
            raise ValueError("BONES-SEED archive SHA-256 mismatch")
        source.seek(0)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_start,
                                 initargs=(args.g1_xml,)) as pool:
            iterator = scan_motions(source, keywords, counts)
            while batch := list(itertools.islice(iterator, 2 * args.workers)):
                verdicts = pool.map(classify_csv, (item[1] for item in batch))
                for (name, _), reason in zip(batch, verdicts, strict=True):
                    if reason is None:
                        keep.append(name)
                        counts["selected"] += 1
                    else:
                        counts[reason] += 1
    reasons = ("elevated_support", "ground_penetration", "floating_feet")
    after_names = counts["raw_clips"] - counts["name_rejected"]
    if counts["raw_clips"] != EXPECTED_CLIPS or after_names != len(keep) + sum(
        counts[reason] for reason in reasons
    ) or not keep:
        raise ValueError("Incomplete BONES corpus scan or no selected clips")
    keep_digest = hashlib.sha256()
    for name in keep:
        keep_digest.update(name.encode("utf-8") + b"\n")
    summary = {
        "schema_version": 1,
        "id": APPROXIMATION_ID,
        "approximate": True,
        "equivalent_to_authors_manifest": False,
        "dataset_sha256": args.sha256,
        "allowlist_sha256": keep_digest.hexdigest(),
        "selector_sha256": sha256(Path(__file__)),
        "converter_sha256": sha256(Path(__file__).with_name("pack_bones.py")),
        "robot": {"mjlab_commit": MJLAB_COMMIT, "xml_sha256": G1_XML_SHA256,
                  "mujoco_version": mujoco.mj_versionString()},
        "paper": {"url": PAPER, "raw": 142220, "after_names": 129667, "selected": 129514},
        "name_rule": {"sonic_commit": SONIC_COMMIT, "source_url": SONIC_SOURCE_URL,
                      "source_sha256": SONIC_SOURCE_SHA256, "omitted_sonic_terms": sorted(OMIT_SONIC),
                      "added_token_or_phrase_terms": PAPER_TERMS},
        "feasibility_rule": {"fps": TARGET_FPS, "floor_z_m": 0.0, **asdict(GATES)},
        "counts": {"raw": counts["raw_clips"], "after_names": after_names, "selected": len(keep),
                   "feasibility_exclusions": {reason: counts[reason] for reason in reasons}},
    }
    checksums = publish(output, keep, summary)
    return {**summary, "allowlist_sha256": checksums["keep.txt"], "summary_sha256": checksums["summary.json"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, help="new directory under an owner-only private parent")
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--sonic-source", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="independently verified BONES archive SHA-256")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accept-approximate", action="store_true", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256) or not 1 <= args.workers <= 32:
        parser.error("expected a SHA-256 digest and 1–32 workers")
    try:
        summary = select(args)
    except (OSError, EOFError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"Selection failed: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

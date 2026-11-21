"""Stream the BONES-SEED G1 tarball; report shape and expanded size, not clip names."""

import argparse
import csv
import gzip
import json
from pathlib import Path, PurePosixPath
import tarfile


BASE_COLUMNS = (
    "Frame", "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
)
EXPECTED_CLIPS = 142_220


def parse_header(header: bytes) -> list[str]:
    columns = next(csv.reader([header.decode("utf-8-sig").rstrip("\r\n")]))
    if tuple(columns[:7]) != BASE_COLUMNS or len(set(columns)) != len(columns):
        raise ValueError("Unexpected or repeated BONES-SEED CSV header")
    joints = columns[7:]
    if len(joints) != 29 or any(not joint.endswith("_joint_dof") for joint in joints):
        raise ValueError("Expected 29 G1 joint-angle columns")
    return [joint.removesuffix("_dof") for joint in joints]


def inspect(archive: Path, expected_clips: int) -> dict:
    members: set[str] = set()
    count = 0
    expanded_bytes = 0
    header_variants: set[bytes] = set()
    days: set[str] = set()
    with gzip.open(archive, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|", bufsize=1 << 20) as data:
            for member in data:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or not (member.isdir() or member.isfile())
                    or issparse(member)
                ):
                    raise ValueError("Unsafe member in BONES-SEED archive")
                normalized = path.as_posix()
                if normalized in members:
                    raise ValueError("Duplicate member in BONES-SEED archive")
                members.add(normalized)
                if member.isdir():
                    if path.parts[:1] != ("g1",) or (
                        len(path.parts) > 1 and path.parts[:2] != ("g1", "csv")
                    ) or len(path.parts) > 3:
                        raise ValueError("Unexpected BONES-SEED archive directory")
                    continue
                if path.parts[:2] != ("g1", "csv") or len(path.parts) != 4 or path.suffix != ".csv":
                    raise ValueError("Unexpected BONES-SEED archive file")
                clip = data.extractfile(member)
                if clip is None:
                    raise ValueError("Missing motion data")
                with clip:
                    header = clip.readline(4097)
                    if not header.endswith(b"\n") or len(header) > 4096:
                        raise ValueError("CSV header exceeds limit or is truncated")
                    if header not in header_variants:
                        parse_header(header)
                        header_variants.add(header)
                count += 1
                expanded_bytes += member.size
                days.add(path.parts[2])
        while stream.read(1 << 20):
            pass  # Require a valid gzip trailer, even beyond end-of-tar markers.
    if count != expected_clips:
        raise ValueError(f"Found {count} CSVs, expected {expected_clips}")
    if len({tuple(parse_header(header)) for header in header_variants}) != 1:
        raise ValueError("BONES-SEED motions disagree on column order")
    return {
        "archive_bytes": archive.stat().st_size,
        "csv_files": count,
        "csv_expanded_bytes": expanded_bytes,
        "date_directories": len(days),
        "joint_count": 29,
        "layout": "g1/csv/<date>/<clip>.csv",
    }


def issparse(member: tarfile.TarInfo) -> bool:
    return member.sparse is not None or any("sparse" in key.lower() for key in member.pax_headers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-clips", type=int, default=EXPECTED_CLIPS)
    args = parser.parse_args()
    print(json.dumps(inspect(args.archive, args.expected_clips), indent=2, sort_keys=True))

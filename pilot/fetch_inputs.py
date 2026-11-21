"""Download only the selected G1 CSVs and the pinned frozen actor."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen


SOURCE = "aaac8c2077c388dd2fcc29f03862029bebd9f6b1"
DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
TEACHER_SHA256 = "22a255c1ae64efc4296244b5040a3444b756296daa35cb57a71710b3d03952ca"


def request(url):
    return urlopen(Request(url, headers={"User-Agent": "supermimic-hoas-pilot/1"}), timeout=30)


def download(url, destination, size, checksum=None):
    tmp = destination.with_suffix(".partial")
    try:
        with request(url) as response, tmp.open("wb") as f:
            remaining = size
            while remaining:
                chunk = response.read(min(1 << 20, remaining))
                if not chunk:
                    raise IOError(f"Truncated {destination.name}")
                f.write(chunk)
                remaining -= len(chunk)
            if response.read(1):
                raise IOError(f"Unexpected length for {destination.name}")
        with tmp.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        if checksum and sha != checksum:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        tmp.replace(destination)
        return sha
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--single-clip", help="For a lightweight conversion check only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output / "csv"
    csv_dir.mkdir(exist_ok=True)
    with request(f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/g1?limit=100") as response:
        files = json.load(response)
    clips = [f for f in files if re.fullmatch(r"g1/(walk|run|sprint)\d+_subject[1-5]\.csv", f["path"])]
    if args.single_clip:
        clips = [f for f in clips if Path(f["path"]).stem == args.single_clip]
    elif len(clips) != 18:
        raise RuntimeError(f"Expected 18 locomotion clips, found {len(clips)}")
    if not clips:
        raise RuntimeError("No matching motion clips")
    meta = {"reproduction_revision": SOURCE, "data_revision": REVISION, "files": {}}
    for f in clips:
        local = csv_dir / Path(f["path"]).name
        sha = download(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{f['path']}",
                       local, f["size"], (f.get("lfs") or {}).get("oid"))
        meta["files"][local.name] = {"bytes": f["size"], "sha256": sha}
    if not args.single_clip:
        actor = args.output / "teacher.pt"
        sha = download(f"https://raw.githubusercontent.com/Breezewrf/supermimic_mjlab/{SOURCE}/checkpoints/best/policy_best.pt",
                       actor, 52_517_089, TEACHER_SHA256)
        meta["files"][actor.name] = {"bytes": actor.stat().st_size, "sha256": sha}
    (args.output / "inputs.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Verified inputs: {sum(f['bytes'] for f in meta['files'].values()):,} bytes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy immutable AIR attempt/checkpoint publications to the private gpu01 archive."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from audit_official import checksums, parse_fields


RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
JOB_ID = re.compile(r"[1-9][0-9]{0,19}")
SHA256 = re.compile(r"[0-9a-f]{64}")
OUT = "outputs.tar.gz"
CHECKPOINT = "checkpoint.tar.gz"
SMALL = 256 * 1024
MAX_ATTEMPT = 4 * (1 << 30)
MAX_CHECKPOINT = 8 * (1 << 30)
MAX_STEP = 320_000
MAX_STEPS = 32
BLOCK = 1024 * 1024
RESERVE = 1 << 30
PUBLIC_LIMIT = 200_000_000_000
DIRECTORY_FLAGS = os.O_NOFOLLOW | os.O_DIRECTORY | os.O_RDONLY | os.O_CLOEXEC
FILE_FLAGS = os.O_NOFOLLOW | os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC
ATTEMPT_NAMES = {"SHA256SUMS", "job-status.txt", OUT}
CHECKPOINT_NAMES = {"SHA256SUMS", "checkpoint.json", CHECKPOINT}
REASONS = {"completed", "setup_error", "training_error", "verification_failed", "walltime_warning", "terminated", "interrupted"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def private_directory(info: os.stat_result) -> bool:
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o777 == 0o700


def private_file(info: os.stat_result) -> bool:
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0


@contextmanager
def open_directory(name: str | Path, parent: int | None = None):
    fd = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
    try:
        require(private_directory(os.fstat(fd)), "Backup directories must be real, owner-only directories")
        yield fd
    finally:
        os.close(fd)


@contextmanager
def open_file(name: str, parent: int, size_limit: int):
    fd = os.open(name, FILE_FLAGS, dir_fd=parent)
    try:
        info = os.fstat(fd)
        require(private_file(info) and 0 < info.st_size <= size_limit,
                f"Unexpected owner, type, permissions or size: {name}")
        yield fd, info
    finally:
        os.close(fd)


def immutable_info(info: os.stat_result) -> tuple:
    # ctime and link count can change if a closed rolling checkpoint is pruned.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def read_bytes(fd: int, size: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    result = bytearray()
    while len(result) < size:
        block = os.read(fd, size - len(result))
        require(bool(block), "Unexpected end of backup control file")
        result.extend(block)
    require(not os.read(fd, 1), "Backup control file expanded during inspection")
    return bytes(result)


@dataclass(frozen=True)
class Member:
    fd: int
    info: os.stat_result
    digest: str = ""


@dataclass(frozen=True)
class Publication:
    group: str
    name: str
    members: dict[str, Member]
    mode: str

    @property
    def bytes(self) -> int:
        return sum(item.info.st_size for item in self.members.values())


def exact_names(fd: int, possibilities: tuple[set[str], ...]) -> set[str]:
    names: set[str] = set()
    with os.scandir(fd) as entries:
        for entry in entries:
            names.add(entry.name)
            require(any(names <= option for option in possibilities), "Unexpected backup publication file")
    require(names in possibilities, "Missing expected backup publication file")
    return names


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_control(entries: dict[str, Member], name: str, *, sha: str | None = None) -> bytes:
    item = entries[name]
    data = read_bytes(item.fd, item.info.st_size)
    require(immutable_info(os.fstat(item.fd)) == immutable_info(item.info), f"Source changed: {name}")
    actual = digest(data)
    require(sha is None or sha == actual, f"Source checksum mismatch: {name}")
    entries[name] = Member(item.fd, item.info, actual)
    return data


def snapshot(stack: ExitStack, source: int, run: str, group: str, name: str) -> Publication:
    attempt = group == "attempts"
    folder = stack.enter_context(open_directory(group, source))
    unit = stack.enter_context(open_directory(name, folder))
    expected = (ATTEMPT_NAMES, ATTEMPT_NAMES | {"SUCCESS"}) if attempt else (CHECKPOINT_NAMES,)
    entries = {}
    archive_name = OUT if attempt else CHECKPOINT
    for member in sorted(exact_names(unit, expected)):
        limit = (MAX_ATTEMPT if attempt else MAX_CHECKPOINT) if member == archive_name else SMALL
        fd, info = stack.enter_context(open_file(member, unit, limit))
        entries[member] = Member(fd, info)
    sums = checksums(checked_control(entries, "SHA256SUMS").decode("ascii"))
    control = "job-status.txt" if attempt else "checkpoint.json"
    require(sums.keys() == {archive_name, control}, "Unexpected checksum manifest names")
    evidence = checked_control(entries, control, sha=sums[control])
    archive = entries[archive_name]
    entries[archive_name] = Member(archive.fd, archive.info, sums[archive_name])
    if attempt:
        meta = parse_fields(evidence.decode("ascii"))
        require(meta.keys() == {"job_id", "exit_code", "reason", "run_id", "mode", "last_published_iteration"}
                and meta["job_id"] == name and meta["run_id"] == run,
                "Attempt status does not identify the requested run and job")
        rc = meta["exit_code"]
        require(re.fullmatch(r"[0-9]{1,3}", rc) is not None and int(rc) <= 255
                and meta["reason"] in REASONS, "Invalid published job termination status")
        last = meta["last_published_iteration"]
        require(last == "" or re.fullmatch(r"[1-9][0-9]{0,5}", last) is not None and int(last) <= MAX_STEP,
                "Invalid last published iteration")
        success = "SUCCESS" in entries
        require(success == (int(rc) == 0) == (meta["reason"] == "completed"),
                "SUCCESS marker disagrees with exit code or reason")
        if success:
            require(checked_control(entries, "SUCCESS") == b"Verified target checkpoint and job outputs.\n",
                    "Unexpected SUCCESS marker")
    else:
        def unique_keys(pairs: list[tuple]) -> dict:
            obj = dict(pairs)
            require(len(obj) == len(pairs), "Duplicate checkpoint metadata field")
            return obj
        meta = json.loads(evidence, object_pairs_hook=unique_keys)
        step = int(name.removeprefix("step-"))
        require(isinstance(meta, dict) and type(meta.get("checkpoint_schema_version")) is int
                and meta["checkpoint_schema_version"] == 1 and meta.get("run_id") == run
                and type(meta.get("completed_iterations")) is int and meta["completed_iterations"] == step
                and isinstance(meta.get("job_id"), str) and JOB_ID.fullmatch(meta["job_id"]) is not None,
                "Checkpoint metadata does not identify the requested run and step")
        require(meta.get("archive") == {"sha256": sums[CHECKPOINT], "bytes": archive.info.st_size},
                "Checkpoint metadata and outer archive size/checksum disagree")
    require(meta.get("mode") in {"full", "smoke", "benchmark"}, "Unexpected publication recipe")
    return Publication(group, name, entries, meta["mode"])


def digest_file(item: Member, target=None) -> str:
    """Hash exactly the number of bytes observed in the opened immutable file."""
    os.lseek(item.fd, 0, os.SEEK_SET)
    remaining = item.info.st_size
    hasher = hashlib.sha256()
    while remaining:
        block = os.read(item.fd, min(BLOCK, remaining))
        require(bool(block), "Artifact truncated during canonical backup")
        if target is not None:
            target.write(block)
        hasher.update(block)
        remaining -= len(block)
    require(not os.read(item.fd, 1) and immutable_info(os.fstat(item.fd)) == immutable_info(item.info),
            "Artifact changed during canonical backup")
    return hasher.hexdigest()


def existing_matches(parent: int, unit: Publication) -> bool:
    try:
        os.stat(unit.name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    with open_directory(unit.name, parent) as existing:
        exact_names(existing, (set(unit.members),))
        for name, source in unit.members.items():
            with open_file(name, existing, source.info.st_size) as (fd, info):
                require(info.st_size == source.info.st_size and digest_file(Member(fd, info)) == source.digest,
                        f"Canonical {unit.group}/{unit.name}/{name} differs; refusing to overwrite")
    return True


def fail_if_present_rename(parent: int, old: str, new: str) -> None:
    # os.rename can replace an empty directory. Linux renameat2 guarantees
    # no-clobber atomically even if a different tool ignores our backup lock.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(parent, os.fsencode(old), parent, os.fsencode(new), 1):
        error = ctypes.get_errno()
        raise OSError(error, f"Cannot atomically publish {new}: {os.strerror(error)}")


def copy_publication(path: Path, parent: int, unit: Publication) -> None:
    partial = Path(tempfile.mkdtemp(prefix=f".backup-{unit.name}.", dir=path))
    try:
        with open_directory(partial.name, parent) as temporary:
            for name, item in unit.members.items():
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=temporary)
                with os.fdopen(fd, "wb") as out:
                    require(digest_file(item, out) == item.digest, f"Source checksum mismatch: {name}")
                    out.flush()
                    os.fsync(out.fileno())
                with open_file(name, temporary, item.info.st_size) as (saved, info):
                    require(info.st_size == item.info.st_size and digest_file(Member(saved, info)) == item.digest,
                            f"Canonical readback checksum mismatch: {name}")
            os.fsync(temporary)
        fail_if_present_rename(parent, partial.name, unit.name)
        os.fsync(parent)
    finally:
        # Only this invocation's unique unpublished directory may be removed.
        if partial.exists():
            with open_directory(partial.name, parent) as temporary:
                for name in unit.members:
                    try:
                        os.unlink(name, dir_fd=temporary)
                    except FileNotFoundError:
                        pass
            os.rmdir(partial.name, dir_fd=parent)


def make_directory(name: str, parent: int) -> None:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent)
        os.fsync(parent)
    except FileExistsError:
        pass  # Followed by checked O_NOFOLLOW directory opens.


def backup(source_root: Path, destination: Path, run: str, job: str, steps: list[int], quota_check) -> list[tuple[str, bool]]:
    """Copy explicitly named units. The CLI supplies the fixed target and quota guard."""
    require(RUN_ID.fullmatch(run) is not None and JOB_ID.fullmatch(job) is not None, "Invalid run or job ID")
    require(len(steps) <= MAX_STEPS and len(steps) == len(set(steps)) and
            all(type(step) is int and 0 < step <= MAX_STEP for step in steps), "Invalid or duplicate checkpoint step selection")
    require(source_root == source_root.resolve(strict=True) and destination == destination.resolve(strict=True)
            and not source_root.is_relative_to(destination) and not destination.is_relative_to(source_root),
            "Source and destination must be disjoint canonical paths with no symlink components")
    with ExitStack() as stack:
        source = stack.enter_context(open_directory(source_root))
        source_run = stack.enter_context(open_directory(run, source))
        # Pin every validated file open before writes. Rolling pruning either
        # fails closed here or cannot change the open file's verified contents.
        attempt = snapshot(stack, source_run, run, "attempts", job)
        units = [attempt, *(snapshot(stack, source_run, run, "checkpoints", f"step-{step:09d}") for step in steps)]
        require(all(unit.mode == attempt.mode for unit in units), "Checkpoint recipe differs from the attempt")
        archive = stack.enter_context(open_directory(destination))
        lock_fd = os.open(".backup.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                          0o600, dir_fd=archive)
        stack.callback(os.close, lock_fd)
        require(private_file(os.fstat(lock_fd)), "Canonical backup lock is not an owner-only regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Ensure a little headroom before adding directory entries, then
        # reserve only for new units once already-present copies are checked.
        quota_check(destination, 0)
        make_directory(run, archive)
        destination_run = stack.enter_context(open_directory(run, archive))
        parents = {}
        for group in {unit.group for unit in units}:
            make_directory(group, destination_run)
            parents[group] = stack.enter_context(open_directory(group, destination_run))
        results = []
        # First validate every existing unit; catch a no-clobber conflict before
        # committing any new units. Units publish atomically, not as one bundle.
        known = [existing_matches(parents[unit.group], unit) for unit in units]
        remaining = sum(unit.bytes for unit, current in zip(units, known, strict=True) if not current)
        if remaining:
            quota_check(destination, remaining)
        for unit, current in zip(units, known, strict=True):
            if not current:
                quota_check(destination, unit.bytes)
                copy_publication(destination / run / unit.group, parents[unit.group], unit)
            results.append((f"{unit.group}/{unit.name}", not current))
        return results


def parse_quota_kib(output: str) -> tuple[int, int]:
    """Support XFS quota's wrapped and single-line numeric block rows."""
    rows = []
    pending = False
    for line in output.splitlines():
        columns = line.split()
        if not columns:
            continue
        if columns[0].startswith("/"):
            pending = len(columns) == 1
            columns = columns[1:]
        elif not pending:
            continue
        if len(columns) >= 3 and all(re.fullmatch(r"[0-9]+", value) for value in columns[:3]):
            used, _soft, hard = map(int, columns[:3])
            rows.append((used, hard))
            pending = False
    require(len(rows) == 1, "Could not read one unambiguous /public XFS user quota")
    return rows[0]


def check_quota(destination: Path, size: int) -> None:
    text = system_output(["xfs_quota", "-c", "quota -N -v -u -b", "/public"])
    used_kib, hard_kib = parse_quota_kib(text)
    cap = min(PUBLIC_LIMIT, hard_kib * 1024) if hard_kib else PUBLIC_LIMIT
    info = os.statvfs(destination)
    require(size + RESERVE <= info.f_bavail * info.f_frsize and used_kib * 1024 + size + RESERVE <= cap,
            "Insufficient canonical /public quota or filesystem headroom")


def system_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, timeout=15,
                                   env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"})


def canonical_destination() -> Path:
    # gpu01 is the documented login alias, not necessarily the OS hostname.
    # Its canonical archive is login-local and lives on a dedicated XFS mount.
    require(Path("/etc/cluster/AGENTS_SERVER.md").is_file() and not os.environ.get("SLURM_JOB_ID"),
            "Canonical backup must run on the gpu01 login host, outside a Slurm allocation")
    account = pwd.getpwuid(os.geteuid()).pw_name
    require(RUN_ID.fullmatch(account) is not None, "Invalid login account name")
    root = Path("/public") / account / "supermimic-archive"
    destination = root / "runs"
    require(destination.resolve(strict=True) == destination and private_directory(root.lstat())
            and private_directory(destination.lstat()), "Canonical archive must be an existing owner-only directory")
    filesystem = system_output(["findmnt", "-n", "-o", "TARGET,FSTYPE", "-T", str(destination)]).split()
    require(filesystem == ["/public", "xfs"], "Canonical archive must reside on gpu01 /public XFS")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Canonical path of approved owner-only job-visible runs root")
    parser.add_argument("run_id")
    parser.add_argument("job_id")
    parser.add_argument("--step", action="append", type=int, default=[], help="Copy this explicit durable step (repeat for more)")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        destination = canonical_destination()
        for name, created in backup(args.source_root, destination, args.run_id, args.job_id, args.step, check_quota):
            print(f"{'Copied' if created else 'Already verified'} {args.run_id}/{name}")
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Canonical backup failed: {error}\n")


if __name__ == "__main__":
    main()

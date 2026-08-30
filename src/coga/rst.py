"""Download, verify, and extract the public Recursive Task Synthesis data."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .common import atomic_json, resolve_project_path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str


DATASETS = (
    DatasetSpec("tasks", "Zhongzhi1228/Recursive-Task-Synthesis"),
    DatasetSpec("trajectories", "Zhongzhi1228/Recursive-Task-Synthesis-Trajectories"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(config: Mapping[str, Any]) -> dict[str, Any]:
    """Use the Hub's resumable downloader and retain the upstream layout."""
    from huggingface_hub import snapshot_download

    raw_dir = resolve_project_path(config["paths"]["dataset_raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for spec in DATASETS:
        destination = raw_dir / spec.repo_id.rsplit("/", 1)[1]
        snapshot_path = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision="main",
            local_dir=destination,
            resume_download=True,
        )
        reports[spec.name] = {
            "repo_id": spec.repo_id,
            "revision": "main",
            "path": str(Path(snapshot_path).resolve()),
        }
    report = {"status": "DOWNLOADED", "datasets": reports}
    atomic_json(raw_dir / "download_report.json", report)
    return report


def _manifest_rows(repository: Path) -> list[dict[str, Any]]:
    path = repository / "metadata" / "shard_manifest.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"invalid manifest row at {path}:{line_number}")
            rows.append(row)
    return rows


def verify(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every public TAR against the publisher's size and SHA-256."""
    raw_dir = resolve_project_path(config["paths"]["dataset_raw_dir"])
    datasets: dict[str, Any] = {}
    for spec in DATASETS:
        repository = raw_dir / spec.repo_id.rsplit("/", 1)[1]
        checked = 0
        total_bytes = 0
        for row in _manifest_rows(repository):
            relative = PurePosixPath(str(row["shard"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe shard path in {repository}: {relative}")
            archive = repository.joinpath(*relative.parts)
            size = archive.stat().st_size
            expected_size = int(row["size_bytes"])
            if size != expected_size:
                raise RuntimeError(f"size mismatch for {archive}: {size} != {expected_size}")
            digest = sha256_file(archive)
            if digest.casefold() != str(row["sha256"]).casefold():
                raise RuntimeError(f"SHA-256 mismatch for {archive}")
            checked += 1
            total_bytes += size
        datasets[spec.name] = {"archives": checked, "bytes": total_bytes, "status": "VERIFIED"}
    report = {"status": "VERIFIED", "datasets": datasets}
    atomic_json(raw_dir / "verification_report.json", report)
    return report


def _validate_tar_members(members: Iterable[tarfile.TarInfo], destination: Path) -> None:
    root = destination.resolve()
    for member in members:
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe TAR member: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"unsupported TAR link/device member: {member.name}")
        target = destination.joinpath(*relative.parts).resolve()
        target.relative_to(root)


def extract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract verified shards idempotently, recording a hash marker per TAR."""
    raw_dir = resolve_project_path(config["paths"]["dataset_raw_dir"])
    extracted_dir = resolve_project_path(config["paths"]["dataset_extracted_dir"])
    state_dir = extracted_dir / ".state"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    completed = skipped = 0
    for spec in DATASETS:
        repository = raw_dir / spec.repo_id.rsplit("/", 1)[1]
        for row in _manifest_rows(repository):
            relative = PurePosixPath(str(row["shard"]))
            archive = repository.joinpath(*relative.parts)
            expected = str(row["sha256"]).casefold()
            marker = state_dir / f"{spec.name}-{archive.name}.sha256"
            if marker.is_file() and marker.read_text(encoding="ascii").strip().casefold() == expected:
                skipped += 1
                continue
            if sha256_file(archive).casefold() != expected:
                raise RuntimeError(f"refusing to extract unverified archive: {archive}")
            with tarfile.open(archive, mode="r:*") as bundle:
                members = bundle.getmembers()
                _validate_tar_members(members, extracted_dir)
                bundle.extractall(extracted_dir, members=members)
            temporary = marker.with_suffix(marker.suffix + f".tmp.{os.getpid()}")
            temporary.write_text(expected + "\n", encoding="ascii")
            os.replace(temporary, marker)
            completed += 1
    report = {
        "status": "EXTRACTED",
        "extracted_archives": completed,
        "already_extracted_archives": skipped,
        "extracted_dir": str(extracted_dir),
    }
    atomic_json(extracted_dir / "extraction_report.json", report)
    return report

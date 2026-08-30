from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch


COGA_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = COGA_ROOT.parent


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def canonical_tree_sha256(path: Path) -> tuple[str, int, int]:
    """Hash a dataset tree using the same contract as the Stage 4 freeze."""
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        size = item.stat().st_size
        total_bytes += size
        rows.append({
            "path": item.relative_to(path).as_posix(),
            "bytes": size,
            "sha256": sha256_file(item),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), len(rows), total_bytes


def terminal_bench_tasks(config: Mapping[str, Any]) -> list[str]:
    """Discover and verify the complete local Terminal-Bench dataset export."""
    benchmark = config["evaluation"]["terminal_bench"]
    if benchmark.get("task_scope") != "all_dataset_tasks":
        raise RuntimeError("Terminal-Bench requires task_scope=all_dataset_tasks")
    dataset_path = resolve_project_path(benchmark["dataset_path"])
    directories = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    missing_contract = [path.name for path in directories if not (path / "task.toml").is_file()]
    if missing_contract:
        raise RuntimeError(f"Terminal-Bench task directories missing task.toml: {missing_contract}")
    tasks = [path.name for path in directories]
    expected_count = int(benchmark["expected_task_count"])
    if len(tasks) != expected_count or len(set(tasks)) != expected_count:
        raise RuntimeError(f"expected {expected_count} unique Terminal-Bench tasks, found {len(tasks)}")

    inventory = read_json(resolve_project_path(benchmark["task_inventory"]))
    if inventory.get("dataset_tree_sha256") != benchmark["expected_dataset_tree_sha256"]:
        raise RuntimeError("Terminal-Bench inventory tree hash differs from the frozen config")
    tree_hash, tree_files, tree_bytes = canonical_tree_sha256(dataset_path)
    if tree_hash != benchmark["expected_dataset_tree_sha256"]:
        raise RuntimeError("Terminal-Bench dataset tree differs from the frozen 89-task export")
    if tree_files != inventory.get("dataset_files") or tree_bytes != inventory.get("dataset_definition_bytes"):
        raise RuntimeError("Terminal-Bench dataset file count/bytes differ from the frozen inventory")
    inventory_tasks = sorted(row["task_id"] for row in inventory.get("tasks", []))
    if inventory_tasks != tasks:
        raise RuntimeError("Terminal-Bench directory tasks differ from the frozen 89-task inventory")
    return tasks


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            yield value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    os.replace(temporary, path)
    return {"rows": count, "sha256": digest.hexdigest()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: Any) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8", errors="replace")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def component_fold(family: str, prompt: str, folds: int, seed: int) -> int:
    if folds < 2:
        raise ValueError("folds must be at least two")
    return stable_int("fold", seed, family, prompt) % folds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved

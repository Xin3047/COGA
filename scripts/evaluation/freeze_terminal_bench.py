#!/usr/bin/env python3
"""Freeze a downloaded Terminal-Bench export before any model evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from coga.common import atomic_json, canonical_tree_sha256, read_json, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    benchmark = config["evaluation"]["terminal_bench"]
    dataset = resolve_project_path(benchmark["dataset_path"])
    tasks = sorted(path.name for path in dataset.iterdir() if path.is_dir() and (path / "task.toml").is_file())
    expected = int(benchmark["expected_task_count"])
    if len(tasks) != expected:
        raise RuntimeError(f"expected {expected} Terminal-Bench tasks, found {len(tasks)}")
    tree_hash, file_count, byte_count = canonical_tree_sha256(dataset)
    output = resolve_project_path(benchmark["task_inventory"])
    atomic_json(output, {
        "benchmark": "Terminal-Bench 2.1",
        "dataset_tree_sha256": tree_hash,
        "dataset_files": file_count,
        "dataset_definition_bytes": byte_count,
        "tasks": [{"task_id": task} for task in tasks],
    })
    print({"inventory": str(output), "tasks": len(tasks), "dataset_tree_sha256": tree_hash})


if __name__ == "__main__":
    main()

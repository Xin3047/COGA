#!/usr/bin/env python3
"""Run the official BFCL V4 Multi-Turn evaluator on base and both adapters."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from coga.common import atomic_json, read_json, resolve_project_path


def overall_from_rows(rows: list[dict[str, Any]], model_name: str):
    selected = next(
        (row for row in rows if any(model_name.casefold() == str(value).casefold() for value in row.values())),
        None,
    )
    if selected is None:
        return None, None
    overall = None
    for key, value in selected.items():
        if key is None or "overall" not in str(key).casefold():
            continue
        try:
            overall = float(str(value).strip().rstrip("%"))
            break
        except ValueError:
            continue
    return selected, overall


def official_overall_row(score_root: Path, model_name: str):
    path = score_root / "data_overall.csv"
    if not path.is_file():
        return None, None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return overall_from_rows(list(csv.DictReader(handle)), model_name)


def generate_command(config: dict[str, Any], model: dict[str, Any]) -> list[str]:
    bfcl = config["evaluation"]["bfcl"]
    command = [
        str(resolve_project_path(bfcl["executable"])), "generate",
        "--model", bfcl["model_name"],
        "--test-category", *list(bfcl["categories"]),
        "--backend", "vllm",
        "--num-gpus", "1",
        "--gpu-memory-utilization", str(config["evaluation"]["gpu_memory_utilization"]),
        "--local-model-path", str(resolve_project_path(config["paths"]["model"])),
        "--num-threads", "1",
    ]
    if model.get("adapter"):
        command.extend([
            "--enable-lora",
            "--max-lora-rank", str(config["model"]["lora"]["r"]),
            "--lora-modules",
            f"{bfcl['model_name']}={resolve_project_path(model['adapter'])}",
        ])
    return command


def evaluate_command(config: dict[str, Any]) -> list[str]:
    bfcl = config["evaluation"]["bfcl"]
    return [
        str(resolve_project_path(bfcl["executable"])), "evaluate",
        "--model", bfcl["model_name"],
        "--test-category", *list(bfcl["categories"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BFCL V4 Multi-Turn comparison.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if config.get("execution_scope") != "FULL_PRODUCTION_ONLY":
        raise RuntimeError("BFCL requires execution_scope=FULL_PRODUCTION_ONLY")
    run_dir = resolve_project_path(config["paths"]["run_dir"])
    output_dir = run_dir / "evaluation" / "bfcl"
    output_dir.mkdir(parents=True, exist_ok=True)
    bfcl = config["evaluation"]["bfcl"]
    categories = list(bfcl["categories"])
    model_names = [model["name"] for model in config["evaluation"]["models"]]
    if model_names != ["base", "rejection_success", "coga_selected"]:
        raise RuntimeError("the frozen full-evaluation contract requires base, rejection_success, coga_selected")
    if categories != ["multi_turn"]:
        raise RuntimeError("the BFCL contract requires the official multi_turn test group")
    reports = {}

    for model in config["evaluation"]["models"]:
        arm_root = output_dir / model["name"]
        arm_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["BFCL_PROJECT_ROOT"] = str(arm_root)
        command = generate_command(config, model)
        started = time.monotonic()
        generated = subprocess.run(command, env=environment, check=False)
        evaluated = subprocess.run(
            evaluate_command(config), env=environment, check=False
        ) if generated.returncode == 0 else None
        score_files = [str(path) for path in sorted((arm_root / "score").rglob("*")) if path.is_file()]
        overall_row, overall_accuracy = official_overall_row(arm_root / "score", bfcl["model_name"])
        complete = (
            generated.returncode == 0
            and evaluated is not None
            and evaluated.returncode == 0
            and overall_row is not None
            and overall_accuracy is not None
        )
        report = {
            "status": "COMPLETE" if complete else "FAILED",
            "benchmark": "BFCL V4 Multi-Turn",
            "model": model["name"],
            "official_bfcl_model_name": bfcl["model_name"],
            "categories": categories,
            "generate_returncode": generated.returncode,
            "evaluate_returncode": evaluated.returncode if evaluated else None,
            "elapsed_seconds": time.monotonic() - started,
            "score_files": score_files,
            "official_overall_row": overall_row,
            "overall_accuracy": overall_accuracy,
        }
        atomic_json(arm_root / "coga_bfcl_report.json", report)
        reports[model["name"]] = report
    atomic_json(output_dir / "bfcl_report.json", {
        "status": "COMPLETE" if all(row["status"] == "COMPLETE" for row in reports.values()) else "FAILED",
        "official_evaluator": "bfcl-eval",
        "models": model_names,
        "categories": categories,
        "arms": reports,
    })


if __name__ == "__main__":
    main()

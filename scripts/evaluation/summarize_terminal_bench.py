#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from coga.common import atomic_json, jsonl_rows, read_json, resolve_project_path, terminal_bench_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if config.get("execution_scope") != "FULL_PRODUCTION_ONLY":
        raise RuntimeError("Terminal-Bench summary requires execution_scope=FULL_PRODUCTION_ONLY")
    run_dir = resolve_project_path(config["paths"]["run_dir"])
    benchmark_dir = run_dir / "evaluation" / "terminal_bench"
    rows = list(jsonl_rows(benchmark_dir / "trials.jsonl"))
    expected_tasks = terminal_bench_tasks(config)
    expected_task_set = set(expected_tasks)
    expected_task_count = int(config["evaluation"]["terminal_bench"]["expected_task_count"])
    by_model = {
        name: {row["task"]: int(row["task_success"]) for row in rows if row["model"] == name}
        for name in ("rejection_success", "coga_selected")
    }
    for name, outcomes in by_model.items():
        if set(outcomes) != expected_task_set or len(outcomes) != expected_task_count:
            raise RuntimeError(f"refusing partial paired summary for {name}")
    tasks = expected_tasks
    differences = np.asarray([
        by_model["coga_selected"][task] - by_model["rejection_success"][task]
        for task in tasks
    ], dtype=np.float64)
    rng = np.random.default_rng(int(config["seed"]))
    draws = rng.choice(differences, size=(10_000, len(differences)), replace=True).mean(axis=1)
    report = read_json(benchmark_dir / "terminal_bench_report.json")
    report["paired_coga_minus_rejection"] = float(differences.mean())
    report["paired_coga_minus_rejection_ci95"] = [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]
    report["paired_tasks"] = len(tasks)
    report["bootstrap_draws"] = 10_000
    atomic_json(benchmark_dir / "terminal_bench_report.json", report)


if __name__ == "__main__":
    main()

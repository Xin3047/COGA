#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from coga.common import (
    atomic_json,
    jsonl_rows,
    read_json,
    resolve_project_path,
    sha256_file,
    stable_int,
    write_jsonl,
)
from coga.data import load_population, load_rewards, load_turns, render_target_only


def round_robin(rows_by_cohort: Mapping[str, list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    queues = {
        cohort_id: deque(rows)
        for cohort_id, rows in sorted(rows_by_cohort.items())
    }
    while queues:
        for cohort_id in list(queues):
            queue = queues[cohort_id]
            if queue:
                yield queue.popleft()
            if not queue:
                del queues[cohort_id]


def selected_cohorts(config, scores):
    real = [row for row in scores if row["control"] == "real"]
    fraction = float(config["selection"]["top_fraction"])
    count = max(1, math.ceil(len(real) * fraction))
    coga = sorted(real, key=lambda row: (-row["contrast_alignment"], row["cohort_id"]))[:count]
    baseline = sorted(
        real,
        key=lambda row: stable_int("baseline", config["seed"], row["cohort_id"]),
    )[:count]
    return {
        "coga_selected": {row["cohort_id"]: float(row["contrast_alignment"]) for row in coga},
        "rejection_success": {row["cohort_id"]: None for row in baseline},
    }


def candidates_for_arm(config, cohort_scores):
    cohorts, members = load_population(config)
    rewards = load_rewards(config)
    rows_by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cohort_id in sorted(cohort_scores):
        for ref in sorted(members[cohort_id], key=lambda value: value.trajectory_id):
            if rewards.get(ref.trajectory_id) != 1:
                continue
            header, turns = load_turns(config, ref.relpath)
            for turn_index in range(len(turns)):
                rows_by_cohort[cohort_id].append({
                    "cohort_id": cohort_id,
                    "family": cohorts[cohort_id].family,
                    "prompt": cohorts[cohort_id].prompt,
                    "task_id": cohorts[cohort_id].task_id,
                    "trajectory_id": ref.trajectory_id,
                    "relpath": ref.relpath,
                    "turn_index": turn_index,
                    "header": header,
                    "turns": turns,
                    "coga_alignment": cohort_scores[cohort_id],
                })
        rows_by_cohort[cohort_id].sort(
            key=lambda row: stable_int(
                "candidate", config["seed"], row["trajectory_id"], row["turn_index"]
            )
        )
    return round_robin(rows_by_cohort)


def materialize_arm(config, tokenizer, arm: str, cohort_scores, output_dir: Path):
    budget = int(config["selection"]["loss_token_budget"])
    max_length = int(config["training"]["max_length"])
    tensor_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    excluded_leakage = 0
    total = 0
    for candidate in candidates_for_arm(config, cohort_scores):
        messages, input_ids, labels, leakage = render_target_only(
            tokenizer,
            candidate["header"],
            candidate["turns"],
            candidate["turn_index"],
            max_length,
        )
        if leakage:
            excluded_leakage += 1
            continue
        positive = [index for index, value in enumerate(labels) if value != -100]
        cap = min(len(positive), budget - total)
        if cap <= 0:
            break
        for index in positive[cap:]:
            labels[index] = -100
        tensor_rows.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        })
        manifest_rows.append({
            "arm": arm,
            "cohort_id": candidate["cohort_id"],
            "family": candidate["family"],
            "prompt": candidate["prompt"],
            "task_id": candidate["task_id"],
            "trajectory_id": candidate["trajectory_id"],
            "turn_index": candidate["turn_index"],
            "coga_alignment": candidate["coga_alignment"],
            "sequence_tokens": len(input_ids),
            "loss_tokens": cap,
            "message_roles": [message["role"] for message in messages],
        })
        total += cap
        if total == budget:
            break
    if total != budget:
        raise RuntimeError(f"{arm} has only {total:,} eligible target tokens; expected {budget:,}")

    dataset_path = output_dir / f"{arm}.parquet"
    table = pa.Table.from_pylist(tensor_rows)
    pq.write_table(table, dataset_path, compression="zstd")
    manifest_stats = write_jsonl(output_dir / f"{arm}.manifest.jsonl", manifest_rows)
    report = {
        "status": "COMPLETE",
        "arm": arm,
        "rows": len(tensor_rows),
        "cohorts": len(cohort_scores),
        "loss_tokens": total,
        "max_length": max_length,
        "excluded_for_conservative_leakage_scan": excluded_leakage,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "manifest": manifest_stats,
    }
    atomic_json(output_dir / f"{arm}.report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    run_dir = resolve_project_path(config["paths"]["run_dir"])
    gradient_report = read_json(run_dir / "gradients" / "gradient_report.json")
    if config["selection"].get("require_gradient_gate", True) and gradient_report["method_gate"] != "PASS":
        raise RuntimeError("COGA gradient gate did not pass; refusing to construct a result-selected arm")
    scores = list(jsonl_rows(run_dir / "gradients" / "scores.jsonl"))
    arms = selected_cohorts(config, scores)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_project_path(config["paths"]["model"]), local_files_only=True
    )
    output_dir = run_dir / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        arm: materialize_arm(config, tokenizer, arm, cohort_scores, output_dir)
        for arm, cohort_scores in arms.items()
    }
    atomic_json(output_dir / "dataset_report.json", {
        "status": "COMPLETE",
        "comparison": "coga_selected_vs_rejection_success",
        "equal_loss_token_budget": config["selection"]["loss_token_budget"],
        "arms": reports,
        "benchmark_results": None,
    })


if __name__ == "__main__":
    main()

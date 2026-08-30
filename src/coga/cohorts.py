"""Build the exact-task, mixed-outcome cohort databases consumed by COGA."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .common import atomic_json, resolve_project_path


PROMPT_NORM_VERSION = "rst_prompt_norm_v1"
POLICY_KEY_VERSION = "rst_policy_key_v1"
TASK_MARKER = "Task Description:"
TERMINAL_MARKER = "Current terminal state:"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _hash(*parts: Any) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def normalize_prompt(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def prompt_key(text: str) -> str:
    return _hash(PROMPT_NORM_VERSION, normalize_prompt(text))


def extract_task_description(message: Any) -> str | None:
    if not isinstance(message, str):
        return None
    if message.count(TASK_MARKER) != 1 or message.count(TERMINAL_MARKER) != 1:
        return None
    start = message.find(TASK_MARKER) + len(TASK_MARKER)
    end = message.find(TERMINAL_MARKER)
    return message[start:end] if end >= start else None


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _operator(task_dir: Path) -> str:
    document = _read_object(task_dir / "rewrite_target.json")
    try:
        return str(document["rewrite_target"]["preferred_card"]["operator"])
    except (TypeError, KeyError):
        return "missing"


def _reward(result: Mapping[str, Any]) -> int | None:
    try:
        value = result["verifier_result"]["rewards"]["reward"]
    except (TypeError, KeyError):
        return None
    return int(value) if value in (0, 1, 0.0, 1.0) else None


def _policy_key(result: Mapping[str, Any], trajectory: Mapping[str, Any]) -> str:
    info = result.get("agent_info") if isinstance(result.get("agent_info"), Mapping) else {}
    model_info = info.get("model_info") if isinstance(info.get("model_info"), Mapping) else {}
    agent = trajectory.get("agent") if isinstance(trajectory.get("agent"), Mapping) else {}
    extra = agent.get("extra") if isinstance(agent.get("extra"), Mapping) else {}
    payload = {
        "agent_name": info.get("name") or agent.get("name"),
        "agent_version": info.get("version") or agent.get("version"),
        "model_name": model_info.get("name") or agent.get("model_name"),
        "model_provider": model_info.get("provider"),
        "parser_name": extra.get("parser"),
        "temperature": extra.get("temperature") if isinstance(extra.get("temperature"), (int, float)) else None,
    }
    return _hash(POLICY_KEY_VERSION, _compact_json(payload))


def _task_map(tasks_dir: Path) -> dict[str, tuple[str, Path]]:
    by_prompt: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        instruction = task_dir / "instruction.md"
        if not instruction.is_file():
            continue
        key = prompt_key(instruction.read_text(encoding="utf-8", errors="strict"))
        by_prompt[key].append((task_dir.name, task_dir))
    return {key: rows[0] for key, rows in by_prompt.items() if len(rows) == 1}


def _trajectory_rows(trajectories_dir: Path, raw_root: Path):
    for trajectory_dir in sorted(path for path in trajectories_dir.iterdir() if path.is_dir()):
        result = _read_object(trajectory_dir / "result.json")
        trajectory_path = trajectory_dir / "trajectory.json"
        trajectory = _read_object(trajectory_path)
        if result is None or trajectory is None:
            continue
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            continue
        first_user = next(
            (row.get("message") for row in steps if isinstance(row, Mapping) and row.get("source") == "user"),
            None,
        )
        description = extract_task_description(first_user)
        family = result.get("task_group_id")
        if description is None or not isinstance(family, str):
            continue
        trajectory_id = str(result.get("trajectory_id") or trajectory_dir.name)
        agent_steps = sum(
            isinstance(row, Mapping) and row.get("source") == "agent" for row in steps
        )
        agent_result = result.get("agent_result") if isinstance(result.get("agent_result"), Mapping) else {}
        output_tokens = agent_result.get("n_output_tokens")
        relpath = trajectory_path.resolve().relative_to(raw_root.resolve()).as_posix()
        yield {
            "trajectory_id": trajectory_id,
            "family": family,
            "prompt": prompt_key(description),
            "policy": _policy_key(result, trajectory),
            "reward": _reward(result),
            "relpath": relpath,
            "steps": int(agent_steps),
            "output_tokens": int(output_tokens) if isinstance(output_tokens, int) else 0,
        }


def build(config: Mapping[str, Any]) -> dict[str, Any]:
    """Scan public JSON once and atomically materialize COGA's compact DBs."""
    raw_root = resolve_project_path(config["paths"]["raw_root"])
    extracted_dir = resolve_project_path(config["paths"]["dataset_extracted_dir"])
    tasks = _task_map(extracted_dir / "tasks")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _trajectory_rows(extracted_dir / "trajectories", raw_root):
        if row["prompt"] in tasks:
            grouped[(row["family"], row["prompt"], row["policy"])].append(row)

    minimum = int(config["data"]["minimum_successes_and_failures"])
    eligible = {
        key: rows for key, rows in grouped.items()
        if sum(row["reward"] == 1 for row in rows) >= minimum
        and sum(row["reward"] == 0 for row in rows) >= minimum
    }
    expected = config["data"].get("expected_cohorts")
    if expected is not None and len(eligible) != int(expected):
        raise RuntimeError(f"expected {expected} exact-task cohorts, found {len(eligible)}")

    population_path = resolve_project_path(config["paths"]["population_db"])
    outcomes_path = resolve_project_path(config["paths"]["outcomes_db"])
    population_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    population_tmp = population_path.with_suffix(population_path.suffix + f".tmp.{os.getpid()}")
    outcomes_tmp = outcomes_path.with_suffix(outcomes_path.suffix + f".tmp.{os.getpid()}")
    population = sqlite3.connect(population_tmp)
    outcomes = sqlite3.connect(outcomes_tmp)
    population.executescript("""
      CREATE TABLE cohorts(
        cohort_id TEXT PRIMARY KEY, family TEXT NOT NULL, prompt TEXT NOT NULL,
        policy TEXT NOT NULL, task_id TEXT NOT NULL, operator TEXT NOT NULL
      );
      CREATE TABLE trajectories(
        trajectory_id TEXT PRIMARY KEY, cohort_id TEXT NOT NULL, relpath TEXT NOT NULL,
        expected_steps INTEGER NOT NULL, output_tokens INTEGER NOT NULL
      );
      CREATE INDEX idx_trajectories_cohort ON trajectories(cohort_id);
    """)
    outcomes.execute("CREATE TABLE outcomes(trajectory_id TEXT PRIMARY KEY, reward INTEGER CHECK(reward IN(0,1)))")
    success = failure = censored = trajectory_count = 0
    for (family, prompt, policy), rows in sorted(eligible.items()):
        cohort_id = _hash("cohort_v1", family, prompt, policy)[:24]
        task_id, task_dir = tasks[prompt]
        population.execute(
            "INSERT INTO cohorts VALUES(?,?,?,?,?,?)",
            (cohort_id, family, prompt, policy, task_id, _operator(task_dir)),
        )
        for row in sorted(rows, key=lambda item: item["trajectory_id"]):
            population.execute(
                "INSERT INTO trajectories VALUES(?,?,?,?,?)",
                (row["trajectory_id"], cohort_id, row["relpath"], row["steps"], row["output_tokens"]),
            )
            trajectory_count += 1
            if row["reward"] in (0, 1):
                outcomes.execute("INSERT INTO outcomes VALUES(?,?)", (row["trajectory_id"], row["reward"]))
            success += row["reward"] == 1
            failure += row["reward"] == 0
            censored += row["reward"] is None
    population.commit()
    outcomes.commit()
    population.close()
    outcomes.close()
    os.replace(population_tmp, population_path)
    os.replace(outcomes_tmp, outcomes_path)
    report = {
        "status": "COMPLETE",
        "cohorts": len(eligible),
        "trajectories": trajectory_count,
        "success": success,
        "failure": failure,
        "censored": censored,
        "minimum_successes_and_failures": minimum,
        "normalization": PROMPT_NORM_VERSION,
        "policy_key": POLICY_KEY_VERSION,
        "population_db": str(population_path),
        "outcomes_db": str(outcomes_path),
    }
    atomic_json(population_path.parent / "cohort_report.json", report)
    return report

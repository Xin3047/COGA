from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from coga.cohorts import build, normalize_prompt, prompt_key


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prompt_normalization_is_stable() -> None:
    assert normalize_prompt("\ufeffhello  \r\nworld\r\n") == "hello\nworld"
    assert prompt_key("hello\r\n") == prompt_key("hello\n")


def test_build_exact_task_mixed_outcome_population(tmp_path: Path) -> None:
    raw_root = tmp_path / "data"
    extracted = raw_root / "extracted"
    task = extracted / "tasks" / "task-1"
    task.mkdir(parents=True)
    instruction = "Create the requested artifact."
    (task / "instruction.md").write_text(instruction, encoding="utf-8")
    write_json(task / "rewrite_target.json", {
        "rewrite_target": {"preferred_card": {"operator": "write"}}
    })

    user_message = f"prefix\nTask Description:\n{instruction}\nCurrent terminal state:\n$"
    for ordinal in range(8):
        trajectory_id = f"trajectory-{ordinal}"
        directory = extracted / "trajectories" / trajectory_id
        write_json(directory / "result.json", {
            "trajectory_id": trajectory_id,
            "task_group_id": "family-1",
            "agent_info": {
                "name": "terminus-2",
                "version": "2.0.0",
                "model_info": {"name": "model", "provider": "hosted_vllm"},
            },
            "agent_result": {"n_output_tokens": 10},
            "verifier_result": {"rewards": {"reward": 1 if ordinal < 4 else 0}},
        })
        write_json(directory / "trajectory.json", {
            "agent": {"name": "terminus-2", "version": "2.0.0", "extra": {"parser": "json", "temperature": 1.0}},
            "steps": [
                {"source": "user", "message": user_message},
                {"source": "agent", "message": "{}", "observation": {"results": []}},
            ],
        })

    population = raw_root / "processed" / "population.sqlite"
    outcomes = raw_root / "processed" / "outcomes.sqlite"
    config = {
        "paths": {
            "raw_root": str(raw_root),
            "dataset_extracted_dir": str(extracted),
            "population_db": str(population),
            "outcomes_db": str(outcomes),
        },
        "data": {"minimum_successes_and_failures": 4, "expected_cohorts": 1},
    }
    report = build(config)
    assert report["cohorts"] == 1
    assert report["trajectories"] == 8
    with sqlite3.connect(population) as database:
        assert database.execute("SELECT COUNT(*) FROM cohorts").fetchone()[0] == 1
        assert database.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0] == 8
        assert database.execute("SELECT operator FROM cohorts").fetchone()[0] == "write"
    with sqlite3.connect(outcomes) as database:
        assert database.execute("SELECT reward, COUNT(*) FROM outcomes GROUP BY reward ORDER BY reward").fetchall() == [(0, 4), (1, 4)]

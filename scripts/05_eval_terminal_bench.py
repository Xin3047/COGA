#!/usr/bin/env python3
"""Serial Terminal-Bench evaluation for base, rejection, and COGA adapters."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from coga.common import atomic_json, read_json, resolve_project_path, terminal_bench_tasks, write_jsonl


def endpoint(port: int, path: str, payload: dict[str, Any] | None = None, timeout: int = 30):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read().decode()
        return json.loads(content) if content.strip() else {"status": response.status}


def vllm_command(config, model, port: int) -> list[str]:
    executable = resolve_project_path(config["evaluation"]["vllm_executable"])
    model_path = resolve_project_path(config["paths"]["model"])
    command = [
        str(executable), "serve", str(model_path),
        "--served-model-name", config["evaluation"]["base_served_name"],
        "--dtype", "bfloat16",
        "--max-model-len", str(config["evaluation"]["max_model_length"]),
        "--gpu-memory-utilization", str(config["evaluation"]["gpu_memory_utilization"]),
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", str(config["evaluation"]["max_model_length"]),
        "--enforce-eager",
        "--generation-config", "vllm",
        "--default-chat-template-kwargs", '{"enable_thinking":false}',
        "--seed", str(config["seed"]),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if model.get("adapter"):
        adapter = resolve_project_path(model["adapter"])
        command.extend([
            "--enable-lora",
            "--max-loras", "1",
            "--max-lora-rank", str(config["model"]["lora"]["r"]),
            "--lora-modules", f"{model['served_name']}={adapter}",
        ])
    return command


def wait_for_service(process: subprocess.Popen, port: int, served_name: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            endpoint(port, "/health", timeout=5)
            response = endpoint(port, "/v1/chat/completions", {
                "model": served_name,
                "messages": [{"role": "user", "content": "Reply with READY."}],
                "temperature": 0,
                "max_tokens": 8,
            })
            if response.get("choices"):
                return
        except Exception as error:  # service may still be loading
            last_error = error
        time.sleep(5)
    raise TimeoutError(f"vLLM was not ready: {last_error}")


def reward_candidates(value: Any, path: str = "$") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.casefold() in {"reward", "rewards"}:
                if isinstance(child, (int, float)) and not isinstance(child, bool):
                    found.append({"path": child_path, "value": float(child)})
                elif isinstance(child, dict):
                    for subkey, number in child.items():
                        if isinstance(number, (int, float)) and not isinstance(number, bool):
                            found.append({"path": f"{child_path}.{subkey}", "value": float(number)})
            found.extend(reward_candidates(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(reward_candidates(child, f"{path}[{index}]"))
    return found


def inspect_job(job_dir: Path) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    result_files = sorted(job_dir.rglob("*.json")) if job_dir.exists() else []
    for path in result_files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in reward_candidates(document):
            candidates.append({**row, "file": str(path)})
    values = [row["value"] for row in candidates]
    return {
        "result_files": len(result_files),
        "reward_candidates": candidates,
        "task_success": any(value >= 1.0 for value in values),
    }


def harbor_job_config(config, model, task: str, job_name: str, jobs_dir: Path, port: int):
    dataset_path = resolve_project_path(config["evaluation"]["terminal_bench"]["dataset_path"])
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "agent_timeout_multiplier": 1.0,
        "verifier_timeout_multiplier": 1.0,
        "n_concurrent_trials": 1,
        "environment": {"type": "docker", "delete": False, "override_gpus": 0},
        "agents": [{
            "name": "terminus-2",
            "model_name": f"hosted_vllm/{model['served_name']}",
            "kwargs": {
                "api_base": f"http://127.0.0.1:{port}/v1",
                "temperature": 0.0,
                "max_turns": int(config["evaluation"]["terminal_bench"]["max_turns"]),
                "enable_summarize": True,
                "proactive_summarization_threshold": 0,
                "model_info": {
                    "max_input_tokens": int(config["evaluation"]["terminal_bench"]["max_input_tokens"]),
                    "max_output_tokens": int(config["evaluation"]["terminal_bench"]["max_output_tokens"]),
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "llm_call_kwargs": {
                    "seed": int(config["seed"]),
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
                "llm_kwargs": {"api_key": "EMPTY"},
            },
        }],
        "datasets": [{"path": str(dataset_path), "task_names": [task]}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete frozen Terminal-Bench comparison.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if config.get("execution_scope") != "FULL_PRODUCTION_ONLY":
        raise RuntimeError("Terminal-Bench requires execution_scope=FULL_PRODUCTION_ONLY")
    run_dir = resolve_project_path(config["paths"]["run_dir"])
    output_dir = run_dir / "evaluation" / "terminal_bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = terminal_bench_tasks(config)
    model_names = [model["name"] for model in config["evaluation"]["models"]]
    if model_names != ["base", "rejection_success", "coga_selected"]:
        raise RuntimeError("the frozen full-evaluation contract requires base, rejection_success, coga_selected")
    expected_task_count = int(config["evaluation"]["terminal_bench"]["expected_task_count"])
    harbor = resolve_project_path(config["evaluation"]["harbor_executable"])
    port = int(config["evaluation"]["port"])
    trials: list[dict[str, Any]] = []

    for model in config["evaluation"]["models"]:
        model_dir = output_dir / model["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (model_dir / "vllm.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            vllm_command(config, model, port),
            cwd=resolve_project_path("."),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_service(
                process,
                port,
                model["served_name"],
                int(config["evaluation"]["service_start_timeout_seconds"]),
            )
            for index, task in enumerate(tasks):
                job_name = f"{index:03d}_{task}"
                job_dir = model_dir / "jobs" / job_name
                summary_path = job_dir / "coga_trial_summary.json"
                if summary_path.is_file():
                    trials.append(read_json(summary_path))
                    continue
                job_dir.mkdir(parents=True, exist_ok=True)
                job_config_path = job_dir / "job.json"
                atomic_json(job_config_path, harbor_job_config(
                    config, model, task, job_name, model_dir / "jobs", port
                ))
                started = time.monotonic()
                completed = subprocess.run(
                    [str(harbor), "run", "-c", str(job_config_path), "-y"],
                    cwd=resolve_project_path("."),
                    check=False,
                )
                inspection = inspect_job(job_dir)
                trial = {
                    "benchmark": "terminal-bench-2.1",
                    "model": model["name"],
                    "task": task,
                    "harbor_returncode": completed.returncode,
                    "elapsed_seconds": time.monotonic() - started,
                    **inspection,
                }
                atomic_json(summary_path, trial)
                trials.append(trial)
                print({"model": model["name"], "task": task, "success": trial["task_success"]}, flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
            log_handle.close()

    write_jsonl(output_dir / "trials.jsonl", trials)
    aggregates = {}
    for model in config["evaluation"]["models"]:
        model_trials = [row for row in trials if row["model"] == model["name"]]
        model_tasks = [row["task"] for row in model_trials]
        if len(model_trials) != expected_task_count or set(model_tasks) != set(tasks):
            raise RuntimeError(f"incomplete Terminal-Bench result set for {model['name']}")
        successes = sum(row["task_success"] for row in model_trials)
        aggregates[model["name"]] = {
            "successes": successes,
            "tasks": len(model_trials),
            "success_rate": successes / max(1, len(model_trials)),
        }
    atomic_json(output_dir / "terminal_bench_report.json", {
        "status": "COMPLETE",
        "benchmark": "Terminal-Bench 2.1",
        "task_scope": "all_dataset_tasks",
        "expected_tasks_per_model": expected_task_count,
        "models": model_names,
        "aggregates": aggregates,
        "paired_coga_minus_rejection_ci95": None,
    })


if __name__ == "__main__":
    main()

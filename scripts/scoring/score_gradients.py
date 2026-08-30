#!/usr/bin/env python3
"""Production COGA gradient scorer for one 24GB RTX 4090.

The scorer warms a Qwen3-8B 4-bit LoRA adapter, extracts a deterministic
CountSketch of each per-example LoRA gradient, and scores each same-task
success-minus-failure contrast against a cross-fitted prototype built only
from other family/prompt folds. Controls are scored against the same prototype.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from coga.common import (
    atomic_json,
    component_fold,
    jsonl_rows,
    read_json,
    resolve_project_path,
    set_seed,
    sha256_file,
    stable_int,
    write_jsonl,
)
from coga.data import Turn, load_turns, render_target_only, transform_turns


def target_only_loss(model, input_ids, attention_mask, labels):
    import torch
    import torch.nn.functional as functional

    target_positions = torch.nonzero(labels[0] != -100, as_tuple=False).squeeze(-1)
    if not target_positions.numel() or int(target_positions.min()) <= 0:
        raise RuntimeError("invalid target positions")
    predictor_positions = target_positions - 1
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        logits_to_keep=predictor_positions,
        use_cache=False,
    )
    loss = functional.cross_entropy(
        output.logits[0].float(), labels[0, target_positions], reduction="mean"
    )
    return loss, int(target_positions.numel())


def tensor_example(tokenizer, header: str, turns: Sequence[Turn], max_length: int):
    import torch

    _messages, input_ids, labels, leakage = render_target_only(
        tokenizer, header, turns, len(turns) - 1, max_length
    )
    if leakage:
        # Conservative string hits remain visible in the report; they are not
        # privileged fields injected by this pipeline.
        pass
    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    mask = torch.ones_like(ids)
    targets = torch.tensor([labels], dtype=torch.long, device="cuda")
    return ids, mask, targets, leakage


def gradient_sketch(model, dimension: int, seed: int) -> tuple[np.ndarray, float]:
    """CountSketch the trainable LoRA gradient without storing a dense vector."""
    import torch

    sketch = torch.zeros(dimension, dtype=torch.float32, device="cuda")
    total_squared = torch.zeros((), dtype=torch.float64, device="cuda")
    global_offset = 0
    multiplier = 6_364_136_223_846_793_005
    mask63 = (1 << 63) - 1
    salt = stable_int("countsketch", seed) & mask63
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        if parameter.grad is not None:
            gradient = parameter.grad.detach().reshape(-1).float()
            total_squared += gradient.double().square().sum()
            indices = torch.arange(count, device=gradient.device, dtype=torch.int64) + global_offset
            hashed = torch.bitwise_and(indices * multiplier + salt, mask63)
            buckets = torch.remainder(hashed, dimension)
            signs = torch.where(
                torch.bitwise_and(torch.bitwise_right_shift(hashed, 32), 1) == 0,
                1.0,
                -1.0,
            )
            sketch.scatter_add_(0, buckets, gradient * signs)
            del gradient, indices, hashed, buckets, signs
        global_offset += count
    return sketch.cpu().numpy(), float(total_squared.sqrt().cpu())


def side_gradient(
    *, model, tokenizer, config: Mapping[str, Any], row: Mapping[str, Any], side: str
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    header, turns = load_turns(config, row[f"{side}_relpath"])
    turns = transform_turns(
        turns,
        str(row[f"{side}_transform"]),
        int(config["seed"]),
        f"{row['control']}:{row['cohort_id']}:{side}",
    )
    if not turns:
        raise RuntimeError(f"empty trajectory: {row[f'{side}_trajectory_id']}")
    ids, mask, labels, leakage = tensor_example(
        tokenizer, header, turns, int(config["gradient"]["max_length"])
    )
    model.zero_grad(set_to_none=True)
    loss, target_tokens = target_only_loss(model, ids, mask, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite per-example loss")
    loss.backward()
    sketch, dense_norm = gradient_sketch(
        model,
        int(config["gradient"]["sketch_dimension"]),
        int(config["gradient"]["sketch_seed"]),
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "target_tokens": target_tokens,
        "dense_lora_gradient_l2": dense_norm,
        "leakage_hits": leakage,
    }
    del ids, mask, labels, loss
    torch.cuda.empty_cache()
    return sketch, metrics


def load_model(config: Mapping[str, Any], adapter_dir: Path):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = resolve_project_path(config["paths"]["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=config["model"]["revision"],
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if (adapter_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
        warmed = True
    else:
        lora = config["model"]["lora"]
        model = get_peft_model(base, LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            task_type="CAUSAL_LM",
            bias="none",
        ))
        warmed = False
    return tokenizer, model, warmed


def warm_scoring_adapter(model, tokenizer, config, real_rows, adapter_dir: Path) -> None:
    import bitsandbytes as bnb
    import torch

    steps = int(config["gradient"]["warmup_steps"])
    optimizer = bnb.optim.PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["gradient"]["warmup_learning_rate"]),
    )
    model.train()
    log: list[dict[str, Any]] = []
    for step in range(steps):
        row = real_rows[step % len(real_rows)]
        header, turns = load_turns(config, row["success_relpath"])
        ids, mask, labels, _leakage = tensor_example(
            tokenizer, header, turns, int(config["gradient"]["max_length"])
        )
        optimizer.zero_grad(set_to_none=True)
        loss, tokens = target_only_loss(model, ids, mask, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite warmup loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        log.append({"step": step + 1, "loss": float(loss.detach().cpu()), "target_tokens": tokens})
        del ids, mask, labels, loss
        torch.cuda.empty_cache()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    atomic_json(adapter_dir.parent / "warmup_report.json", {
        "status": "COMPLETE",
        "steps": steps,
        "learning_rate": config["gradient"]["warmup_learning_rate"],
        "log": log,
    })


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def paired_bootstrap(main: Mapping[str, float], control: Mapping[str, float], seed: int) -> dict[str, Any]:
    keys = sorted(set(main) & set(control))
    differences = np.asarray([main[key] - control[key] for key in keys], dtype=np.float64)
    if not len(differences):
        return {"matched_cohorts": 0, "mean_difference": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(10_000, len(differences)), replace=True).mean(axis=1)
    return {
        "matched_cohorts": len(keys),
        "mean_difference": float(differences.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def aggregate(config, rows, embeddings, metrics, output_dir: Path) -> None:
    folds = int(config["gradient"]["crossfit_folds"])
    seed = int(config["seed"])
    normalized = []
    for embedding in embeddings:
        norm = float(np.linalg.norm(embedding))
        normalized.append(embedding / max(norm, 1e-12))
    normalized_array = np.asarray(normalized, dtype=np.float32)
    row_folds = np.asarray([
        component_fold(row["family"], row["prompt"], folds, seed) for row in rows
    ])
    real_mask = np.asarray([row["control"] == "real" for row in rows])
    prototypes: dict[int, np.ndarray] = {}
    for fold in range(folds):
        eligible = real_mask & (row_folds != fold)
        prototype = normalized_array[eligible].mean(axis=0)
        prototypes[fold] = prototype / max(float(np.linalg.norm(prototype)), 1e-12)

    score_rows: list[dict[str, Any]] = []
    by_control: dict[str, dict[str, float]] = defaultdict(dict)
    for row, embedding, unit, fold, metric in zip(rows, embeddings, normalized_array, row_folds, metrics):
        score = float(np.dot(unit, prototypes[int(fold)]))
        record = {
            **row,
            "crossfit_fold": int(fold),
            "contrast_alignment": score,
            "gradient_contrast_l2_sketch": float(np.linalg.norm(embedding)),
            **metric,
        }
        score_rows.append(record)
        by_control[row["control"]][row["cohort_id"]] = score
    write_jsonl(output_dir / "scores.jsonl", score_rows)

    control_means = {
        name: float(np.mean(list(values.values()))) for name, values in by_control.items() if values
    }
    comparisons = {
        name: paired_bootstrap(by_control["real"], values, seed + stable_int(name) % 1_000_000)
        for name, values in by_control.items()
        if name != "real"
    }
    strongest = max(
        (name for name in control_means if name != "real"),
        key=lambda name: control_means[name],
    )
    strongest_comparison = comparisons[strongest]
    lower = strongest_comparison["ci95"][0]
    report = {
        "status": "COMPLETE",
        "model": config["model"]["hf_id"],
        "revision": config["model"]["revision"],
        "rows": len(rows),
        "sketch_dimension": config["gradient"]["sketch_dimension"],
        "crossfit_folds": folds,
        "mean_alignment_by_control": control_means,
        "paired_real_minus_control": comparisons,
        "strongest_control": strongest,
        "method_gate": "PASS" if lower is not None and lower > 0 else "FAIL",
        "gate_rule": "paired real-minus-strongest-control bootstrap CI95 lower bound > 0",
        "training_and_benchmark_results": None,
    }
    atomic_json(output_dir / "gradient_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    set_seed(int(config["seed"]))
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-8B production scoring requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    run_dir = resolve_project_path(config["paths"]["run_dir"])
    manifest_path = run_dir / "manifests" / "gradient_pairs.jsonl"
    allowed_controls = set(config["gradient"]["controls"])
    contract = {
        "manifest_sha256": sha256_file(manifest_path),
        "model_revision": config["model"]["revision"],
        "lora": config["model"]["lora"],
        "gradient": config["gradient"],
        "seed": config["seed"],
    }
    contract_path = run_dir / "gradients" / "gradient_contract.json"
    if contract_path.is_file() and read_json(contract_path) != contract:
        raise RuntimeError("gradient config/manifest changed; refusing to reuse stale row cache")
    atomic_json(contract_path, contract)
    rows = [row for row in jsonl_rows(manifest_path) if row["control"] in allowed_controls]
    real_rows = [row for row in rows if row["control"] == "real"]
    if not real_rows:
        raise RuntimeError("manifest has no real COGA rows")

    output_dir = run_dir / "gradients"
    adapter_dir = output_dir / "scoring_adapter"
    tokenizer, model, warmed = load_model(config, adapter_dir)
    if not warmed:
        warm_scoring_adapter(model, tokenizer, config, real_rows, adapter_dir)
    model.eval()

    embeddings: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        embedding_path = output_dir / "cache" / f"{ordinal:05d}.npy"
        metric_path = output_dir / "cache" / f"{ordinal:05d}.json"
        if embedding_path.is_file() and metric_path.is_file():
            embeddings.append(np.load(embedding_path).astype(np.float32))
            metrics.append(read_json(metric_path))
            continue
        success, success_metrics = side_gradient(
            model=model, tokenizer=tokenizer, config=config, row=row, side="success"
        )
        failure, failure_metrics = side_gradient(
            model=model, tokenizer=tokenizer, config=config, row=row, side="failure"
        )
        contrast = success - failure
        success_failure_cosine = float(np.dot(success, failure) / max(
            float(np.linalg.norm(success) * np.linalg.norm(failure)), 1e-12
        ))
        metric = {
            "success_loss": success_metrics["loss"],
            "failure_loss": failure_metrics["loss"],
            "success_target_tokens": success_metrics["target_tokens"],
            "failure_target_tokens": failure_metrics["target_tokens"],
            "success_dense_lora_gradient_l2": success_metrics["dense_lora_gradient_l2"],
            "failure_dense_lora_gradient_l2": failure_metrics["dense_lora_gradient_l2"],
            "success_failure_gradient_cosine": success_failure_cosine,
            "success_leakage_hits": success_metrics["leakage_hits"],
            "failure_leakage_hits": failure_metrics["leakage_hits"],
        }
        stored_contrast = contrast.astype(np.float16)
        atomic_npy(embedding_path, stored_contrast)
        atomic_json(metric_path, metric)
        embeddings.append(stored_contrast.astype(np.float32))
        metrics.append(metric)
        print({"row": ordinal + 1, "total": len(rows), "control": row["control"]}, flush=True)

    aggregate(config, rows, embeddings, metrics, output_dir)
    final_report = read_json(output_dir / "gradient_report.json")
    final_report["elapsed_seconds"] = time.monotonic() - started
    final_report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_report["cached_or_scored_rows"] = len(rows)
    atomic_json(output_dir / "gradient_report.json", final_report)


if __name__ == "__main__":
    main()

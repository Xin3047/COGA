#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import bitsandbytes as bnb
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from coga.common import atomic_json, read_json, resolve_project_path, set_seed, sha256_file
from coga.scripts_loss import target_only_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Qwen3-8B QLoRA comparison arm on a 24GB 4090.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("rejection_success", "coga_selected"), required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if config.get("execution_scope") != "FULL_PRODUCTION_ONLY":
        raise RuntimeError("training requires execution_scope=FULL_PRODUCTION_ONLY")
    set_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-8B QLoRA requires CUDA")

    run_dir = resolve_project_path(config["paths"]["run_dir"])
    dataset_path = run_dir / "datasets" / f"{args.arm}.parquet"
    dataset_report = read_json(run_dir / "datasets" / f"{args.arm}.report.json")
    if sha256_file(dataset_path) != dataset_report["dataset_sha256"]:
        raise RuntimeError("dataset hash differs from materialization report")
    table = pq.read_table(dataset_path, columns=["input_ids", "attention_mask", "labels"])
    loss_counts = [
        sum(value != -100 for value in table["labels"][index].as_py())
        for index in range(table.num_rows)
    ]
    if sum(loss_counts) != int(config["selection"]["loss_token_budget"]):
        raise RuntimeError("loss-token budget changed before training")

    model_path = resolve_project_path(config["paths"]["model"])
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
    lora = config["model"]["lora"]
    model = get_peft_model(base, LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
        bias="none",
    ))
    model.train()

    training = config["training"]
    if int(training["epochs"]) != 1:
        raise RuntimeError("the frozen full-training contract requires exactly one epoch")
    if int(training["batch_size"]) != 1 or int(training["gradient_accumulation_steps"]) != 1:
        raise RuntimeError("the frozen full-training contract requires batch size and accumulation of one")
    if training["packing"] or not training["target_only_loss"]:
        raise RuntimeError("the frozen full-training contract requires no packing and target-only loss")
    optimizer = bnb.optim.PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
    )
    warmup = math.ceil(table.num_rows * float(training["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=table.num_rows,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    order = torch.randperm(table.num_rows, generator=generator).tolist()

    output_dir = run_dir / "training" / args.arm
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty training directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    started = time.monotonic()
    consumed = 0
    token_weighted_loss_sum = 0.0
    last_loss = None
    torch.cuda.reset_peak_memory_stats()
    for step_index, ordinal in enumerate(order):
        step = step_index + 1
        ids = torch.tensor([table["input_ids"][ordinal].as_py()], dtype=torch.long, device="cuda")
        mask = torch.tensor([table["attention_mask"][ordinal].as_py()], dtype=torch.long, device="cuda")
        labels = torch.tensor([table["labels"][ordinal].as_py()], dtype=torch.long, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss, target_count = target_only_loss(model, ids, mask, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        consumed += target_count
        last_loss = float(loss.detach().cpu())
        token_weighted_loss_sum += last_loss * target_count
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "step": step,
                "row_ordinal": ordinal,
                "loss": last_loss,
                "loss_tokens": target_count,
                "cumulative_loss_tokens": consumed,
                "learning_rate": scheduler.get_last_lr()[0],
                "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            }) + "\n")
        if step % int(training["save_steps"]) == 0:
            model.save_pretrained(output_dir / f"checkpoint-{step}", safe_serialization=True)
        del ids, mask, labels, loss
        torch.cuda.empty_cache()
        if step <= 10 or step % 50 == 0:
            print({"arm": args.arm, "step": step, "total": table.num_rows, "loss": last_loss}, flush=True)

    if consumed != int(config["selection"]["loss_token_budget"]):
        raise RuntimeError("training did not consume the frozen loss-token budget")
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_files = {
        path.relative_to(adapter_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    }
    atomic_json(output_dir / "completion_report.json", {
        "status": "COMPLETE",
        "arm": args.arm,
        "model": config["model"]["hf_id"],
        "revision": config["model"]["revision"],
        "steps": table.num_rows,
        "epochs": 1,
        "loss_tokens": consumed,
        "last_loss": last_loss,
        "target_token_weighted_loss": token_weighted_loss_sum / consumed,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "adapter": str(adapter_dir),
        "adapter_files": adapter_files,
        "terminal_bench_result": None,
        "bfcl_result": None,
    })


if __name__ == "__main__":
    main()

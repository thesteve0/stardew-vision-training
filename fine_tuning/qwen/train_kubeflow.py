#!/usr/bin/env python3
"""KubeFlow PyTorchJob wrapper for Qwen2.5-VL LoRA fine-tuning.

Wraps the SFTTrainer pipeline for distributed training via torchrun.
Reuses data preparation and config loading from train.py.

Usage:
    # Local single-GPU (no torchrun needed)
    python fine_tuning/qwen/train_kubeflow.py \
        --config fine_tuning/qwen/lora_config_kubeflow_local.yaml --dry-run

    # Local multi-GPU via torchrun
    torchrun --nproc_per_node=2 -m fine_tuning.qwen.train_kubeflow \
        --config fine_tuning/qwen/lora_config_kubeflow_local.yaml

    # On cluster: PyTorchJob sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
    # and launches torchrun automatically via deploy/pytorchjob.yaml
"""

import argparse
import logging
import os
import tempfile
import time

import torch

from fine_tuning.qwen.train import (
    EVAL_IMAGE_SIZE,
    load_config,
    load_images_transform,
    prepare_split_jsonl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_func(config: dict):
    """Training function — runs inside each torchrun process."""
    import mlflow
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    if torch.version.hip is not None:
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    yaml_config = load_config(config["config_path"])
    dry_run = config.get("dry_run", False)

    config_dir = os.path.dirname(os.path.abspath(config["config_path"]))
    project_root = os.environ.get("TRAINING_DATA_ROOT") or os.path.dirname(os.path.dirname(config_dir))
    os.chdir(project_root)

    def _resolve(path: str) -> str:
        if os.path.isabs(path) or path.startswith(("s3://", "http://", "https://")):
            return path
        return os.path.join(project_root, path)

    output_dir = config.get("output_dir_override") or _resolve(yaml_config["training"]["output_dir"])

    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_mlflow = yaml_config["training"].get("report_to") == "mlflow"
    if world_rank == 0 and use_mlflow:
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if not os.environ.get("MLFLOW_TRACKING_TOKEN") and os.path.exists(token_path):
            with open(token_path) as f:
                os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

        os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")

        tracking_uri = _resolve(yaml_config["mlflow"]["tracking_uri"])
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

        workspace = yaml_config["mlflow"].get("workspace")
        if workspace:
            mlflow.set_workspace(workspace)

        mlflow.set_experiment(yaml_config["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=yaml_config["mlflow"]["run_name"])
        mlflow.log_params({
            "model_name": yaml_config["model_name"],
            "lora_r": yaml_config["lora"]["r"],
            "lora_alpha": yaml_config["lora"]["lora_alpha"],
            "learning_rate": yaml_config["training"]["learning_rate"],
            "batch_size": yaml_config["training"]["per_device_train_batch_size"],
            "grad_accum": yaml_config["training"]["gradient_accumulation_steps"],
            "dry_run": dry_run,
            "kubeflow_num_workers": world_size,
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Worker %d/%d — device: %s", world_rank, world_size, device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Loading model: %s", yaml_config["model_name"])
    processor = AutoProcessor.from_pretrained(yaml_config["model_name"])
    load_kwargs = {"torch_dtype": getattr(torch, yaml_config["torch_dtype"])}
    if world_size == 1:
        load_kwargs["device_map"] = "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        yaml_config["model_name"], **load_kwargs
    )

    lora_cfg = yaml_config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    tmp_dir = tempfile.mkdtemp(prefix="sft_kf_data_")

    train_source = _resolve(yaml_config["data"]["train_file"])
    eval_source = _resolve(yaml_config["data"]["eval_file"])

    train_jsonl = os.path.join(tmp_dir, "train.jsonl")
    eval_jsonl = os.path.join(tmp_dir, "eval.jsonl")

    prepare_split_jsonl(train_source, train_jsonl)
    prepare_split_jsonl(eval_source, eval_jsonl)

    ds = load_dataset("json", data_files={"train": train_jsonl, "eval": eval_jsonl})
    ds = ds.map(load_images_transform, remove_columns=["image_path"])
    train_dataset = ds["train"]
    eval_dataset = ds["eval"]

    tc = yaml_config["training"]
    training_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": tc["num_train_epochs"],
        "per_device_train_batch_size": tc["per_device_train_batch_size"],
        "per_device_eval_batch_size": tc.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": tc["gradient_accumulation_steps"],
        "learning_rate": tc["learning_rate"],
        "warmup_steps": tc["warmup_steps"],
        "logging_steps": tc["logging_steps"],
        "save_steps": tc["save_steps"],
        "eval_steps": tc["eval_steps"],
        "eval_strategy": tc.get("eval_strategy", "steps"),
        "save_strategy": tc.get("save_strategy", "steps"),
        "save_total_limit": tc["save_total_limit"],
        "load_best_model_at_end": tc["load_best_model_at_end"],
        "metric_for_best_model": tc["metric_for_best_model"],
        "greater_is_better": tc["greater_is_better"],
        "fp16": tc["fp16"],
        "bf16": tc["bf16"],
        "dataloader_num_workers": tc["dataloader_num_workers"],
        "remove_unused_columns": False,
        "gradient_checkpointing": tc["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": tc["optim"],
        "lr_scheduler_type": tc["lr_scheduler_type"],
        "max_grad_norm": tc["max_grad_norm"],
        "report_to": tc["report_to"],
        "seed": yaml_config["seed"],
        "max_length": None,
        "packing": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }

    if dry_run:
        training_kwargs["max_steps"] = 4
        training_kwargs["logging_steps"] = 1
        training_kwargs["save_strategy"] = "no"
        training_kwargs["eval_strategy"] = "no"

    sft_config = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    logger.info("Starting training%s (KubeFlow, %d workers)", " (dry run)" if dry_run else "", world_size)
    t_train_start = time.time()
    trainer.train()
    train_duration = time.time() - t_train_start

    train_state = trainer.state
    steps = train_state.global_step
    epochs_completed = train_state.epoch or 0
    logger.info(
        "Training finished: %d steps, %.1f epochs in %.1f min (%.1f s/step)",
        steps, epochs_completed, train_duration / 60,
        train_duration / max(steps, 1),
    )

    if world_size > 1:
        torch.distributed.barrier()

    if world_rank == 0 and not dry_run:
        logger.info("Saving model to %s", output_dir)
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        metrics = {
            "train_wall_time_min": round(train_duration / 60, 1),
            "seconds_per_step": round(train_duration / max(steps, 1), 1),
            "train_loss": train_state.log_history[-1].get("train_loss", 0),
        }
        if use_mlflow:
            mlflow.log_metrics(metrics)
            mlflow.end_run()

    if world_size > 1:
        torch.distributed.destroy_process_group()

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="KubeFlow PyTorchJob: Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 2 training steps to verify pipeline",
    )
    args = parser.parse_args()

    train_loop_config = {
        "config_path": os.path.abspath(args.config),
        "dry_run": args.dry_run,
        "output_dir_override": os.path.abspath(args.output_dir) if args.output_dir else None,
    }

    train_func(train_loop_config)


if __name__ == "__main__":
    main()

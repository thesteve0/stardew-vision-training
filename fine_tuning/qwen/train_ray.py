#!/usr/bin/env python3
"""Ray Train wrapper for Qwen2.5-VL LoRA fine-tuning.

Wraps the SFTTrainer pipeline in a Ray Train train_func for distributed
training. Reuses data preparation and config loading from train.py.

Usage:
    # Local single-GPU via Ray (dry run)
    python fine_tuning/qwen/train_ray.py --config fine_tuning/qwen/lora_config_tiny.yaml --dry-run

    # Local single-GPU via Ray (full)
    python fine_tuning/qwen/train_ray.py --config fine_tuning/qwen/lora_config_local.yaml

    # Multi-GPU on KubeRay cluster
    python fine_tuning/qwen/train_ray.py \
        --config fine_tuning/qwen/lora_config_ray_cluster.yaml \
        --num-workers 2 \
        --storage-path s3://bucket/ray-results/
"""

import argparse
import logging
import os
import tempfile
import time

import ray
import ray.train
import torch
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import (
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer

from fine_tuning.qwen.train import (
    EVAL_IMAGE_SIZE,
    load_config,
    load_images_transform,
    prepare_split_jsonl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_s3_file(s3_path: str, local_path: str):
    """Download a file from S3 to a local path."""
    import fsspec

    with fsspec.open(s3_path, "rb") as fin, open(local_path, "wb") as fout:
        fout.write(fin.read())


def _make_image_transform(image_base_path: str | None = None):
    """Create an image loading transform, optionally prepending an S3 base path."""
    if not image_base_path:
        return load_images_transform

    from PIL import Image

    def transform(example):
        path = image_base_path.rstrip("/") + "/" + example["image_path"]
        if path.startswith("s3://"):
            import fsspec

            with fsspec.open(path, "rb") as f:
                img = Image.open(f).convert("RGB").resize(EVAL_IMAGE_SIZE)
        else:
            img = Image.open(path).convert("RGB").resize(EVAL_IMAGE_SIZE)
        example["images"] = [img]
        return example

    return transform


def train_func(config: dict):
    """Training function that runs inside each Ray Train worker."""
    import mlflow
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    if torch.version.hip is not None:
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    yaml_config = load_config(config["config_path"])
    dry_run = config.get("dry_run", False)

    # Ray workers may run in a different working directory. The YAML config and
    # JSONL data files contain relative paths (e.g. "datasets/no_tools/images/...").
    # Resolve the project root from the config path and chdir there so all
    # relative paths work — including those inside prepare_split_jsonl
    # (Path.exists check) and load_images_transform (Image.open).
    config_dir = os.path.dirname(os.path.abspath(config["config_path"]))
    project_root = os.environ.get("TRAINING_DATA_ROOT") or os.path.dirname(os.path.dirname(config_dir))
    os.chdir(project_root)

    def _resolve(path: str) -> str:
        if os.path.isabs(path) or path.startswith(("s3://", "http://", "https://")):
            return path
        return os.path.join(project_root, path)

    output_dir = config.get("output_dir_override") or _resolve(yaml_config["training"]["output_dir"])

    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    # MLflow — rank 0 only
    use_mlflow = yaml_config["training"].get("report_to") == "mlflow"
    if world_rank == 0 and use_mlflow:
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if os.path.exists(token_path):
            with open(token_path) as f:
                os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()
        mlflow.set_tracking_uri(_resolve(yaml_config["mlflow"]["tracking_uri"]))
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
            "ray_num_workers": world_size,
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Worker %d/%d — device: %s", world_rank, world_size, device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load model — use device_map="auto" only for single-worker
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

    # Prepare data inside the worker
    tmp_dir = tempfile.mkdtemp(prefix="sft_ray_data_")

    train_source = _resolve(yaml_config["data"]["train_file"])
    eval_source = _resolve(yaml_config["data"]["eval_file"])

    if train_source.startswith("s3://"):
        local_train_src = os.path.join(tmp_dir, "source_train.jsonl")
        local_eval_src = os.path.join(tmp_dir, "source_eval.jsonl")
        _load_s3_file(train_source, local_train_src)
        _load_s3_file(eval_source, local_eval_src)
        train_source = local_train_src
        eval_source = local_eval_src

    train_jsonl = os.path.join(tmp_dir, "train.jsonl")
    eval_jsonl = os.path.join(tmp_dir, "eval.jsonl")

    prepare_split_jsonl(train_source, train_jsonl)
    prepare_split_jsonl(eval_source, eval_jsonl)

    image_base_path = yaml_config.get("data", {}).get("image_base_path")
    image_transform = _make_image_transform(image_base_path)

    ds = load_dataset("json", data_files={"train": train_jsonl, "eval": eval_jsonl})
    ds = ds.map(image_transform, remove_columns=["image_path"])
    train_dataset = ds["train"]
    eval_dataset = ds["eval"]

    # Build SFTConfig
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
        training_kwargs["max_steps"] = 2
        training_kwargs["logging_steps"] = 1
        training_kwargs["save_strategy"] = "no"
        training_kwargs["eval_strategy"] = "no"
        training_kwargs["report_to"] = "none"

    sft_config = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    # Ray Train integration
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    logger.info("Starting training%s (Ray, %d workers)", " (dry run)" if dry_run else "", world_size)
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

    if world_rank == 0 and not dry_run:
        logger.info("Saving model to %s", output_dir)
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        t_eval_start = time.time()
        metrics = trainer.evaluate()
        eval_duration = time.time() - t_eval_start
        logger.info("Evaluation finished in %.1f min", eval_duration / 60)

        metrics["train_wall_time_min"] = round(train_duration / 60, 1)
        metrics["eval_wall_time_min"] = round(eval_duration / 60, 1)
        metrics["total_wall_time_min"] = round((train_duration + eval_duration) / 60, 1)
        metrics["seconds_per_step"] = round(train_duration / max(steps, 1), 1)
        if use_mlflow:
            mlflow.log_metrics(metrics)
            mlflow.end_run()

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Ray Train: Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 2 training steps to verify pipeline",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of Ray Train workers (each gets 1 GPU)",
    )
    parser.add_argument(
        "--storage-path", type=str, default=None,
        help="Shared storage path for multi-node checkpoints (S3 or NFS)",
    )
    args = parser.parse_args()

    ray.init()
    logger.info("Ray initialized: %s", ray.cluster_resources())

    train_loop_config = {
        "config_path": os.path.abspath(args.config),
        "dry_run": args.dry_run,
        "output_dir_override": os.path.abspath(args.output_dir) if args.output_dir else None,
    }

    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
    )

    run_config_kwargs = {
        "name": "stardew-vision-lora-train",
        "checkpoint_config": CheckpointConfig(num_to_keep=3),
    }
    if args.storage_path:
        run_config_kwargs["storage_path"] = args.storage_path

    run_config = RunConfig(**run_config_kwargs)

    ray_trainer = TorchTrainer(
        train_func,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = ray_trainer.fit()
    logger.info("Training result: %s", result)


if __name__ == "__main__":
    main()
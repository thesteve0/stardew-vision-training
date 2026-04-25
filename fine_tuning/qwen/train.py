#!/usr/bin/env python3
"""Qwen2.5-VL LoRA fine-tuning for Stardew Vision Phase 1 (tool selection).

Uses TRL's SFTTrainer with built-in VLM support. Training data is loaded
from JSONL splits produced by data_prep.py and converted to prompt-completion
format with PIL images for the built-in DataCollatorForVisionLanguageModeling.

Usage:
    python fine_tuning/qwen/train.py --config fine_tuning/qwen/lora_config.yaml
    python fine_tuning/qwen/train.py --config fine_tuning/qwen/lora_config_tiny.yaml --dry-run
"""

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

EVAL_IMAGE_SIZE = (1600, 1200)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_split_jsonl(input_path: str, output_path: str) -> int:
    """Transform a data_prep.py JSONL split into prompt-completion format.

    Writes a new JSONL where each record has:
      - prompt: [system_msg, user_msg] with image placeholder
      - completion: [assistant_msg]
      - image_path: path to the image file

    Images are loaded lazily during training via a dataset map transform.
    """
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            raw = json.loads(line)
            msgs = raw["messages"]

            system_msg = msgs[0]
            user_msg = msgs[1]
            assistant_msg = msgs[2]

            image_path = None
            user_content_clean = []
            for part in user_msg["content"]:
                if part.get("type") == "image":
                    image_path = part.get("image", "").removeprefix("file://")
                    user_content_clean.append({"type": "image"})
                else:
                    user_content_clean.append(part)

            if not image_path or not Path(image_path).exists():
                logger.warning("Skipping record with missing image: %s", image_path)
                continue

            system_content = system_msg["content"]
            if isinstance(system_content, str):
                system_content = [{"type": "text", "text": system_content}]
            assistant_content = assistant_msg["content"]
            if isinstance(assistant_content, str):
                assistant_content = [{"type": "text", "text": assistant_content}]
            record = {
                "prompt": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content_clean},
                ],
                "completion": [
                    {"role": "assistant", "content": assistant_content},
                ],
                "image_path": image_path,
            }
            fout.write(json.dumps(record) + "\n")
            count += 1

    logger.info("Prepared %d samples from %s", count, input_path)
    return count


def load_images_transform(example):
    """Map transform: load PIL image from path into the 'images' column."""
    img = Image.open(example["image_path"]).convert("RGB").resize(EVAL_IMAGE_SIZE)
    example["images"] = [img]
    return example


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 2 training steps to verify pipeline"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config["training"]["output_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.log_params({
            "model_name": config["model_name"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["per_device_train_batch_size"],
            "grad_accum": config["training"]["gradient_accumulation_steps"],
            "dry_run": args.dry_run,
        })

        # Load processor and model
        logger.info("Loading model: %s", config["model_name"])
        processor = AutoProcessor.from_pretrained(config["model_name"])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model_name"],
            torch_dtype=getattr(torch, config["torch_dtype"]),
            device_map="auto",
        )

        # LoRA config — passed to SFTTrainer, which applies PEFT internally
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            bias=config["lora"]["bias"],
            task_type=config["lora"]["task_type"],
        )

        # Prepare and load datasets
        tmp_dir = tempfile.mkdtemp(prefix="sft_data_")
        train_jsonl = os.path.join(tmp_dir, "train.jsonl")
        eval_jsonl = os.path.join(tmp_dir, "eval.jsonl")

        prepare_split_jsonl(config["data"]["train_file"], train_jsonl)
        prepare_split_jsonl(config["data"]["eval_file"], eval_jsonl)

        ds = load_dataset("json", data_files={"train": train_jsonl, "eval": eval_jsonl})
        ds = ds.map(load_images_transform, remove_columns=["image_path"])
        train_dataset = ds["train"]
        eval_dataset = ds["eval"]

        # Build SFTConfig
        training_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": config["training"]["num_train_epochs"],
            "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
            "per_device_eval_batch_size": config["training"].get("per_device_eval_batch_size", 1),
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "learning_rate": config["training"]["learning_rate"],
            "warmup_steps": config["training"]["warmup_steps"],
            "logging_steps": config["training"]["logging_steps"],
            "save_steps": config["training"]["save_steps"],
            "eval_steps": config["training"]["eval_steps"],
            "eval_strategy": config["training"].get("eval_strategy", "steps"),
            "save_strategy": config["training"].get("save_strategy", "steps"),
            "save_total_limit": config["training"]["save_total_limit"],
            "load_best_model_at_end": config["training"]["load_best_model_at_end"],
            "metric_for_best_model": config["training"]["metric_for_best_model"],
            "greater_is_better": config["training"]["greater_is_better"],
            "fp16": config["training"]["fp16"],
            "bf16": config["training"]["bf16"],
            "dataloader_num_workers": config["training"]["dataloader_num_workers"],
            "remove_unused_columns": False,
            "gradient_checkpointing": config["training"]["gradient_checkpointing"],
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": config["training"]["optim"],
            "lr_scheduler_type": config["training"]["lr_scheduler_type"],
            "max_grad_norm": config["training"]["max_grad_norm"],
            "report_to": config["training"]["report_to"],
            "seed": config["seed"],
            # VLM-specific
            "max_length": None,
            "packing": False,
            "dataset_kwargs": {"skip_prepare_dataset": True},
        }

        if args.dry_run:
            training_kwargs["max_steps"] = 2
            training_kwargs["logging_steps"] = 1
            training_kwargs["save_strategy"] = "no"
            training_kwargs["eval_strategy"] = "no"
            training_kwargs["report_to"] = "none"

        sft_config = SFTConfig(**training_kwargs)

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            peft_config=lora_config,
        )

        # Train
        logger.info("Starting training%s", " (dry run)" if args.dry_run else "")
        t_train_start = time.time()
        trainer.train()
        train_duration = time.time() - t_train_start

        train_result = trainer.state
        steps = train_result.global_step
        epochs_completed = train_result.epoch or 0
        logger.info(
            "Training finished: %d steps, %.1f epochs in %.1f min (%.1f s/step)",
            steps, epochs_completed, train_duration / 60,
            train_duration / max(steps, 1),
        )

        if not args.dry_run:
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
            mlflow.log_metrics(metrics)

        logger.info("Training complete!")


if __name__ == "__main__":
    main()

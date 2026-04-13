#!/usr/bin/env python3
"""
Qwen2.5-VL LoRA fine-tuning script for Stardew Vision.

Fine-tunes Qwen2.5-VL-7B on multi-turn ChatML conversations to learn:
1. Screen classification → correct tool calling
2. OCR result parsing → natural language narration

Usage:
    python fine_tuning/qwen/train.py --config fine_tuning/qwen/lora_config.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict):
    """
    Load and prepare training dataset from JSONL files.

    Expected format: ChatML conversations with tool calls.
    """
    train_file = config["data"]["train_file"]
    eval_file = config["data"]["eval_file"]

    logger.info(f"Loading training data from {train_file}")
    logger.info(f"Loading eval data from {eval_file}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": eval_file,
        }
    )

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Eval samples: {len(dataset['validation'])}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    output_dir = args.output_dir or config["training"]["output_dir"]

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize MLFlow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        # Log config
        mlflow.log_params(config)

        # Load processor and model
        logger.info(f"Loading model: {config['model_name']}")
        processor = AutoProcessor.from_pretrained(config["model_name"])
        model = AutoModelForVision2Seq.from_pretrained(
            config["model_name"],
            torch_dtype=getattr(torch, config["torch_dtype"]),
            device_map="auto",
        )

        # Configure LoRA
        logger.info("Configuring LoRA")
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            bias=config["lora"]["bias"],
            task_type=config["lora"]["task_type"],
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load dataset
        dataset = prepare_dataset(config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config["training"]["num_train_epochs"],
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            learning_rate=config["training"]["learning_rate"],
            warmup_steps=config["training"]["warmup_steps"],
            logging_steps=config["training"]["logging_steps"],
            save_steps=config["training"]["save_steps"],
            eval_steps=config["training"]["eval_steps"],
            evaluation_strategy=config["training"]["evaluation_strategy"],
            save_strategy=config["training"]["save_strategy"],
            save_total_limit=config["training"]["save_total_limit"],
            load_best_model_at_end=config["training"]["load_best_model_at_end"],
            metric_for_best_model=config["training"]["metric_for_best_model"],
            greater_is_better=config["training"]["greater_is_better"],
            fp16=config["training"]["fp16"],
            bf16=config["training"]["bf16"],
            dataloader_num_workers=config["training"]["dataloader_num_workers"],
            remove_unused_columns=config["training"]["remove_unused_columns"],
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
            optim=config["training"]["optim"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            max_grad_norm=config["training"]["max_grad_norm"],
            report_to=config["training"]["report_to"],
            seed=config["seed"],
        )

        # TODO: Implement custom data collator for ChatML format with images
        # TODO: Add chat template processing

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            # data_collator=data_collator,  # TODO
        )

        # Train
        logger.info("Starting training")
        trainer.train()

        # Save final model
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        # Log final metrics
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

        logger.info("Training complete!")
        logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

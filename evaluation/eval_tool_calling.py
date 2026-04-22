#!/usr/bin/env python3
"""
Evaluate tool calling accuracy for Qwen2.5-VL model.

Metrics:
- Screen classification accuracy (correct tool selected?)
- Per-screen-type confusion matrix

Usage:
    python evaluation/eval_tool_calling.py \
        --model experiments/qwen-tv-fish-v1 \
        --test-set datasets/splits/test.jsonl

For baseline evaluation without a pre-built test JSONL, use run_baseline.py instead:
    python evaluation/run_baseline.py
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.dataset import TestSample
from evaluation.inference import load_model, run_inference
from evaluation.scoring import compute_metrics, format_results_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_set(test_file: str) -> list[TestSample]:
    """Load test set from JSONL file."""
    samples = []
    with open(test_file) as f:
        for line in f:
            record = json.loads(line)
            samples.append(TestSample(
                image_path=record["image_path"],
                screen_type=record["screen_type"],
                expected_tool=record.get("expected_tool"),
            ))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool calling accuracy")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--test-set", type=str, required=True, help="Path to test JSONL")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model}")
    model, processor = load_model(args.model)

    logger.info(f"Loading test set from {args.test_set}")
    samples = load_test_set(args.test_set)
    logger.info(f"Test samples: {len(samples)}")

    logger.info("Running evaluation")
    predictions = []
    for sample in tqdm(samples, desc="Evaluating"):
        prediction = run_inference(model, processor, sample.image_path)
        predictions.append(prediction)

    metrics = compute_metrics(samples, predictions)

    print("\n" + format_results_table(metrics))

    output_file = Path(args.model) / "eval_tool_calling.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

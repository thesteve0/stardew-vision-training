#!/usr/bin/env python3
"""
Evaluate tool calling accuracy for fine-tuned Qwen2.5-VL model.

Metrics:
- Screen classification accuracy (correct tool selected?)
- Per-screen-type confusion matrix

Usage:
    python evaluation/eval_tool_calling.py \
        --model experiments/qwen-tv-fish-v1 \
        --test-set datasets/splits/test.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoModelForVision2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_set(test_file: str) -> list:
    """Load test set from JSONL file."""
    test_samples = []
    with open(test_file) as f:
        for line in f:
            test_samples.append(json.loads(line))
    return test_samples


def evaluate_tool_calling(model, processor, test_samples: list) -> dict:
    """
    Evaluate tool calling accuracy.

    For each test sample:
    1. Pass screenshot to model
    2. Extract predicted tool call
    3. Compare to ground truth tool call
    """
    predictions = []
    ground_truths = []

    for sample in test_samples:
        # TODO: Extract image and expected tool call from sample
        # TODO: Run model inference
        # TODO: Parse predicted tool call
        # TODO: Compare to ground truth

        # Placeholder
        predicted_tool = "crop_tv_dialog"
        expected_tool = sample.get("tool_call", "unknown")

        predictions.append(predicted_tool)
        ground_truths.append(expected_tool)

    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, predictions)

    # Confusion matrix
    cm = confusion_matrix(ground_truths, predictions)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool calling accuracy")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test-set", type=str, required=True, help="Path to test JSONL")
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Load test set
    logger.info(f"Loading test set from {args.test_set}")
    test_samples = load_test_set(args.test_set)
    logger.info(f"Test samples: {len(test_samples)}")

    # Evaluate
    logger.info("Running evaluation")
    results = evaluate_tool_calling(model, processor, test_samples)

    # Log results
    logger.info(f"Tool calling accuracy: {results['accuracy']:.2%}")
    logger.info(f"Confusion matrix:\n{results['confusion_matrix']}")

    # Save results
    output_file = Path(args.model) / "eval_tool_calling.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

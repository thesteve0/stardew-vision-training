#!/usr/bin/env python3
"""
Evaluate narration quality for fine-tuned Qwen2.5-VL model.

Metrics:
- Field extraction F1 score (are OCR fields present in narration?)
- Fluency (perplexity score)

Usage:
    python evaluation/eval_narration.py \
        --model experiments/qwen-tv-fish-v1 \
        --test-set datasets/splits/test.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import torch
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


def calculate_field_extraction_f1(predicted_narration: str, ocr_fields: dict) -> float:
    """
    Calculate F1 score for field extraction.

    Checks if key OCR fields (item name, price, etc.) appear in narration.
    """
    # TODO: Extract field values from ocr_fields
    # TODO: Check if each field appears in predicted_narration
    # TODO: Calculate precision, recall, F1

    # Placeholder
    return 0.9


def calculate_perplexity(model, processor, narration: str) -> float:
    """
    Calculate perplexity of generated narration.

    Lower perplexity = more fluent text.
    """
    # TODO: Tokenize narration
    # TODO: Calculate model perplexity
    # TODO: Return perplexity score

    # Placeholder
    return 42.3


def evaluate_narration(model, processor, test_samples: list) -> dict:
    """
    Evaluate narration quality.

    For each test sample:
    1. Run full agent loop (tool call → OCR → narration)
    2. Extract predicted narration
    3. Calculate field extraction F1
    4. Calculate perplexity
    """
    f1_scores = []
    perplexities = []
    sample_outputs = []

    for sample in test_samples:
        # TODO: Run agent loop to generate narration
        # TODO: Extract predicted narration
        # TODO: Calculate metrics

        # Placeholder
        predicted_narration = "TV weather forecast: Welcome to KOZU 5..."
        ocr_fields = sample.get("ocr_fields", {})

        f1 = calculate_field_extraction_f1(predicted_narration, ocr_fields)
        perplexity = calculate_perplexity(model, processor, predicted_narration)

        f1_scores.append(f1)
        perplexities.append(perplexity)

        sample_outputs.append({
            "image_id": sample.get("image_id"),
            "predicted_narration": predicted_narration,
            "expected_narration": sample.get("narration"),
            "f1": f1,
            "perplexity": perplexity,
        })

    # Calculate averages
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0

    return {
        "avg_f1": avg_f1,
        "avg_perplexity": avg_perplexity,
        "sample_outputs": sample_outputs[:10],  # First 10 for review
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate narration quality")
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
    results = evaluate_narration(model, processor, test_samples)

    # Log results
    logger.info(f"Narration F1: {results['avg_f1']:.2%}")
    logger.info(f"Narration perplexity: {results['avg_perplexity']:.2f}")

    # Save results
    output_file = Path(args.model) / "eval_narration.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

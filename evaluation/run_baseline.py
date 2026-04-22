#!/usr/bin/env python3
"""CLI entry point for baseline tool selection evaluation.

Usage:
    # Baseline (untuned model)
    python evaluation/run_baseline.py

    # Fine-tuned model comparison
    python evaluation/run_baseline.py \
        --model experiments/qwen-tv-fish-v1 \
        --run-name finetuned-v1
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm

from evaluation.dataset import load_test_set
from evaluation.inference import load_model, run_inference
from evaluation.scoring import compute_metrics, format_results_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tool selection accuracy for Qwen2.5-VL"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/eval-baseline-v1",
        help="Directory for result artifacts",
    )
    parser.add_argument(
        "--run-name",
        default="baseline",
        help="MLflow run name (e.g., baseline, finetuned-v1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Path to datasets directory",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger.info("Loading test set")
    samples = load_test_set(args.datasets_dir)
    if not samples:
        logger.error("No test samples found")
        return

    model, processor = load_model(args.model)

    logger.info(f"Running inference on {len(samples)} samples")
    predictions = []
    for sample in tqdm(samples, desc="Evaluating"):
        prediction = run_inference(model, processor, sample.image_path)
        predictions.append(prediction)

    logger.info("Computing metrics")
    metrics = compute_metrics(samples, predictions)

    print("\n" + format_results_table(metrics))

    model_type = "baseline" if "Instruct" in args.model else "finetuned"
    results = {
        "model": args.model,
        "model_type": model_type,
        "seed": args.seed,
        "num_samples": len(samples),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    outputs_file = output_dir / "sample_outputs.jsonl"
    with open(outputs_file, "w") as f:
        for sample, pred in zip(samples, predictions):
            from evaluation.scoring import classify_prediction

            predicted_type = classify_prediction(pred)
            record = {
                "image_path": sample.image_path,
                "screen_type": sample.screen_type,
                "expected_tool": sample.expected_tool,
                "predicted_tool": pred.tool_called,
                "predicted_type": predicted_type,
                "correct": sample.screen_type == predicted_type,
                "raw_output": pred.raw_output[:500],
            }
            if pred.parse_error:
                record["parse_error"] = pred.parse_error
            f.write(json.dumps(record) + "\n")
    logger.info(f"Per-sample outputs saved to {outputs_file}")

    mlflow.set_experiment("qwen-tool-selection-eval")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model_name", args.model)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("dtype", "float16")
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("num_samples", len(samples))

        mlflow.log_metric("overall_accuracy", metrics["overall_accuracy"])
        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        for screen_type, info in metrics["per_class"].items():
            mlflow.log_metric(f"{screen_type}_accuracy", info["accuracy"])
            mlflow.log_metric(f"{screen_type}_f1", info["f1"])

        mlflow.log_artifact(str(results_file))
        mlflow.log_artifact(str(outputs_file))

    logger.info("Results logged to MLflow (experiment: qwen-tool-selection-eval)")


if __name__ == "__main__":
    main()

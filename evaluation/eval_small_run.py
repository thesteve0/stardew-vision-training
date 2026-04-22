#!/usr/bin/env python3
"""Quick smoke test: 2 images per screen type (8 total) with full trace.

Usage:
    python -m evaluation.eval_small_run
"""

import logging
import subprocess
import time

import torch

from evaluation.dataset import load_test_set
from evaluation.inference import load_model, run_inference_traced
from evaluation.scoring import classify_prediction, compute_metrics, format_results_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLES_PER_TYPE = 4
SEPARATOR = "=" * 80


def gpu_stats() -> str:
    """Return GPU utilization and VRAM usage from rocm-smi + torch."""
    parts = []
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showmemuse", "--csv"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines()[1:]:
            fields = line.split(",")
            if len(fields) >= 3:
                parts.append(f"GPU util: {fields[1].strip()}%")
                parts.append(f"VRAM alloc: {fields[2].strip()}%")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated(0) / 1e9
        reserved_gb = torch.cuda.memory_reserved(0) / 1e9
        parts.append(f"torch alloc: {alloc_gb:.2f} GB")
        parts.append(f"torch reserved: {reserved_gb:.2f} GB")

    return " | ".join(parts) if parts else "no GPU stats available"


def main():
    samples = load_test_set()

    by_type: dict[str, list] = {}
    for s in samples:
        by_type.setdefault(s.screen_type, []).append(s)

    subset = []
    for screen_type, group in sorted(by_type.items()):
        subset.extend(group[:SAMPLES_PER_TYPE])

    logger.info(f"Smoke test: {len(subset)} samples ({SAMPLES_PER_TYPE} per type)")

    t0 = time.perf_counter()
    model, processor = load_model()
    load_time = time.perf_counter() - t0
    print(f"\nModel loaded in {load_time:.1f}s")
    print(f"  {gpu_stats()}")

    predictions = []
    sample_times = []

    for i, sample in enumerate(subset, 1):
        print(f"\n{SEPARATOR}")
        print(f"SAMPLE {i}/{len(subset)}: {sample.image_path}")
        print(f"  Expected screen_type: {sample.screen_type}")
        print(f"  Expected tool:        {sample.expected_tool or 'none (plain text)'}")
        print(SEPARATOR)

        t_start = time.perf_counter()
        prediction, prompt_text = run_inference_traced(
            model, processor, sample.image_path
        )
        t_elapsed = time.perf_counter() - t_start
        sample_times.append(t_elapsed)
        predictions.append(prediction)

        print(f"\n--- PROMPT (text portion) ---\n{prompt_text}")
        print(f"\n--- RAW MODEL OUTPUT ---\n{prediction.raw_output}")
        if prediction.parse_error:
            print(f"\n--- PARSE ERROR ---\n{prediction.parse_error}")
        print(f"\n--- PARSED RESULT ---")
        print(f"  tool_called:    {prediction.tool_called or 'none'}")
        print(f"  tool_arguments: {prediction.tool_arguments}")

        predicted_type = classify_prediction(prediction)
        status = "OK" if sample.screen_type == predicted_type else "MISS"
        print(f"  verdict:        [{status}]")
        print(f"  inference time: {t_elapsed:.1f}s")
        print(f"  {gpu_stats()}")

    metrics = compute_metrics(subset, predictions)
    total_time = time.perf_counter() - t0

    print(f"\n{SEPARATOR}")
    print(format_results_table(metrics))
    print(f"\n--- TIMING ---")
    print(f"  Model load:       {load_time:.1f}s")
    print(f"  Per-sample avg:   {sum(sample_times) / len(sample_times):.1f}s")
    print(f"  Per-sample range: {min(sample_times):.1f}s - {max(sample_times):.1f}s")
    print(f"  Total inference:  {sum(sample_times):.1f}s")
    print(f"  Total wall clock: {total_time:.1f}s")
    print(f"  {gpu_stats()}")


if __name__ == "__main__":
    main()

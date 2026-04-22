#!/usr/bin/env python3
"""Metric computation for tool selection evaluation."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from evaluation.dataset import SCREEN_TYPE_TO_TOOL, TestSample
from evaluation.prompt import EXTRACTION_TOOLS, NO_TOOL_RESPONSE
from evaluation.tool_parser import ParsedPrediction

TOOL_TO_SCREEN_TYPE = {v: k for k, v in SCREEN_TYPE_TO_TOOL.items() if v}


def classify_prediction(prediction: ParsedPrediction) -> str:
    """Map a parsed model output to a predicted screen type.

    Rules:
    - Extraction tool called → corresponding screen type
    - No tool call + response contains the no-tool phrase → no_tools
    - Anything else → "unknown" (always scored as wrong)
    """
    tool = prediction.tool_called

    if tool in EXTRACTION_TOOLS:
        return TOOL_TO_SCREEN_TYPE[tool]

    if tool is None and NO_TOOL_RESPONSE.lower() in prediction.raw_output.lower():
        return "no_tools"

    return "unknown"


def compute_metrics(
    samples: list[TestSample],
    predictions: list[ParsedPrediction],
) -> dict:
    """Compute all tool selection metrics.

    Returns a dict with overall accuracy, macro F1, per-class metrics,
    and confusion matrix.
    """
    y_true = [s.screen_type for s in samples]
    y_pred = [classify_prediction(p) for p in predictions]

    labels = ["tv_dialog", "caught_fish", "pierre_shop", "no_tools"]

    overall_accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels + ["unknown"])
    cm_labels = labels + ["unknown"]

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    per_class = {}
    for screen_type in labels:
        class_indices = [i for i, s in enumerate(samples) if s.screen_type == screen_type]
        class_correct = sum(1 for i in class_indices if y_true[i] == y_pred[i])
        class_total = len(class_indices)

        per_class[screen_type] = {
            "count": class_total,
            "correct": class_correct,
            "accuracy": class_correct / class_total if class_total > 0 else 0.0,
            "precision": report.get(screen_type, {}).get("precision", 0.0),
            "recall": report.get(screen_type, {}).get("recall", 0.0),
            "f1": report.get(screen_type, {}).get("f1-score", 0.0),
        }

    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": cm_labels,
    }


def format_results_table(metrics: dict) -> str:
    """Format metrics as a human-readable console table."""
    lines = []
    lines.append(f"{'Screen Type':<16} {'Count':>5}  {'Correct':>7}  {'Accuracy':>8}")
    lines.append("─" * 42)

    total_count = 0
    total_correct = 0
    for screen_type in ["tv_dialog", "caught_fish", "pierre_shop", "no_tools"]:
        info = metrics["per_class"][screen_type]
        lines.append(
            f"{screen_type:<16} {info['count']:>5}  "
            f"{info['correct']:>7}  {info['accuracy']:>7.1%}"
        )
        total_count += info["count"]
        total_correct += info["correct"]

    lines.append("─" * 42)
    lines.append(
        f"{'Overall':<16} {total_count:>5}  "
        f"{total_correct:>7}  {metrics['overall_accuracy']:>7.1%}"
    )
    lines.append(f"Macro F1: {metrics['macro_f1']:.1%}")

    return "\n".join(lines)

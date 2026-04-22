# Baseline Evaluation Plan: Qwen2.5-VL Tool Selection Accuracy

## Context

Synthetic training data generation is complete (692 conversations). Before fine-tuning, we need
baseline numbers from the untuned Qwen2.5-VL-7B-Instruct model to measure how much benefit
fine-tuning provides. This is **Phase 1 evaluation only**: does the model call the correct
extraction tool given a screenshot? Phase 2 (text correction accuracy) comes later.

The evaluation is a standalone CLI script with MLflow logging. The code is structured as clean
modules so it can optionally be wrapped in an EvalHub BYOF adapter later (~30 lines of glue).

## What We're Measuring

Given a Stardew Valley screenshot + tool definitions + "What's on this screen?", does the model:
- Call `crop_tv_dialog` for TV screens?
- Call `crop_caught_fish_notification` for caught fish screens?
- Call `crop_pierres_detail_panel` for Pierre's shop screens?
- **Explicitly decline** (e.g., call `text_to_speech` saying no tool is available) for unrecognized screens?

Plain text responses or unrelated tool calls do NOT count as correct for no_tools — the model
must actively indicate it has no tool for the screen.

## Test Set

All 277 real annotated screenshots (gold standard from actual gameplay):

| Screen Type | Count | Expected Tool | Source |
|---|---|---|---|
| tv_dialog | 45 | `crop_tv_dialog` | `datasets/tv_dialog/annotations.jsonl` |
| caught_fish | 33 | `crop_caught_fish_notification` | `datasets/caught_fish/annotations.jsonl` |
| pierre_shop | 26 | `crop_pierres_detail_panel` | `datasets/pierre_shop/annotations.jsonl` |
| no_tools | 173 | None (must explicitly decline) | `datasets/no_tools/annotations.jsonl` |

Synthetic data is for training only — never used in evaluation.

## Metrics

- **Overall tool selection accuracy** (277 samples)
- **Per-screen-type accuracy** (sliced: tv_dialog, caught_fish, pierre_shop, no_tools)
- **Confusion matrix** (4x4: which tools get confused for which)
- **Per-class precision/recall/F1** (especially important given no_tools is 62% of the data)
- **Macro-averaged F1** (prevents no_tools from inflating overall accuracy)

## Tool Definitions (verified against production app)

Matches between training repo and production app (`thesteve0/stardew-vision`):
- `crop_pierres_detail_panel` — confirmed in production (implemented)
- `crop_tv_dialog` — confirmed in production ADR-009 (Phase 2)
- `crop_caught_fish_notification` — training repo only (not yet in production roadmap, but we have data, so we evaluate it)
- `text_to_speech` — removed from production tool loop (ADR-011), but still used in training data as the "decline" mechanism

## Implementation

### New files to create

```
evaluation/
  __init__.py              # package marker
  prompt.py                # tool definitions + system prompt + message builder
  tool_parser.py           # parse Qwen output → which tool was called
  dataset.py               # load all 4 annotations.jsonl into unified format
  inference.py             # load Qwen, run single-image inference
  scoring.py               # accuracy, confusion matrix, per-class metrics
  run_baseline.py          # CLI entry point (argparse + MLflow logging)
```

### Files to modify

- `pyproject.toml` — add `scikit-learn` dependency (used by existing `eval_tool_calling.py` already, but not in deps)
- `evaluation/eval_tool_calling.py` — refactor to import from new modules instead of placeholder logic
- `evaluation/eval_narration.py` — leave as-is for now (Phase 2)

### Existing files to reuse

- `fine_tuning/qwen/lora_config.yaml` — canonical tool definitions and model config
- `synthetic_data/blocks/build_chatml.py` lines 20-23 — `_TOOL_NAMES` mapping (reference only; evaluation defines its own copy)

### Module details

**`evaluation/dataset.py`**
- Read `datasets/{screen_type}/annotations.jsonl` for all 4 screen types
- Build unified list of test samples: `{image_path, screen_type, expected_tool}`
- Screen type → tool mapping:
  - `tv_dialog` → `crop_tv_dialog`
  - `caught_fish` → `crop_caught_fish_notification`
  - `pierre_shop` → `crop_pierres_detail_panel`
  - `no_tools` → `None`
- Validate that all referenced image files exist on disk
- Return count summary for logging

**`evaluation/prompt.py`**
- Define `TOOL_DEFINITIONS` list in OpenAI function-calling format (4 tools: 3 extraction + text_to_speech)
- System prompt instructs the model to examine the screenshot and call the appropriate extraction tool, or call text_to_speech to indicate no tool is available
- `build_messages(image_path)` → returns the messages list for `processor.apply_chat_template()`
- Tool parameter schemas match production: `image_b64: str` for extraction tools, `text: str` for text_to_speech

**`evaluation/tool_parser.py`**
- Parse Qwen2.5-VL output text to extract tool call information
- Handle Qwen's native tool-calling format (Hermes-style `<tool_call>` tags)
- Return a dataclass: `ParsedPrediction(tool_called: str | None, raw_output: str, parse_error: str | None)`
- Handle edge cases: no tool call in output, malformed JSON, multiple tool calls (take first), unrecognized tool names
- For no_tools scoring: check if model called `text_to_speech` with a decline message (explicit decline = correct)

**`evaluation/inference.py`**
- `load_model(model_path: str)` → returns `(model, processor)`
  - Uses `Qwen2_5_VLForConditionalGeneration` (not generic `AutoModelForVision2Seq`)
  - `torch_dtype=torch.float16` (ROCm constraint)
  - `device_map="auto"`
- `run_inference(model, processor, image_path: str, tool_definitions: list) -> str`
  - Build messages via `prompt.build_messages()`
  - Apply chat template with tools
  - Use `qwen_vl_utils.process_vision_info()` for image processing
  - Generate with `do_sample=False` (greedy, deterministic)
  - `max_new_tokens=512`
  - Decode with `skip_special_tokens=False` (parser needs delimiters)
  - Return raw output string

**`evaluation/scoring.py`**
- `classify_prediction(parsed: ParsedPrediction) -> str` — maps model output to predicted screen type
  - Extraction tool called → corresponding screen type
  - `text_to_speech` called with decline-like text → `no_tools`
  - No tool called / unrecognized → `unknown` (always wrong)
- `compute_metrics(samples, predictions) -> dict` — computes all metrics
  - Uses `sklearn.metrics.accuracy_score`, `confusion_matrix`, `classification_report`
  - Returns: overall accuracy, per-class accuracy, confusion matrix, macro F1, per-class precision/recall/F1
- All metrics returned as a plain dict (easy to log to MLflow, easy to wrap in EvalHub later)

**`evaluation/run_baseline.py`** (CLI entry point)
- argparse with:
  - `--model` (default: `Qwen/Qwen2.5-VL-7B-Instruct`)
  - `--output-dir` (default: `experiments/eval-baseline-v1`)
  - `--run-name` (default: `baseline`)
  - `--seed` (default: 42)
- Flow:
  1. Load test set → log sample counts
  2. Load model
  3. Run inference on each sample (tqdm progress bar)
  4. Parse each output
  5. Compute metrics
  6. Print summary table to console
  7. Save detailed results to `{output-dir}/results.json` and `{output-dir}/sample_outputs.jsonl`
  8. Log to MLflow (experiment: `qwen-tool-selection-eval`)
- Designed for reuse: same script, `--model experiments/qwen-tv-fish-v1 --run-name finetuned-v1` after fine-tuning

### Output format

**Console output:**
```
Screen Type      Count  Correct  Accuracy
──────────────────────────────────────────
tv_dialog           45       ??     ??.?%
caught_fish         33       ??     ??.?%
pierre_shop         26       ??     ??.?%
no_tools           173       ??     ??.?%
──────────────────────────────────────────
Overall            277       ??     ??.?%
Macro F1: ??.?%
```

**`results.json`:**
```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "model_type": "baseline",
  "num_samples": 277,
  "overall_accuracy": 0.xx,
  "macro_f1": 0.xx,
  "per_class": { "tv_dialog": {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}, ... },
  "confusion_matrix": [[...], ...],
  "confusion_labels": ["tv_dialog", "caught_fish", "pierre_shop", "no_tools"]
}
```

**`sample_outputs.jsonl`** (one line per sample):
```json
{"image_path": "...", "screen_type": "tv_dialog", "expected_tool": "crop_tv_dialog", "predicted_tool": "...", "correct": true, "raw_output": "..."}
```

**MLflow:**
- Experiment: `qwen-tool-selection-eval`
- Params: model_name, model_type (baseline/finetuned), dtype, seed, num_samples
- Metrics: overall_accuracy, macro_f1, tv_dialog_accuracy, caught_fish_accuracy, pierre_shop_accuracy, no_tools_accuracy
- Artifacts: results.json, sample_outputs.jsonl

## Dependencies to add

```bash
uv add scikit-learn
```

No other new dependencies needed — `transformers`, `qwen-vl-utils`, `torch`, `mlflow`, `tqdm`, `pillow` are already in pyproject.toml.

The `eval-hub-sdk` is NOT added now. If we wrap with EvalHub later, that's a separate `uv add` at that time.

## Verification

1. Run `python evaluation/run_baseline.py` — should complete without errors on all 277 samples
2. Check `experiments/eval-baseline-v1/results.json` exists with all metrics
3. Check MLflow UI shows the run under `qwen-tool-selection-eval` experiment
4. Verify per-class counts match expected (45 + 33 + 26 + 173 = 277)
5. Run again with same seed — results should be identical (deterministic)

## What we expect from baseline

The base Qwen2.5-VL-7B-Instruct has never seen these tool definitions or Stardew Valley screenshots. We expect:
- Low overall accuracy (model may not understand the domain-specific tools)
- Possibly reasonable no_tools accuracy (if the model tends to not call tools it doesn't know)
- Near-zero accuracy on specific screen types (it has no training signal for crop_tv_dialog etc.)
- These numbers become the baseline that fine-tuning must beat

## Future: EvalHub wrapping (not part of this plan)

When ready, the EvalHub adapter would:
1. `uv add eval-hub-sdk` (from GitHub or PyPI)
2. Create `evaluation/adapter.py` — ~30 lines subclassing `FrameworkAdapter`, calling the same `load_model()`, `run_inference()`, `compute_metrics()` functions
3. Create `evaluation/meta/job.json` and `evaluation/meta/collection.yaml` with thresholds
4. The Collection would define per-screen-type benchmarks with pass thresholds based on the baseline numbers we measure now

# Next Steps — Fresh Training Baseline

Previous results cleared. Code now aligned between `train.py` and `train_ray.py` (both use `gradient_checkpointing_kwargs: {"use_reentrant": False}`).

## What Changed

- Added `gradient_checkpointing_kwargs: {"use_reentrant": False}` to `train.py` (was only in `train_ray.py`)
- Changed `lora_config_cluster.yaml` to 3 epochs (was 2), bumped run name to v3
- Deleted all previous experiment results, eval runs, and model checkpoints

## Runs to Do

1. **Local standalone** (`train.py` + `lora_config.yaml`) — establishes new baseline with corrected code
2. **Local Ray** (`train_ray.py` + `lora_config.yaml`) — confirms Ray wrapper doesn't change results
3. **Cluster Ray** (`train_ray.py` + `lora_config_cluster.yaml`) — compare distributed to local

## Key Question

The cluster config has a different effective batch size (16 vs 8 local). If cluster results are worse, reduce `gradient_accumulation_steps` from 2 to 1 in `lora_config_cluster.yaml` to match the local effective batch size of 8.

## Key Files

| File | What it does |
|------|-------------|
| `fine_tuning/qwen/train.py` | Standalone SFTTrainer |
| `fine_tuning/qwen/train_ray.py` | Ray Train wrapper |
| `fine_tuning/qwen/lora_config.yaml` | Local config (FP16, batch 2, grad_accum 4) |
| `fine_tuning/qwen/lora_config_cluster.yaml` | Cluster config (BF16, batch 4, grad_accum 2, 2 workers) |
| `deploy/rayjob.yaml` | KubeRay RayJob manifest |
| `evaluation/run_baseline.py` | Eval script |
| `docs/comparison-small-training-runs.md` | Results comparison (empty — to be filled) |

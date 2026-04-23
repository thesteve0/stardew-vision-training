# Plan: Migrate Training to Ray Train (Local + KubeRay)

## Context

Training currently runs via a standalone SFTTrainer script (`fine_tuning/qwen/train.py`) on a single AMD ROCm GPU. The next step is to wrap this in Ray Train so it can run locally first (validating the Ray integration) and then be submitted to a KubeRay cluster on OpenShift AI with 2 NVIDIA L40S GPU workers. The existing `train.py` must remain untouched for standalone use.

**Key hardware difference**: Local = AMD ROCm FP16 only. Cluster = NVIDIA L40S with BF16 support. The YAML config already controls dtype, so no code branching is needed.

---

## Files to Create

| File | Purpose |
|------|---------|
| `fine_tuning/__init__.py` | Empty. Enable package imports (missing today; `evaluation/__init__.py` exists as precedent) |
| `fine_tuning/qwen/__init__.py` | Empty. Enable `from fine_tuning.qwen.train import ...` |
| `fine_tuning/qwen/train_ray.py` | Ray Train wrapper (~200 lines) |
| `fine_tuning/qwen/lora_config_ray.yaml` | Local Ray config (FP16, local paths, same as lora_config.yaml but with ray-compatible output_dir notes) |
| `fine_tuning/qwen/lora_config_cluster.yaml` | Cluster config (BF16, S3 data paths, cluster MLflow URI) |
| `deploy/rayjob.yaml` | KubeRay RayJob manifest for OpenShift AI |

**No changes** to: `train.py`, `lora_config.yaml`, `lora_config_tiny.yaml`, `data_prep.py`

---

## Step 1: Create `__init__.py` files

Create empty `fine_tuning/__init__.py` and `fine_tuning/qwen/__init__.py` so `train_ray.py` can import shared functions from `train.py`:
- `load_config` (line 37)
- `prepare_split_jsonl` (line 42)
- `load_images_transform` (line 92)
- `EVAL_IMAGE_SIZE` (line 31)

---

## Step 2: Create `fine_tuning/qwen/train_ray.py`

### Structure

```
train_ray.py
├── train_func(config: dict)     # Runs inside each Ray worker
│   ├── Guard ROCm env var (torch.version.hip check)
│   ├── Load YAML config from config["config_path"]
│   ├── MLflow setup (rank 0 only)
│   ├── Load processor + model (NO device_map="auto" for multi-worker)
│   ├── Configure LoRA
│   ├── Prepare data (reuse prepare_split_jsonl + load_images_transform)
│   ├── Build SFTConfig + SFTTrainer
│   ├── Add RayTrainReportCallback
│   ├── prepare_trainer(trainer)
│   ├── trainer.train()
│   └── Post-training save + eval (rank 0, non-dry-run only)
│
├── main()                        # Runs on the driver
│   ├── Parse CLI: --config, --dry-run, --output-dir, --num-workers, --storage-path
│   ├── ray.init()
│   ├── Build train_loop_config (small serializable dict)
│   ├── ScalingConfig(num_workers=N, use_gpu=True)
│   ├── RunConfig(storage_path=..., checkpoint_config=...)
│   ├── TorchTrainer(train_func, ...)
│   └── trainer.fit()
```

### Key design decisions

1. **`device_map="auto"` handling**: Use `device_map="auto"` when `world_size == 1` (local single-GPU), omit it for multi-worker (DDP handles placement)

2. **Data loading inside `train_func`**: Per Ray docs, all data/model loading must happen inside the training function to avoid serialization issues. The YAML config path (a string) is passed via `train_loop_config`.

3. **MLflow**: Only rank 0 manages the `mlflow.start_run()` wrapper. HF Trainer's built-in `report_to: "mlflow"` already logs only from the main process.

4. **ROCm env var**: Guarded with `if torch.version.hip is not None` so it doesn't run on NVIDIA cluster workers.

5. **S3 data support**: When data paths in the YAML start with `s3://`, download JSONL files to a temp dir first, then use the existing local-file pipeline. For images, add a `data.image_base_path` config field and create a custom image transform that handles S3 via `fsspec`.

6. **`--dry-run`**: Same overrides as standalone (max_steps=2, no saving, no eval).

### Imports from Ray

```python
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
```

---

## Step 3: Create YAML configs

### `lora_config_ray.yaml` (local Ray testing)
- Copy of `lora_config.yaml` with a comment noting it's for Ray-based training
- `torch_dtype: "float16"`, `fp16: true`, `bf16: false` (ROCm)
- Same local data paths and MLflow URI

### `lora_config_cluster.yaml` (KubeRay on OpenShift AI)
Key differences from local config:
- `torch_dtype: "bfloat16"` (NVIDIA L40S supports BF16)
- `fp16: false`, `bf16: true`
- `per_device_train_batch_size: 4` (48GB VRAM on L40S vs 20GB container limit locally)
- `gradient_accumulation_steps: 2` (effective batch = 4 * 2 workers * 2 = 16)
- `data.train_file: "s3://BUCKET/datasets/splits/train.jsonl"` (placeholder)
- `data.eval_file: "s3://BUCKET/datasets/splits/val.jsonl"`
- `data.image_base_path: "s3://BUCKET"` (prepended to image paths in JSONL)
- `mlflow.tracking_uri: "http://mlflow.apps.CLUSTER_DOMAIN"` (placeholder)

---

## Step 4: Create `deploy/rayjob.yaml`

KubeRay RayJob manifest for OpenShift AI:
- **Head node**: No GPU, 2 CPU, 8Gi RAM (coordination only)
- **Worker group**: 2 replicas, each with 1 `nvidia.com/gpu`, 4 CPU, 32Gi RAM
- **Container image**: `quay.io/opendatahub/odh-pipeline-runtime-pytorch-cuda-py312-ubi9` (placeholder, awaiting confirmation)
- **Entrypoint**: `python fine_tuning/qwen/train_ray.py --config fine_tuning/qwen/lora_config_cluster.yaml --num-workers 2 --storage-path s3://BUCKET/ray-results/`
- **Secrets**: `s3-credentials` and `mlflow-credentials` mounted as env vars
- **GPU tolerations**: `nvidia.com/gpu` NoSchedule

---

## Step 5: Verify

### Local testing sequence
```bash
# 1. Dry run with Ray (2 steps, no saving)
python fine_tuning/qwen/train_ray.py \
    --config fine_tuning/qwen/lora_config_tiny.yaml --dry-run

# 2. Tiny training via Ray (1 epoch, 40 samples)
python fine_tuning/qwen/train_ray.py \
    --config fine_tuning/qwen/lora_config_tiny.yaml

# 3. Confirm standalone train.py still works unchanged
python fine_tuning/qwen/train.py \
    --config fine_tuning/qwen/lora_config_tiny.yaml --dry-run
```

### What success looks like
- Ray dry run completes 2 steps without errors
- Ray tiny training completes 1 epoch, checkpoints appear in experiments/
- MLflow logs show the Ray run with `ray_num_workers` parameter
- Standalone `train.py` produces identical behavior to before

---

## Dependencies

- `ray[default,train]>=2.40.0` — already in pyproject.toml
- `fsspec` and `boto3` — needed for S3 data access on cluster; add via `uv add` when implementing cluster support
- No new dependencies needed for local Ray testing

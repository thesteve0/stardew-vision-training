# Plan: KubeRay Distributed Training on OpenShift AI with EFS Storage

## Context

Previous attempt used S3 (NooBaa) for everything — training data, code delivery, checkpoints, MLflow artifacts — causing cascading SSL/cert issues with NooBaa's OpenShift service-serving CAs. Training itself worked (69/98 steps, loss 0.06→0.028, 98.9% token accuracy), but the infrastructure approach was wrong. All code was reverted to commit c72b453.

**Goal**: Get 2-epoch distributed training running on 2 L40S GPUs with live MLflow metrics, proper checkpointing, and final model export to HF Hub + S3. **Minimize code changes** — the training code that works locally should work on the cluster with correct infrastructure setup.

**Key insight**: Mount datasets at `/app/datasets/` via EFS PVC so relative paths in config and JSONL files resolve identically to local development. Training code needs only 2 bug fixes.

---

## Architecture

```
                 ┌─────────────────────────┐
                 │   MLflow (RHOAI)        │
                 │   https://mlflow.       │
                 │   redhat-ods-apps.svc   │
                 │   :8443                 │
                 └──────────▲──────────────┘
                            │ metrics (rank 0 only)
         ┌──────────────────┼──────────────────┐
         │                  │                   │
    ┌────┴────┐       ┌─────┴─────┐      ┌─────┴─────┐
    │Ray Head │       │Ray Worker │      │Ray Worker │
    │(no GPU) │       │(L40S #1)  │      │(L40S #2)  │
    └────┬────┘       └─────┬─────┘      └─────┬─────┘
         │                  │                   │
    ┌────┴──────────────────┴───────────────────┴────┐
    │           AWS EFS (RWX PVCs)                   │
    │  /data/datasets/...  (training data, RO)       │
    │  /checkpoints/...    (Ray checkpoints, RW)     │
    └────────────────────────────────────────────────┘
```

**Data flow**:
1. Training data → EFS PVC mounted at `/data` (read-only)
2. Training code → Baked into container image (BuildConfig already exists)
3. Ray checkpoints → EFS PVC mounted at `/checkpoints` (read-write, shared across nodes)
4. MLflow metrics → Internal HTTPS service (already configured in `mlflow-credentials` secret)
5. Final model → Saved to `/checkpoints/model-output/`, then uploaded to HF Hub + S3 bucket for vLLM/KServe (`storageUri: s3://...`)

---

## Step 0: Install AWS EFS CSI Driver + Create EFS Filesystem

**Operator**: `aws-efs-csi-driver-operator` (Red Hat supported, available in OperatorHub)

1. Install operator via OperatorHub or CLI
2. Get cluster VPC ID and subnet IDs using AWS CLI (credentials from `kube-system/aws-creds`)
   - Region: `us-east-2`
   - Infra name: `stardew-vision-bls5m`
   - Subnets: `stardew-vision-bls5m-subnet-private-us-east-2{a,b,c}`
   - Security groups: `stardew-vision-bls5m-node`
3. Create security group allowing NFS (port 2049) inbound from worker node security group
4. Create EFS filesystem in VPC
5. Create mount targets in each private subnet
6. Create `StorageClass` referencing EFS filesystem ID
7. Verify with a test PVC

---

## Step 1: Create EFS-backed PVCs

Two PVCs in `stardew-vision-training` namespace:

| PVC | Size | Access | Mount Path | Purpose |
|-----|------|--------|------------|---------|
| `training-data` | 5Gi | RWX (mounted RO) | `/data` | Datasets, JSONL splits, images |
| `training-checkpoints` | 20Gi | RWX | `/checkpoints` | Ray checkpoint sync + model output |

---

## Step 2: Populate Training Data PVC

Run a one-shot pod that mounts the `training-data` PVC and copies data into it using `oc rsync` or `oc cp`:

```
/data/
└── datasets/
    ├── splits/
    │   ├── train.jsonl
    │   └── val.jsonl
    ├── caught_fish/images/...
    ├── no_tools/images/...
    ├── pierre_shop/images/...
    ├── tv_dialog/images/...
    └── synthetic/
        ├── caught_fish/images/...
        ├── pierre_shop/images/...
        └── tv_dialog/images/...
```

**Why this layout**: The JSONL files contain relative image paths like `datasets/caught_fish/images/IMG_0016.PNG`. The training code does `os.chdir(project_root)` then opens these paths. With the PVC mounted at `/data` and `project_root` set to `/data`, all paths resolve correctly — **zero code changes to data loading**.

---

## Step 3: Fix 2 Bugs in `train_ray.py` (minimal, targeted)

**File**: `fine_tuning/qwen/train_ray.py`

### Bug 1: `_resolve()` mangles HTTPS URLs (line 101-104)

The `_resolve()` function doesn't recognize `http://` or `https://` URLs, treating them as relative paths and prepending `project_root`. This broke MLflow tracking — metrics went to a local directory instead of the server.

**Fix**: Add URL scheme check:
```python
def _resolve(path: str) -> str:
    if os.path.isabs(path) or path.startswith(("s3://", "http://", "https://")):
        return path
    return os.path.join(project_root, path)
```

### Bug 2: DDP + LoRA + gradient_checkpointing needs `use_reentrant=False`

Without this, DDP raises `RuntimeError: Expected to mark a variable ready only once` when combining LoRA with gradient checkpointing.

**Fix**: Add to `training_kwargs` dict (around line 195):
```python
"gradient_checkpointing_kwargs": {"use_reentrant": False},
```

### Additional change: Override project_root for PVC mount

The current code derives `project_root` from `config_path` (2 dirs up). On the cluster, the config is baked into the image at `/app/fine_tuning/qwen/lora_config_cluster.yaml`, so `project_root` = `/app/`. But the training data is on the PVC at `/data/`.

**Fix**: Allow overriding via environment variable or config:
```python
project_root = os.environ.get("TRAINING_DATA_ROOT", project_root)
os.chdir(project_root)
```

This way, the rayjob.yaml sets `TRAINING_DATA_ROOT=/data` and all relative paths resolve from the PVC mount.

**Total code changes**: 3 small edits in `train_ray.py`. No changes to `train.py` or data loading logic.

---

## Step 4: Update `lora_config_cluster.yaml`

**File**: `fine_tuning/qwen/lora_config_cluster.yaml`

Changes from current state:
- `num_train_epochs`: 3 → **2** (user preference, validated by val loss curve)
- `data.train_file`: `datasets/splits/train.jsonl` (relative, same as local config — resolves from TRAINING_DATA_ROOT)
- `data.eval_file`: `datasets/splits/val.jsonl` (same)
- Remove `data.image_base_path` (not needed — default `load_images_transform` works with relative paths)
- `mlflow.tracking_uri`: `https://mlflow.redhat-ods-applications.svc:8443`
- `training.output_dir`: `/checkpoints/model-output`

---

## Step 5: Build Container Image

**File**: Create `deploy/Dockerfile` (new file)

```dockerfile
FROM registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

COPY fine_tuning/ fine_tuning/
```

The base image already has transformers, peft, trl, datasets, accelerate, deepspeed, mlflow. We only need to add: `ray[default,train]`, `qwen-vl-utils`, and a few small packages.

**Build**: Use existing BuildConfig `stardew-training`:
```bash
oc start-build stardew-training --from-dir=. -n stardew-vision-training
```

**Image**: `image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest`

---

## Step 6: Update `rayjob.yaml`

**File**: `deploy/rayjob.yaml`

Key changes:
- **namespace**: `stardew-vision-training`
- **entrypoint**: Remove S3 storage-path, use PVC path: `--storage-path /checkpoints/ray-results/`
- **Remove** `runtimeEnvYAML` working_dir (code is in the image)
- **Keep** `runtimeEnvYAML` env_vars for `NCCL_DEBUG` and add `TRAINING_DATA_ROOT: "/data"`
- **image**: `image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest`
- **Volume mounts** on head + all workers:
  - `training-data` PVC → `/data` (readOnly: true)
  - `training-checkpoints` PVC → `/checkpoints`
- **env** on head + workers: `MLFLOW_TRACKING_INSECURE_TLS: "true"` (already in secret, but explicit doesn't hurt)
- **envFrom**: Keep `s3-credentials` (for final model export) and `mlflow-credentials`

---

## Step 7: Model Export for vLLM Serving

After training completes, the LoRA adapter at `/checkpoints/model-output/` needs to be accessible by vLLM/KServe in the `stardew-vision` namespace.

**Approach**: Upload to S3 bucket, serve via KServe `storageUri: s3://...`

1. From the rank-0 worker (post-training), upload model to S3 using boto3 (credentials already in pod from `s3-credentials` secret)
2. Copy `s3-credentials` secret to `stardew-vision` namespace (if not already there)
3. KServe InferenceService references the model via `storageUri: s3://stardew-training-bucket/models/lora-v2/`
4. Also upload to HF Hub for public access

**Why S3 here and not for training data**: Writing a small model artifact (~100MB LoRA adapter) to S3 once after training is trivial. The SSL issues from last session were caused by pyarrow continuously reading large datasets from S3 during training — completely different workload pattern. For a single boto3 upload, `AWS_ENDPOINT_URL` + the NooBaa endpoint works fine (boto3 respects `REQUESTS_CA_BUNDLE` or we use the external S3 endpoint with trusted certs).

---

## Step 8: Submit and Verify

1. `oc apply -f deploy/rayjob.yaml`
2. Watch pod startup: `oc get pods -n stardew-vision-training -w`
3. Follow driver logs: `oc logs -f job/stardew-lora-train -n stardew-vision-training`
4. Check MLflow UI for live metrics at `https://data-science-gateway.apps.stardew-vision.sandbox5291.opentlc.com/mlflow`
5. After training: verify model saved at `/checkpoints/model-output/`
6. Verify model uploaded to S3 bucket
7. Test vLLM/KServe can load the model from S3

---

## Verification Checklist

- [ ] EFS PVC accessible from multiple pods across AZs
- [ ] Training data visible at `/data/datasets/` inside pods
- [ ] Both L40S GPU workers join the Ray cluster
- [ ] MLflow receives training metrics (loss, accuracy, timing)
- [ ] Checkpoints written to `/checkpoints/ray-results/`
- [ ] Final model saved to `/checkpoints/model-output/`
- [ ] Model uploaded to S3 bucket, accessible via `storageUri: s3://...`
- [ ] s3-credentials secret available in `stardew-vision` namespace for vLLM
- [ ] Training completes 2 epochs with loss decreasing

---

## Files Modified

| File | Change | Scope |
|------|--------|-------|
| `fine_tuning/qwen/train_ray.py` | Fix `_resolve()`, add gradient_checkpointing_kwargs, add TRAINING_DATA_ROOT env var | 3 small edits |
| `fine_tuning/qwen/lora_config_cluster.yaml` | Set local paths, 2 epochs, MLflow URI | Config only |
| `deploy/rayjob.yaml` | PVC mounts, correct image, remove S3 working_dir | Full rewrite |
| `deploy/Dockerfile` | New file — training image definition | New |

**Not modified**: `train.py`, `data_prep.py`, data loading logic, MLflow logging logic.

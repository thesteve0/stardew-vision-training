# Continuing KubeFlow PyTorchJob Work

This document picks up from work done on the laptop. All code has been written but not yet tested — the laptop can't run Qwen training. The desktop (AMD ROCm) is needed for dry-run verification.

## What Was Created

We added a KubeFlow Training Operator (PyTorchJob) path alongside the existing Ray Train setup. Same training parameters, same data pipeline, same metrics — only the orchestration layer changed.

**New files (5):**

| File | Purpose |
|------|---------|
| `fine_tuning/qwen/train_kubeflow.py` | Training driver — mirrors `train_ray.py` with torchrun/DDP instead of Ray |
| `fine_tuning/qwen/lora_config_kubeflow_cluster.yaml` | Cluster config for L40S GPUs (BF16, batch 4) |
| `fine_tuning/qwen/lora_config_kubeflow_local.yaml` | Local config for ROCm (FP16, batch 2) |
| `deploy/pytorchjob.yaml` | PyTorchJob manifest (Master + 1 Worker = 2 GPUs) |
| `deploy/Dockerfile.kubeflow` | Container image without Ray (~500MB smaller) |

## What Needs to Happen Next

### Step 1: Local dry run (single GPU, no torchrun)

Verifies the script runs end-to-end, imports work, and the training pipeline is functional.

```bash
python fine_tuning/qwen/train_kubeflow.py \
    --config fine_tuning/qwen/lora_config_kubeflow_local.yaml --dry-run
```

**What to look for:**
- Should load Qwen2.5-VL-7B model successfully
- Should prepare train/eval datasets from `datasets/splits/`
- Should run 2 training steps and exit cleanly
- Log line: `Training finished: 2 steps, ...`
- Log line: `Training complete!`
- No Ray imports or errors about missing Ray

### Step 2: Local torchrun dry run (single GPU via torchrun)

Verifies the torchrun launch mechanism works (sets RANK, WORLD_SIZE, etc.).

```bash
torchrun --nproc_per_node=1 -m fine_tuning.qwen.train_kubeflow \
    --config fine_tuning/qwen/lora_config_kubeflow_local.yaml --dry-run
```

**What to look for:**
- Same output as Step 1 but launched via torchrun
- Worker logs should show `Worker 0/1 — device: cuda`
- Clean exit without distributed errors

### Step 3: Fix any issues found in Steps 1-2

If something breaks, the changes should be minimal — the training flow is identical to `train_ray.py` and `train.py`. Common things to check:
- Import paths (especially when run as `-m fine_tuning.qwen.train_kubeflow`)
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` env var guard (ROCm only)
- Path resolution via `TRAINING_DATA_ROOT` or config path fallback

### Step 4: Build and push the KubeFlow container image (when ready for cluster)

```bash
# On the host (not devcontainer)
podman build -t stardew-training-kf:latest -f deploy/Dockerfile.kubeflow .

# Port-forward, login, tag, push (same flow as Ray image)
oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false
podman tag stardew-training-kf:latest \
    localhost:5000/stardew-vision-training/stardew-training-kf:latest
podman push localhost:5000/stardew-vision-training/stardew-training-kf:latest \
    --tls-verify=false
kill %1
```

### Step 5: Submit PyTorchJob to cluster

```bash
oc apply -f deploy/pytorchjob.yaml
oc get pytorchjob stardew-lora-train-kf -n stardew-vision-training -w
oc logs -f -l training.kubeflow.org/job-name=stardew-lora-train-kf --prefix
```

## Key Architectural Differences from Ray

- **No Ray**: No `ray.init()`, no `TorchTrainer`, no `RayTrainReportCallback`. The HuggingFace SFTTrainer handles DDP automatically when launched via `torchrun`.
- **Master trains**: In Ray, the head node has no GPU. In PyTorchJob, the Master is rank 0 and participates in training with its own GPU.
- **`/dev/shm`**: The PyTorchJob manifest mounts an 8Gi shared memory volume. NCCL needs this for inter-process communication (Docker default is 64MB, too small).
- **Istio sidecar disabled**: `sidecar.istio.io/inject: "false"` on pod templates prevents Istio from intercepting NCCL TCP connections.
- **Rank discovery**: `os.environ["RANK"]` and `os.environ["WORLD_SIZE"]` instead of `ray.train.get_context()`.

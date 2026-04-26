# Plan: Add KubeFlow PyTorchJob Distributed Training

## Context

The project currently uses Ray Train (`train_ray.py` + `rayjob.yaml`) for distributed fine-tuning on OpenShift AI. We're adding a parallel KubeFlow Training Operator path using PyTorchJob. The goal is identical training behavior — same parameters, same data pipeline, same metrics — with only the orchestration layer changing from Ray to KubeFlow/torchrun.

## Files to Create (6 total)

### 1. `fine_tuning/qwen/train_kubeflow.py` — KubeFlow driver script

Mirrors `train_ray.py` structure with Ray-specific code replaced by native torchrun/DDP integration.

**Reused from `train.py`** (same imports as `train_ray.py`):
- `load_config`, `prepare_split_jsonl`, `load_images_transform`, `EVAL_IMAGE_SIZE`

**Removed** (all Ray-specific):
- `ray`, `ray.train`, `TorchTrainer`, `ScalingConfig`, `RunConfig`, `CheckpointConfig`
- `RayTrainReportCallback`, `prepare_trainer`
- `_load_s3_file` (data comes from PVCs, not S3)
- `--num-workers` and `--storage-path` CLI args (PyTorchJob manifest controls workers)

**Changed**:
- Rank/world_size from env vars: `int(os.environ.get("RANK", 0))` instead of `ray.train.get_context()`
- No trainer wrapping — SFTTrainer detects torchrun automatically via Accelerate
- `torch.distributed.barrier()` before rank-0 save/eval, `destroy_process_group()` at end
- MLflow param logs `kubeflow_num_workers` instead of `ray_num_workers`
- `main()` is just argparse + call `train_func()` — torchrun handles process spawning externally

**Kept identical**: model loading, LoRA config, data prep pipeline, SFTConfig construction, training loop, timing metrics collection, rank-0 model save + eval + MLflow logging, dry-run overrides, `device_map="auto"` only when `world_size == 1`.

### 2. `fine_tuning/qwen/lora_config_kubeflow_cluster.yaml` — Cluster config

Copy of `lora_config_ray_cluster.yaml` with only:
- Updated header comments (references `deploy/pytorchjob.yaml`, KubeFlow instead of KubeRay)
- `mlflow.run_name`: `"tool-select-kubeflow-v1"` (distinct from Ray's `"tool-select-cluster-v3"`)
- All training params identical: BF16, batch 4, grad_accum 2, 3 epochs, same LoRA, same data paths

### 3. `fine_tuning/qwen/lora_config_kubeflow_local.yaml` — Local config

Copy of `lora_config_ray_local.yaml` with:
- Updated header comments (references `train_kubeflow.py`)
- `training.output_dir`: `"experiments/qwen-tool-select-kubeflow-v1"`
- `mlflow.run_name`: `"tool-select-kubeflow-v1"`
- All training params identical: FP16, batch 2, grad_accum 4, 3 epochs, local MLflow

### 4. `deploy/pytorchjob.yaml` — PyTorchJob manifest

Key differences from `rayjob.yaml`:
- **CRD**: `kubeflow.org/v1 PyTorchJob` instead of `ray.io/v1 RayJob`
- **Master gets a GPU**: In Ray, head has no GPU. In PyTorchJob, Master is rank 0 and trains. For 2 GPUs total: Master=1 GPU + Worker replicas=1 with 1 GPU
- **Command**: `torchrun --nnodes=2 --nproc_per_node=1 --master_addr=$(MASTER_ADDR) --master_port=$(MASTER_PORT) --node_rank=$(RANK) -m fine_tuning.qwen.train_kubeflow --config /app/fine_tuning/qwen/lora_config_kubeflow_cluster.yaml`
- **`/dev/shm` volume**: NCCL needs shared memory (default 64MB is too small). Mount emptyDir with `medium: Memory`, 8Gi
- **`sidecar.istio.io/inject: "false"`**: On pod template metadata for both Master and Worker (Istio sidecars break NCCL TCP)
- **`NCCL_SOCKET_IFNAME=eth0`**: Tells NCCL which interface to use on OpenShift pods
- **Same PVC mounts**: `/data` (training-data, read-only) and `/checkpoints` (training-checkpoints)
- **Same secrets**: s3-credentials, mlflow-credentials
- **Same resources**: 4-8 CPU, 32-48Gi memory, 1 GPU per pod
- **`PYTHONPATH=/app`**: Ensures module imports work with torchrun (same pattern as `eval-job.yaml`)

### 5. `deploy/Dockerfile.kubeflow` — Dockerfile without Ray

Same RHOAI base image. Only installs `qwen-vl-utils` (drops `ray[default,train]`). Saves ~500MB image size.

```dockerfile
FROM registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2
WORKDIR /app
RUN pip install --no-cache-dir "qwen-vl-utils>=0.0.8"
COPY fine_tuning/ fine_tuning/
COPY evaluation/ evaluation/
```

### 6. `CONTINUING_KUBEFLOW_WORK.md` — Handoff document

Checked into the repo root. Describes:
- What was created on the laptop and why
- What needs to happen next on the desktop (dry-run testing, full training)
- Exact commands to run for verification
- What to look for in the output to confirm success

## Implementation Order (all on laptop)

1. `deploy/Dockerfile.kubeflow` — no dependencies, simplest
2. `fine_tuning/qwen/lora_config_kubeflow_cluster.yaml` — copy + edit
3. `fine_tuning/qwen/lora_config_kubeflow_local.yaml` — copy + edit
4. `fine_tuning/qwen/train_kubeflow.py` — main code work
5. `deploy/pytorchjob.yaml` — references image and config from steps 1-3
6. `CONTINUING_KUBEFLOW_WORK.md` — handoff document
7. Commit and push to GitHub

## Verification (on desktop)

1. **Local dry run** (no torchrun, single GPU): `python fine_tuning/qwen/train_kubeflow.py --config fine_tuning/qwen/lora_config_kubeflow_local.yaml --dry-run`
2. **Local torchrun test**: `torchrun --nproc_per_node=1 -m fine_tuning.qwen.train_kubeflow --config fine_tuning/qwen/lora_config_kubeflow_local.yaml --dry-run`
3. **Build image**: `podman build -t stardew-training-kf:latest -f deploy/Dockerfile.kubeflow .`
4. **Cluster test**: `oc apply -f deploy/pytorchjob.yaml` then `oc logs -f -l training.kubeflow.org/job-name=stardew-lora-train-kf --prefix`

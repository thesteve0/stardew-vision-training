# Next Steps — Fresh Training Baseline

Previous results cleared. Code now aligned between `train.py` and `train_ray.py` (both use `gradient_checkpointing_kwargs: {"use_reentrant": False}`).

## What Changed

- Added `gradient_checkpointing_kwargs: {"use_reentrant": False}` to `train.py` (was only in `train_ray.py`)
- Changed `lora_config_ray_cluster.yaml` to 3 epochs (was 2), bumped run name to v3
- Deleted all previous experiment results, eval runs, and model checkpoints

## Runs to Do

1. **Local standalone** (`train.py` + `lora_config.yaml`) — establishes new baseline with corrected code
2. **Local Ray** (`train_ray.py` + `lora_config.yaml`) — confirms Ray wrapper doesn't change results
3. **Rebuild and push image** — build locally, port-forward to OpenShift registry, push (see below)
4. **Cluster Ray** (`oc apply -f deploy/rayjob.yaml`) — compare distributed to local

## Building and Pushing the Training Image

Build on the host (not in devcontainer), then push to the OpenShift internal registry via port-forward:

```bash
# Build
podman build -t stardew-training:latest -f deploy/Dockerfile .

# Port-forward to OpenShift registry
oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &

# Auth, tag, push
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false
podman tag stardew-training:latest localhost:5000/stardew-vision-training/stardew-training:latest
podman push localhost:5000/stardew-vision-training/stardew-training:latest --tls-verify=false

# Cleanup
kill %1
```

After push, submit the RayJob:
```bash
oc apply -f deploy/rayjob.yaml
oc logs -f job/stardew-lora-train -n stardew-vision-training
```

## Key Question

The cluster config has a different effective batch size (16 vs 8 local). If cluster results are worse, reduce `gradient_accumulation_steps` from 2 to 1 in `lora_config_ray_cluster.yaml` to match the local effective batch size of 8.

## Key Files

| File | What it does |
|------|-------------|
| `fine_tuning/qwen/train.py` | Standalone SFTTrainer |
| `fine_tuning/qwen/train_ray.py` | Ray Train wrapper |
| `fine_tuning/qwen/lora_config.yaml` | Local config (FP16, batch 2, grad_accum 4) |
| `fine_tuning/qwen/lora_config_ray_cluster.yaml` | Cluster config (BF16, batch 4, grad_accum 2, 2 workers) |
| `deploy/rayjob.yaml` | KubeRay RayJob manifest |
| `evaluation/run_baseline.py` | Eval script |
| `docs/comparison-small-training-runs.md` | Results comparison (empty — to be filled) |

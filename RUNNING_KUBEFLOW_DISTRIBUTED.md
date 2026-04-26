# Running KubeFlow PyTorchJob Training on OpenShift AI

Step-by-step runbook for submitting a KubeFlow PyTorchJob distributed training job, monitoring it, running evaluation, and copying results locally.

KubeFlow PyTorchJob uses native `torchrun` for distributed training — no Ray overhead. The Master pod participates in training with a GPU (unlike Ray where the head node is a coordinator).

---

## 1. Build and Push the Training Image

Run these on the **host** (not in the devcontainer), from the repo root:

```bash
# Build (uses the lighter Dockerfile.kubeflow — no Ray dependency)
podman build -t stardew-training-kf:latest -f deploy/Dockerfile.kubeflow .

# Port-forward to the OpenShift internal registry
oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &

# Auth, tag, push
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false
podman tag stardew-training-kf:latest localhost:5000/stardew-vision-training/stardew-training-kf:latest
podman push localhost:5000/stardew-vision-training/stardew-training-kf:latest --tls-verify=false

# Kill the port-forward
kill %1
```

Only rebuild when `fine_tuning/` or `evaluation/` code changes. Config-only changes can use a ConfigMap instead (see `deploy/pytorchjob-1gpu.yaml` for an example).

---

## 2. Submit the Training Job

### 2-GPU distributed training (Master + Worker)

```bash
oc apply -f deploy/pytorchjob.yaml
```

### 1-GPU single-node baseline

```bash
# Config is embedded as a ConfigMap — no image rebuild needed
oc apply -f deploy/pytorchjob-1gpu.yaml
```

---

## 3. Monitor GPU Node Provisioning

GPU nodes (L40S on g6e.2xlarge) may need to scale up from zero. Track the full provisioning sequence:

```bash
# Watch MachineSet scale-up
oc get machineset -n openshift-machine-api -w

# Watch Machines being created and transitioning to Running
oc get machines -n openshift-machine-api -w

# Watch the node join the cluster and become Ready
oc get nodes -l nvidia.com/gpu.present=true -w

# Verify GPUs are allocatable
oc get nodes -l nvidia.com/gpu.present=true \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```

---

## 4. Monitor NVIDIA Driver Installation

Once the node joins, the GPU operator installs drivers (~5 min):

```bash
# Watch GPU operator pods
oc get pods -n nvidia-gpu-operator -w

# Verify GPU is advertised
oc describe node -l nvidia.com/gpu.present=true | grep -A5 "Allocatable"
```

---

## 5. Monitor Training

### 2-GPU job

```bash
# Watch pod status
oc get pods -l training.kubeflow.org/job-name=stardew-lora-train-kf -w

# Tail master logs (training output appears here)
oc logs -f stardew-lora-train-kf-master-0

# Check worker logs (useful for NCCL connection issues)
oc logs -f stardew-lora-train-kf-worker-0

# Check PyTorchJob status
oc get pytorchjob stardew-lora-train-kf -o jsonpath='{.status.conditions[*].type}'
```

### 1-GPU job

```bash
oc get pods -l training.kubeflow.org/job-name=stardew-lora-train-1gpu -w
oc logs -f stardew-lora-train-1gpu-master-0
```

### Expected timings (775 samples, 3 epochs, effective batch 8)

| Config | Steps | Wall time | s/step |
|--------|-------|-----------|--------|
| 2-GPU (Master + Worker) | 291 | ~42 min | ~8.6 |
| 1-GPU (Master only) | 291 | ~82 min | ~16.9 |

---

## 6. Run Evaluation on the Cluster

After training completes, the model adapter is saved to `/checkpoints/model-output` on the PVC.

```bash
# Update the run name in eval-job.yaml if needed, then:
oc delete job stardew-eval -n stardew-vision-training --ignore-not-found
oc apply -f deploy/eval-job.yaml

# Watch it run
oc logs -f job/stardew-eval
```

Eval takes ~3 min on L40S (125 images). Results are written to `/checkpoints/eval-results/` on the PVC.

---

## 7. Copy Results Locally

Use one-shot busybox pods to cat files from the PVC:

### Copy eval results

```bash
mkdir -p experiments/eval-cluster-kubeflow-v1

# results.json
oc run copy-results --rm -it --restart=Never --image=busybox \
  --overrides='{"spec":{"containers":[{"name":"c","image":"busybox","command":["cat","/checkpoints/eval-results/results.json"],"volumeMounts":[{"name":"ckpt","mountPath":"/checkpoints"}]}],"volumes":[{"name":"ckpt","persistentVolumeClaim":{"claimName":"training-checkpoints"}}]}}' \
  -n stardew-vision-training > experiments/eval-cluster-kubeflow-v1/results.json 2>/dev/null

# sample_outputs.jsonl
oc run copy-outputs --rm -it --restart=Never --image=busybox \
  --overrides='{"spec":{"containers":[{"name":"c","image":"busybox","command":["cat","/checkpoints/eval-results/sample_outputs.jsonl"],"volumeMounts":[{"name":"ckpt","mountPath":"/checkpoints"}]}],"volumes":[{"name":"ckpt","persistentVolumeClaim":{"claimName":"training-checkpoints"}}]}}' \
  -n stardew-vision-training > experiments/eval-cluster-kubeflow-v1/sample_outputs.jsonl 2>/dev/null

# Clean trailing "pod deleted" message if present
sed -i '$ { /^pod /d }' experiments/eval-cluster-kubeflow-v1/sample_outputs.jsonl
```

### Copy adapter config (text files only — safetensors is too large for cat)

```bash
mkdir -p experiments/qwen-tool-select-cluster-kubeflow-v1

oc run copy-config --rm -it --restart=Never \
  --image=image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training-kf:latest \
  --overrides='{"spec":{"containers":[{"name":"c","image":"image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training-kf:latest","command":["cat","/checkpoints/model-output/adapter_config.json"],"volumeMounts":[{"name":"ckpt","mountPath":"/checkpoints"}]}],"volumes":[{"name":"ckpt","persistentVolumeClaim":{"claimName":"training-checkpoints"}}]}}' \
  -n stardew-vision-training > experiments/qwen-tool-select-cluster-kubeflow-v1/adapter_config.json 2>/dev/null
```

---

## 8. Clean Up

```bash
# Delete the PyTorchJob (removes Master + Worker pods)
oc delete pytorchjob stardew-lora-train-kf -n stardew-vision-training

# Or for 1-GPU job
oc delete pytorchjob stardew-lora-train-1gpu -n stardew-vision-training

# Delete the eval job
oc delete job stardew-eval -n stardew-vision-training
```

---

## Known Issues

- **Post-training eval hangs on 2-GPU**: The `trainer.evaluate()` call after saving uses DDP, but the worker has already exited. Fixed in `train_kubeflow.py` by removing the post-training eval (we run eval separately). If you see the master pod stuck after "Saving model", the model is already saved — safe to delete the job.
- **Eval job MLflow**: `MLFLOW_TRACKING_URI` must be set to `file:///tmp/mlruns` (not empty string) and `workingDir` must be `/data` (not `/tmp`) for image paths to resolve.

---

## Quick Reference

| What | Where |
|------|-------|
| Training config (2-GPU) | `fine_tuning/qwen/lora_config_kubeflow_cluster.yaml` |
| Training config (1-GPU) | Embedded as ConfigMap in `deploy/pytorchjob-1gpu.yaml` |
| PyTorchJob manifest (2-GPU) | `deploy/pytorchjob.yaml` |
| PyTorchJob manifest (1-GPU) | `deploy/pytorchjob-1gpu.yaml` |
| Eval job manifest | `deploy/eval-job.yaml` |
| Training script | `fine_tuning/qwen/train_kubeflow.py` |
| Dockerfile | `deploy/Dockerfile.kubeflow` |
| Training data PVC | `training-data` (mounted at `/data`) |
| Checkpoints PVC | `training-checkpoints` (mounted at `/checkpoints`) |
| Model output | `/checkpoints/model-output/` on PVC |
| Eval results | `/checkpoints/eval-results/` on PVC |
| Namespace | `stardew-vision-training` |

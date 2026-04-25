# Research: KubeRay RayJob Training on OpenShift AI with PVC-Based Storage

## Context

Your current `deploy/rayjob.yaml` uses an S3-heavy approach (S3 for training data, code zip, checkpoint storage) which caused SSL/CA issues. You decided to rework with PVCs. This research covers the six topics you asked about, with practical YAML examples and source URLs.

---

## 1. KubeRay RayJob with PVC Volume Mounts

PVC mounts in a RayJob follow standard Kubernetes pod template patterns. You add `volumes` and `volumeMounts` under each pod template spec (head and workers independently).

### Working YAML Pattern

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: stardew-lora-train
  namespace: stardew-vision-training
spec:
  entrypoint: python /app/fine_tuning/qwen/train_ray.py --config /app/fine_tuning/qwen/lora_config_ray_cluster.yaml --num-workers 2 --storage-path /mnt/checkpoints/ray-results/
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 3600

  rayClusterSpec:
    rayVersion: '2.9.0'
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
      template:
        spec:
          containers:
            - name: ray-head
              image: image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest
              volumeMounts:
                - mountPath: /mnt/datasets
                  name: training-data
                  readOnly: true
                - mountPath: /mnt/checkpoints
                  name: checkpoint-storage
                - mountPath: /tmp/ray
                  name: ray-logs
          volumes:
            - name: training-data
              persistentVolumeClaim:
                claimName: training-data-pvc
            - name: checkpoint-storage
              persistentVolumeClaim:
                claimName: checkpoint-pvc
            - name: ray-logs
              emptyDir: {}

    workerGroupSpecs:
      - replicas: 2
        groupName: gpu-workers
        rayStartParams:
          num-gpus: "1"
        template:
          spec:
            containers:
              - name: ray-worker
                image: image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest
                volumeMounts:
                  - mountPath: /mnt/datasets
                    name: training-data
                    readOnly: true
                  - mountPath: /mnt/checkpoints
                    name: checkpoint-storage
            volumes:
              - name: training-data
                persistentVolumeClaim:
                  claimName: training-data-pvc
              - name: checkpoint-storage
                persistentVolumeClaim:
                  claimName: checkpoint-pvc
```

### Key Points

- **volumes/volumeMounts must be defined separately** in both headGroupSpec and each workerGroupSpec -- they are independent pod templates.
- Mount training data as `readOnly: true` on workers to prevent accidental writes.
- Ray logs volume (`/tmp/ray`) is recommended as an `emptyDir` (per Ray docs) so logs don't fill the shared PVC.
- **Known bug**: Do NOT mount PVCs on the submitter pod (the K8s Job that submits the Ray job). Mount them on the head and worker pods only. There was a [reported issue](https://github.com/ray-project/kuberay/issues/3929) where mounting on the submitter caused KubeRay to treat the cluster as single-node.

### Sources
- [RayJob Quickstart (Ray docs)](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html)
- [KubeRay RayJob submitter bug #3929](https://github.com/ray-project/kuberay/issues/3929)
- [NeMo-Run KubeRay PVC example](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)
- [Kueue RayJob integration](https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/)

---

## 2. OpenShift AI Distributed Training with PVCs

Red Hat's recommended pattern for distributed training data storage depends on cluster size and storage backend.

### Storage Backend Recommendations

| Cluster Size | Recommended Backend | Storage Class |
|---|---|---|
| 1 node | LVM Storage Operator | `lvms-vg1` |
| 2 nodes | NFS Operator (unsupported) | `nfs-provisioner` |
| 3+ nodes | OpenShift Data Foundation (ODF) | `ocs-storagecluster-cephfs` |

**Your cluster** (OpenShift sandbox with ODF already deployed) should use **CephFS via ODF** for RWX PVCs.

### PVC Definition for Training Data

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: stardew-vision-training
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: ocs-storagecluster-cephfs
```

### Red Hat's Recommended Data Loading Pattern

Red Hat recommends **separating data preparation from training**:

1. Create a dedicated "data loader" pod that downloads/copies datasets to the PVC
2. Once loading is complete, delete the loader pod
3. Training pods mount the PVC and read data directly

For your case, this would be:
- Create a one-shot pod that copies `datasets/splits/*.jsonl` and `datasets/*/images/` from your laptop (via `oc rsync`) or from S3 into the PVC
- Training pods mount it at `/mnt/datasets` read-only

### Required Access Mode: ReadWriteMany (RWX)

This is **critical** -- without RWX, OpenShift will schedule all pods on the same node (RWO forces co-location). CephFS via ODF supports RWX natively.

### DataScienceCluster Configuration (if not already done)

```yaml
kind: DataScienceCluster
apiVersion: datasciencecluster.opendatahub.io/v1
metadata:
  name: default-dsc
spec:
  components:
    trainingoperator:
      managementState: Managed
    kueue:
      managementState: Managed
    ray:
      managementState: Managed
    codeflare:
      managementState: Managed
```

### Sources
- [Set up storage for distributed AI training on OpenShift (Red Hat Developer)](https://developers.redhat.com/learning/learn:openshift:roce-multi-node-ai-training-red-hat-openshift/resource/resources:set-storage-distributed-ai-training-openshift)
- [Run distributed AI training on OpenShift (Red Hat Developer)](https://developers.redhat.com/learning/learn:openshift:roce-multi-node-ai-training-red-hat-openshift/resource/resources:run-distributed-ai-training-openshift)
- [AI Workloads on OpenShift: Best Practices (Simplyblock)](https://simplyblock.io/blog/ai-workloads-on-openshift-best-practices/)
- [OpenShift AI 3.0 Distributed Training docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html/working_with_distributed_workloads/running-kfto-based-distributed-training-workloads_distributed-workloads)

---

## 3. Ray Train Checkpoint Storage with PVCs

### How It Works

Ray Train requires shared storage for checkpoints in multi-node training. When you call `ray.train.report(..., checkpoint=...)`, every worker needs to write to the same storage location. You have two options: cloud storage (S3/GCS) or a shared filesystem (NFS/CephFS PVC).

For PVC-based storage, you:
1. Create an RWX PVC
2. Mount it at the same path on all head + worker pods
3. Set `RunConfig(storage_path=...)` to that mount path

### PVC for Checkpoints

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: checkpoint-pvc
  namespace: stardew-vision-training
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: ocs-storagecluster-cephfs
```

100Gi is generous -- your LoRA adapters are small (rank=16, ~50MB per checkpoint), but Ray also stores trial metadata and logs here.

### Python Configuration

```python
from ray import train
from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    run_config=train.RunConfig(
        storage_path="/mnt/checkpoints",  # must match volumeMount path
        name="stardew-lora-train",
    ),
    # ...
)
```

All checkpoints, metrics, and artifacts will be written to `/mnt/checkpoints/stardew-lora-train/`.

### Critical Requirement

If you use `ray.train.report(..., checkpoint=...)` on a multi-node cluster **without** shared storage configured, Ray Train will raise an error. This is a hard requirement, not optional.

### Alternative: Separate Training Data PVC vs Checkpoint PVC

You could use a single RWX PVC for both data and checkpoints (mounted at different subdirectories), or two separate PVCs. Two separate PVCs is cleaner because:
- Training data PVC can be mounted `readOnly: true` on workers
- Checkpoint PVC needs read-write on all pods
- Different lifecycle -- data PVC persists across runs, checkpoint PVC can be cleaned between experiments

### Sources
- [Configuring Persistent Storage (Ray docs)](https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html)
- [RunConfig API reference (Ray docs)](https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html)
- [Distributed checkpointing with KubeRay (Ray docs)](https://docs.ray.io/en/latest/cluster/kubernetes/examples/distributed-checkpointing-with-gcsfuse.html)
- [Syncing files deprecation (Ray GitHub issue #37177)](https://github.com/ray-project/ray/issues/37177)

---

## 4. MLflow Tracking from KubeRay Pods

### Your Setup

MLflow is already running in `redhat-ods-applications` namespace with UI at `https://data-science-gateway.apps.stardew-vision.sandbox5291.opentlc.com/mlflow`. You have a `mlflow-credentials` secret in the training namespace.

### Two Approaches to Set the Tracking URI

**Approach A: Kubernetes pod env (recommended)**

Set `MLFLOW_TRACKING_URI` as an environment variable in the pod spec. This makes it available to all processes in the container, including any MLflow calls outside Ray tasks.

```yaml
containers:
  - name: ray-head
    image: ...
    env:
      - name: MLFLOW_TRACKING_URI
        valueFrom:
          secretKeyRef:
            name: mlflow-credentials
            key: MLFLOW_TRACKING_URI
      - name: MLFLOW_TRACKING_TOKEN
        valueFrom:
          secretKeyRef:
            name: mlflow-credentials
            key: MLFLOW_TRACKING_TOKEN
    # ... repeat for worker containers
```

**Approach B: runtimeEnvYAML env_vars**

```yaml
runtimeEnvYAML: |
  env_vars:
    MLFLOW_TRACKING_URI: "http://mlflow-server.redhat-ods-applications.svc:8080"
```

Approach A is preferred because the env var is available at container startup, not just within Ray task context.

### Training Code Integration

Inside your `train_func`, use rank-zero-only logging to avoid duplicated metrics:

```python
import mlflow
import ray.train

def train_func(config):
    is_rank_zero = ray.train.get_context().get_world_rank() == 0

    if is_rank_zero:
        # MLFLOW_TRACKING_URI is already set via pod env
        mlflow.set_experiment("stardew-lora-train")
        mlflow.start_run()
        mlflow.log_params({
            "lora_rank": config["lora_rank"],
            "learning_rate": config["learning_rate"],
            "num_epochs": config["num_epochs"],
        })

    # ... training loop ...

    for epoch in range(config["num_epochs"]):
        # ... training step ...
        if is_rank_zero:
            mlflow.log_metrics({"train_loss": loss, "epoch": epoch})

    if is_rank_zero:
        mlflow.end_run()
```

### Alternative: setup_mlflow Helper

Ray provides `ray.air.integrations.mlflow.setup_mlflow()` which handles rank-zero-only logic automatically:

```python
from ray.air.integrations.mlflow import setup_mlflow

def train_func(config):
    mlflow = setup_mlflow(
        config,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        experiment_name="stardew-lora-train",
        rank_zero_only=True,  # default
    )
    # mlflow.log_metric(...) is a no-op on non-rank-zero workers
```

### Internal vs External Tracking URI

From inside the cluster, use the **internal service URL** (not the external route):
- Internal: `http://mlflow-server.redhat-ods-applications.svc:8080` (or whatever port/service name MLflow uses)
- External: `https://data-science-gateway.apps.stardew-vision.sandbox5291.opentlc.com/mlflow`

Using the internal URL avoids TLS certificate issues and is faster (no ingress hop). Check the actual service name with `oc get svc -n redhat-ods-applications | grep mlflow`.

### Sources
- [Experiment Tracking (Ray docs)](https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html)
- [setup_mlflow API (Ray docs)](https://docs.ray.io/en/latest/tune/api/doc/ray.air.integrations.mlflow.setup_mlflow.html)
- [MLflow + Ray Train + MinIO (MinIO blog)](https://blog.min.io/distributed-training-and-experiment-tracking-with-ray-train-mlflow-and-minio/)
- [KubeRay on OpenShift AI with Kueue (Red Hat Developer)](https://developers.redhat.com/articles/2025/12/03/tame-ray-workloads-openshift-ai-kuberay-and-kueue)
- [MLflow Tracking Operator for OpenShift (GitHub)](https://github.com/AICoE/mlflow-tracking-operator)

---

## 5. S3-Backed PVCs vs Regular RWX PVCs

### Short Answer

**Do NOT use S3-backed PVCs for training data.** Use regular CephFS RWX PVCs instead.

### Why S3-Backed PVCs Are Wrong for This Use Case

S3 ObjectBucketClaims (OBCs) in OpenShift Data Foundation are NOT mountable as PVCs. They provide S3 API credentials (access key, secret key, endpoint) via a ConfigMap and Secret. Your application code must use an S3 client library to read/write data.

There is no transparent "mount S3 as a filesystem" PVC in ODF. The pattern that exists (FUSE-based S3 mounts like s3fs or goofys) is fragile, has poor POSIX compliance, and is not recommended for training workloads.

### What You Already Have

Your cluster already has:
- **OBC**: `stardew-training-bucket` (NooBaa-backed S3) -- keep this for final model export to HF Hub and vLLM access
- **ODF**: Available for CephFS RWX PVCs -- use this for training data and checkpoints

### Recommended Architecture

| Data | Storage | Why |
|---|---|---|
| Training data (images + JSONL) | CephFS RWX PVC | POSIX filesystem, all pods mount it, no code changes |
| Ray checkpoints | CephFS RWX PVC | Ray Train needs shared filesystem path |
| Final LoRA adapter export | S3 (existing OBC) | Upload to S3 for vLLM serving + HF Hub push |
| MLflow artifacts | MLflow server (already deployed) | Metrics via tracking URI, artifacts stored by MLflow |

### PVC Definitions You Need

```yaml
# Training data -- images and split files
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: stardew-vision-training
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: ocs-storagecluster-cephfs
---
# Checkpoints and Ray results
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: checkpoint-pvc
  namespace: stardew-vision-training
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: ocs-storagecluster-cephfs
```

### Verify CephFS Is Available

```bash
oc get storageclass | grep cephfs
# Should show: ocs-storagecluster-cephfs
```

If CephFS is not available but ODF is deployed, you may need to enable the CephFS storage class in the ODF configuration.

### Sources
- [Object storage with OpenShift Container Storage (oddbit.com)](https://blog.oddbit.com/post/2021-02-10-object-storage-with-openshift/)
- [ODF Training guide (Red Hat Storage)](https://red-hat-storage.github.io/ocs-training/training/ocs4/odf.html)
- [S3 Storage with ODF (Neteye blog)](https://www.neteye-blog.com/2025/08/use-s3-storage-with-openshift-data-foundation/)
- [OpenShift Container Storage 4: Ceph Introduction (Red Hat)](https://www.redhat.com/en/blog/openshift-container-storage-4-introduction-to-ceph)

---

## 6. Training Code in Container Image vs runtime_env

### Recommendation: Bake Code into the Container Image

For production KubeRay jobs on OpenShift, **bake your training code into the container image**. This is the standard Kubernetes best practice.

### Why Bake vs runtime_env

| Factor | Custom Image (bake code in) | runtime_env working_dir |
|---|---|---|
| Reproducibility | High -- immutable image tag | Lower -- depends on zip URL availability |
| Startup speed | Fast -- code already on disk | Slow -- downloads + extracts zip per worker |
| Dependency management | All pre-installed | pip install at runtime, fragile |
| OpenShift compatibility | Standard -- works with ImageStreams, BuildConfigs | Non-standard -- needs outbound S3/HTTP access from pods |
| Debugging | `oc debug` with same image | Hard to reproduce runtime_env state |

### Your Current Setup Already Supports This

You already have a BuildConfig `stardew-training` with an ImageStream. The base image is `registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2` which has PyTorch 2.9 and Python 3.12 pre-installed.

### Dockerfile Pattern

```dockerfile
FROM registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2

# Install additional Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

# Copy training code
COPY fine_tuning/ /app/fine_tuning/
COPY evaluation/ /app/evaluation/

WORKDIR /app
```

Note: `pip install uv` is acceptable as a Dockerfile bootstrap per your CLAUDE.md. Then `uv sync` installs from the lockfile.

### RayJob Entrypoint With Baked Code

When code is baked into the image, the entrypoint references the image path directly:

```yaml
spec:
  entrypoint: python /app/fine_tuning/qwen/train_ray.py --config /app/fine_tuning/qwen/lora_config_ray_cluster.yaml --num-workers 2 --storage-path /mnt/checkpoints/ray-results/
  # NO runtimeEnvYAML working_dir needed!
```

### Hybrid Approach (for iteration)

During development, you can combine both: bake stable dependencies into the image but use `runtimeEnvYAML` with `env_vars` for config overrides:

```yaml
runtimeEnvYAML: |
  env_vars:
    NUM_EPOCHS: "2"
    DRY_RUN: "true"
```

This avoids rebuilding the image for config changes while keeping code reproducible.

### Build and Deploy Cycle

```bash
# Rebuild the image (uses existing BuildConfig)
oc start-build stardew-training --from-dir=. --follow

# The ImageStream tag updates automatically
# RayJob spec references the ImageStream tag
```

### Sources
- [Best Practices for Storage and Dependencies (Ray docs)](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/storage.html)
- [Custom Docker Images (Ray docs)](https://docs.ray.io/en/latest/serve/production-guide/docker.html)
- [Environment Dependencies (Ray docs)](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html)
- [Running Ray on K8s: Production Setup (Medium)](https://medium.com/@cenkayyaman1/running-ray-on-kubernetes-a-production-setup-guide-8cce1c6fb225)
- [KubeRay on OpenShift AI with Kueue (Red Hat Developer)](https://developers.redhat.com/articles/2025/12/03/tame-ray-workloads-openshift-ai-kuberay-and-kueue)
- [Using uv for Python in KubeRay (Ray docs)](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/uv.html)

---

## Putting It All Together: Revised Architecture

Based on all research, here is the recommended architecture for your cluster deployment:

```
+----------------------------------+
|  Data Loading (one-shot pod)     |
|  oc rsync datasets/ → PVC       |
+----------------------------------+
              |
              v
+----------------------------------+     +---------------------------+
|  training-data-pvc (CephFS RWX) |     |  checkpoint-pvc (CephFS)  |
|  /mnt/datasets (read-only)      |     |  /mnt/checkpoints (r/w)   |
+----------------------------------+     +---------------------------+
         |          |                          |          |
         v          v                          v          v
+-------------+  +-------------+       +-------------+  +-------------+
| Ray Head    |  | Ray Worker  |       | Ray Head    |  | Ray Worker  |
| (no GPU)    |  | (L40S GPU)  |       |             |  |             |
+-------------+  +-------------+       +-------------+  +-------------+
         |          |
         v          v
+----------------------------------+     +---------------------------+
|  MLflow Server (internal svc)    |     |  S3 OBC (final export)    |
|  http://mlflow.svc:8080         |     |  LoRA adapter → HF Hub    |
+----------------------------------+     +---------------------------+
```

### Data Flow
1. **Before training**: `oc rsync` copies datasets from laptop to training-data-pvc via a loader pod
2. **During training**: Ray workers read data from PVC, write checkpoints to checkpoint PVC, log metrics to MLflow
3. **After training**: Best checkpoint exported to S3 OBC, then uploaded to HuggingFace Hub

### Complete Revised RayJob YAML Skeleton

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: stardew-vision-training
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 50Gi
  storageClassName: ocs-storagecluster-cephfs
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: checkpoint-pvc
  namespace: stardew-vision-training
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 100Gi
  storageClassName: ocs-storagecluster-cephfs
---
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: stardew-lora-train
  namespace: stardew-vision-training
spec:
  entrypoint: >-
    python /app/fine_tuning/qwen/train_ray.py
    --config /app/fine_tuning/qwen/lora_config_cluster.yaml
    --num-workers 2
    --storage-path /mnt/checkpoints/ray-results/

  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 3600

  rayClusterSpec:
    rayVersion: '2.9.0'
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
      template:
        spec:
          containers:
            - name: ray-head
              image: image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest
              resources:
                requests:
                  cpu: "2"
                  memory: "8Gi"
                limits:
                  cpu: "4"
                  memory: "8Gi"
              env:
                - name: MLFLOW_TRACKING_URI
                  valueFrom:
                    secretKeyRef:
                      name: mlflow-credentials
                      key: MLFLOW_TRACKING_URI
                - name: MLFLOW_TRACKING_TOKEN
                  valueFrom:
                    secretKeyRef:
                      name: mlflow-credentials
                      key: MLFLOW_TRACKING_TOKEN
                - name: HF_TOKEN
                  valueFrom:
                    secretKeyRef:
                      name: hf-credentials
                      key: HF_TOKEN
              volumeMounts:
                - mountPath: /mnt/datasets
                  name: training-data
                  readOnly: true
                - mountPath: /mnt/checkpoints
                  name: checkpoint-storage
                - mountPath: /tmp/ray
                  name: ray-logs
          volumes:
            - name: training-data
              persistentVolumeClaim:
                claimName: training-data-pvc
            - name: checkpoint-storage
              persistentVolumeClaim:
                claimName: checkpoint-pvc
            - name: ray-logs
              emptyDir: {}

    workerGroupSpecs:
      - replicas: 2
        minReplicas: 2
        maxReplicas: 2
        groupName: gpu-workers
        rayStartParams:
          num-gpus: "1"
        template:
          spec:
            containers:
              - name: ray-worker
                image: image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest
                resources:
                  requests:
                    cpu: "4"
                    memory: "32Gi"
                    nvidia.com/gpu: 1
                  limits:
                    cpu: "8"
                    memory: "48Gi"
                    nvidia.com/gpu: 1
                env:
                  - name: MLFLOW_TRACKING_URI
                    valueFrom:
                      secretKeyRef:
                        name: mlflow-credentials
                        key: MLFLOW_TRACKING_URI
                  - name: MLFLOW_TRACKING_TOKEN
                    valueFrom:
                      secretKeyRef:
                        name: mlflow-credentials
                        key: MLFLOW_TRACKING_TOKEN
                volumeMounts:
                  - mountPath: /mnt/datasets
                    name: training-data
                    readOnly: true
                  - mountPath: /mnt/checkpoints
                    name: checkpoint-storage
            volumes:
              - name: training-data
                persistentVolumeClaim:
                  claimName: training-data-pvc
              - name: checkpoint-storage
                persistentVolumeClaim:
                  claimName: checkpoint-pvc
            tolerations:
              - key: nvidia.com/gpu
                operator: Exists
                effect: NoSchedule
```

### Known Bugs to Fix in train_ray.py (from cluster_setup_progress.md)

These still need addressing before cluster deployment:
1. `_resolve()` mangles `https://` MLflow tracking URIs
2. Needs `gradient_checkpointing_kwargs={"use_reentrant": False}` for DDP + LoRA
3. `lora_config_cluster.yaml` should be 2 epochs (validated by val loss curve)

### Open Questions

1. **CephFS availability**: Verify `ocs-storagecluster-cephfs` storage class exists on your sandbox cluster: `oc get storageclass`
2. **MLflow internal service name**: Find the actual service name: `oc get svc -n redhat-ods-applications | grep mlflow`
3. **Data loading**: Decide between `oc rsync` to a helper pod vs keeping S3 as the data source for initial PVC population
4. **Image tag strategy**: Use `:latest` for dev or pin to a specific tag for reproducibility

# Plan: Enable MLflow Tracking for Ray Distributed Training on OpenShift AI

## Context

Training runs on the OpenShift AI cluster complete successfully (96-97.6% accuracy) but log nothing to the cluster MLflow server. The root cause: both cluster configs (`lora_config_ray_cluster.yaml` and `lora_config_kubeflow_cluster.yaml`) set `report_to: "none"`, which disables all MLflow code. The MLflow server is deployed in `redhat-ods-applications` via the ODH MLflow CRD. The `mlflow-credentials` secret is already mounted in the training pods, but we don't yet know what env vars it contains or how auth/TLS work.

The existing MLflow code in `train_ray.py` is structurally sound — rank-0-only setup, manual run lifecycle, param and metric logging. It just needs to be enabled and hardened for the cluster endpoint.

## Step 0: Investigate cluster MLflow (oc commands)

Before any code changes, determine auth mechanism and TLS setup by running these on the cluster:

```bash
# What env vars does the secret provide?
oc get secret mlflow-credentials -n stardew-vision-training -o jsonpath='{.data}' | \
  python3 -c "import sys,json,base64; data=json.load(sys.stdin); [print(f'{k} = {base64.b64decode(v).decode()}') for k,v in sorted(data.items())]"

# MLflow service details
oc get svc -n redhat-ods-applications | grep -i mlflow

# OAuth proxy check
oc get pods -n redhat-ods-applications -l app=mlflow -o yaml | grep -A5 "oauth-proxy\|containers:"

# TLS cert info
oc get svc mlflow -n redhat-ods-applications -o yaml | grep "service.beta.openshift.io/serving-cert-secret-name"

# Connectivity test from a training pod
oc run mlflow-test --rm -it --restart=Never \
  --image=image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest \
  -- python3 -c "
import urllib.request, ssl, os
ctx = ssl.create_default_context()
ctx.load_verify_locations('/var/run/secrets/kubernetes.io/serviceaccount/ca.crt')
token = open('/var/run/secrets/kubernetes.io/serviceaccount/token').read().strip()
req = urllib.request.Request('https://mlflow.redhat-ods-applications.svc:8443/api/2.0/mlflow/experiments/search',
    headers={'Authorization': f'Bearer {token}'}, method='POST', data=b'{}')
try:
    resp = urllib.request.urlopen(req, context=ctx)
    print('SUCCESS:', resp.status, resp.read()[:200])
except Exception as e:
    print('FAILED:', e)
"
```

Results determine whether we use SA token auth + CA cert, or secret-provided token + insecure TLS.

## Step 1: Enable MLflow in `lora_config_ray_cluster.yaml`

**File**: `fine_tuning/qwen/lora_config_ray_cluster.yaml`

- Line 50: `report_to: "none"` → `report_to: "mlflow"`
- Line 59: bump `run_name` to next version (e.g. `tool-select-cluster-v5`)

This unblocks the gate check at `train_ray.py:112` and also tells HuggingFace's SFTTrainer to create its built-in `MLflowCallback` for per-step metric logging.

**How the two MLflow paths interact**: The manual `mlflow.start_run()` runs before the trainer is created. HuggingFace's `MLflowCallback.setup()` detects the active run, reuses it, and sets `_auto_end_run = False`. This means:
- HuggingFace callback logs per-step metrics (loss, lr, grad norm) during training
- Callback does NOT end the run
- Post-training eval metrics are logged manually to the still-active run
- Manual `mlflow.end_run()` closes the run

## Step 2: Harden MLflow setup in `train_ray.py`

**File**: `fine_tuning/qwen/train_ray.py` (lines 112-130)

Replace the MLflow setup block with:

```python
# MLflow — rank 0 only
use_mlflow = yaml_config["training"].get("report_to") == "mlflow"
if world_rank == 0 and use_mlflow:
    # Auth: prefer token from mlflow-credentials secret, fall back to SA token
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    if not os.environ.get("MLFLOW_TRACKING_TOKEN") and os.path.exists(token_path):
        with open(token_path) as f:
            os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

    # TLS: use OpenShift service-serving CA if available
    ca_cert_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    if os.path.exists(ca_cert_path):
        os.environ.setdefault("MLFLOW_TRACKING_SERVER_CERT_PATH", ca_cert_path)

    tracking_uri = _resolve(yaml_config["mlflow"]["tracking_uri"])
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(yaml_config["mlflow"]["experiment_name"])
    mlflow.start_run(run_name=yaml_config["mlflow"]["run_name"])
    mlflow.log_params({
        "model_name": yaml_config["model_name"],
        "lora_r": yaml_config["lora"]["r"],
        "lora_alpha": yaml_config["lora"]["lora_alpha"],
        "learning_rate": yaml_config["training"]["learning_rate"],
        "batch_size": yaml_config["training"]["per_device_train_batch_size"],
        "grad_accum": yaml_config["training"]["gradient_accumulation_steps"],
        "dry_run": dry_run,
        "ray_num_workers": world_size,
    })
```

**Changes from current code**:
1. Token precedence: don't override `MLFLOW_TRACKING_TOKEN` if already set by `mlflow-credentials` secret
2. TLS: set `MLFLOW_TRACKING_SERVER_CERT_PATH` to the K8s CA bundle for cluster-internal HTTPS
3. Env var: set `MLFLOW_TRACKING_URI` so HuggingFace's callback picks it up

## Step 3: Apply same changes to `train_kubeflow.py` (consistency)

**File**: `fine_tuning/qwen/train_kubeflow.py` (lines 67-85)

Same three changes as Step 2 (token precedence, TLS cert, env var). Also update `lora_config_kubeflow_cluster.yaml` line 50 from `report_to: "none"` to `report_to: "mlflow"`.

## Step 4: Rebuild and push container image

Code is baked into the container image, so changes require rebuild:

```bash
# On the host (not devcontainer)
podman build -t stardew-training:latest -f deploy/Dockerfile .
oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false
podman tag stardew-training:latest localhost:5000/stardew-vision-training/stardew-training:latest
podman push localhost:5000/stardew-vision-training/stardew-training:latest --tls-verify=false
kill %1
```

## Step 5: Test and verify

1. Submit the RayJob: `oc apply -f deploy/rayjob.yaml`
2. Watch logs for MLflow success/errors: `oc logs -f -l job-name=stardew-lora-train | grep -i mlflow`
3. Access MLflow dashboard via port-forward: `oc port-forward svc/mlflow -n redhat-ods-applications 8443:8443`
4. Verify experiment `qwen-tool-selection-train` has run with params + per-step metrics + final eval metrics

## Files to modify

| File | Change |
|------|--------|
| `fine_tuning/qwen/lora_config_ray_cluster.yaml` | `report_to: "mlflow"`, bump run_name |
| `fine_tuning/qwen/train_ray.py` | TLS cert, token precedence, env var (lines 112-130) |
| `fine_tuning/qwen/lora_config_kubeflow_cluster.yaml` | `report_to: "mlflow"`, bump run_name |
| `fine_tuning/qwen/train_kubeflow.py` | Same TLS/auth changes (lines 67-85) |

---
name: MLflow integration — in progress
description: Red Hat MLflow setup findings, what works, what's left; code changes done but untested
type: project
originSessionId: 3f78e579-778d-4320-8e48-61bb5ae7d07d
---
## Status (2026-04-27)

Code changes complete, waiting on a clean image build + 2-GPU dry-run test. Last failure was a podman layer cache issue (`podman build --no-cache` needed).

## Investigation Findings

- **Official docs**: Red Hat KB article 7136121 — "Configuring MLflow in OpenShift AI (Technology Preview)"
- **SDK**: Must use Red Hat fork: `pip install "git+https://github.com/red-hat-data-services/mlflow@rhoai-3.2"` — provides `mlflow.set_workspace()` and `MLFLOW_WORKSPACE` env var. Upstream MLflow 3.8.1 does NOT support workspaces.
- **Auth**: `self_subject_access_review` mode — SA token via `MLFLOW_TRACKING_TOKEN`
- **Tracking URI**: Use **internal service URL** `https://mlflow.redhat-ods-applications.svc:8443` (NOT the gateway — gateway redirects to OAuth login, fails for pod-to-pod)
- **TLS**: Internal service uses OpenShift serving certs; `MLFLOW_TRACKING_INSECURE_TLS=true` required
- **Workspace**: `mlflow.set_workspace("stardew-vision-training")` or `MLFLOW_WORKSPACE` env var (already set in pytorchjob.yaml)
- **RBAC**: `edit` role on `default` SA — **already granted** via `oc adm policy add-role-to-user edit`
- **Connectivity verified**: Debug pod successfully created experiment, logged param+metric, ended run via raw API with `X-MLFLOW-WORKSPACE` header

## Code Changes Done (not committed)

1. `deploy/Dockerfile` — added Red Hat MLflow fork install
2. `deploy/pytorchjob.yaml` — fixed image name (`stardew-training` not `stardew-training-kf`), added `--dry-run`
3. `fine_tuning/qwen/lora_config_*_cluster.yaml` — `report_to: "mlflow"`, added `workspace`, bumped run_name
4. `fine_tuning/qwen/train_ray.py` + `train_kubeflow.py` — token precedence, `MLFLOW_TRACKING_URI` env var, `MLFLOW_TRACKING_INSECURE_TLS`, `set_workspace()`, removed `report_to: "none"` from dry-run

## What's Left

1. Rebuild image with `podman build --no-cache` (layer cache served stale config last time)
2. Push to internal registry
3. Submit 2-GPU KubeFlow dry-run job
4. Verify: rank 0 logs to MLflow, rank 1 skips cleanly, run ends FINISHED
5. Remove `--dry-run` from pytorchjob.yaml after test passes
6. Commit all changes

**Why:** Training runs complete but log nothing to MLflow. Need experiment tracking for the conference demo.

**How to apply:** Plan file at `.claude/plans/clever-growing-hoare.md`. Resume at Step 5 (rebuild) and Step 6 (test).

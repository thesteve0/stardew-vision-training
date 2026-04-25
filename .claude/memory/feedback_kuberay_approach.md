---
name: KubeRay deployment approach rethink
description: First attempt failed due to S3-heavy architecture; plan with PVCs and minimal code changes tomorrow
type: feedback
originSessionId: 08a9a0d5-1cf4-4bcc-a246-d057ff1cac92
---
First attempt at distributed KubeRay training spent hours fighting infrastructure instead of training. User rightly called out the approach was wrong from the start.

**What went wrong:**
- Used S3 (NooBaa) for everything: training data, code delivery (runtime env working_dir zip), Ray checkpoint storage, and MLflow artifacts
- Each S3 interaction hit SSL cert issues — NooBaa uses OpenShift service-serving CAs that pyarrow/libcurl/boto3 each handle differently
- Rewrote data loading code (normalize_content, Dataset.from_list, skip_exists_check) to work around pyarrow version differences between local and cluster — these changes were probably unnecessary
- Cascading fixes (env vars, init containers, combined CA bundles) were a red flag of a bad architecture

**What actually worked:**
- Infrastructure setup: OBC, MLflow, secrets, GPU autoscaling, container image build — all solid
- Training itself: NCCL connected across 2 L40S nodes, model loaded, 69/98 steps completed at ~15s/step, loss 0.06→0.028, 98.9% token accuracy
- Two real bugs found: `_resolve()` doesn't handle `https://` URLs (MLflow URI gets mangled), and DDP + LoRA + gradient_checkpointing needs `use_reentrant=False`

**Why:** Tried to make everything work via S3 when OpenShift AI has PVCs and other native patterns for exactly this.

**How to apply:**
1. USE PLAN MODE FIRST — don't start coding until the architecture is agreed
2. Training data → PVC mounted into pods (same code path as local, no S3 data loading rewrites needed)
3. Training code → bake into container image (no S3 runtime env working_dir)
4. Ray checkpoints → PVC or rethink multi-node checkpoint strategy
5. MLflow → fix the `_resolve()` bug and verify tracking URI reaches the server
6. S3 → only for final model export after training completes
7. Don't rewrite working code to accommodate infrastructure — fix the infrastructure
8. The base image `registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2` already has most deps (transformers, peft, trl, datasets, accelerate, deepspeed, mlflow, etc.) — only need to add ray, qwen-vl-utils, and a few small packages

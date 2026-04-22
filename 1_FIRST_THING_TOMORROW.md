# First Thing Tomorrow — Distributed Training Setup

Baseline eval is done (70% untuned, results in `experiments/eval-baseline-v1/`). Now we train.

## What We Need to Solve

### 1. Training Data → S3
- Upload `datasets/` contents to an S3 bucket on the AWS cluster
- Training scripts need to pull from S3 rather than local filesystem
- The eval set (`datasets/eval_set.json`) must stay separate — never train on it
- Decide bucket structure: `s3://bucket/datasets/{screen_type}/images/` + annotations

### 2. Ray Train Locally (First)
- Get LoRA fine-tuning working with Ray Train in the devcontainer
- Single-GPU first, confirm it trains and metrics log correctly
- Training config: LoRA rank=16, alpha=32, FP16, target modules q/k/v/o_proj
- Use the synthetic + real data (everything NOT in `datasets/eval_set.json`)

### 3. KubeRay on OpenShift AI
- Deploy Ray cluster via KubeRay operator on OpenShift AI
- Port the local Ray Train script to submit to the remote cluster
- GPU resource requests for worker pods
- Mount S3 credentials / configure boto for data access

### 4. Kubeflow on OpenShift AI
- Set up Kubeflow pipeline that wraps the Ray training job
- Pipeline steps: data validation → training → evaluation → model registration
- Parameterize for different screen types, hyperparameters, data versions

### 5. MLflow on OpenShift
- Deploy MLflow server on the same OpenShift cluster
- Configure tracking URI so both local and remote training jobs log to it
- The baseline run is currently in local `mlflow.db` — migrate or re-log to remote instance

### 6. Model Storage + Serving
- Where to store trained LoRA adapters (S3? HuggingFace Hub? OpenShift PVC?)
- Production app (`stardew-vision`) currently pulls from HuggingFace Hub
- KServe vLLM serving needs access to base model + LoRA adapter
- Decide: upload to HF Hub vs serve directly from S3/PVC

## Suggested Order of Attack

1. **Ray Train locally** — get training working end-to-end on devcontainer first
2. **S3 upload** — push datasets to S3, update data loading to support S3 paths
3. **MLflow on OpenShift** — deploy so we have a central place for metrics before scaling out
4. **KubeRay** — move training to the cluster
5. **Model storage** — decide where adapters land after training
6. **Kubeflow** — wrap everything in a pipeline
7. **Re-evaluate** — run eval against the fine-tuned model, compare to baseline

## Key Files

- Eval set (DO NOT TRAIN ON): `datasets/eval_set.json`
- Baseline results: `experiments/eval-baseline-v1/results.json`
- LoRA config: `fine_tuning/qwen/lora_config.yaml`
- Eval scripts: `evaluation/run_baseline.py`, `evaluation/eval_small_run.py`
- Production app reference: https://github.com/thesteve0/stardew-vision

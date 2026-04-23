# Next Steps — KubeRay, Kubeflow, and Talk Prep

Training is done locally — 97.6% eval accuracy on tool selection (Phase 1). Now we move to the cluster and prepare the talk.

## Where We Left Off

- Full LoRA training (3 epochs, 775 samples) completed via Ray Train on local AMD GPU
- 97.6% overall accuracy, only 3 errors out of 125 eval images
- Val loss suggests 2 epochs is optimal (overfit starts at epoch 3)
- All code committed and pushed to main

## Key Files

| File | What it does |
|------|-------------|
| `fine_tuning/qwen/train_ray.py` | Ray Train wrapper — works locally and on KubeRay |
| `fine_tuning/qwen/lora_config_cluster.yaml` | Cluster config (BF16, S3 paths, CHANGEME placeholders) |
| `deploy/rayjob.yaml` | KubeRay RayJob manifest (CHANGEME placeholders) |
| `fine_tuning/qwen/train.py` | Standalone SFTTrainer (still works, not needed for cluster) |
| `fine_tuning/qwen/data_prep.py` | Generates train/val/tiny splits with 2:1 no_tools ratio |
| `evaluation/run_baseline.py` | Eval script — runs against eval_set.json |
| `docs/comparison-small-training-runs.md` | All eval results across runs |

## Step 1: KubeRay on OpenShift AI

### Before you can submit

1. **Fill in CHANGEME placeholders** in `lora_config_cluster.yaml` and `deploy/rayjob.yaml`:
   - S3 bucket name and paths
   - MLflow tracking URI (coordinate with James Harmison)
   - Namespace
   - Container image (confirm `quay.io/opendatahub/odh-pipeline-runtime-pytorch-cuda-py312-ubi9` works)

2. **Upload training data to S3**:
   - `datasets/splits/train.jsonl`, `val.jsonl`
   - `datasets/*/images/` (all screen types)
   - `datasets/synthetic/*/images/`
   - `datasets/eval_set.json`
   - The training code itself (zipped or via container image)

3. **Create Kubernetes secrets**:
   - `s3-credentials`: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
   - `mlflow-credentials`: MLFLOW_TRACKING_URI (and token if auth required)

### Submit and monitor

```bash
kubectl apply -f deploy/rayjob.yaml
kubectl logs -f job/stardew-lora-train
# Ray dashboard: kubectl port-forward svc/stardew-lora-train-head-svc 8265:8265
```

### After training completes

- Run eval against the cluster-trained checkpoint
- Capture timings (wall time, s/step) — compare to local (4h 40m / 48s per step)
- Screenshot MLflow: training loss curve, val loss curve, timing metrics

## Step 2: Kubeflow Pipeline

Run the same training through a Kubeflow pipeline on OpenShift AI. The pipeline should wrap the Ray training job:

1. Data validation step (verify splits exist, image counts match expectations)
2. Training step (calls `train_ray.py` via RayJob)
3. Evaluation step (calls `run_baseline.py` on the trained checkpoint)
4. Model registration step (upload LoRA adapter to HuggingFace Hub or S3)

Capture timings and eval results for comparison to standalone Ray.

## Step 3: MLflow Screenshots

Collect screenshots from MLflow for the talk:

- [ ] Training loss curve (steps vs loss) — show the drop from 0.085 to near-zero
- [ ] Val loss curve — show epoch 2 minimum and epoch 3 overfit
- [ ] Token accuracy curve
- [ ] Parameter comparison table (if multiple runs visible)
- [ ] Timing metrics (train_wall_time_min, seconds_per_step)
- [ ] Compare local vs cluster vs Kubeflow runs side-by-side

## Step 4: Talk Prep

### Story arc

1. **The problem**: Stardew Valley is inaccessible to visually impaired players — no screen reader support
2. **The approach**: Fine-tune a VLM to recognize UI screens and call extraction tools
3. **The challenge**: Teaching a model when NOT to act (no_tools rejection class)
4. **Data decisions**: 2:1 oversampling for negative class, backed by research (Larson et al., Gorilla)
5. **Training progression**: Standalone → Ray Train → KubeRay → Kubeflow (show it scales)
6. **Results**: 55% baseline → 97.6% after fine-tuning, with the no_tools story as the highlight
7. **Live demo**: Show the model classifying real game screenshots

### Key talking points to develop

- The no_tools data balance story (0% → 98%) is the most compelling technical narrative
- Show MLflow screenshots to make training tangible
- Compare local GPU timing vs cluster timing — the "why distribute" argument
- OpenShift AI as the platform tying Ray, Kubeflow, MLflow together
- LoRA efficiency — fine-tuned a 7B model on a single consumer GPU in under 5 hours

### Demo ideas

- Live inference on new screenshots (show tool call output)
- Side-by-side: baseline model vs fine-tuned model on the same image
- MLflow dashboard walkthrough
- KubeRay job submission live (if time permits)

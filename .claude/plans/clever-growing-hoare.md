# Plan: Create Conference Demo Training Configuration (KubeRay + MLflow)

## Context

The user is giving a conference talk and needs a training run that completes quickly (~12-15 min) while showing the full distributed training flow in real time. The audience will watch both the **Ray Dashboard** (job progress, worker utilization) and the **MLflow Dashboard** (experiment metrics accumulating live). This uses KubeRay (not KubeFlow PyTorchJob) because the demo focuses on the Ray integration.

The demo uses 15% of the training data (~116 of 775 samples), runs 3 epochs with eval after each, and logs to an MLflow experiment with "demo" in the name. A **separate container image** (`stardew-training-demo:latest`) is built so the production image stays untouched.

## Step 1: Create demo split script

**File**: `scripts/make_demo_split.py` (new)

A standalone script that reads the existing `datasets/splits/train.jsonl`, samples 15% while maintaining class proportions, and writes `datasets/splits/train_demo.jsonl`. Does not modify `data_prep.py`.

```python
#!/usr/bin/env python3
"""Generate a 15% demo subset of the training data for conference talks."""

import argparse
import json
import random
from pathlib import Path

DEMO_FRACTION = 0.15
SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("datasets/splits/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/splits/train_demo.jsonl"))
    parser.add_argument("--fraction", type=float, default=DEMO_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    with open(args.input) as f:
        examples = [json.loads(line) for line in f]

    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        st = ex["metadata"]["screen_type"]
        by_type.setdefault(st, []).append(ex)

    demo = []
    for st, pool in by_type.items():
        n = max(1, int(len(pool) * args.fraction))
        demo.extend(pool[:n])

    random.shuffle(demo)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for record in demo:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(demo)} demo samples to {args.output} ({args.fraction:.0%} of {len(examples)})")
    for st, pool in by_type.items():
        n = max(1, int(len(pool) * args.fraction))
        print(f"  {st}: {n}/{len(pool)}")

if __name__ == "__main__":
    main()
```

Run: `python scripts/make_demo_split.py`

## Step 2: Create `lora_config_ray_demo.yaml`

**File**: `fine_tuning/qwen/lora_config_ray_demo.yaml` (new)

Based on `lora_config_ray_cluster.yaml` with these differences:

| Parameter | Cluster | Demo | Why |
|-----------|---------|------|-----|
| `output_dir` | `/checkpoints/model-output` | `/checkpoints/model-demo` | Separate from real runs |
| `warmup_steps` | 50 | 5 | Proportional to smaller dataset |
| `logging_steps` | 10 | 1 | Every step — audience sees frequent live updates |
| `save_steps` | 100 | 999999 | Skip mid-training saves |
| `eval_steps` | 100 | 999999 | Not used (`eval_strategy: "epoch"`) |
| `save_strategy` | `"epoch"` | `"no"` | Skip checkpoint saves for speed |
| `save_total_limit` | 3 | 1 | Minimal |
| `load_best_model_at_end` | true | false | No checkpoint selection |
| `data.train_file` | `datasets/splits/train.jsonl` | `datasets/splits/train_demo.jsonl` | 15% subset |
| `mlflow.experiment_name` | `qwen-tool-selection-train` | `qwen-tool-selection-demo` | "demo" in name |
| `mlflow.run_name` | `tool-select-cluster-v5` | `demo-ray-v1` | Descriptive |

All other parameters (model_name, lora config, learning_rate, batch_size, bf16, eval_strategy=epoch, etc.) stay identical.

**Estimated timing** (2 x L40S GPU workers):
- ~116 training samples, batch=4, 2 workers → effective batch 8 → ~15 steps/epoch
- 3 epochs = ~45 steps at ~9s/step ≈ 7 min training
- 3 eval passes on 146 val samples ≈ 2 min
- Model load + data mapping ≈ 5 min
- **Total: ~14 min**

## Step 3: Create `deploy/rayjob-demo.yaml`

**File**: `deploy/rayjob-demo.yaml` (new)

Based on `deploy/rayjob.yaml` with these changes:

- `metadata.name`: `stardew-lora-train-demo`
- `spec.entrypoint`: config path → `/app/fine_tuning/qwen/lora_config_ray_demo.yaml`
- All 3 container images (head + 2 workers): `stardew-training-demo:latest` instead of `stardew-training:latest`

Everything else (PVCs, env vars, secrets, resources, tolerations, `--num-workers 2`) stays the same.

## Step 4: Build and push the demo image

Same Dockerfile, different image name. Since `lora_config_ray_demo.yaml` is in `fine_tuning/qwen/`, it gets baked in by `COPY fine_tuning/ fine_tuning/`.

```bash
# On host (not devcontainer)
podman build --no-cache -t stardew-training-demo:latest -f deploy/Dockerfile .

oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false
podman tag stardew-training-demo:latest \
  localhost:5000/stardew-vision-training/stardew-training-demo:latest
podman push localhost:5000/stardew-vision-training/stardew-training-demo:latest \
  --tls-verify=false
kill %1
```

## Step 5: Copy demo data to PVC

`train_demo.jsonl` must exist on the `training-data` PVC at `/data/datasets/splits/train_demo.jsonl`. Copy using the same method used for the other split files.

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/make_demo_split.py` | **New** — standalone script to generate 15% demo split |
| `fine_tuning/qwen/lora_config_ray_demo.yaml` | **New** — Ray demo config for cluster |
| `deploy/rayjob-demo.yaml` | **New** — RayJob manifest with demo image + config |
| `datasets/splits/train_demo.jsonl` | **Generated** by running `make_demo_split.py` |

## Usage (during talk)

```bash
# Submit the demo job
oc apply -f deploy/rayjob-demo.yaml

# Open Ray Dashboard — shows job progress, worker GPU utilization
# Open MLflow UI — experiment "qwen-tool-selection-demo", watch metrics accumulate

# Clean up after talk
oc delete rayjob stardew-lora-train-demo
```

## Verification

1. Run `python scripts/make_demo_split.py` — confirm `train_demo.jsonl` has ~116 samples
2. Build demo image as `stardew-training-demo:latest`
3. Push to cluster registry
4. Copy `train_demo.jsonl` to training-data PVC
5. `oc apply -f deploy/rayjob-demo.yaml`
6. Confirm Ray Dashboard shows 2 GPU workers active
7. Confirm MLflow experiment `qwen-tool-selection-demo` appears with run `demo-ray-v1`
8. Watch 3 epochs + 3 eval passes complete with per-step metrics

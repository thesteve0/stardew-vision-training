---
name: Distributed training plan
description: Ray Train code aligned; fresh runs needed local then cluster; effective batch size mismatch identified
type: project
originSessionId: 561452f1-3c56-4b65-ab91-c8ad240d347b
---
Ray Train integration complete but results not yet validated with aligned code (as of 2026-04-25).

Progression:
1. Local standalone (train.py) — NEXT (fresh baseline)
2. Local Ray (train_ray.py) — after standalone baseline
3. KubeRay on OpenShift AI — after local Ray matches
4. Kubeflow on OpenShift AI — future

**Why:** Previous distributed run produced worse results. Root causes identified: `gradient_checkpointing_kwargs` mismatch (now fixed) and effective batch size difference (8 local vs 16 cluster, not yet fixed).

**How to apply:**
- Both `train.py` and `train_ray.py` now use `use_reentrant=False` — aligned
- `lora_config_ray_cluster.yaml` has effective batch 16 (4 × 2 workers × 2 grad_accum). To match local's 8, set `gradient_accumulation_steps: 1`
- `deploy/rayjob.yaml` ready with PVC-based data mounting
- Cluster: 1x L40S (autoscales to 4), BF16 capable

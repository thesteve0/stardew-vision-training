---
name: Distributed training — completed
description: All training paths validated (local, Ray, KubeFlow 1-GPU, KubeFlow 2-GPU); training phase is done, moving to MLflow and adapter deployment
type: project
---
Distributed training is complete as of 2026-04-26. All orchestration paths validated on the cluster.

**Completed runs:**
1. Local standalone (train.py) — 96.0%, ~424 min on Strix Halo
2. Cluster Ray Train (train_ray.py) — 96.0%, ~43 min on 2x L40S
3. Cluster KubeFlow 2-GPU (train_kubeflow.py) — 97.6%, ~42 min on 2x L40S
4. Cluster KubeFlow 1-GPU (train_kubeflow.py) — 96.8%, ~82 min on 1x L40S

**Key finding:** Effective batch size mismatch (8 local vs 16 cluster) was the root cause of Ray v3's poor results (90.4%). Fixed by setting `gradient_accumulation_steps: 1` in cluster configs. 1-GPU baseline confirmed the 10x local-to-cluster speedup decomposes into ~5.2x chip + ~2x parallelism.

**Why:** Training phase is done. Next steps are MLflow dashboard and adapter deployment to production app.

**How to apply:** KubeFlow is the preferred path going forward — simpler (no Ray dependency), slightly faster, and the Dockerfile is smaller. Runbooks: `RUNNING_KUBEFLOW_DISTRIBUTED.md` and `RUNNING_RAY_DISTRIBUTED.md`.

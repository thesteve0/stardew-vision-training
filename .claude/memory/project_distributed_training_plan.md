---
name: Distributed training — completed
description: Both Ray Train and KubeFlow PyTorchJob runs completed on OpenShift AI cluster; batch size issue resolved
type: project
originSessionId: 0b1c1454-7379-436e-a94a-f747fbf758da
---
Distributed training is complete as of 2026-04-26. Both orchestration paths validated on the cluster.

**Completed runs:**
1. Local standalone (train.py) — baseline established
2. Cluster Ray Train — 96.0% accuracy (effective batch 8, 2x L40S)
3. Cluster KubeFlow PyTorchJob — 97.6% accuracy (effective batch 8, 2x L40S)

**Key finding:** Effective batch size mismatch (8 local vs 16 cluster) was the root cause of Ray v3's poor results (90.4%). Fixed by setting `gradient_accumulation_steps: 1` in cluster configs.

**How to apply:** Both paths are production-ready. KubeFlow is slightly simpler (thinner wrapper, master trains with GPU). Ray adds fault tolerance and actor-based orchestration. Detailed comparison in `POINTS_FOR_TALK.md`.

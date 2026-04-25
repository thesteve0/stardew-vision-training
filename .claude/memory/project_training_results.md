---
name: Training results cleared for fresh baseline
description: All previous training results deleted 2026-04-25; rerunning with aligned gradient_checkpointing_kwargs between local and distributed
type: project
originSessionId: 561452f1-3c56-4b65-ab91-c8ad240d347b
---
Previous results (97.6% accuracy) cleared on 2026-04-25 to establish a clean baseline.

**What changed:** `train.py` now uses `gradient_checkpointing_kwargs: {"use_reentrant": False}`, matching `train_ray.py`. Previous runs had this mismatch, making local-vs-distributed comparisons invalid.

**Why:** Distributed training produced worse results than local. Investigation found the `use_reentrant` mismatch and an effective batch size difference (8 local vs 16 cluster). Need clean runs with aligned code to isolate the real cause.

**How to apply:** Run local standalone first to establish baseline, then local Ray, then cluster Ray. Record all results in `docs/comparison-small-training-runs.md`. If cluster is still worse, adjust `gradient_accumulation_steps` in cluster config to match effective batch size of 8.

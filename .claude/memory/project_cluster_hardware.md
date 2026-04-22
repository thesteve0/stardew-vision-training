---
name: Cluster hardware and ops
description: OpenShift cluster has 2-3 Nvidia L40S GPUs (48GB VRAM, BF16 capable); James Harmison is ops deploying MLflow
type: project
originSessionId: 46553f92-c1a3-4c2b-8ed3-5d6452f9494a
---
OpenShift AI cluster for training has 2-3 Nvidia L40S GPUs (48GB VRAM each, Ada Lovelace, supports BF16/FP16/INT8).

**Why:** Need multi-GPU training at scale beyond single-GPU devcontainer.

**How to apply:** Training code for cluster uses BF16 (not FP16), DDP is sufficient (no FSDP needed for 7B + LoRA), batch size 4 per GPU feasible. NVIDIA container image needed (separate from ROCm devcontainer). Candidate base image: `quay.io/opendatahub/odh-pipeline-runtime-pytorch-cuda-py312-ubi9` (awaiting confirmation from container team as of 2026-04-22). James Harmison is the ops person deploying MLflow on the same OpenShift cluster — coordinate with him on tracking URI and auth.

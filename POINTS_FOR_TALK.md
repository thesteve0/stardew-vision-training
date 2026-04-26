# Points for Talk

Interesting findings and talking points from the Stardew Valley VLM fine-tuning project.

---

## Prompt engineering vs fine-tuning impact

The untuned Qwen2.5-VL-7B-Instruct baseline scored **0% on no_tools** with the original eval prompt. After updating the prompt to match the production prompt (which explicitly instructs the model to refuse when no tool matches), no_tools accuracy jumped to **30%** — with zero training.

Full baseline with production prompt: **51.2% overall accuracy** (tv_dialog 80%, caught_fish 20%, pierre_shop 96%, no_tools 30%). Fine-tuning brought this to 97.6%. Shows that prompt engineering is the first lever, and fine-tuning is the second — but both matter.

## Effective batch size influences training results

When moving from local to cluster training, the effective batch size accidentally doubled (4 per-device × 2 GPUs × 2 gradient accumulation = 16, vs 8 locally). This dropped accuracy from 97.6% to 90.4% — a 7.2 point hit from a single hyperparameter. The most affected class was caught_fish (68% vs 100% local). Fixing gradient accumulation back to 1 (effective batch = 8) recovered to 96.0% on Ray and 97.6% on KubeFlow.

Lesson: when scaling to multiple GPUs, effective batch size = per_device_batch × num_GPUs × gradient_accumulation_steps. If you don't halve one of those factors, you're training with a different recipe than you validated locally.

## Ray Train vs KubeFlow PyTorchJob — architectural differences

Both ran on identical hardware (2x NVIDIA L40S) with identical hyperparameters. Ray Train scored 96.0%, KubeFlow PyTorchJob scored 97.6% — but the 2-sample difference (120 vs 122 out of 125) is likely statistical noise rather than a real architectural advantage.

Ray Train adds a layer over PyTorch DDP: an actor-based model where the head node orchestrates workers, handles checkpointing via callbacks, and provides fault tolerance at the framework level. The head node has no GPU — it's a coordinator. KubeFlow PyTorchJob is thinner: it launches pods with native `torchrun` and relies on Kubernetes for lifecycle. The Master node gets a GPU and participates in training (unlike Ray's head).

If the accuracy difference is real, the most likely cause is data shuffling: Ray shards data across workers with its own mechanism (`prepare_trainer`), while torchrun uses PyTorch's native `DistributedSampler`. Different micro-batch assignments per rank means different gradient compositions at each step. On a small dataset (775 samples), specific ordering can matter.

KubeFlow was slightly faster per step (8.6s vs 8.8s) due to less framework overhead.

## Training and eval timings

775 training samples, 3 epochs, effective batch size 8 (unless noted).

| Run | Hardware | Steps | Wall time | s/step | Accuracy |
|-----|----------|-------|-----------|--------|----------|
| Local standalone | 1x AMD Strix Halo (gfx1151) | 291 | ~424 min | ~87.4 | 96.0% |
| Cluster Ray v3 (batch=16) | 2x NVIDIA L40S | 147 | ~42 min | ~17 | 90.4% |
| Cluster Ray v4 (batch=8) | 2x NVIDIA L40S | 291 | ~43 min | ~8.8 | 96.0% |
| Cluster KubeFlow v1 (batch=8) | 2x NVIDIA L40S | 291 | ~42 min | ~8.6 | 97.6% |

Wall time is nearly identical across cluster runs despite 2x more steps in v4/KubeFlow — smaller batches run faster per step since there's no gradient accumulation synchronization wait.

## iGPU vs data center GPU — the real numbers

**Training**: Strix Halo takes ~10x longer end-to-end (424 min vs 43 min). This breaks down as:
- **~5x per-GPU compute gap**: Each forward+backward pass takes ~22s on Strix Halo vs ~4.4s on L40S (comparing per fwd+bwd pass: local does 4 per step with grad_accum=4, cluster does 2 per GPU with grad_accum=2)
- **2x from parallelism**: Two L40S GPUs split the gradient accumulation work

**Eval inference (125 images, fine-tuned model)**: Strix Halo ~8 min (~3.85s/sample), L40S cluster ~5 min (~2.4s/sample) — only **~1.6x difference**. Inference is memory-bandwidth-bound (autoregressive decoding), so raw TFLOPS matters far less than during training. The Strix Halo is surprisingly competitive for serving a 7B VLM.

**Baseline eval (untuned model, no LoRA)**: Strix Halo was ~63 min (~30s/sample) — 8x slower than the fine-tuned eval on the same hardware. The LoRA adapter dramatically reduces generation length (tool calls are short vs rambling untuned outputs), which dominates autoregressive inference time.

## iGPU vs data center GPU — theoretical specs

| Spec | AMD Strix Halo (Radeon 8060S) | NVIDIA L40S | Ratio (L40S / Strix Halo) |
|------|-------------------------------|-------------|---------------------------|
| Architecture | RDNA 3.5 (4nm TSMC) | Ada Lovelace (4nm TSMC) |  |
| Compute units / SMs | 40 CUs | 142 SMs (18,176 CUDA cores) |  |
| Peak FP16 (theoretical) | 59.4 TFLOPS | 362 TFLOPS (Tensor Core, dense) | **6.1x** |
| Achieved FP16 (practical) | ~37 TFLOPS (with hipBLASLt) | ~300+ TFLOPS (typical) | **~8x** |
| Memory capacity | Up to 96 GB (unified DDR5) | 48 GB (GDDR6 ECC) | 0.5x (Strix has more!) |
| Memory bandwidth (theoretical) | 256 GB/s | 864 GB/s | **3.4x** |
| Memory bandwidth (measured) | ~212 GB/s | ~864 GB/s | **~4.1x** |
| TDP | 55W (whole APU) | 350W | 6.4x |
| TFLOPS/Watt (FP16 achieved) | ~0.67 | ~0.86 | 1.3x |

### How theory maps to our measurements

**Training (compute-bound)**: The theoretical FP16 ratio is 6.1x, practical is ~8x. Our measured per-GPU training gap is ~5x (22s vs 4.4s per fwd+bwd pass). The lower-than-theoretical gap is because LoRA fine-tuning is partially memory-bandwidth-bound (adapter weight updates, optimizer states), not purely matmul-limited. The L40S can't fully exploit its Tensor Core advantage on LoRA workloads.

**Inference (bandwidth-bound)**: The memory bandwidth ratio is 3.4–4.1x, but our measured inference gap is only ~1.6x. This is because autoregressive decoding of short sequences (tool calls are ~50 tokens) is latency-bound, not throughput-bound — each token requires a full model pass through all layers, and the overhead per token (kernel launch, attention) compresses the bandwidth advantage. For longer generations or batched inference, the gap would widen toward the 3–4x range.

**Power efficiency**: The Strix Halo achieves ~0.67 TFLOPS/W vs L40S at ~0.86 TFLOPS/W — surprisingly close. The iGPU gets competitive efficiency from shared memory (no separate VRAM power) and a power-optimized process node. For single-user VLM inference, the Strix Halo delivers ~60% of L40S throughput at ~16% of the power draw.

**The memory advantage**: Strix Halo can address up to 96 GB of unified memory as VRAM — 2x the L40S's 48 GB. This means larger models (30B+ parameters) that don't fit on an L40S can run on Strix Halo, albeit slowly. For VLM fine-tuning with LoRA at 7B scale, both have ample memory.

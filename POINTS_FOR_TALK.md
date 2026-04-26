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

All cluster runs used 2x NVIDIA L40S (48GB each), 775 training samples, 3 epochs, effective batch size 8.

| Run | Steps | Wall time | s/step | Accuracy |
|-----|-------|-----------|--------|----------|
| Cluster Ray v3 (batch=16) | 147 | ~42 min | ~17 | 90.4% |
| Cluster Ray v4 (batch=8) | 291 | ~43 min | ~8.8 | 96.0% |
| Cluster KubeFlow v1 (batch=8) | 291 | ~42 min | ~8.6 | 97.6% |

Wall time is nearly identical across runs despite 2x more steps in v4/KubeFlow — smaller batches run faster per step since there's no gradient accumulation synchronization wait.

Baseline eval on local AMD iGPU (Strix Halo gfx1151, ROCm 7.2): ~63 minutes for 125 images (~30s/sample). Same eval on cluster L40S: ~5 minutes. That's a 12x throughput difference for inference — the datacenter GPU advantage is even larger for training where memory bandwidth and VRAM matter more.

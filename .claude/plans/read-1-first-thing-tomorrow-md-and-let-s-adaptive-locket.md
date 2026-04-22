# Plan: Training Infrastructure Setup & First Training Runs

## Context

Baseline eval is done (70% accuracy untuned). Now we need to get training working — starting with small local iterations, then scaling to 2-3 Nvidia L40S GPUs on OpenShift via Ray Train. The project moves between a desktop and laptop (both AMD ROCm) via git push/pull, and Claude memory needs to travel with it. MLflow deployment on OpenShift is being coordinated with James Harmison (ops).

Key finding during exploration: the synthetic training data (`datasets/synthetic/*/conversations.jsonl`) does **not** include a system prompt with tool definitions, but the eval code (`evaluation/prompt.py`) injects one. Training must add this system prompt to each conversation to match eval-time behavior — otherwise the model won't learn to associate the tool definition format with correct tool selection.

The `train.py` has two unresolved TODOs (data collator and chat template) that block any training.

---

## Part 1: Directory Structure

Add `infrastructure/` at the top level for all deployment/cluster configs. Keep training code in `fine_tuning/`, keep eval and synthetic data where they are.

```
stardew-vision-training/
├── fine_tuning/qwen/
│   ├── train.py                  # FIX — resolve data collator + chat template TODOs
│   ├── train_ray.py              # NEW — Ray Train wrapper
│   ├── lora_config.yaml          # existing base config
│   ├── lora_config_tiny.yaml     # NEW — 40 samples, 1 epoch, for iteration
│   ├── data_collator.py          # NEW — ChatML + image collator for Qwen2.5-VL
│   └── data_prep.py              # NEW — assemble train/val splits from synthetic + real
│
├── infrastructure/               # NEW — all cluster/deployment configs
│   ├── ray/
│   │   └── ray_cluster_l40s.yaml    # Ray cluster config for L40S nodes
│   ├── kuberay/
│   │   ├── raycluster.yaml          # KubeRay CRD for OpenShift
│   │   └── rayjob.yaml             # RayJob for training submission
│   ├── kubeflow/
│   │   └── pipeline.py             # Pipeline: validate → train → eval → register
│   └── openshift/
│       └── training-job.yaml       # Container image + resource requests for training
│
├── scripts/s3/                   # NEW — data upload/download for cluster training
│   ├── upload_datasets.py
│   └── download_datasets.py
│
├── .claude/                       # Claude Code project config (git-tracked)
│   ├── settings.local.json       # existing project settings
│   ├── memory/                   # NEW — auto-memory files (symlinked from user profile)
│   └── plans/                    # NEW — plan files (symlinked from user profile)
│
└── (everything else unchanged)
```

**Why `infrastructure/` and not `deploy/`:** It holds Ray configs (not K8s-specific), Kubeflow pipelines (Python, not manifests), and OpenShift specifics. "Infrastructure" covers all of these without implying a single tool.

**MLflow deployment configs excluded:** James Harmison is handling MLflow on OpenShift. We just need the tracking URI. If we end up needing manifests here, they go in `infrastructure/mlflow/`.

### Files to modify
- `.gitignore` — add `mlflow.db` (currently tracked at 648KB, will grow; being replaced by remote MLflow)
- `CLAUDE.md` — update directory structure section

---

## Part 2: Consolidate All Claude Code Config into `.claude/` (Git-Tracked)

All Claude Code data for this project must live in the workspace's `.claude/` directory so it travels with `git push/pull`. Nothing project-related should live exclusively in the devcontainer's user profile (`~/.claude/`).

### Current state in user profile (`~/.claude/`):
- `projects/-workspaces-stardew-vision-training/memory/` — 7 memory files + MEMORY.md
- `plans/` — 4 plan files (including this plan)
- `settings.json` — global settings (just `{"model": "claude-opus-4-6[1m]"}`)
- `projects/-workspaces-stardew-vision-training/*.jsonl` — session transcripts (large, ephemeral)
- `history.jsonl`, `paste-cache/`, `backups/`, `cache/`, `sessions/`, `shell-snapshots/` — all ephemeral

### Current state in workspace (`.claude/`):
- `settings.local.json` — project settings (already git-tracked)

### Migration steps

1. **Create directories** in the workspace:
   ```
   .claude/memory/    # for auto-memory files
   .claude/plans/     # for plan files
   ```

2. **Move existing memory files** from `~/.claude/projects/-workspaces-stardew-vision-training/memory/*` → `.claude/memory/`

3. **Move existing plan files** from `~/.claude/plans/*` → `.claude/plans/` (includes this plan file itself)

4. **Copy global settings** — copy `~/.claude/settings.json` to `.claude/settings.json` (model preference should travel with the project)

5. **Add symlinks in `.devcontainer/setup-environment-existing.sh`:**
   ```bash
   # --- Claude Code portability ---
   # All Claude Code state for this project lives in the workspace .claude/ directory
   # (git-tracked). Create symlinks so Claude Code finds it at the expected paths.
   WORKSPACE_CLAUDE="/workspaces/stardew-vision-training/.claude"
   USER_CLAUDE="/home/stpousty-devcontainer/.claude"
   PROJECT_SLUG="-workspaces-stardew-vision-training"

   # Memory: symlink per-project memory to workspace
   mkdir -p "$USER_CLAUDE/projects/$PROJECT_SLUG"
   ln -sfn "$WORKSPACE_CLAUDE/memory" "$USER_CLAUDE/projects/$PROJECT_SLUG/memory"

   # Plans: symlink plans directory to workspace
   ln -sfn "$WORKSPACE_CLAUDE/plans" "$USER_CLAUDE/plans"

   # Settings: copy workspace settings.json as the global default
   if [ -f "$WORKSPACE_CLAUDE/settings.json" ]; then
     cp "$WORKSPACE_CLAUDE/settings.json" "$USER_CLAUDE/settings.json"
   fi
   ```

6. **Add `.gitignore` entries within `.claude/`** — the workspace `.claude/` directory should NOT be fully gitignored. Add a `.claude/.gitignore` file:
   ```
   # Track: settings, memory, plans
   # Ignore: ephemeral Claude Code data
   *.jsonl
   paste-cache/
   backups/
   cache/
   sessions/
   shell-snapshots/
   plugins/
   ```

7. **Add `containerEnv` to `devcontainer.json`** — no env vars needed for the symlink approach, but document it:
   ```jsonc
   // Claude Code config lives in .claude/ (workspace), symlinked from user profile.
   // See setup-environment-existing.sh for details.
   ```

### Laptop migration

On the laptop, after `git pull`:
1. Rebuild the devcontainer (the setup script creates the symlinks automatically)
2. Claude Code will find memory and plans at the symlinked paths
3. No manual steps required — the `postCreateCommand` handles everything

### Files to create
- `.claude/memory/` — move existing files here
- `.claude/plans/` — move existing files here
- `.claude/settings.json` — copy from user profile
- `.claude/.gitignore` — exclude ephemeral data

### Files to modify
- `.devcontainer/setup-environment-existing.sh` — add symlink block
- `.gitignore` (root) — ensure `.claude/` is NOT ignored (remove any `.claude` entry if present)

---

## Part 3: Fix Training Pipeline (Before Any Training)

The training script at `fine_tuning/qwen/train.py` has two blocking TODOs at lines 156-157.

### 3a. Training Data Format for Phase 1 (Tool Selection)

Phase 1 trains only on tool selection — not narration. Each training example is a **2-turn conversation**:

1. **System message:** Tool definitions in `<tools>` tags (from `evaluation/prompt.py`)
2. **User message:** Image (file path) + "What's on this screen?"
3. **Assistant message:** The expected tool call in Qwen's `<tool_call>` format, e.g.:
   ```
   <tool_call>
   {"name": "crop_tv_dialog", "arguments": {}}
   </tool_call>
   ```
   Or for `no_tools`: `"I don't have a tool to handle that screen"`

The current synthetic `conversations.jsonl` has 4 turns (through OCR + narration). `data_prep.py` will **truncate to the first 2 turns** (user + first assistant tool call) and prepend the system prompt.

### 3b. Create `fine_tuning/qwen/data_prep.py`

Assembles training splits for Phase 1:
1. Load all `datasets/synthetic/*/conversations.jsonl` (692 records)
2. Load `datasets/eval_set.json` — extract image paths to exclude
3. Remove any conversation whose image appears in the eval set
4. **Truncate each conversation to 2 turns** (user image + assistant tool call only)
5. **Prepend the system prompt** from `evaluation/prompt.py` as the first message
6. **Reformat assistant tool call** to use Qwen's `<tool_call>` XML format (matching what the model actually generates at inference time)
7. Shuffle with fixed seed, split 85/15 into train/val
8. Write to `datasets/splits/train.jsonl`, `datasets/splits/val.jsonl`
9. Also write `datasets/splits/train_tiny.jsonl` (40 samples, 10 per screen type) for fast iteration

### 3c. Create `fine_tuning/qwen/data_collator.py`

Custom collator for ChatML conversations with embedded images. Pattern from `evaluation/inference.py`:
1. For each conversation, call `processor.apply_chat_template()` to tokenize
2. Load images from `file://` paths using PIL, resize to 1600×1200 (matching eval)
3. Use `process_vision_info()` from `qwen_vl_utils` to extract pixel values
4. Mask user/system token labels to -100 (only train on assistant responses — the tool call output)
5. Pad batch to uniform length

### 3d. Fix `fine_tuning/qwen/train.py`

- Wire in the data collator
- Add system prompt injection at dataset load time (or delegate to data_prep.py)
- Remove `evaluation_strategy` (deprecated, use `eval_strategy`)
- Add `--dry-run` flag for Tier 0 testing (sets `max_steps=2`)

### Files to create
- `fine_tuning/qwen/data_prep.py`
- `fine_tuning/qwen/data_collator.py`
- `fine_tuning/qwen/lora_config_tiny.yaml`

### Files to modify
- `fine_tuning/qwen/train.py` — fix TODOs, wire collator

---

## Part 4: Training Iteration Strategy (Start Small)

All local machines have 48GB VRAM. Hyperparameters below assume 48GB available.

### Tier 0: Dry Run (~1 minute, 2 training steps)
- 8 samples (2 per screen type)
- `max_steps=2` — verify the pipeline loads, tokenizes, computes a loss, and that loss changes between steps
- Command: `python fine_tuning/qwen/train.py --config fine_tuning/qwen/lora_config_tiny.yaml --dry-run`
- Batch size 2 (48GB VRAM supports it for this model)
- **Success criteria:** loss is finite on both steps, loss changes between step 1 and step 2

### Tier 1: Tiny Training (~10-15 min on single GPU)
- 40 samples from `train_tiny.jsonl`, 2 steps only
- `max_steps=2`, batch=2, grad_accum=4 (effective batch=8)
- lr=1e-4 (conservative start for LoRA)
- Log to local MLflow
- **Success criteria:** training loss decreases between the 2 steps, confirming gradients flow and the model is learning from the data
- After: run `evaluation/eval_small_run.py` on the checkpoint, compare to baseline

### Tier 2: Full Local Training (~1-2 hours on single GPU)
- All ~620 training samples (692 minus eval holdout)
- 3 epochs, batch=2, grad_accum=4 (effective batch=8)
- Save checkpoints every 100 steps, keep best 3
- Full eval against `datasets/eval_set.json`
- **Success criteria:** accuracy > 70% baseline, especially no_tools and caught_fish improve

### Tier 3: Multi-GPU on Cluster (2-3 L40S, ~30-60 min)
- Same dataset, DDP via Ray Train across GPUs
- BF16 precision (L40S supports it)
- Larger effective batch size (batch=4 per GPU × 3 GPUs × grad_accum=2 = 24)
- Hyperparameter exploration: learning rate sweep, LoRA rank comparison
- Log to remote MLflow on OpenShift

### Hyperparameters (48GB VRAM baseline)
```yaml
per_device_train_batch_size: 2     # 7B model FP16 (~14GB) + batch 2 fits in 48GB
gradient_accumulation_steps: 4     # effective batch = 8
learning_rate: 1.0e-4              # conservative for LoRA (can tune up)
warmup_steps: 50
lr_scheduler_type: cosine
max_grad_norm: 1.0
fp16: true                         # ROCm local (bf16: true for NVIDIA cluster)
```

---

## Part 5: L40S Cluster Training Config

Both local machines and the cluster have 48GB VRAM. The key differences for the cluster are: NVIDIA (BF16 capable), multi-GPU via DDP, and the need for a separate container image.

For this 7B model with LoRA (~4.2M trainable params), each GPU holds a full model copy easily, so **DDP (not FSDP)** is sufficient and simpler.

### Ray Train wrapper (`fine_tuning/qwen/train_ray.py`)

```
TorchTrainer(
    train_func,              # reuses logic from train.py
    scaling_config=ScalingConfig(num_workers=N, use_gpu=True),
    run_config=RunConfig(name="qwen-lora-vN"),
)
```

Inside `train_func`: load model + LoRA, wrap with `ray.train.torch.prepare_model()` (handles DDP), load data with `prepare_data_loader()`, run HF Trainer.

### NVIDIA cluster config differences (vs local ROCm)
- `torch_dtype: bfloat16` (vs float16 on ROCm — better numerical stability)
- `bf16: true, fp16: false`
- `per_device_train_batch_size: 4` (BF16 uses slightly less memory than FP16, can push batch higher)
- `optim: adamw_torch_fused` (NVIDIA-optimized fused kernel)
- Effective batch with 3 GPUs: batch=4 × 3 workers × grad_accum=2 = 24

### Container image for cluster
- The OpenShift training job needs an NVIDIA PyTorch container image (not the ROCm devcontainer)
- **Candidate base:** `quay.io/opendatahub/odh-pipeline-runtime-pytorch-cuda-py312-ubi9` (OpenDataHub pipeline runtime, CUDA, Python 3.12, UBI9)
- **Status:** Awaiting confirmation from the container team — noted but not yet confirmed
- Install project deps with `uv` (same `pyproject.toml`, minus the ROCm exclusions)
- Defined in `infrastructure/openshift/training-job.yaml`

---

## Part 6: MLflow Integration

### Immediate (local, before cluster)
- Keep using local `file://experiments/mlruns` for Tier 0-2
- Add `mlflow.db` to `.gitignore`
- All training runs log params, loss curves, and eval metrics

### When James has MLflow on OpenShift
- Change `tracking_uri` in config to the remote URL
- Add `MLFLOW_TRACKING_URI` and `MLFLOW_TRACKING_TOKEN` as env vars
- Re-run baseline eval against remote MLflow to establish the comparison point
- Real-time tracking: MLflow UI shows loss curves as training progresses (both HF Trainer and Ray Train support this natively via the `mlflow` reporter)

### No migration needed
The local baseline run can just be re-run against the remote instance — same eval set, same model, deterministic results. Simpler than SQLite export/import.

---

## Part 7: Reading Material — Fine-Tuning Tool Calling with LoRA

Since you already understand LoRA fundamentals, these focus specifically on **how tool-calling behavior is learned and preserved during fine-tuning**.

### Papers on tool-calling fine-tuning

1. **"Gorilla: Large Language Model Connected with Massive APIs"** (Patil et al., 2023)
   - arxiv.org/abs/2305.15334
   - The most directly relevant paper. Fine-tunes LLaMA for API/tool calling using API documentation in the prompt + synthetic call examples. Key insight: the model learns to **map natural language intent to structured tool invocations** — exactly what we're doing (screenshot → tool call). They show that including API docs in the system prompt (like our `<tools>` block) is critical for generalization. Also demonstrates that retrieval-augmented fine-tuning (where the model sees tool definitions at train time matching inference) outperforms memorization-based approaches.

2. **"ToolLLM: Facilitating Large Language Models to Master 16000+ APIs"** (Qin et al., 2023)
   - arxiv.org/abs/2307.16789
   - Systematic approach to tool-calling fine-tuning with **synthetic data** — same strategy as this project. Key contributions: (a) Their DFSDT decision tree for multi-step tool use maps to our multi-turn format, (b) They demonstrate that **diverse synthetic examples matter more than volume** — 100 diverse examples outperform 1000 repetitive ones. Relevant to our 173-per-type synthetic set. (c) Shows how to handle the "no tool needed" case (our `no_tools` class).

3. **"LoRA Learns Less and Forgets Less"** (Biderman et al., 2024)
   - arxiv.org/abs/2405.09673
   - Addresses catastrophic forgetting specifically. Finding: LoRA at low ranks (like our rank=16) modifies attention patterns just enough to learn the new task routing without disrupting the model's general capabilities. For tool calling, this means the model learns "when I see this screen type, emit this tool call format" without forgetting how to understand images or generate coherent text.

4. **"Nexus Raven: a large language model for function calling"** (Nexusflow, 2023)
   - nexusflow.ai/blogs/ravenv2
   - Practical case study of fine-tuning specifically for function calling. They share training recipes: how to format tool definitions, how to structure the expected output, handling edge cases (wrong tool, no tool needed). Their format is very similar to the Hermes `<tool_call>` format we use.

### Practical guides

5. **NousResearch Hermes Function Calling format**
   - github.com/NousResearch/Hermes-Function-Calling
   - The `<tool_call>` format that Qwen2.5-VL uses and that `evaluation/tool_parser.py` parses. This is the target format our training data must produce. Understanding the exact XML structure matters — the model needs to see it consistently during training.

6. **Qwen2.5 tool calling documentation**
   - qwen.readthedocs.io (search for "tool calling" / "function calling")
   - Qwen-specific details on how the base model handles tools. Important: Qwen2.5-VL's chat template does NOT natively handle tools (unlike the text-only Qwen2.5), which is why we bake tool definitions into the system prompt. This is the correct approach for fine-tuning too.

### Key insights for this project

- **What the model actually learns during tool-calling LoRA:** The attention layers learn a routing pattern — "if the visual features match pattern X (e.g., a TV dialog box), attend to tool definition Y in the system prompt and emit the corresponding `<tool_call>` structure." The MLP layers (general knowledge) are untouched.
- **Why format consistency is critical:** The training data must use the exact same `<tool_call>` XML format the model will produce at inference. Any mismatch (e.g., JSON keys in different order, extra whitespace) can confuse the LoRA adapter. The eval parser in `tool_parser.py` is the reference.
- **The "no tool" case is hardest to learn:** Both Gorilla and ToolLLM found that teaching a model to abstain from calling tools is harder than teaching it to call the right one. Our baseline confirms this (no_tools at 36% vs pierre_shop at 100%). The 173 no_tools synthetic examples are essential.
- **Forgetting protection:** Monitor no_tools accuracy during training — if it drops below baseline (36%), the model is over-fitting to "always call a tool." Also spot-check general vision capability (can the model still describe non-Stardew images?) after training.
- **Learning rate for tool calling:** Start at 1e-4 (conservative). Tool calling is a relatively simple routing task for LoRA — it should converge fast. If loss plateaus early, try 2e-4. If loss oscillates, drop to 5e-5.

---

## Implementation Order

| Step | What | Where | Depends On |
|------|------|-------|------------|
| 1 | Consolidate Claude Code config into `.claude/` (memory, plans, settings) | `.claude/`, `.devcontainer/` | — |
| 2 | Add `mlflow.db` to `.gitignore` | `.gitignore` | — |
| 3 | Create `data_prep.py` (assemble splits, inject system prompt) | `fine_tuning/qwen/` | — |
| 4 | Create `data_collator.py` (ChatML + images for Qwen) | `fine_tuning/qwen/` | — |
| 5 | Create `lora_config_tiny.yaml` | `fine_tuning/qwen/` | — |
| 6 | Fix `train.py` TODOs, wire collator | `fine_tuning/qwen/train.py` | 3, 4, 5 |
| 7 | Tier 0 dry run | — | 6 |
| 8 | Tier 1 tiny training + eval | — | 7 |
| 9 | Create `infrastructure/` directory structure | `infrastructure/` | — |
| 10 | Tier 2 full training + eval | — | 8 |
| 11 | Create `train_ray.py` | `fine_tuning/qwen/` | 8 |
| 12 | S3 upload scripts | `scripts/s3/` | — |
| 13 | KubeRay manifests + cluster training | `infrastructure/kuberay/` | 11, 12 |
| 14 | Connect to remote MLflow (when James has it ready) | config change | — |

Steps 1-2 can be done immediately. Steps 3-5 are parallelizable. Steps 9 and 12 can happen anytime.

**After each step: ask the user if they want to git commit and push.**

---

## Verification

After each tier:
- **Tier 0:** `train.py` exits cleanly, loss is finite on both steps, loss changes between step 1 and 2, MLflow logs a run
- **Tier 1:** Training loss decreases between 2 steps. Run `python evaluation/eval_small_run.py` on checkpoint — compare per-class accuracy to baseline (70% overall, 36% no_tools, 52% caught_fish)
- **Tier 2:** Full eval with `python evaluation/run_baseline.py --model experiments/qwen-lora-v1/` — target >80% overall, >50% no_tools, >70% caught_fish
- **Tier 3:** Same eval, verify multi-GPU didn't degrade results vs Tier 2

---

## Key Files Reference

| File | Role |
|------|------|
| `fine_tuning/qwen/train.py:156-157` | TODOs blocking training |
| `evaluation/inference.py:41-78` | Working pattern for Qwen image+chat processing |
| `evaluation/prompt.py:14-85` | System prompt + tool definitions (must match training) |
| `datasets/synthetic/*/conversations.jsonl` | Training data (692 records, no system prompt) |
| `datasets/eval_set.json` | 100 eval images — never train on these |
| `experiments/eval-baseline-v1/results.json` | Baseline numbers to beat |
| `.devcontainer/setup-environment-existing.sh` | Add symlink block for Claude Code portability |

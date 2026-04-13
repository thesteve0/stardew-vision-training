# Migration Instructions

**Status**: Staging complete  
**Next step**: Move to host filesystem and initialize as git repository

---

## Migration Steps (Run on Host Machine)

### 1. Verify Staging Directory

On host, check that staging directory exists:

```bash
ls -la ~/path/to/workspace/stardew-vision/TRAINING_REPO_STAGING/
```

You should see:
- datasets/
- fine_tuning/
- evaluation/
- synthetic_data/
- experiments/
- scripts/
- docs/
- configs/
- tests/
- .devcontainer/
- .gitignore
- README.md
- CLAUDE.md
- GETTING_STARTED.md
- pyproject.toml

### 2. Move Staging to New Repository Location

```bash
# Navigate to workspace parent directory
cd ~/path/to/workspace/parent/

# Move staging directory
mv stardew-vision/TRAINING_REPO_STAGING stardew-vision-training

# Verify
ls -la stardew-vision-training/
```

### 3. Initialize Git Repository

```bash
cd stardew-vision-training

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Stardew Vision training repository

- Dataset structure for 6 screen types (pierre_shop, tv_dialog, caught_fish, quest_board, game_letter, level_up)
- Fine-tuning scripts for Qwen2.5-VL LoRA training
- Evaluation scripts for tool calling and narration quality
- Synthetic data generation tools
- Complete documentation and getting started guide
- ROCm 7.2 devcontainer configuration
"
```

### 4. Create GitHub Repository (Optional)

```bash
# Create private GitHub repo
gh repo create stardew-vision-training --private --source=. --remote=origin

# Or manually:
# 1. Create repo on GitHub
# 2. Add remote:
git remote add origin https://github.com/yourusername/stardew-vision-training.git

# Push
git branch -M main
git push -u origin main
```

### 5. Clean Up Original Repository

```bash
cd ../stardew-vision

# Add staging directory to gitignore (temporary - will be deleted)
echo "TRAINING_REPO_STAGING/" >> .gitignore

# Commit gitignore update
git add .gitignore
git commit -m "Add TRAINING_REPO_STAGING to gitignore (migration complete)"

# Remove staging directory (now that it's moved)
rm -rf TRAINING_REPO_STAGING/
```

### 6. Update Main Repository Documentation

In `stardew-vision` repo, update CLAUDE.md to reference training repo:

```markdown
## Related Repositories

**stardew-vision-training**: Model fine-tuning, dataset preparation, and evaluation
- Repository: https://github.com/yourusername/stardew-vision-training
- Purpose: VLM LoRA training, synthetic data generation, evaluation metrics
- Output: Fine-tuned LoRA adapters uploaded to HuggingFace Hub
```

### 7. Open Training Repo in VS Code

```bash
# Open training repo in VS Code
code stardew-vision-training/

# VS Code will prompt to reopen in devcontainer
# Click "Reopen in Container"
```

Wait for devcontainer build (~5-10 minutes first time).

### 8. Verify Training Environment

Once devcontainer is running:

```bash
# Activate virtual environment
uv sync
source .venv/bin/activate

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Expected output:
# True
# AMD Radeon Graphics
```

### 9. Next Steps in Training Repo

1. **Collect screenshots**: See `docs/phase2-collection-plan.md`
   - tv_dialog: 15-20 examples
   - caught_fish: 15-20 examples

2. **Build extraction tools**: (or use simplified OCR wrappers)

3. **Generate synthetic data**:
   ```bash
   python synthetic_data/generate_variations.py \
     --screen-type tv_dialog \
     --num-variations 100
   ```

4. **Fine-tune Qwen**:
   ```bash
   python fine_tuning/qwen/train.py \
     --config fine_tuning/qwen/lora_config.yaml
   ```

---

## What Was Migrated

### Copied from Main Repo
- **datasets/** — All screenshots, annotations, sprites, templates
- **docs/** — Training-relevant documentation (ADRs, OCR analysis, collection plan)
- **scripts/annotation/** — Screenshot organization and annotation tools

### Created New
- **fine_tuning/** — LoRA training scripts, configs
- **evaluation/** — Metric scripts (tool calling, narration quality)
- **synthetic_data/** — LLM-based variation generation
- **experiments/** — MLFlow tracking directory
- **README.md** — Training repo overview
- **CLAUDE.md** — Repository context for Claude Code
- **GETTING_STARTED.md** — Quick start guide
- **pyproject.toml** — Training dependencies
- **.devcontainer/** — ROCm development environment

### Stays in Main Repo Only
- **services/** — Coordinator, OCR tool, TTS tool
- **deploy/** — OpenShift manifests, Docker configs
- **configs/serving/** — KServe, vLLM configs
- **Production documentation** — deployment-status.md, LESSONS_LEARNED.md

---

## Repository Separation Benefits

✅ **Clean separation**: Application code vs. research/training code  
✅ **Independent lifecycles**: Deploy app without touching training experiments  
✅ **Workshop-ready**: Training repo is self-contained for conference demos  
✅ **Lower cognitive load**: Each repo has single, clear purpose  
✅ **Artifact handoff**: LoRA adapters uploaded to HF Hub, consumed by app

---

## Troubleshooting

### "Directory not found" during move

Check that you're in the correct parent directory and that `TRAINING_REPO_STAGING` exists.

### Git initialization fails

Ensure you have git installed and configured:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Devcontainer build fails

Check Docker is running and you have sufficient disk space (~20GB for ROCm image).

### GPU not detected in devcontainer

Verify ROCm installation on host:
```bash
rocm-smi
```

---

## Success Checklist

- [ ] Staging directory moved to `stardew-vision-training/`
- [ ] Git repository initialized
- [ ] Initial commit created
- [ ] GitHub repository created (optional)
- [ ] Main repo cleaned up (TRAINING_REPO_STAGING removed)
- [ ] Main repo CLAUDE.md updated with training repo reference
- [ ] Training repo opened in VS Code devcontainer
- [ ] Virtual environment verified (`uv sync`)
- [ ] GPU access verified (`torch.cuda.is_available()`)
- [ ] Ready to collect screenshots and start training!

---

## Questions?

See `README.md` and `GETTING_STARTED.md` in the training repository, or refer to main application documentation.

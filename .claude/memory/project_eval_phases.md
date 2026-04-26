---
name: Evaluation phases strategy
description: Phase 1 (tool selection) complete — best 97.6%; Phase 2 (text correction) is next after adapter deployment
type: project
---
Evaluation and training are split into two sequential phases:

**Phase 1: Tool Selection** — Complete. Best result: 97.6% (KubeFlow 2-GPU, 2x L40S). Baseline (untuned): 51.2%. All orchestration paths validated (local, Ray, KubeFlow).

**Phase 2: Text Correction** — After a tool returns OCR text, can the model clean it up and produce accurate narration? Text-to-text correction. Deferred until adapter is deployed to production app.

**Why:** If the model calls the wrong tool, nothing downstream matters. Tool selection is the gate.

**How to apply:** Don't mix Phase 2 concerns (narration quality, field extraction F1, perplexity) into Phase 1 evaluation. Keep them separate.

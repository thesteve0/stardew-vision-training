---
name: Evaluation phases strategy
description: Phase 1 (tool selection) complete — 97.6% on KubeFlow; Phase 2 (text correction) is next
type: project
originSessionId: 0b1c1454-7379-436e-a94a-f747fbf758da
---
Evaluation and training are split into two sequential phases:

**Phase 1: Tool Selection** — Complete. Best result: 97.6% (KubeFlow PyTorchJob, 2x L40S). Baseline (untuned): 51.2%. Both Ray and KubeFlow paths validated.

**Phase 2: Text Correction** — After a tool returns OCR text, can the model clean it up and produce accurate narration? Text-to-text correction. Next focus area.

**Why:** If the model calls the wrong tool, nothing downstream matters. Tool selection is the gate.

**How to apply:** Don't mix Phase 2 concerns (narration quality, field extraction F1, perplexity) into Phase 1 evaluation. Keep them separate.

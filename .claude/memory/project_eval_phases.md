---
name: Evaluation phases strategy
description: Phase 1 (tool selection) complete at 97.6% accuracy; Phase 2 (text correction) is next after cluster deployment
type: project
originSessionId: 1cac980d-2f8c-45f6-8704-b8fe85e81113
---
Evaluation and training are split into two sequential phases:

**Phase 1: Tool Selection** — Can the model call the correct extraction tool (or decline) given a screenshot? Pure classification. **COMPLETE** — 97.6% accuracy achieved (2026-04-23).

**Phase 2: Text Correction** — After a tool returns OCR text, can the model clean it up and produce accurate narration? Text-to-text correction. Comes after Phase 1 is deployed to cluster.

**Why:** If the model calls the wrong tool, nothing downstream matters. Tool selection is the gate.

**How to apply:** Don't mix Phase 2 concerns (narration quality, field extraction F1, perplexity) into Phase 1 evaluation. Keep them separate. Phase 1 is now about deploying to KubeRay, not further local training.

---
name: Evaluation phases strategy
description: Phase 1 (tool selection) needs fresh baseline after code alignment; Phase 2 (text correction) after cluster deployment
type: project
originSessionId: 561452f1-3c56-4b65-ab91-c8ad240d347b
---
Evaluation and training are split into two sequential phases:

**Phase 1: Tool Selection** — Can the model call the correct extraction tool (or decline) given a screenshot? Pure classification. Previously achieved 97.6% but results cleared for fresh baseline with corrected code (2026-04-25).

**Phase 2: Text Correction** — After a tool returns OCR text, can the model clean it up and produce accurate narration? Text-to-text correction. Comes after Phase 1 is deployed to cluster.

**Why:** If the model calls the wrong tool, nothing downstream matters. Tool selection is the gate.

**How to apply:** Don't mix Phase 2 concerns (narration quality, field extraction F1, perplexity) into Phase 1 evaluation. Keep them separate.

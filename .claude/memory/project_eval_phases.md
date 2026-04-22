---
name: Evaluation phases strategy
description: Evaluation is split into two phases — Phase 1 is tool selection accuracy, Phase 2 is text correction accuracy. Training follows the same phasing.
type: project
originSessionId: 30228c4f-4f55-4462-87a1-7dd56a52355f
---
Evaluation and training are split into two sequential phases:

**Phase 1: Tool Selection** — Can the model call the correct extraction tool (or decline) given a screenshot? Pure classification. This is the current focus.

**Phase 2: Text Correction** — After a tool returns OCR text, can the model clean it up and produce accurate narration? Text-to-text correction. Comes after Phase 1 training is complete.

**Why:** If the model calls the wrong tool, nothing downstream matters. Tool selection is the gate.

**How to apply:** Don't mix Phase 2 concerns (narration quality, field extraction F1, perplexity) into Phase 1 evaluation. Keep them separate.

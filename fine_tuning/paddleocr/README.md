# PaddleOCR Fine-Tuning (Future)

**Status**: Not implemented

**Rationale**: PaddleOCR (PP-OCRv5) performance is sufficient for current screen types. Fine-tuning may be needed for:
- Low-contrast text (game letters with brown text on tan background)
- Specialty pixel art fonts
- Unconventional UI layouts

**If needed**, fine-tuning would involve:
1. Collecting text region crops with ground truth labels
2. Training PaddleOCR recognition model on Stardew Valley fonts
3. Evaluating CER (Character Error Rate) improvement

**Placeholder for future work.**

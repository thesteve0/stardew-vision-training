"""
Caught fish notification OCR extraction tool.

Two-stage extraction from the fish-caught notification in Stardew Valley:
1. Crop the notification speech bubble from the full screenshot
2. Crop the fish sprite from the notification (for sprite matching)
3. Run OCR on the notification to extract length text

The fish name is NOT available via OCR (cropped by enlarged UI) — it must
be identified by sprite matching against datasets/assets/sprites/.

Architecture follows crop_pierres_detail_panel.py from the main repo.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Allow imports from tools-code directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    crop_regions,
    decode_image_b64,
    load_image_from_path,
    load_layout,
    run_ocr,
)

_LAYOUT_FILE = "caught_fish_layout.json"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FishNotFoundError(Exception):
    """Raised when the caught fish notification cannot be located or parsed."""


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

_LENGTH_PATTERN = re.compile(r"(\d+)\s*in", re.IGNORECASE)


def parse_caught_fish_fields(ocr_results: list[dict]) -> dict:
    """Parse OCR results from the caught fish notification.

    Extracts the fish length from "Length: NN in." text.
    Fish name identification is handled separately via sprite matching.

    Returns
    -------
    dict with keys: screen_type, length_inches, ocr_text
    """
    # OCR results are already in reading order from run_ocr()
    full_text = " ".join(rec["text"].strip() for rec in ocr_results if rec["text"].strip())

    # Extract length
    length_inches = None
    length_match = _LENGTH_PATTERN.search(full_text)
    if length_match:
        length_inches = int(length_match.group(1))

    return {
        "screen_type": "caught_fish",
        "length_inches": length_inches,
        "ocr_text": full_text,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def crop_caught_fish(image_b64: str, debug: bool = False) -> dict:
    """
    Extract information from a caught fish notification screenshot.

    Uses two-stage cropping: notification bubble from full image, then
    fish sprite from the notification. OCR runs on the notification
    to extract the fish length.

    Parameters
    ----------
    image_b64:
        Base64-encoded PNG or JPEG screenshot.
    debug:
        If True, include ``ocr_raw`` and ``fish_sprite_shape`` in the
        returned dict for inspection.

    Returns
    -------
    dict with keys: screen_type, length_inches, ocr_text.
    When debug=True, also includes: ocr_raw, fish_sprite_shape.

    Raises
    ------
    FishNotFoundError
        If OCR returns no text from the notification region.
    """
    img = decode_image_b64(image_b64)
    layout = load_layout(_LAYOUT_FILE)

    cropped = crop_regions(img, layout)
    notification_crop = cropped["notification"]
    fish_sprite_crop = cropped["fish_sprite"]

    # 3x upscale needed for the large game-font numbers in this notification
    ocr_results = run_ocr(notification_crop, upscale=3.0)

    if not ocr_results:
        raise FishNotFoundError(
            "OCR returned no text from the notification region. "
            "The caught fish notification may not be visible in this screenshot."
        )

    fields = parse_caught_fish_fields(ocr_results)

    if debug:
        fields["ocr_raw"] = sorted(ocr_results, key=lambda r: r["rel_y"])
        sprite_h, sprite_w = fish_sprite_crop.shape[:2]
        fields["fish_sprite_shape"] = [sprite_w, sprite_h]

    return fields


def crop_caught_fish_from_path(image_path: str | Path, debug: bool = False) -> dict:
    """Convenience wrapper that reads an image file from disk."""
    image_b64 = load_image_from_path(image_path)
    return crop_caught_fish(image_b64, debug=debug)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path> [--debug]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    debug = "--debug" in sys.argv

    result = crop_caught_fish_from_path(path, debug=debug)
    print(json.dumps(result, indent=2))

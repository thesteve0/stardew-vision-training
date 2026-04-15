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

import cv2
import numpy as np

from common import (
    crop_regions,
    decode_image_b64,
    load_fish_sprites,
    load_image_from_path,
    load_layout,
    load_manifest_fish,
    run_ocr,
)

_LAYOUT_FILE = "caught_fish_layout.json"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FishNotFoundError(Exception):
    """Raised when the caught fish notification cannot be located or parsed."""


# ---------------------------------------------------------------------------
# Fish sprite matching
# ---------------------------------------------------------------------------


def match_fish_sprite(
    fish_sprite_crop: np.ndarray,
    frame_margin: float = 0.15,
    scale_range: tuple[int, int] = (6, 18),
    match_threshold: float = 0.80,
) -> dict:
    """Identify the fish by template matching against the sprite library.

    Steps:
        1. Strip the wooden frame border (inner crop using frame_margin)
        2. Scale each of the 83 fish sprites up at multiple integer factors
        3. Run cv2.matchTemplate (TM_CCOEFF_NORMED) for each
        4. Return the best match above threshold, with fish name from manifest

    Parameters
    ----------
    fish_sprite_crop:
        Cropped image of the fish in its wooden frame (BGR numpy array).
    frame_margin:
        Fraction of width/height to trim on each side to remove the frame.
    scale_range:
        (min_scale, max_scale) integer range for nearest-neighbor upscaling.
    match_threshold:
        Minimum match score (0.0–1.0) to accept a result.

    Returns
    -------
    dict with keys: fish_name, sprite_index, match_score.
    fish_name is None if no match above threshold.
    """
    # Strip the wooden frame
    h, w = fish_sprite_crop.shape[:2]
    mx = int(w * frame_margin)
    my = int(h * frame_margin)
    inner = fish_sprite_crop[my : h - my, mx : w - mx]

    # Ensure BGR (3-channel)
    if inner.shape[2] == 4:
        inner = cv2.cvtColor(inner, cv2.COLOR_BGRA2BGR)

    fish_sprites = load_fish_sprites()
    fish_names = load_manifest_fish()

    best_score = -1.0
    best_index = -1
    best_scale = -1

    inner_h, inner_w = inner.shape[:2]

    for sprite_index, sprite_rgba in fish_sprites.items():
        # Composite RGBA sprite onto white background for matching
        if sprite_rgba.shape[2] == 4:
            alpha = sprite_rgba[:, :, 3:4].astype(np.float32) / 255.0
            bgr = sprite_rgba[:, :, :3].astype(np.float32)
            white_bg = np.full_like(bgr, 255.0)
            composited = (bgr * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        else:
            composited = sprite_rgba

        for scale in range(scale_range[0], scale_range[1]):
            scaled = cv2.resize(
                composited, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )

            # Template must fit inside search image
            if scaled.shape[0] > inner_h or scaled.shape[1] > inner_w:
                continue

            result = cv2.matchTemplate(inner, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_index = sprite_index
                best_scale = scale

    if best_score < match_threshold:
        return {
            "fish_name": None,
            "sprite_index": None,
            "match_score": round(float(best_score), 4),
        }

    return {
        "fish_name": fish_names.get(best_index, f"Unknown (sprite_{best_index})"),
        "sprite_index": best_index,
        "match_score": round(float(best_score), 4),
    }


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
    dict with keys: screen_type, fish_name, length_inches, ocr_text.
    When debug=True, also includes: ocr_raw, sprite_match (full match details).

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

    # Fish identification via sprite matching
    sprite_result = match_fish_sprite(fish_sprite_crop)
    fields["fish_name"] = sprite_result["fish_name"]

    if debug:
        fields["ocr_raw"] = sorted(ocr_results, key=lambda r: r["rel_y"])
        fields["sprite_match"] = sprite_result

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

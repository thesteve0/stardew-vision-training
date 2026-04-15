"""
TV dialog OCR extraction tool.

Crops the dialog text box from a Stardew Valley TV screen screenshot
and extracts the dialog text using PaddleOCR.

The dialog box is at a fixed position (defined in tv_dialog_layout.json),
so no template matching is needed — just crop and OCR.

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

_LAYOUT_FILE = "tv_dialog_layout.json"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DialogNotFoundError(Exception):
    """Raised when the TV dialog box cannot be located or OCR returns no text."""


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

# Known TV show identifiers
_TV_SHOW_PATTERNS = {
    "weather_forecast": re.compile(r"weather|forecast|KOZU\s*5", re.IGNORECASE),
    "fortune_teller": re.compile(r"fortune|spirits|luck", re.IGNORECASE),
    "queen_of_sauce": re.compile(r"queen\s*of\s*sauce|recipe|cooking", re.IGNORECASE),
    "livin_off_the_land": re.compile(
        r"livin|off\s*the\s*land|tip", re.IGNORECASE
    ),
}


def parse_tv_dialog_fields(ocr_results: list[dict]) -> dict:
    """Parse OCR results from a TV dialog box into structured fields.

    Returns
    -------
    dict with keys: screen_type, tv_show, dialog_text
    """
    # OCR results are already in reading order from run_ocr()
    dialog_text = " ".join(rec["text"].strip() for rec in ocr_results if rec["text"].strip())

    # Detect TV show type from content
    tv_show = "unknown"
    for show_name, pattern in _TV_SHOW_PATTERNS.items():
        if pattern.search(dialog_text):
            tv_show = show_name
            break

    return {
        "screen_type": "tv_dialog",
        "tv_show": tv_show,
        "dialog_text": dialog_text,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def crop_tv_dialog(image_b64: str, debug: bool = False) -> dict:
    """
    Extract dialog text from a TV screen screenshot.

    Parameters
    ----------
    image_b64:
        Base64-encoded PNG or JPEG screenshot.
    debug:
        If True, include ``ocr_raw`` in the returned dict.

    Returns
    -------
    dict with keys: screen_type, tv_show, dialog_text.
    When debug=True, also includes: ocr_raw (list[dict]).

    Raises
    ------
    DialogNotFoundError
        If OCR returns no text from the dialog region.
    """
    img = decode_image_b64(image_b64)
    layout = load_layout(_LAYOUT_FILE)

    cropped = crop_regions(img, layout)
    dialog_crop = cropped["dialog_box"]

    ocr_results = run_ocr(dialog_crop)

    if not ocr_results:
        raise DialogNotFoundError(
            "OCR returned no text from the dialog region. "
            "The dialog box may not be visible in this screenshot."
        )

    fields = parse_tv_dialog_fields(ocr_results)

    if debug:
        fields["ocr_raw"] = sorted(ocr_results, key=lambda r: r["rel_y"])

    return fields


def crop_tv_dialog_from_path(image_path: str | Path, debug: bool = False) -> dict:
    """Convenience wrapper that reads an image file from disk."""
    image_b64 = load_image_from_path(image_path)
    return crop_tv_dialog(image_b64, debug=debug)


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

    result = crop_tv_dialog_from_path(path, debug=debug)
    print(json.dumps(result, indent=2))

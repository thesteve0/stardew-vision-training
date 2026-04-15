"""
Shared utilities for Stardew Vision OCR extraction tools.

Provides lazy-loaded PaddleOCR, image decoding, region cropping,
and layout JSON loading used by all screen-type extraction tools.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Set PaddleX/PaddleOCR environment variables BEFORE any paddleocr imports
os.environ.setdefault("PADDLEX_HOME", "/tmp/.paddlex")
os.environ.setdefault("PADDLEX_CACHE_DIR", "/tmp/.paddlex/cache")
os.environ.setdefault("PADDLE_HUB_HOME", "/tmp/.paddlex/hub")
os.environ.setdefault("PADDLE_OCR_BASE_DIR", "/tmp/.paddleocr")

# Templates directory — layout JSONs live here
_TEMPLATES_DIR = Path(
    os.getenv(
        "TEMPLATES_DIR",
        str(Path(__file__).resolve().parent.parent / "datasets" / "assets" / "templates"),
    )
)

# ---------------------------------------------------------------------------
# PaddleOCR lazy loading
# ---------------------------------------------------------------------------

_OCR_INSTANCE = None


def load_ocr():
    """Lazy-load PaddleOCR PP-OCRv5 (CPU-only).

    The instance is cached module-level to avoid the ~30s model reload
    penalty between calls.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        from paddleocr import PaddleOCR

        _OCR_INSTANCE = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    return _OCR_INSTANCE


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def decode_image_b64(image_b64: str) -> np.ndarray:
    """Decode a base64-encoded image to a BGR numpy array."""
    img_bytes = base64.b64decode(image_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64 data.")
    return img


def load_image_from_path(image_path: str | Path) -> str:
    """Read an image file and return its base64-encoded contents."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Layout JSON
# ---------------------------------------------------------------------------


def load_layout(layout_filename: str) -> dict:
    """Load a layout JSON file from the templates directory.

    Parameters
    ----------
    layout_filename:
        Filename (not full path), e.g. "tv_dialog_layout.json".
        Looked up in the templates directory.

    Returns
    -------
    Parsed layout dict with screen_type, extracted_from_resolution, and regions.
    """
    layout_path = _TEMPLATES_DIR / layout_filename
    if not layout_path.exists():
        raise FileNotFoundError(
            f"Layout file not found at {layout_path}. "
            f"Expected in templates directory: {_TEMPLATES_DIR}"
        )
    with open(layout_path) as f:
        layout = json.load(f)

    required = ["screen_type", "extracted_from_resolution", "regions"]
    for key in required:
        if key not in layout:
            raise KeyError(f"Layout JSON missing required key '{key}': {layout_path}")

    return layout


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------


def crop_region(image: np.ndarray, region: dict) -> np.ndarray:
    """Crop a region from an image using relative coordinates.

    Parameters
    ----------
    image:
        Source image (BGR numpy array).
    region:
        Dict with keys x, y, w, h (all floats 0.0–1.0).

    Returns
    -------
    Cropped image as a numpy array.
    """
    img_h, img_w = image.shape[:2]

    x_px = int(region["x"] * img_w)
    y_px = int(region["y"] * img_h)
    w_px = int(region["w"] * img_w)
    h_px = int(region["h"] * img_h)

    # Clamp to image bounds
    x_px = max(0, min(x_px, img_w - 1))
    y_px = max(0, min(y_px, img_h - 1))
    w_px = min(w_px, img_w - x_px)
    h_px = min(h_px, img_h - y_px)

    return image[y_px : y_px + h_px, x_px : x_px + w_px]


def crop_regions(image: np.ndarray, layout: dict) -> dict[str, np.ndarray]:
    """Crop all regions defined in a layout, respecting parent relationships.

    Regions with a "parent" key are cropped from the parent's cropped image
    rather than from the full image.

    Returns a dict mapping region name to cropped numpy array.
    """
    regions = layout["regions"]
    cropped = {}

    # Pass 1: regions without a parent
    for name, region in regions.items():
        if "parent" not in region:
            cropped[name] = crop_region(image, region)

    # Pass 2: regions with a parent
    for name, region in regions.items():
        if "parent" in region:
            parent_name = region["parent"]
            if parent_name not in cropped:
                raise KeyError(
                    f"Region '{name}' references parent '{parent_name}' "
                    f"which was not found or has not been cropped yet."
                )
            cropped[name] = crop_region(cropped[parent_name], region)

    return cropped


# ---------------------------------------------------------------------------
# OCR runner
# ---------------------------------------------------------------------------


def run_ocr(
    cropped: np.ndarray,
    upscale: float = 2.0,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Run PaddleOCR on a cropped image region.

    Parameters
    ----------
    cropped:
        Cropped image (BGR numpy array).
    upscale:
        Scale factor before OCR (default 2x improves accuracy on game text).
    min_confidence:
        Minimum OCR confidence score (0.0–1.0). Text blocks below this
        threshold are discarded as noise.

    Returns
    -------
    List of dicts: {text, score, rel_x, rel_y} where rel_x and rel_y are
    the horizontal and vertical centres of the text block as fractions of
    the image dimensions. Results are sorted in reading order (top-to-bottom,
    then left-to-right within the same line).
    """
    ocr = load_ocr()
    upscaled = cv2.resize(
        cropped, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
    )
    panel_h = upscaled.shape[0]
    panel_w = upscaled.shape[1]
    result = ocr.predict(upscaled)

    records = []
    if not result or not result[0]:
        return records

    page = result[0]
    if not isinstance(page, dict):
        return records

    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    polys = page.get("rec_polys", [])

    for text, score, poly in zip(texts, scores, polys):
        if poly is None:
            continue
        if score < min_confidence:
            continue
        ys = [pt[1] for pt in poly]
        xs = [pt[0] for pt in poly]
        centre_y = (min(ys) + max(ys)) / 2
        centre_x = (min(xs) + max(xs)) / 2
        rel_y = centre_y / panel_h if panel_h > 0 else 0.0
        rel_x = centre_x / panel_w if panel_w > 0 else 0.0
        records.append({
            "text": text,
            "score": float(score),
            "rel_x": rel_x,
            "rel_y": rel_y,
        })

    # Sort in reading order: group into lines, then left-to-right within lines
    records = sort_reading_order(records)

    return records


def sort_reading_order(
    records: list[dict], line_threshold: float = 0.03
) -> list[dict]:
    """Sort OCR records into reading order (top-to-bottom, left-to-right).

    Text blocks whose rel_y values are within *line_threshold* of each other
    are considered to be on the same line and sorted left-to-right by rel_x.

    Parameters
    ----------
    records:
        List of OCR result dicts with rel_x and rel_y keys.
    line_threshold:
        Maximum rel_y difference for two blocks to be on the same line.
    """
    if not records:
        return records

    # Sort by y first
    by_y = sorted(records, key=lambda r: r["rel_y"])

    # Group into lines
    lines: list[list[dict]] = []
    current_line: list[dict] = [by_y[0]]

    for rec in by_y[1:]:
        if abs(rec["rel_y"] - current_line[0]["rel_y"]) <= line_threshold:
            current_line.append(rec)
        else:
            lines.append(current_line)
            current_line = [rec]
    lines.append(current_line)

    # Sort each line left-to-right, then flatten
    result = []
    for line in lines:
        result.extend(sorted(line, key=lambda r: r["rel_x"]))

    return result

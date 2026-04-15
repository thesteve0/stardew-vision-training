#!/usr/bin/env python3
"""
Crop testing tool for screen region bounding boxes.

Reads a layout JSON defining named regions with relative coordinates,
crops those regions from an input screenshot, and saves the cropped
images to a temp directory for visual inspection.

Usage:
    python scripts/test_crop_regions.py <image_path> --layout <layout.json>
    python scripts/test_crop_regions.py <image_path> --layout <layout.json> --output-dir /tmp/my_crops
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


def load_layout(layout_path: Path) -> dict:
    """Load and validate a layout JSON file."""
    with open(layout_path) as f:
        layout = json.load(f)

    required_keys = ["screen_type", "extracted_from_resolution", "regions"]
    for key in required_keys:
        if key not in layout:
            print(f"Error: layout JSON missing required key '{key}'", file=sys.stderr)
            sys.exit(1)

    if not isinstance(layout["regions"], dict) or len(layout["regions"]) == 0:
        print("Error: 'regions' must be a non-empty dict", file=sys.stderr)
        sys.exit(1)

    for name, region in layout["regions"].items():
        for coord in ["x", "y", "w", "h"]:
            if coord not in region:
                print(
                    f"Error: region '{name}' missing required coordinate '{coord}'",
                    file=sys.stderr,
                )
                sys.exit(1)
            val = region[coord]
            if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
                print(
                    f"Error: region '{name}' coordinate '{coord}' must be a float "
                    f"between 0.0 and 1.0, got {val}",
                    file=sys.stderr,
                )
                sys.exit(1)

    return layout


def crop_region(image, region: dict, img_width: int, img_height: int):
    """Crop a region from an image using relative coordinates.

    Returns the cropped image as a numpy array.
    """
    x_px = int(region["x"] * img_width)
    y_px = int(region["y"] * img_height)
    w_px = int(region["w"] * img_width)
    h_px = int(region["h"] * img_height)

    # Clamp to image bounds
    x_px = max(0, min(x_px, img_width - 1))
    y_px = max(0, min(y_px, img_height - 1))
    w_px = min(w_px, img_width - x_px)
    h_px = min(h_px, img_height - y_px)

    return image[y_px : y_px + h_px, x_px : x_px + w_px]


def main():
    parser = argparse.ArgumentParser(
        description="Crop testing tool for screen region bounding boxes"
    )
    parser.add_argument("image_path", type=Path, help="Path to the input screenshot")
    parser.add_argument(
        "--layout", type=Path, required=True, help="Path to the layout JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/crop_test"),
        help="Output directory for cropped images (default: /tmp/crop_test)",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"Error: image not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)
    if not args.layout.exists():
        print(f"Error: layout file not found: {args.layout}", file=sys.stderr)
        sys.exit(1)

    # Load layout
    layout = load_layout(args.layout)
    screen_type = layout["screen_type"]
    regions = layout["regions"]

    print(f"Loading layout: {args.layout}")
    print(f"  Screen type: {screen_type}")
    print(f"  Regions defined: {', '.join(regions.keys())}")

    # Load image
    image = cv2.imread(str(args.image_path))
    if image is None:
        print(f"Error: could not read image: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    img_height, img_width = image.shape[:2]
    print(f"\nLoading image: {args.image_path}")
    print(f"  Resolution: {img_width}x{img_height}")

    # Create output directory
    output_dir = args.output_dir / screen_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crop regions in two passes: parent regions first, then children
    cropped_images = {}
    print("\nCropping regions:")

    # Pass 1: regions without a parent (crop from full image)
    for name, region in regions.items():
        if "parent" in region:
            continue
        cropped = crop_region(image, region, img_width, img_height)
        cropped_images[name] = cropped
        crop_h, crop_w = cropped.shape[:2]

        output_path = output_dir / f"{name}.png"
        cv2.imwrite(str(output_path), cropped)

        desc = region.get("description", "")
        print(f"  {name:20s} -> {output_path} ({crop_w}x{crop_h})")
        if desc:
            print(f"  {'':20s}    {desc}")

    # Pass 2: regions with a parent (crop from parent's cropped image)
    for name, region in regions.items():
        if "parent" not in region:
            continue
        parent_name = region["parent"]
        if parent_name not in cropped_images:
            print(
                f"Error: region '{name}' references parent '{parent_name}' "
                f"which was not found",
                file=sys.stderr,
            )
            sys.exit(1)
        parent_img = cropped_images[parent_name]
        parent_h, parent_w = parent_img.shape[:2]
        cropped = crop_region(parent_img, region, parent_w, parent_h)
        cropped_images[name] = cropped
        crop_h, crop_w = cropped.shape[:2]

        output_path = output_dir / f"{name}.png"
        cv2.imwrite(str(output_path), cropped)

        desc = region.get("description", "")
        print(f"  {name:20s} -> {output_path} ({crop_w}x{crop_h})")
        print(f"  {'':20s}    (cropped from parent: {parent_name})")
        if desc:
            print(f"  {'':20s}    {desc}")

    print(f"\nDone. Inspect crops in {output_dir}/")


if __name__ == "__main__":
    main()

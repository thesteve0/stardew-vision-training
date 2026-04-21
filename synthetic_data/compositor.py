"""
Screenshot compositor for synthetic data generation.

Takes real Stardew Valley screenshots as background templates, clears
text regions, and renders new text using the bitmap font renderer.
Supports tv_dialog, caught_fish, and pierre_shop screen types.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic_data.font_renderer import SpriteFont, load_dialog_font, load_ui_font


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_letterbox(img: Image.Image, threshold: float = 10.0) -> Image.Image:
    """Remove black letterbox borders from a screenshot."""
    arr = np.array(img)
    row_means = arr.mean(axis=(1, 2))
    col_means = arr.mean(axis=(0, 2))

    non_black_rows = np.where(row_means > threshold)[0]
    non_black_cols = np.where(col_means > threshold)[0]

    if len(non_black_rows) == 0 or len(non_black_cols) == 0:
        return img

    top, bottom = non_black_rows[0], non_black_rows[-1] + 1
    left, right = non_black_cols[0], non_black_cols[-1] + 1

    if top == 0 and bottom == arr.shape[0] and left == 0 and right == arr.shape[1]:
        return img

    return Image.fromarray(arr[top:bottom, left:right])


def _sample_bg_color(
    img: Image.Image, region: tuple[int, int, int, int], margin: int = 5
) -> tuple[int, int, int]:
    """Sample the dominant background color from the edges of a region.

    Takes the median color from a thin strip along the right edge of the
    region (usually the least obstructed by text).
    """
    x1, y1, x2, y2 = region
    arr = np.array(img)
    # Sample from right edge strip
    strip = arr[y1 + margin : y2 - margin, x2 - margin - 5 : x2 - margin, :3]
    if strip.size == 0:
        return (237, 168, 96)  # fallback: tan/parchment
    median = np.median(strip.reshape(-1, 3), axis=0).astype(int)
    return tuple(median)


def _find_inner_region(
    img: Image.Image,
    outer: tuple[int, int, int, int],
    bg_threshold: tuple[int, int, int] = (180, 140, 80),
) -> tuple[int, int, int, int]:
    """Find the inner content area within a bordered UI element.

    Scans inward from the outer region edges to find where the
    background color (tan/parchment) begins, identifying the border
    thickness.

    Returns (x1, y1, x2, y2) of the inner content area.
    """
    ox1, oy1, ox2, oy2 = outer
    arr = np.array(img)[oy1:oy2, ox1:ox2, :3]

    r_thr, g_thr, b_thr = bg_threshold
    bg_mask = (arr[:, :, 0] > r_thr) & (arr[:, :, 1] > g_thr) & (arr[:, :, 2] > b_thr)

    # Find bounds of background-colored area
    bg_rows = np.where(bg_mask.any(axis=1))[0]
    bg_cols = np.where(bg_mask.any(axis=0))[0]

    if len(bg_rows) < 2 or len(bg_cols) < 2:
        # Fallback: use outer with small margin
        m = 10
        return (ox1 + m, oy1 + m, ox2 - m, oy2 - m)

    return (
        ox1 + int(bg_cols[0]),
        oy1 + int(bg_rows[0]),
        ox1 + int(bg_cols[-1]),
        oy1 + int(bg_rows[-1]),
    )


def _fit_text_scale(
    font: SpriteFont,
    text: str,
    area_w: int,
    area_h: int,
    max_scale: int = 6,
) -> int:
    """Find the largest integer scale where text fits the target area.

    Tries scales from max_scale down to 1, wrapping text at each scale
    and checking if the result fits within area_w x area_h.
    """
    for scale in range(max_scale, 0, -1):
        native_w = area_w // scale
        wrapped = font.wrap_text(text, native_w)
        rendered_w, rendered_h = font.measure_text(wrapped)
        if rendered_w * scale <= area_w and rendered_h * scale <= area_h:
            return scale
    return 1


# ---------------------------------------------------------------------------
# TV Dialog Compositor
# ---------------------------------------------------------------------------

# TV dialog rendering parameters (determined from screenshot analysis):
# - Font: SmallFont at scale=2 (line spacing = 33*2 = 66px)
# - Shadow: 1px offset, same text color at ~47% alpha
# - Text box: tan/parchment background, sized to fit text content
# - Bottom-anchored in the game viewport, right of the toolbar sidebar
# - Text padding inside box: ~16px top/bottom, ~16px left/right (at scale=2)
_TV_FONT_SCALE = 2
_TV_BOX_COLOR = (243, 191, 120)  # tan/parchment fill
_TV_TEXT_PAD_X = 16  # horizontal padding inside box (pixels at final scale)
_TV_TEXT_PAD_Y = 14  # vertical padding inside box
_TV_BOX_MARGIN_BOTTOM = 10  # margin from viewport bottom to box bottom

# Approximate text area width per resolution class (native pixels, before scale)
# Determined by matching word wrapping to real screenshots
_TV_NATIVE_TEXT_WIDTH = {
    "mobile_portrait": 350,   # 1640x2360
    "mobile_landscape": 520,  # 2360x1640
    "desktop": 380,           # 1600x1200
}


def _get_resolution_class(w: int, h: int) -> str:
    """Classify screenshot resolution for layout rules."""
    if w < h:
        return "mobile_portrait"
    elif w > 1800:
        return "mobile_landscape"
    return "desktop"


# Dialog box layout per resolution class (measured from original screenshots).
# box_left: left edge of the dialog box (anchored to toolbar right edge)
# box_right: right edge of the dialog box
# viewport_bottom: bottom of the game viewport
_TV_BOX_LAYOUT = {
    "mobile_portrait": {"box_left": 390, "box_right": 1230, "viewport_bottom": 1750},
    "mobile_landscape": {"box_left": 570, "box_right": 1770, "viewport_bottom": 1640},
    "desktop": {"box_left": 155, "box_right": 1200, "viewport_bottom": 1200},
}


def _get_tv_box_rect(
    img_w: int, img_h: int, text_h: int,
) -> tuple[int, int, int, int]:
    """Calculate the dialog text box rectangle (x1, y1, x2, y2).

    The box is:
    - Left-anchored against the toolbar boundary
    - Bottom-anchored at the game viewport bottom
    - Fixed width (matching the original dialog box width)
    - Height grows with text content
    """
    res = _get_resolution_class(img_w, img_h)
    layout = _TV_BOX_LAYOUT.get(res, _TV_BOX_LAYOUT["mobile_portrait"])

    box_x1 = layout["box_left"]
    box_x2 = layout["box_right"]
    viewport_bottom = min(layout["viewport_bottom"], img_h)

    box_h = text_h + _TV_TEXT_PAD_Y * 2
    box_y1 = viewport_bottom - box_h - _TV_BOX_MARGIN_BOTTOM
    box_y2 = viewport_bottom - _TV_BOX_MARGIN_BOTTOM

    return (box_x1, max(0, box_y1), box_x2, box_y2)


def composite_tv_dialog(
    background_path: str | Path,
    dialog_text: str,
    font: SpriteFont | None = None,
    text_color: tuple[int, int, int, int] = (86, 22, 12, 255),
) -> Image.Image:
    """Generate a TV dialog screenshot from a clean background.

    Renders a tan text box sized to fit the dialog text, positioned at
    the bottom of the game viewport. Uses SmallFont at 2x scale with
    a 1px drop shadow.

    Parameters
    ----------
    background_path:
        Path to a clean background image (no dialog box).
        Use images from ``datasets/tv_dialog/backgrounds/``.
    dialog_text:
        Dialog text to render.
    font:
        SpriteFont to use. Defaults to SmallFont.
    text_color:
        RGBA color for the rendered text.

    Returns
    -------
    PIL Image with the dialog text box composited onto the background.
    """
    if font is None:
        font = load_ui_font()

    img = Image.open(background_path).convert("RGB")
    w, h = img.size
    scale = _TV_FONT_SCALE

    # Get the fixed box rectangle for this resolution
    # (we need text height first, so do a preliminary wrap)
    res = _get_resolution_class(w, h)
    layout = _TV_BOX_LAYOUT.get(res, _TV_BOX_LAYOUT["mobile_portrait"])
    box_inner_w = layout["box_right"] - layout["box_left"] - _TV_TEXT_PAD_X * 2
    native_text_w = box_inner_w // scale

    wrapped = font.wrap_text(dialog_text, native_text_w)

    # Render text with drop shadow
    shadow_color = (86, 22, 12, 120)
    shadow_offset = (1, 1)
    text_img = font.render_text(
        wrapped, color=text_color, scale=scale,
        shadow_color=shadow_color, shadow_offset=shadow_offset,
    )

    # Get the box rectangle (height based on rendered text)
    bx1, by1, bx2, by2 = _get_tv_box_rect(w, h, text_img.height)
    box_w = bx2 - bx1
    box_h = by2 - by1

    # Create the tan text box
    box_img = Image.new("RGB", (box_w, box_h), _TV_BOX_COLOR)

    # Paste text onto the box with padding
    box_img.paste(text_img, (_TV_TEXT_PAD_X, _TV_TEXT_PAD_Y), text_img)

    # Composite onto background
    draw_img = img.copy()
    draw_img.paste(box_img, (bx1, by1))

    return draw_img


# ---------------------------------------------------------------------------
# Caught Fish Compositor
# ---------------------------------------------------------------------------

# Notification placement bounds (from 33 annotated screenshots)
_FISH_PLACEMENT = {
    "rel_x": (0.247, 0.326),  # x position as fraction of image width
    "rel_y": (0.000, 0.259),  # y position as fraction of image height
    "rel_w": (0.334, 0.491),  # notification width as fraction of image
    "rel_h": (0.103, 0.285),  # notification height as fraction of image
}
# Fish frame position within the notification (from annotations)
_FISH_FRAME_REL = {
    "x": 0.09,   # frame left edge within notification
    "y": 0.08,   # frame top edge within notification
    "w": 0.31,   # frame width as fraction of notification width
    "h": 0.65,   # frame height as fraction of notification height
}
_FISH_BUBBLE_COLOR = (240, 235, 225)  # off-white speech bubble
_FISH_FRAME_COLOR = (160, 100, 50)   # wooden frame border
_FISH_FRAME_BG = (210, 170, 110)     # interior of wooden frame
_FISH_POSITIONS_PATH = Path("datasets/caught_fish/backgrounds/positions_filtered.json")


def composite_caught_fish(
    background_path: str | Path,
    length_inches: int | None,
    fish_sprite_path: str | Path | None = None,
    font: SpriteFont | None = None,
    text_color: tuple[int, int, int, int] = (86, 22, 12, 255),
) -> Image.Image:
    """Generate a caught-fish notification on a clean background.

    Renders the notification bubble with fish sprite frame and length
    text, placed within the observed position bounds from real
    screenshots.

    Parameters
    ----------
    background_path:
        Path to a clean background image (no notification).
        Use images from ``datasets/caught_fish/backgrounds/``.
    length_inches:
        Fish length to display. None for trash items (no length shown).
    fish_sprite_path:
        Path to a 16x16 fish sprite PNG. If None, frame is shown empty.
    font:
        SpriteFont to use. Defaults to SmallFont.
    text_color:
        RGBA color for the rendered text.
    """
    if font is None:
        font = load_ui_font()

    img = Image.open(background_path).convert("RGB")
    w, h = img.size

    # Look up the recorded position for this background, or use random
    bg_name = Path(background_path).name
    notif_rect = _get_fish_notification_rect(bg_name, w, h)
    nx, ny, nw, nh = notif_rect

    # Build the notification image
    notif_img = Image.new("RGB", (nw, nh), _FISH_BUBBLE_COLOR)

    # Draw the fish frame (left portion of notification)
    fr = _FISH_FRAME_REL
    frame_x = int(fr["x"] * nw)
    frame_y = int(fr["y"] * nh)
    frame_w = int(fr["w"] * nw)
    frame_h = int(fr["h"] * nh)

    # Draw wooden frame border
    border = max(3, int(frame_w * 0.08))
    notif_arr = np.array(notif_img)
    # Frame border (dark wood)
    notif_arr[frame_y:frame_y + frame_h, frame_x:frame_x + frame_w, :3] = _FISH_FRAME_COLOR
    # Frame interior (lighter wood)
    notif_arr[
        frame_y + border : frame_y + frame_h - border,
        frame_x + border : frame_x + frame_w - border,
        :3,
    ] = _FISH_FRAME_BG
    notif_img = Image.fromarray(notif_arr)

    # Paste fish sprite inside the frame
    if fish_sprite_path is not None:
        sprite_img = Image.open(fish_sprite_path).convert("RGBA")
        inner_w = frame_w - border * 2
        inner_h = frame_h - border * 2
        sprite_scale = max(1, min(inner_w // sprite_img.width, inner_h // sprite_img.height))
        scaled_sprite = sprite_img.resize(
            (sprite_img.width * sprite_scale, sprite_img.height * sprite_scale),
            Image.NEAREST,
        )
        paste_x = frame_x + border + (inner_w - scaled_sprite.width) // 2
        paste_y = frame_y + border + (inner_h - scaled_sprite.height) // 2
        notif_img.paste(scaled_sprite, (paste_x, paste_y), scaled_sprite)

    # Render length text to the right of the frame
    if length_inches is not None and length_inches > 0:
        text_x = frame_x + frame_w + int(nw * 0.03)
        text_area_w = nw - text_x - int(nw * 0.03)
        text_area_h = nh - int(nh * 0.1)

        shadow_color = (86, 22, 12, 120)
        shadow_offset = (1, 1)

        length_text = f"Length:\n{length_inches} in."
        scale = _fit_text_scale(font, length_text, text_area_w, text_area_h)
        text_img = font.render_text(
            length_text, color=text_color, scale=scale,
            shadow_color=shadow_color, shadow_offset=shadow_offset,
        )

        # Center text vertically in the notification
        text_paste_y = (nh - text_img.height) // 2
        notif_img.paste(text_img, (text_x, max(0, text_paste_y)), text_img)

    # Composite notification onto background
    draw_img = img.copy()
    draw_img.paste(notif_img, (nx, ny))

    return draw_img


def _get_fish_notification_rect(
    bg_name: str, img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Get the notification placement rectangle for a background.

    Uses the recorded position from annotations if available,
    otherwise samples randomly within the observed bounds.
    """
    # Try to load recorded position
    if _FISH_POSITIONS_PATH.exists():
        import json
        with open(_FISH_POSITIONS_PATH) as f:
            positions = json.load(f)
        if bg_name in positions:
            p = positions[bg_name]["notification"]
            return (p["x"], p["y"], p["w"], p["h"])

    # Fall back to random placement within observed bounds
    pl = _FISH_PLACEMENT
    rx = random.uniform(*pl["rel_x"])
    ry = random.uniform(*pl["rel_y"])
    rw = random.uniform(*pl["rel_w"])
    rh = random.uniform(*pl["rel_h"])

    nx = int(rx * img_w)
    ny = int(ry * img_h)
    nw = int(rw * img_w)
    nh = int(rh * img_h)

    # Clamp to image bounds
    nw = min(nw, img_w - nx)
    nh = min(nh, img_h - ny)

    return (nx, ny, nw, nh)

    return draw_img


# ---------------------------------------------------------------------------
# Pierre's Shop Compositor
# ---------------------------------------------------------------------------

_PIERRE_PANEL = {"x": 0.6875, "y": 0.129, "w": 0.25, "h": 0.7875}
# Sub-regions within the panel (relative to panel)
_PIERRE_TEXT_TOP = 0.02  # start of text area from panel top
_PIERRE_TEXT_BOTTOM = 0.85  # end of text area (above the price bar)
_PIERRE_TEXT_HPAD = 0.05  # horizontal padding


def composite_pierre_shop(
    background_path: str | Path,
    item_name: str,
    description: str,
    price_per_unit: int,
    quantity: int | None = None,
    total_cost: int | None = None,
    energy: str = "",
    health: str = "",
    dialog_font: SpriteFont | None = None,
    ui_font: SpriteFont | None = None,
    text_color: tuple[int, int, int, int] = (86, 22, 12, 255),
) -> Image.Image:
    """Replace the item details in a Pierre's shop detail panel.

    Parameters
    ----------
    background_path:
        Path to a real Pierre's shop screenshot.
    item_name:
        New item name to display.
    description:
        New item description text.
    price_per_unit:
        Price per unit in gold.
    quantity:
        Selected quantity (shown in the price bar at bottom).
    total_cost:
        Total cost (price * quantity).
    energy, health:
        Energy and health values (e.g. "+13", "+5").
    dialog_font:
        Font for the item name. Defaults to SpriteFont1.
    ui_font:
        Font for description and prices. Defaults to SmallFont.
    text_color:
        RGBA color for the rendered text.
    """
    if dialog_font is None:
        dialog_font = load_dialog_font()
    if ui_font is None:
        ui_font = load_ui_font()

    img = Image.open(background_path).convert("RGB")
    img = _strip_letterbox(img)
    w, h = img.size

    # Locate the detail panel
    pp = _PIERRE_PANEL
    panel = (
        int(pp["x"] * w),
        int(pp["y"] * h),
        int((pp["x"] + pp["w"]) * w),
        int((pp["y"] + pp["h"]) * h),
    )
    px1, py1, px2, py2 = panel
    panel_w = px2 - px1
    panel_h = py2 - py1

    # Text region within the panel
    hpad = int(panel_w * _PIERRE_TEXT_HPAD)
    text_x1 = px1 + hpad
    text_y1 = py1 + int(panel_h * _PIERRE_TEXT_TOP)
    text_x2 = px2 - hpad
    text_y2 = py1 + int(panel_h * _PIERRE_TEXT_BOTTOM)
    text_w = text_x2 - text_x1

    # Sample and fill background
    bg_color = _sample_bg_color(img, (text_x1, text_y1, text_x2, text_y2))
    draw_img = img.copy()
    arr = np.array(draw_img)
    arr[text_y1:text_y2, text_x1:text_x2, :3] = bg_color
    draw_img = Image.fromarray(arr)

    # Pierre's panel uses SmallFont for everything at scale=1.
    # The game renders at native resolution and the panel is already
    # at the correct pixel size in the screenshot.
    # Shadow: same text color at ~50% alpha, 1px offset — blends with
    # the tan background to create the subtle multi-tone effect.
    shadow_color = (86, 22, 12, 120)
    shadow_offset = (1, 1)

    # Render item name
    name_img = ui_font.render_text(
        item_name, color=text_color,
        shadow_color=shadow_color, shadow_offset=shadow_offset,
    )
    y_cursor = text_y1 + 4
    draw_img.paste(name_img, (text_x1, y_cursor), name_img)
    y_cursor += name_img.height

    # Render description (word-wrapped to fit panel width)
    wrapped_desc = ui_font.wrap_text(description, text_w)
    desc_img = ui_font.render_text(
        wrapped_desc, color=text_color,
        shadow_color=shadow_color, shadow_offset=shadow_offset,
    )
    desc_img = desc_img.crop((0, 0, min(desc_img.width, text_w), desc_img.height))
    draw_img.paste(desc_img, (text_x1, y_cursor), desc_img)
    y_cursor += desc_img.height

    # Render unit price
    price_text = f"{price_per_unit}g"
    if energy:
        price_text += f"  Energy: {energy}"
    if health:
        price_text += f"  Health: {health}"
    price_img = ui_font.render_text(
        price_text, color=text_color,
        shadow_color=shadow_color, shadow_offset=shadow_offset,
    )
    draw_img.paste(price_img, (text_x1, y_cursor), price_img)

    # Clear and re-render the price bar at the bottom of the panel
    # The price bar shows "xQUANTITY: TOTAL" on a gold/brown bar
    if quantity is not None and total_cost is not None:
        bar_y1 = py1 + int(panel_h * 0.88)
        bar_y2 = py2 - int(panel_h * 0.02)
        bar_x1 = px1 + int(panel_w * 0.08)
        bar_x2 = px2 - int(panel_w * 0.08)

        # Sample and fill the bar background
        arr = np.array(draw_img)
        bar_bg = _sample_bg_color(draw_img, (bar_x1, bar_y1, bar_x2, bar_y2))
        arr[bar_y1:bar_y2, bar_x1:bar_x2, :3] = bar_bg
        draw_img = Image.fromarray(arr)

        # Render the quantity × total text
        bar_text = f"x{quantity}: {total_cost}g"
        bar_img = ui_font.render_text(
            bar_text, color=text_color,
            shadow_color=shadow_color, shadow_offset=shadow_offset,
        )
        # Center in the bar
        bar_w = bar_x2 - bar_x1
        bar_h = bar_y2 - bar_y1
        bar_paste_x = bar_x1 + (bar_w - bar_img.width) // 2
        bar_paste_y = bar_y1 + (bar_h - bar_img.height) // 2
        draw_img.paste(bar_img, (bar_paste_x, bar_paste_y), bar_img)

    return draw_img


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

_SCREEN_COMPOSITORS = {
    "tv_dialog": composite_tv_dialog,
    "caught_fish": composite_caught_fish,
    "pierre_shop": composite_pierre_shop,
}


def get_random_background(screen_type: str, images_dir: str | Path | None = None) -> Path:
    """Pick a random real screenshot to use as a background template."""
    if images_dir is None:
        images_dir = Path(f"datasets/{screen_type}/images")
    else:
        images_dir = Path(images_dir)

    images = list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return random.choice(images)

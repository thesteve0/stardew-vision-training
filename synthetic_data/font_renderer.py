"""
Bitmap font renderer for Stardew Valley's XNA SpriteFont format.

Parses the JSON metadata + PNG sprite sheet from unpacked game assets
and renders text as PIL Images matching the in-game appearance.

Supports SmallFont (UI text, lineSpacing=33) and SpriteFont1 (dialog
text, lineSpacing=50).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class GlyphInfo:
    """Rendering data for a single character."""

    char: str
    # Source rectangle in the font sprite sheet
    src_x: int
    src_y: int
    src_w: int
    src_h: int
    # Offset within the line-height cell where the glyph is drawn
    crop_x: int
    crop_y: int
    # Kerning: left bearing, glyph width region, right bearing
    kern_left: int
    kern_width: int
    kern_right: int


class SpriteFont:
    """XNA SpriteFont renderer using game bitmap font assets.

    Parameters
    ----------
    font_json_path:
        Path to the unpacked SpriteFont JSON (e.g. SmallFont.json).
    font_png_path:
        Path to the corresponding PNG sprite sheet. If None, derived
        from the JSON's ``texture.export`` field.
    """

    def __init__(self, font_json_path: str | Path, font_png_path: str | Path | None = None):
        font_json_path = Path(font_json_path)
        with open(font_json_path) as f:
            data = json.load(f)

        content = data["content"]
        self.line_spacing: int = content["verticalLineSpacing"]
        self.char_spacing: int = content["horizontalSpacing"]
        self.default_char: str = content.get("defaultCharacter", "*")

        # Load sprite sheet
        if font_png_path is None:
            font_png_path = font_json_path.parent / content["texture"]["export"]
        self._sheet = Image.open(font_png_path).convert("RGBA")

        # Build character → GlyphInfo lookup
        char_map = content["characterMap"]
        glyphs = content["glyphs"]
        cropping = content["cropping"]
        kerning = content["kerning"]

        self._glyphs: dict[str, GlyphInfo] = {}
        for i, char in enumerate(char_map):
            g = glyphs[i]
            cr = cropping[i]
            k = kerning[i]
            self._glyphs[char] = GlyphInfo(
                char=char,
                src_x=g["x"],
                src_y=g["y"],
                src_w=g["width"],
                src_h=g["height"],
                crop_x=cr["x"],
                crop_y=cr["y"],
                kern_left=k["x"],
                kern_width=k["y"],
                kern_right=k["z"],
            )

    def _get_glyph(self, char: str) -> GlyphInfo:
        """Look up glyph info, falling back to the default character."""
        return self._glyphs.get(char) or self._glyphs.get(self.default_char)

    def measure_char(self, char: str) -> int:
        """Return the horizontal advance width for a single character."""
        g = self._get_glyph(char)
        if g is None:
            return 0
        return max(g.kern_left, 0) + g.kern_width + g.kern_right + self.char_spacing

    def measure_text(self, text: str) -> tuple[int, int]:
        """Return (width, height) in pixels for a string (may contain newlines)."""
        lines = text.split("\n")
        max_width = 0
        for line in lines:
            w = sum(self.measure_char(c) for c in line)
            max_width = max(max_width, w)
        height = len(lines) * self.line_spacing
        return max_width, height

    def wrap_text(self, text: str, max_width: int) -> str:
        """Word-wrap text to fit within max_width pixels.

        Splits at word boundaries. If a single word exceeds max_width,
        it is placed on its own line without breaking.
        """
        paragraphs = text.split("\n")
        wrapped_lines = []

        for paragraph in paragraphs:
            words = paragraph.split(" ")
            current_line = ""
            current_width = 0
            space_width = self.measure_char(" ")

            for word in words:
                word_width = sum(self.measure_char(c) for c in word)
                if current_line and current_width + space_width + word_width > max_width:
                    wrapped_lines.append(current_line)
                    current_line = word
                    current_width = word_width
                elif current_line:
                    current_line += " " + word
                    current_width += space_width + word_width
                else:
                    current_line = word
                    current_width = word_width

            wrapped_lines.append(current_line)

        return "\n".join(wrapped_lines)

    def render_text(
        self,
        text: str,
        color: tuple[int, int, int, int] = (86, 22, 12, 255),
        scale: int = 1,
        shadow_color: tuple[int, int, int, int] | None = None,
        shadow_offset: tuple[int, int] = (2, 2),
    ) -> Image.Image:
        """Render text as a PIL RGBA Image.

        Parameters
        ----------
        text:
            The string to render (may contain newlines).
        color:
            RGBA color to apply to the glyphs. The font atlas contains
            white-on-transparent glyphs; this color replaces the white.
        scale:
            Integer scale factor applied via nearest-neighbor after
            rendering, matching the game's pixel-art upscaling.
        shadow_color:
            If set, render a drop shadow behind the text in this color.
        shadow_offset:
            (dx, dy) pixel offset for the drop shadow at native scale.

        Returns
        -------
        PIL RGBA Image containing the rendered text.
        """
        width, height = self.measure_text(text)
        # Add space for shadow offset if needed
        if shadow_color is not None:
            width += abs(shadow_offset[0])
            height += abs(shadow_offset[1])

        if width <= 0 or height <= 0:
            return Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))

        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Render shadow first (behind the text) at reduced opacity
        if shadow_color is not None:
            shadow_alpha = shadow_color[3] / 255.0 if len(shadow_color) == 4 else 1.0
            self._draw_text_on_canvas(
                canvas, text, shadow_color, shadow_offset,
                alpha_scale=shadow_alpha,
            )

        # Render foreground text
        self._draw_text_on_canvas(canvas, text, color, (0, 0))

        if scale > 1:
            canvas = canvas.resize(
                (width * scale, height * scale),
                Image.NEAREST,
            )

        return canvas

    def _draw_text_on_canvas(
        self,
        canvas: Image.Image,
        text: str,
        color: tuple[int, int, int, int],
        offset: tuple[int, int],
        alpha_scale: float = 1.0,
    ) -> None:
        """Draw text glyphs onto an existing canvas at the given offset.

        Parameters
        ----------
        alpha_scale:
            Multiplier for the glyph alpha channel (0.0-1.0). Used for
            semi-transparent shadow rendering.
        """
        width, height = canvas.size
        lines = text.split("\n")
        ox, oy = offset

        for line_idx, line in enumerate(lines):
            cursor_x = 0
            base_y = line_idx * self.line_spacing

            for char in line:
                g = self._get_glyph(char)
                if g is None:
                    continue

                cursor_x += max(g.kern_left, 0)

                if g.src_w > 0 and g.src_h > 0:
                    glyph_img = self._sheet.crop((
                        g.src_x, g.src_y,
                        g.src_x + g.src_w, g.src_y + g.src_h,
                    ))
                    glyph_img = _colorize(glyph_img, color, alpha_scale)

                    paste_x = ox + cursor_x + g.crop_x
                    paste_y = oy + base_y + g.crop_y

                    if 0 <= paste_x < width and 0 <= paste_y < height:
                        canvas.paste(glyph_img, (paste_x, paste_y), glyph_img)

                cursor_x += g.kern_width + g.kern_right + self.char_spacing


def _colorize(
    glyph: Image.Image,
    color: tuple[int, int, int, int],
    alpha_scale: float = 1.0,
) -> Image.Image:
    """Replace white pixels with the target color, preserving alpha.

    Parameters
    ----------
    alpha_scale:
        Multiplier for the alpha channel (0.0-1.0). Used for
        semi-transparent shadow rendering.
    """
    pixels = glyph.load()
    w, h = glyph.size
    r, g, b = color[0], color[1], color[2]

    for py in range(h):
        for px in range(w):
            _, _, _, pa = pixels[px, py]
            if pa > 0:
                scaled_alpha = max(0, min(255, int(pa * alpha_scale)))
                pixels[px, py] = (r, g, b, scaled_alpha)

    return glyph


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

_FONT_DIR = Path("datasets/assets/game_files/unpacked")


def load_dialog_font(font_dir: str | Path | None = None) -> SpriteFont:
    """Load SpriteFont1 (dialog/notification text, lineSpacing=50)."""
    d = Path(font_dir) if font_dir else _FONT_DIR
    return SpriteFont(d / "SpriteFont1.json")


def load_ui_font(font_dir: str | Path | None = None) -> SpriteFont:
    """Load SmallFont (UI text like prices and descriptions, lineSpacing=33)."""
    d = Path(font_dir) if font_dir else _FONT_DIR
    return SpriteFont(d / "SmallFont.json")

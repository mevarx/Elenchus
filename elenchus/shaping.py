"""Devanagari complex-script shaping for the bill generator.

VERIFIED ON THIS MACHINE (Windows 11, Pillow 12.3.0): stock Pillow has
NO raqm — `features.check('raqm')` is False, wheels are fully static,
and Layout.RAQM silently degrades to BASIC. Without shaping, क्ष renders
as क + virama + ष — disconnected conjuncts that degrade VLM reading.

The fix (proven end-to-end): shape with uharfbuzz, render glyph-by-glyph
through freetype-py into PIL. Matras position correctly, conjuncts join.

    pip install uharfbuzz freetype-py
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

try:
    import uharfbuzz as hb
    import freetype
    _SHAPING_AVAILABLE = True
except ImportError:
    _SHAPING_AVAILABLE = False

FONT_DEVA = "C:/Windows/Fonts/Nirmala.ttc"
FONT_LATIN = "C:/Windows/Fonts/arial.ttf"

_hb_font = None
_upem = 2048
_ft_faces: dict = {}


def shaping_available() -> bool:
    return _SHAPING_AVAILABLE and os.path.isfile(FONT_DEVA)


def _get_hb_font():
    global _hb_font, _upem
    if _hb_font is None:
        face = hb.Face(hb.Blob.from_file_path(FONT_DEVA), 0)
        _upem = face.upem
        _hb_font = hb.Font(face)
        _hb_font.scale = (_upem, _upem)
    return _hb_font


def _get_ft_face(face_index: int = 0):
    if face_index not in _ft_faces:
        _ft_faces[face_index] = freetype.Face(FONT_DEVA, face_index)
    return _ft_faces[face_index]


def draw_shaped(img: Image.Image, pos: tuple, text: str, size: int,
                fill: tuple = (20, 20, 20), bold: bool = False) -> None:
    """Render Devanagari (or any complex script) with proper shaping.

    pos is the TOP-LEFT of the text line (PIL convention), converted to
    a baseline internally so it composes with plain draw.text calls.
    """
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(_get_hb_font(), buf, {"kern": True, "liga": True})

    ft = _get_ft_face(1 if bold else 0)
    ft.set_pixel_sizes(0, size)
    ascent = ft.size.ascender / 64.0

    pen_x, top = pos
    baseline = top + ascent
    for info, p in zip(buf.glyph_infos, buf.glyph_positions):
        ft.load_glyph(info.codepoint)
        g = ft.glyph
        g.render(freetype.FT_RENDER_MODE_NORMAL)
        bm = g.bitmap
        if bm.rows and bm.width:
            glyph_img = Image.frombytes("L", (bm.width, bm.rows),
                                        bytes(bm.buffer))
            x = round(pen_x + p.x_offset * size / _upem + g.bitmap_left)
            y = round(baseline - p.y_offset * size / _upem - g.bitmap_top)
            # tint to the fill color
            solid = Image.new("L", (bm.width, bm.rows), 255)
            colored = Image.new("RGB", (bm.width, bm.rows), fill)
            mask = solid.point(lambda v: 0)  # placeholder, replaced below
            mask = glyph_img
            region = img.crop((x, y, x + bm.width, y + bm.rows))
            region_rgb = Image.composite(colored, region, mask)
            img.paste(region_rgb, (x, y))
        pen_x += p.x_advance * size / _upem


def text_length(text: str, size: int, bold: bool = False) -> int:
    """Width of shaped text — for right-alignment."""
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(_get_hb_font(), buf, {"kern": True, "liga": True})
    return round(sum(p.x_advance * size / _upem
                     for p in buf.glyph_positions))


class TextRenderer:
    """Unified text drawing: shaped for Devanagari, plain for Latin.

    Falls back to plain PIL text when uharfbuzz is unavailable — the
    generator still works, the conjuncts just render unjoined (which is
    itself the measured finding: numbers read fine, descriptions garble).
    """

    def __init__(self, img: Image.Image, draw: ImageDraw.ImageDraw,
                 latin_font_path: str = FONT_LATIN):
        self.img = img
        self.draw = draw
        self.latin_font_path = latin_font_path
        self._pil_fonts: dict = {}
        self._use_shaping = shaping_available()

    def _pil_font(self, size: int, bold: bool) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self._pil_fonts:
            path = self.latin_font_path
            if bold and os.path.isfile(path.replace(".ttf", "bd.ttf")):
                path = path.replace(".ttf", "bd.ttf")
            try:
                self._pil_fonts[key] = ImageFont.truetype(path, size)
            except Exception:
                self._pil_fonts[key] = ImageFont.load_default()
        return self._pil_fonts[key]

    def _is_devanagari(self, text: str) -> bool:
        return any("\u0900" <= ch <= "\u097F" for ch in text)

    def text(self, pos: tuple, text: str, size: int,
             fill: tuple = (20, 20, 20), bold: bool = False) -> None:
        if self._use_shaping and self._is_devanagari(text):
            draw_shaped(self.img, pos, text, size, fill, bold)
        else:
            self.draw.text(pos, text, font=self._pil_font(size, bold),
                           fill=fill)

    def text_right(self, right_x: int, pos_y: int, text: str, size: int,
                   fill: tuple = (20, 20, 20), bold: bool = False) -> None:
        """Right-aligned text (for the amount column)."""
        if self._use_shaping and self._is_devanagari(text):
            w = text_length(text, size, bold)
            self.text((right_x - w, pos_y), text, size, fill, bold)
        else:
            f = self._pil_font(size, bold)
            w = self.draw.textlength(text, font=f)
            self.draw.text((right_x - w, pos_y), text, font=f, fill=fill)

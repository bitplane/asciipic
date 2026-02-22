import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _find_fallback_font(char: str) -> str | None:
    """Ask fontconfig which font provides a given character."""
    codepoint = f"{ord(char):04x}"
    result = subprocess.run(
        ["fc-match", "-f", "%{file}", f":charset={codepoint}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _fit_fallback_font(fallback_path: str, cell_height: int) -> tuple[ImageFont.FreeTypeFont, int]:
    """Find the largest font size for a fallback font that fits within cell_height."""
    for size in range(cell_height, 4, -1):
        font = ImageFont.truetype(fallback_path, size)
        bbox = font.getbbox("M")
        if bbox[3] - bbox[1] <= cell_height:
            return font, -bbox[1]
    return ImageFont.truetype(fallback_path, 6), 0


def _render_char(char: str, font: ImageFont.FreeTypeFont, y_offset: int, cell_width: int, cell_height: int) -> bool:
    """Check if a font renders a character distinctly (not a blank fallback rectangle)."""
    img = Image.new("L", (cell_width, cell_height), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, y_offset), char, fill=255, font=font)
    return np.asarray(img).sum() > 0


def build_atlas(
    font_path: str,
    font_size: int,
    characters: str,
) -> tuple[list[str], np.ndarray, int, int]:
    """Pre-render characters as binary masks for differentiable rendering.

    Uses fontconfig fallback for characters not properly rendered by the primary font.

    Returns:
        char_list: ordered list of characters
        masks: float32 array of shape (num_chars, cell_h, cell_w), values 0-1
        cell_width: pixel width of one character cell
        cell_height: pixel height of one character cell
    """
    primary_font = ImageFont.truetype(font_path, font_size)
    bbox = primary_font.getbbox("M")
    cell_width = bbox[2] - bbox[0]
    cell_height = bbox[3] - bbox[1]
    primary_y_offset = -bbox[1]

    # Cache fallback fonts so we only look them up once per font file
    fallback_cache: dict[str, tuple[ImageFont.FreeTypeFont, int]] = {}

    char_list = list(characters)
    masks = np.zeros((len(char_list), cell_height, cell_width), dtype=np.float32)

    # First, detect which chars the primary font renders identically (broken glyphs).
    # Render two different chars that should look different — if identical, need fallback.
    # We do this by checking a pair of chars that share a unicode block.
    broken_blocks: set[str] = set()

    for i, char in enumerate(char_list):
        img = Image.new("L", (cell_width, cell_height), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, primary_y_offset), char, fill=255, font=primary_font)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # Check if this might be a broken fallback glyph by seeing if a
        # very different character in the same block renders identically
        if char not in (" ", "\t") and arr.sum() > 0:
            # Quick check: is this char in a block we already know is broken?
            block_start = ord(char) & 0xFFFFFF00
            if block_start in broken_blocks:
                arr = _render_with_fallback(char, cell_width, cell_height, fallback_cache)
            else:
                # Compare with a different char in the same block
                other_code = ord(char) ^ 0x01  # flip lowest bit
                if chr(other_code) in characters and other_code != ord(char):
                    other_img = Image.new("L", (cell_width, cell_height), 0)
                    other_draw = ImageDraw.Draw(other_img)
                    other_draw.text((0, primary_y_offset), chr(other_code), fill=255, font=primary_font)
                    other_arr = np.asarray(other_img, dtype=np.float32) / 255.0
                    if np.array_equal(arr, other_arr) and arr.sum() > 0:
                        broken_blocks.add(block_start)
                        arr = _render_with_fallback(char, cell_width, cell_height, fallback_cache)

        masks[i] = arr

    return char_list, masks, cell_width, cell_height


def _render_with_fallback(
    char: str,
    cell_width: int,
    cell_height: int,
    cache: dict[str, tuple[ImageFont.FreeTypeFont, int]],
) -> np.ndarray:
    """Render a character using fontconfig fallback, centered in the cell."""
    fallback_path = _find_fallback_font(char)
    if fallback_path is None:
        return np.zeros((cell_height, cell_width), dtype=np.float32)

    if fallback_path not in cache:
        cache[fallback_path] = _fit_fallback_font(fallback_path, cell_height)
    fallback_font, fb_y_offset = cache[fallback_path]

    img = Image.new("L", (cell_width, cell_height), 0)
    draw = ImageDraw.Draw(img)
    # Center the glyph horizontally
    gb = fallback_font.getbbox(char)
    gw = gb[2] - gb[0]
    x_offset = (cell_width - gw) // 2
    draw.text((x_offset, fb_y_offset), char, fill=255, font=fallback_font)
    return np.asarray(img, dtype=np.float32) / 255.0

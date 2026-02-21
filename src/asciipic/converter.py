from pathlib import Path

from PIL import Image

from asciipic.model import FontModel
from asciipic.sampling import enhance_contrast, sample_colours, sample_grid


def _format_colour(lines: list[str], colours) -> str:
    """Wrap each character in ANSI truecolor escape sequences."""
    out = []
    for r, line in enumerate(lines):
        parts = []
        for c, char in enumerate(line):
            br, bg, bb = int(colours[r, c, 0, 0]), int(colours[r, c, 0, 1]), int(colours[r, c, 0, 2])
            fr, fg, fb = int(colours[r, c, 1, 0]), int(colours[r, c, 1, 1]), int(colours[r, c, 1, 2])
            parts.append(f"\033[38;2;{fr};{fg};{fb}m\033[48;2;{br};{bg};{bb}m{char}")
        parts.append("\033[0m")
        out.append("".join(parts))
    return "\n".join(out)


def image_to_ascii(
    image: Image.Image | str | Path,
    model: FontModel,
    width: int | None = None,
    exponent: float | None = None,
    colour: bool = False,
) -> str:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    cw = model.cell_width
    ch = model.cell_height

    if width is not None:
        new_pixel_width = width * cw
        scale = new_pixel_width / image.width
        # Terminal characters are taller than wide; compensate so output isn't stretched
        aspect_correction = cw / ch
        new_pixel_height = int(image.height * scale * aspect_correction)
        image = image.resize((new_pixel_width, new_pixel_height), Image.LANCZOS)

    cols = image.width // cw
    rows = image.height // ch

    if rows == 0 or cols == 0:
        return ""

    gray = image.convert("L")
    grid = sample_grid(gray, cw, ch) / 255.0
    if exponent is not None:
        grid = enhance_contrast(grid, exponent)
    lines = model.find_nearest_grid(grid)

    if colour:
        colours = sample_colours(image, cw, ch)
        return _format_colour(lines, colours)
    return "\n".join(lines)

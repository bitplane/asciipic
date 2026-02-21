from pathlib import Path

from PIL import Image

from asciipic.engine import Engine


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
    engine: Engine,
    width: int | None = None,
    colour: bool = False,
) -> str:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    cw = engine.cell_width
    ch = engine.cell_height

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

    result = engine.render(image)

    if colour:
        return _format_colour(result.chars, result.colours)
    return "\n".join(result.chars)

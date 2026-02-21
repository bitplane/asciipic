from pathlib import Path

from PIL import Image

from asciipic.model import FontModel
from asciipic.sampling import sample_vector


def image_to_ascii(
    image: Image.Image | str | Path,
    model: FontModel,
    width: int | None = None,
) -> str:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("L")

    cw = model.cell_width
    ch = model.cell_height

    if width is not None:
        new_pixel_width = width * cw
        scale = new_pixel_width / image.width
        new_pixel_height = int(image.height * scale)
        image = image.resize((new_pixel_width, new_pixel_height), Image.LANCZOS)

    cols = image.width // cw
    rows = image.height // ch

    lines = []
    for row in range(rows):
        chars = []
        for col in range(cols):
            x0 = col * cw
            y0 = row * ch
            cell = image.crop((x0, y0, x0 + cw, y0 + ch))
            raw = sample_vector(cell, cw, ch)
            normalized = tuple(v / 255.0 for v in raw)
            chars.append(model.find_nearest(normalized))
        lines.append("".join(chars))

    return "\n".join(lines)

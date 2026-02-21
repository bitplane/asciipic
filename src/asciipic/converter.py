from pathlib import Path

from PIL import Image

from asciipic.model import FontModel
from asciipic.sampling import enhance_contrast, sample_grid


def image_to_ascii(
    image: Image.Image | str | Path,
    model: FontModel,
    width: int | None = None,
    exponent: float | None = None,
) -> str:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("L")

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

    grid = sample_grid(image, cw, ch) / 255.0
    if exponent is not None:
        grid = enhance_contrast(grid, exponent)
    return "\n".join(model.find_nearest_grid(grid))

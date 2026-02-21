from PIL import Image, ImageDraw, ImageFont

from asciipic.model import FontModel
from asciipic.sampling import sample_vector


def generate_model(
    font_path: str,
    characters: str,
    font_size: int = 16,
) -> FontModel:
    font = ImageFont.truetype(font_path, font_size)

    # Measure cell size from a reference character
    bbox = font.getbbox("M")
    cell_width = bbox[2] - bbox[0]
    cell_height = bbox[3] - bbox[1]
    y_offset = -bbox[1]

    raw_vectors: dict[str, list[float]] = {}
    for char in characters:
        img = Image.new("L", (cell_width, cell_height), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, y_offset), char, fill=255, font=font)
        raw_vectors[char] = list(sample_vector(img, cell_width, cell_height))

    # Normalize each dimension to 0–1
    maxes = [0.0] * 6
    for vector in raw_vectors.values():
        for i, v in enumerate(vector):
            maxes[i] = max(maxes[i], v)

    characters_normalized: dict[str, tuple[float, ...]] = {}
    for char, vector in raw_vectors.items():
        characters_normalized[char] = tuple(v / m if m > 0 else 0.0 for v, m in zip(vector, maxes))

    return FontModel(
        font_name=font_path,
        font_size=font_size,
        cell_width=cell_width,
        cell_height=cell_height,
        characters=characters_normalized,
    )

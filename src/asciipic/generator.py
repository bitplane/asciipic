from PIL import Image, ImageDraw, ImageFont

from asciipic.model import FontModel

# Circle centers as fractions of (cell_width, cell_height)
SAMPLE_POSITIONS = [
    (0.25, 0.167),  # UL
    (0.75, 0.167),  # UR
    (0.25, 0.5),  # ML
    (0.75, 0.5),  # MR
    (0.25, 0.833),  # LL
    (0.75, 0.833),  # LR
]


def _sample_circle(image: Image.Image, cx: float, cy: float, radius: float) -> float:
    pixels = image.load()
    w, h = image.size
    total = 0.0
    count = 0
    r2 = radius * radius
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius) + 1)
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius) + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                total += pixels[x, y]
                count += 1
    return total / count if count else 0.0


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

    radius = min(cell_width, cell_height) * 0.25

    raw_vectors: dict[str, list[float]] = {}
    for char in characters:
        img = Image.new("L", (cell_width, cell_height), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, y_offset), char, fill=255, font=font)
        vector = [_sample_circle(img, cx * cell_width, cy * cell_height, radius) for cx, cy in SAMPLE_POSITIONS]
        raw_vectors[char] = vector

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

from PIL import Image

# Circle centers as fractions of (cell_width, cell_height)
SAMPLE_POSITIONS = [
    (0.25, 0.167),  # UL
    (0.75, 0.167),  # UR
    (0.25, 0.5),  # ML
    (0.75, 0.5),  # MR
    (0.25, 0.833),  # LL
    (0.75, 0.833),  # LR
]


def sample_circle(image: Image.Image, cx: float, cy: float, radius: float) -> float:
    """Average brightness of pixels within a circle."""
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


def sample_vector(image: Image.Image, cell_width: int, cell_height: int) -> tuple[float, ...]:
    """Sample all 6 positions on a cell-sized image, returning raw brightness values."""
    radius = min(cell_width, cell_height) * 0.25
    return tuple(sample_circle(image, cx * cell_width, cy * cell_height, radius) for cx, cy in SAMPLE_POSITIONS)

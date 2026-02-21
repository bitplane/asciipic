import numpy as np
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


def _build_circle_masks(cell_width: int, cell_height: int) -> list[np.ndarray]:
    """Pre-compute boolean circle masks for each sample position."""
    radius = min(cell_width, cell_height) * 0.25
    r2 = radius * radius
    ys = np.arange(cell_height)[:, None]
    xs = np.arange(cell_width)[None, :]
    masks = []
    for cx_frac, cy_frac in SAMPLE_POSITIONS:
        cx = cx_frac * cell_width
        cy = cy_frac * cell_height
        masks.append((xs - cx) ** 2 + (ys - cy) ** 2 <= r2)
    return masks


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


def sample_grid(image: Image.Image, cell_width: int, cell_height: int) -> np.ndarray:
    """Sample all cells in an image at once. Returns array of shape (rows, cols, 6)."""
    arr = np.asarray(image, dtype=np.float64)
    masks = _build_circle_masks(cell_width, cell_height)
    rows = arr.shape[0] // cell_height
    cols = arr.shape[1] // cell_width

    # Trim to exact grid and reshape into (rows, cell_h, cols, cell_w)
    trimmed = arr[: rows * cell_height, : cols * cell_width]
    cells = trimmed.reshape(rows, cell_height, cols, cell_width).transpose(0, 2, 1, 3)
    # cells is now (rows, cols, cell_h, cell_w)

    result = np.empty((rows, cols, len(masks)))
    for i, mask in enumerate(masks):
        # mask is (cell_h, cell_w), broadcast across all cells
        masked = cells * mask
        result[:, :, i] = masked.sum(axis=(2, 3)) / mask.sum()

    return result

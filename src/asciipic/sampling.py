import numpy as np
from PIL import Image

# Circle centers as fractions of (cell_width, cell_height) — 3x5 grid
SAMPLE_POSITIONS = [
    (0.167, 0.1),
    (0.5, 0.1),
    (0.833, 0.1),
    (0.167, 0.3),
    (0.5, 0.3),
    (0.833, 0.3),
    (0.167, 0.5),
    (0.5, 0.5),
    (0.833, 0.5),
    (0.167, 0.7),
    (0.5, 0.7),
    (0.833, 0.7),
    (0.167, 0.9),
    (0.5, 0.9),
    (0.833, 0.9),
]
SAMPLE_COLS = 3
SAMPLE_ROWS = 5
NUM_SAMPLES = len(SAMPLE_POSITIONS)

# For each neighbor direction: (axis, shift_direction, affected_indices, mirror_indices)
# affected_indices are the border sample positions influenced by this neighbor,
# mirror_indices are the corresponding positions in the neighbor cell.
_NEIGHBOR_MAPS = [
    (1, -1, [0, 3, 6, 9, 12], [2, 5, 8, 11, 14]),  # left neighbor
    (1, 1, [2, 5, 8, 11, 14], [0, 3, 6, 9, 12]),  # right neighbor
    (0, -1, [0, 1, 2], [12, 13, 14]),  # above neighbor
    (0, 1, [12, 13, 14], [0, 1, 2]),  # below neighbor
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
    """Sample all positions on a cell-sized image, returning raw brightness values."""
    radius = min(cell_width, cell_height) * 0.25
    return tuple(sample_circle(image, cx * cell_width, cy * cell_height, radius) for cx, cy in SAMPLE_POSITIONS)


def sample_grid(image: Image.Image, cell_width: int, cell_height: int) -> np.ndarray:
    """Sample all cells in an image at once. Returns array of shape (rows, cols, NUM_SAMPLES)."""
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


def enhance_contrast(grid: np.ndarray, exponent: float) -> np.ndarray:
    """Apply directional contrast enhancement using neighboring cells.

    For border sample positions, compares with the mirror position in the
    adjacent cell and applies a power curve to increase local contrast.
    """
    rows, cols, _ = grid.shape
    external_max = np.zeros_like(grid)

    for axis, direction, affected, mirrors in _NEIGHBOR_MAPS:
        pad_widths = [(0, 0)] * 3
        if direction == -1:
            pad_widths[axis] = (1, 0)
        else:
            pad_widths[axis] = (0, 1)
        padded = np.pad(grid, pad_widths, constant_values=0.0)

        slices = [slice(None)] * 3
        if direction == -1:
            slices[axis] = slice(0, padded.shape[axis] - 1)
        else:
            slices[axis] = slice(1, padded.shape[axis])
        neighbor = padded[tuple(slices)]

        for a_idx, m_idx in zip(affected, mirrors):
            np.maximum(external_max[:, :, a_idx], neighbor[:, :, m_idx], out=external_max[:, :, a_idx])

    max_val = np.maximum(grid, external_max)
    safe_max = np.where(max_val > 0, max_val, 1.0)
    return (grid / safe_max) ** exponent * max_val


def sample_colours(image: Image.Image, cell_width: int, cell_height: int) -> np.ndarray:
    """Compute per-cell foreground and background colours from an RGB image.

    Splits each cell's pixels by brightness threshold (mean), averages the
    bright pixels to get foreground and dark pixels to get background.

    Returns array of shape (rows, cols, 2, 3) as uint8 where [:,:,0,:] is bg
    and [:,:,1,:] is fg.
    """
    arr = np.asarray(image, dtype=np.float64)
    rows = arr.shape[0] // cell_height
    cols = arr.shape[1] // cell_width

    trimmed = arr[: rows * cell_height, : cols * cell_width]
    # (rows, cols, cell_h, cell_w, 3)
    cells = trimmed.reshape(rows, cell_height, cols, cell_width, 3).transpose(0, 2, 1, 3, 4)

    # Per-cell mean brightness from RGB
    brightness = cells.mean(axis=4)  # (rows, cols, cell_h, cell_w)
    threshold = brightness.mean(axis=(2, 3), keepdims=True)  # (rows, cols, 1, 1)
    bright_mask = brightness > threshold  # (rows, cols, cell_h, cell_w)
    dark_mask = ~bright_mask

    result = np.zeros((rows, cols, 2, 3), dtype=np.uint8)
    for ch in range(3):
        channel = cells[:, :, :, :, ch]  # (rows, cols, cell_h, cell_w)

        bright_sum = (channel * bright_mask).sum(axis=(2, 3))
        bright_count = bright_mask.sum(axis=(2, 3))
        dark_sum = (channel * dark_mask).sum(axis=(2, 3))
        dark_count = dark_mask.sum(axis=(2, 3))

        # Average, falling back to overall mean when one group is empty
        cell_mean = channel.mean(axis=(2, 3))
        fg = np.where(bright_count > 0, bright_sum / np.maximum(bright_count, 1), cell_mean)
        bg = np.where(dark_count > 0, dark_sum / np.maximum(dark_count, 1), cell_mean)

        result[:, :, 0, ch] = np.clip(bg, 0, 255).astype(np.uint8)
        result[:, :, 1, ch] = np.clip(fg, 0, 255).astype(np.uint8)

    return result

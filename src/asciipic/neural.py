from pathlib import Path

import numpy as np
from PIL import Image

from asciipic.engine import CellGrid
from asciipic.glyph_atlas import build_atlas


class NeuralEngine:
    """Rendering engine using a trained MLP with 3x3 cell context. Pure numpy inference."""

    def __init__(self, weights_path: str | Path):
        data = np.load(weights_path, allow_pickle=True)

        self.char_list = list(data["char_list"])
        self.cell_width = int(data["cell_width"])
        self.cell_height = int(data["cell_height"])
        self.downsample_h = int(data["downsample_h"])
        self.downsample_w = int(data["downsample_w"])

        font_path = str(data["font_path"])
        font_size = int(data["font_size"])
        _, self.atlas, _, _ = build_atlas(font_path, font_size, "".join(self.char_list))

        # Load MLP weights
        self.w_shared = data["weights.shared.weight"]  # (hidden, flat_size)
        self.b_shared = data["weights.shared.bias"]  # (hidden,)
        self.w_char = data["weights.char_head.weight"]  # (num_chars, hidden)
        self.b_char = data["weights.char_head.bias"]  # (num_chars,)
        self.w_fg = data["weights.fg_head.weight"]  # (3, hidden)
        self.b_fg = data["weights.fg_head.bias"]  # (3,)
        self.w_bg = data["weights.bg_head.weight"]  # (3, hidden)
        self.b_bg = data["weights.bg_head.bias"]  # (3,)

    def _forward(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run MLP on a batch of 3x3 cell patches.

        Args:
            patches: (N, 3*cell_h, 3*cell_w, 3) float32 in [0, 1]

        Returns:
            char_indices: (N,) int
            fg: (N, 3) float32 in [0, 1]
            bg: (N, 3) float32 in [0, 1]
        """
        n = patches.shape[0]
        src_h, src_w = patches.shape[1], patches.shape[2]
        dst_h, dst_w = self.downsample_h, self.downsample_w

        # Area-average downsample
        downsampled = np.empty((n, dst_h, dst_w, 3), dtype=np.float32)
        for dy in range(dst_h):
            y0 = dy * src_h // dst_h
            y1 = (dy + 1) * src_h // dst_h
            for dx in range(dst_w):
                x0 = dx * src_w // dst_w
                x1 = (dx + 1) * src_w // dst_w
                downsampled[:, dy, dx, :] = patches[:, y0:y1, x0:x1, :].mean(axis=(1, 2))
        x = downsampled.transpose(0, 3, 1, 2).reshape(n, -1)

        # Shared layer + ReLU
        h = x @ self.w_shared.T + self.b_shared
        h = np.maximum(h, 0)

        # Heads
        char_logits = h @ self.w_char.T + self.b_char
        char_indices = np.argmax(char_logits, axis=1)

        fg = 1.0 / (1.0 + np.exp(-(h @ self.w_fg.T + self.b_fg)))
        bg = 1.0 / (1.0 + np.exp(-(h @ self.w_bg.T + self.b_bg)))

        return char_indices, fg, bg

    def render(self, image: Image.Image) -> CellGrid:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        rows = arr.shape[0] // self.cell_height
        cols = arr.shape[1] // self.cell_width
        cw, ch = self.cell_width, self.cell_height

        # Pad by 1 cell on each side (reflect to avoid black border artifacts)
        padded = np.pad(arr, ((ch, ch), (cw, cw), (0, 0)), mode="reflect")

        # Extract 3x3 cell patches centered on each cell
        patches = np.empty((rows * cols, 3 * ch, 3 * cw, 3), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                # In padded coords, center cell starts at (r+1)*ch, (c+1)*cw
                # so the 3x3 patch starts 1 cell earlier
                y = r * ch
                x = c * cw
                patches[r * cols + c] = padded[y : y + 3 * ch, x : x + 3 * cw]

        char_indices, fg, bg = self._forward(patches)

        # Build output
        chars = []
        for r in range(rows):
            row_start = r * cols
            chars.append("".join(self.char_list[char_indices[row_start + c]] for c in range(cols)))

        colours = np.zeros((rows, cols, 2, 3), dtype=np.uint8)
        fg_reshaped = (fg.reshape(rows, cols, 3) * 255).clip(0, 255).astype(np.uint8)
        bg_reshaped = (bg.reshape(rows, cols, 3) * 255).clip(0, 255).astype(np.uint8)
        colours[:, :, 0, :] = bg_reshaped
        colours[:, :, 1, :] = fg_reshaped

        return CellGrid(chars=chars, colours=colours)

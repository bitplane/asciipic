from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image

from asciipic.engine import CellGrid


class NeuralEngine:
    """Rendering engine using a trained MLP with 3x3 cell context. Pure numpy inference."""

    def __init__(self, weights_path: str | Path):
        data = np.load(weights_path, allow_pickle=True)

        self.char_list = list(data["char_list"])
        self.cell_width = int(data["cell_width"])
        self.cell_height = int(data["cell_height"])
        self.downsample_h = int(data["downsample_h"])
        self.downsample_w = int(data["downsample_w"])

        # Load MLP weights
        self.w_shared = data["weights.shared.weight"]  # (hidden, flat_size)
        self.b_shared = data["weights.shared.bias"]  # (hidden,)
        self.w_char = data["weights.char_head.weight"]  # (num_chars, hidden)
        self.b_char = data["weights.char_head.bias"]  # (num_chars,)
        self.w_fg = data["weights.fg_head.weight"]  # (3, hidden)
        self.b_fg = data["weights.fg_head.bias"]  # (3,)
        self.w_bg = data["weights.bg_head.weight"]  # (3, hidden)
        self.b_bg = data["weights.bg_head.bias"]  # (3,)

    def _forward(self, flat_input: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run MLP on pre-flattened input.

        Args:
            flat_input: (N, flat_size) float32

        Returns:
            char_indices: (N,) int
            fg: (N, 3) float32 in [0, 1]
            bg: (N, 3) float32 in [0, 1]
        """
        # Shared layer + ReLU
        h = flat_input @ self.w_shared.T + self.b_shared
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
        dst_h, dst_w = self.downsample_h, self.downsample_w

        # Pad by 1 cell on each side (reflect to avoid black border artifacts)
        padded = np.pad(arr, ((ch, ch), (cw, cw), (0, 0)), mode="reflect")

        # Resize the entire padded image to the downsampled grid in one PIL call
        padded_h, padded_w = padded.shape[:2]
        # Each cell maps to (dst_h/3, dst_w/3) pixels in downsampled space
        # padded is (rows+2) cells tall and (cols+2) cells wide
        ds_total_h = (rows + 2) * dst_h // 3
        ds_total_w = (cols + 2) * dst_w // 3
        padded_img = Image.fromarray((padded * 255).astype(np.uint8))
        small = padded_img.resize((ds_total_w, ds_total_h), Image.BOX)
        small_arr = np.asarray(small, dtype=np.float32) / 255.0  # (ds_total_h, ds_total_w, 3)

        # Extract downsampled 3x3 patches using strided view (no Python loop)
        cell_ds_h = dst_h // 3
        cell_ds_w = dst_w // 3
        s = small_arr.strides
        patches = as_strided(
            small_arr,
            shape=(rows, cols, dst_h, dst_w, 3),
            strides=(cell_ds_h * s[0], cell_ds_w * s[1], s[0], s[1], s[2]),
        )
        # CHW ordering per patch, then flatten to (N, flat_size)
        flat_input = np.ascontiguousarray(patches.transpose(0, 1, 4, 2, 3)).reshape(rows * cols, -1)

        char_indices, fg, bg = self._forward(flat_input)

        # Build output
        char_arr = np.array(self.char_list)
        chars = ["".join(char_arr[char_indices[r * cols : (r + 1) * cols]]) for r in range(rows)]

        colours = np.zeros((rows, cols, 2, 3), dtype=np.uint8)
        fg_reshaped = (fg.reshape(rows, cols, 3) * 255).clip(0, 255).astype(np.uint8)
        bg_reshaped = (bg.reshape(rows, cols, 3) * 255).clip(0, 255).astype(np.uint8)
        colours[:, :, 0, :] = bg_reshaped
        colours[:, :, 1, :] = fg_reshaped

        return CellGrid(chars=chars, colours=colours)

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image

from asciipic.engine import CellGrid


def _conv2d(x, weight, bias, stride=1, padding=0):
    """Conv2d forward pass in numpy.

    Args:
        x: (N, C_in, H, W) float32
        weight: (C_out, C_in, kH, kW) float32
        bias: (C_out,) float32
    """
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    N, C_in, H_pad, W_pad = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H_pad - kH) // stride + 1
    W_out = (W_pad - kW) // stride + 1

    # im2col via stride tricks
    s = x.strides
    col = as_strided(
        x,
        shape=(N, H_out, W_out, C_in, kH, kW),
        strides=(s[0], s[2] * stride, s[3] * stride, s[1], s[2], s[3]),
    )
    col_flat = np.ascontiguousarray(col).reshape(N * H_out * W_out, C_in * kH * kW)
    weight_flat = weight.reshape(C_out, -1)

    out = col_flat @ weight_flat.T + bias
    return out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)


def _adaptive_avg_pool2d(x, output_size):
    """AdaptiveAvgPool2d in numpy. x: (N, C, H, W)."""
    N, C, H, W = x.shape
    oH, oW = output_size
    out = np.empty((N, C, oH, oW), dtype=x.dtype)
    for i in range(oH):
        h0 = i * H // oH
        h1 = (i + 1) * H // oH
        for j in range(oW):
            w0 = j * W // oW
            w1 = (j + 1) * W // oW
            out[:, :, i, j] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
    return out


class NeuralEngine:
    """Rendering engine using a trained CNN+MLP with 3x3 cell context. Pure numpy inference."""

    def __init__(self, weights_path: str | Path):
        data = np.load(weights_path, allow_pickle=True)

        self.char_list = list(data["char_list"])
        self.cell_width = int(data["cell_width"])
        self.cell_height = int(data["cell_height"])

        # Load conv weights
        self.conv1_w = data["weights.conv.0.weight"]  # (64, 1, 3, 3)
        self.conv1_b = data["weights.conv.0.bias"]  # (64,)
        self.conv2_w = data["weights.conv.2.weight"]  # (256, 64, 3, 3)
        self.conv2_b = data["weights.conv.2.bias"]  # (256,)

        # Load MLP weights
        self.w_shared = data["weights.shared.weight"]  # (hidden, 256)
        self.b_shared = data["weights.shared.bias"]  # (hidden,)
        self.w_char = data["weights.char_head.weight"]  # (num_chars, hidden)
        self.b_char = data["weights.char_head.bias"]  # (num_chars,)
        self.w_invert = data["weights.invert_head.weight"]  # (2, hidden)
        self.b_invert = data["weights.invert_head.bias"]  # (2,)

    def _forward(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run CNN+MLP on binarized input patches.

        Args:
            patches: (N, patch_h, patch_w) float32, values 0.0 or 1.0

        Returns:
            char_indices: (N,) int
            invert: (N,) bool
        """
        x = patches[:, np.newaxis, :, :]  # (N, 1, H, W)

        # Conv frontend → (N, 256, 3, 3), extract center position
        x = np.maximum(_conv2d(x, self.conv1_w, self.conv1_b, padding=1), 0)
        x = np.maximum(_conv2d(x, self.conv2_w, self.conv2_b, stride=2, padding=1), 0)
        x = _adaptive_avg_pool2d(x, (3, 3))
        x = x[:, :, 1, 1]  # center position → (N, 256)

        # Shared layer + ReLU
        h = x @ self.w_shared.T + self.b_shared
        h = np.maximum(h, 0)

        # Heads
        char_logits = h @ self.w_char.T + self.b_char
        char_indices = np.argmax(char_logits, axis=1)
        invert_logits = h @ self.w_invert.T + self.b_invert  # (N, 2)
        invert = invert_logits[:, 1] > invert_logits[:, 0]  # class 1 = inverted

        return char_indices, invert

    def render(self, image: Image.Image) -> CellGrid:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        rows = arr.shape[0] // self.cell_height
        cols = arr.shape[1] // self.cell_width
        cw, ch = self.cell_width, self.cell_height
        patch_h, patch_w = 3 * ch, 3 * cw

        # Pad by 1 cell on each side (reflect to avoid black border artifacts)
        padded = np.pad(arr, ((ch, ch), (cw, cw), (0, 0)), mode="reflect")

        # Extract 3x3 cell patches at full resolution using strided view
        s = padded.strides
        patches_rgb = as_strided(
            padded,
            shape=(rows, cols, patch_h, patch_w, 3),
            strides=(ch * s[0], cw * s[1], s[0], s[1], s[2]),
        )
        patches_rgb = np.ascontiguousarray(patches_rgb).reshape(rows * cols, patch_h, patch_w, 3)

        # Binarize patches for CNN input (same method as training data)
        gray = patches_rgb.mean(axis=-1)  # (N, patch_h, patch_w)
        center_gray = gray[:, ch : 2 * ch, cw : 2 * cw]  # (N, ch, cw)
        threshold = center_gray.mean(axis=(1, 2), keepdims=True)  # (N, 1, 1)
        binary = (gray > threshold).astype(np.float32)

        char_indices, invert = self._forward(binary)

        # Compute per-cell colours from original RGB center cells
        center_cells = patches_rgb[:, ch : 2 * ch, cw : 2 * cw, :]  # (N, ch, cw, 3)
        bright_mask = center_gray > threshold  # reuse already-computed values
        dark_mask = ~bright_mask

        bright_mask_3 = bright_mask[..., np.newaxis]  # (N, ch, cw, 1)
        dark_mask_3 = dark_mask[..., np.newaxis]
        bright_sum = (center_cells * bright_mask_3).sum(axis=(1, 2))
        bright_count = bright_mask.sum(axis=(1, 2))[:, np.newaxis].clip(min=1)
        bright = bright_sum / bright_count  # (N, 3)
        dark_sum = (center_cells * dark_mask_3).sum(axis=(1, 2))
        dark_count = dark_mask.sum(axis=(1, 2))[:, np.newaxis].clip(min=1)
        dark = dark_sum / dark_count  # (N, 3)

        # Assign fg/bg based on invert flag
        fg = np.where(invert[:, np.newaxis], dark, bright)
        bg = np.where(invert[:, np.newaxis], bright, dark)

        # Build output
        char_arr = np.array(self.char_list)
        chars = ["".join(char_arr[char_indices[r * cols : (r + 1) * cols]]) for r in range(rows)]

        colours = np.zeros((rows, cols, 2, 3), dtype=np.uint8)
        fg_reshaped = (fg.reshape(rows, cols, 3) * 255).clip(0, 255).astype(np.uint8)
        bg_reshaped = (bg.reshape(rows, cols, 3) * 255).clip(0, 255).astype(np.uint8)
        colours[:, :, 0, :] = bg_reshaped
        colours[:, :, 1, :] = fg_reshaped

        return CellGrid(chars=chars, colours=colours)

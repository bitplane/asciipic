"""Training script for the neural ASCII rendering engine.

Trains an MLP to map 3x3 cell patches to character + fg/bg colour predictions
for the center cell, using a differentiable renderer and pixel-level MSE loss.

Usage:
    python -m asciipic.generator.neural --data /path/to/images --output data/neural/weights.npz
"""

import argparse
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from asciipic.charsets import VISUAL
from asciipic.glyph_atlas import build_atlas

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
PATCHES_PER_IMAGE = 100
TARGET_COLS = 120
CONTEXT = 3  # 3x3 cells per patch


def _cache_path(data_dir: str | Path) -> Path:
    """Build cache path from the data directory, encoding slashes as dashes."""
    resolved = Path(data_dir).resolve()
    encoded = str(resolved).replace("/", "-").lstrip("-")
    return Path.home() / ".cache" / "asciipic" / f"{encoded}.npy"


def _build_dataset(data_dir: str | Path, cell_width: int, cell_height: int, max_patches: int) -> np.ndarray:
    """Extract random 3x3-cell patches from all images in a directory.

    Returns array of shape (N, 3*cell_h, 3*cell_w, 3) float32 in [0, 1].
    """
    data_dir = Path(data_dir)
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    target_pixel_width = TARGET_COLS * cell_width
    patch_w = CONTEXT * cell_width
    patch_h = CONTEXT * cell_height
    rng = np.random.default_rng(42)
    patches = []

    print(f"Building dataset from {len(paths)} images (max {max_patches} patches)...")
    for i, path in enumerate(paths):
        if len(patches) >= max_patches:
            break
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)} images, {len(patches)} patches so far")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                img = Image.open(path).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        if w < patch_w or h < patch_h:
            continue

        # Scale to TARGET_COLS cells wide, preserving aspect ratio
        scale = target_pixel_width / w
        new_h = int(h * scale)
        if new_h < patch_h:
            continue
        img = img.resize((target_pixel_width, new_h), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        remaining = max_patches - len(patches)
        n_crops = min(PATCHES_PER_IMAGE, remaining)
        max_y = arr.shape[0] - patch_h
        max_x = arr.shape[1] - patch_w
        for _ in range(n_crops):
            y = rng.integers(max_y + 1)
            x = rng.integers(max_x + 1)
            patches.append(arr[y : y + patch_h, x : x + patch_w].copy())

    print(f"  Done: {len(patches)} patches from {i + 1} images")
    return np.stack(patches)


def load_or_build_dataset(data_dir: str | Path, cell_width: int, cell_height: int, max_patches: int) -> np.ndarray:
    """Load cached dataset or build and cache it."""
    cache = _cache_path(data_dir)
    if cache.exists():
        print(f"Loading cached dataset: {cache}")
        return np.load(cache)

    dataset = _build_dataset(data_dir, cell_width, cell_height, max_patches)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, dataset)
    print(f"Cached dataset to: {cache}")
    return dataset


class NeuralModel(nn.Module):
    """MLP that maps a downsampled 3x3 cell patch to center cell predictions."""

    DOWNSAMPLE_H = 15
    DOWNSAMPLE_W = 18

    def __init__(self, num_chars: int, hidden: int = 256):
        super().__init__()
        flat_size = self.DOWNSAMPLE_H * self.DOWNSAMPLE_W * 3
        self.shared = nn.Linear(flat_size, hidden)
        self.char_head = nn.Linear(hidden, num_chars)
        self.fg_head = nn.Linear(hidden, 3)
        self.bg_head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, 3*cell_h, 3*cell_w, 3)
        batch = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(self.DOWNSAMPLE_H, self.DOWNSAMPLE_W), mode="bilinear", align_corners=False)
        x = x.reshape(batch, -1)
        h = F.relu(self.shared(x))
        char_logits = self.char_head(h)
        fg = torch.sigmoid(self.fg_head(h))
        bg = torch.sigmoid(self.bg_head(h))
        return char_logits, fg, bg


def soft_render(
    char_logits: torch.Tensor,
    fg: torch.Tensor,
    bg: torch.Tensor,
    atlas: torch.Tensor,
) -> torch.Tensor:
    """Differentiable rendering of the center cell using soft glyph selection.

    Returns:
        rendered: (batch, cell_h, cell_w, 3)
    """
    weights = F.softmax(char_logits, dim=1)  # (batch, num_chars)
    num_chars, cell_h, cell_w = atlas.shape
    flat_atlas = atlas.reshape(num_chars, -1)  # (num_chars, cell_h * cell_w)
    soft_glyph = weights @ flat_atlas  # (batch, cell_h * cell_w)
    soft_glyph = soft_glyph.reshape(-1, cell_h, cell_w, 1)  # (batch, cell_h, cell_w, 1)
    fg = fg.reshape(-1, 1, 1, 3)
    bg = bg.reshape(-1, 1, 1, 3)
    return soft_glyph * fg + (1 - soft_glyph) * bg


def train(
    data_dir: str | Path,
    font_path: str,
    font_size: int,
    characters: str,
    output_path: str | Path,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    log_dir: str | Path | None = None,
    max_patches: int = 100_000,
):
    char_list, masks_np, cell_width, cell_height = build_atlas(font_path, font_size, characters)
    num_chars = len(char_list)

    patches = load_or_build_dataset(data_dir, cell_width, cell_height, max_patches)
    print(
        f"Dataset: {patches.shape[0]} patches of {patches.shape[1]}x{patches.shape[2]}px, {patches.nbytes / 1e6:.0f}MB"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atlas = torch.from_numpy(masks_np).to(device)
    model = NeuralModel(num_chars).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.from_numpy(patches))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    writer = SummaryWriter(log_dir=log_dir) if log_dir else SummaryWriter()

    print(f"Training on {device}, {num_chars} characters, cell {cell_width}x{cell_height}")
    print(f"Logging to: {writer.log_dir}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)

            # Extract center cell pixels as target
            center = batch[:, cell_height : 2 * cell_height, cell_width : 2 * cell_width, :]

            char_logits, fg, bg = model(batch)
            rendered = soft_render(char_logits, fg, bg, atlas)
            loss = F.mse_loss(rendered, center)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar("loss/mse", loss.item(), global_step)

        avg_loss = epoch_loss / num_batches
        print(f"epoch {epoch}/{epochs}  avg_loss={avg_loss:.6f}  ({num_batches} batches)")
        writer.add_scalar("loss/epoch_avg", avg_loss, epoch)
        _save_weights(model, char_list, cell_width, cell_height, font_path, font_size, output_path)

    writer.close()
    print(f"Saved weights to {output_path}")


def _save_weights(model, char_list, cell_width, cell_height, font_path, font_size, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    save_dict = {
        "char_list": np.array(char_list),
        "cell_width": np.array(cell_width),
        "cell_height": np.array(cell_height),
        "font_path": np.array(font_path),
        "font_size": np.array(font_size),
        "downsample_h": np.array(NeuralModel.DOWNSAMPLE_H),
        "downsample_w": np.array(NeuralModel.DOWNSAMPLE_W),
    }
    for key, tensor in state.items():
        save_dict[f"weights.{key}"] = tensor.cpu().numpy()
    np.savez(output_path, **save_dict)


def main():
    parser = argparse.ArgumentParser(description="Train neural ASCII rendering engine")
    parser.add_argument("--data", required=True, help="Directory of training images")
    parser.add_argument("--output", default="src/data/neural/weights.npz", help="Output weights path")
    parser.add_argument("--font", default=None, help="Font path (default: system monospace via fontconfig)")
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--characters", default=VISUAL)
    parser.add_argument("--max-patches", type=int, default=100_000, help="Maximum number of patches to extract")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-dir", default=None, help="Tensorboard log directory")
    args = parser.parse_args()

    font_path = args.font
    if font_path is None:
        result = subprocess.run(["fc-match", "-f", "%{file}", "monospace"], capture_output=True, text=True)
        font_path = result.stdout.strip()
        print(f"Using font: {font_path}")

    train(
        data_dir=args.data,
        font_path=font_path,
        font_size=args.font_size,
        characters=args.characters,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_dir=args.log_dir,
        max_patches=args.max_patches,
    )


if __name__ == "__main__":
    main()

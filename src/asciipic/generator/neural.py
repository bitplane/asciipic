"""Training script for the neural ASCII rendering engine.

Trains a CNN+MLP to map 3x3 cell patches to character + invert predictions.
Both real and rendered images are binarized to black and white. The rendered
grid is seamless — tile boundaries are invisible except at invert transitions.
Training uses direct MSE between the soft-rendered grid and the real binary patch.

Usage:
    python -m asciipic.generator.neural --data /path/to/images --output data/neural/weights.npz
"""

import argparse
import hashlib
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
TARGET_COLS = 120
CONTEXT = 3  # 3x3 cells per patch


def _cache_path(data_dir: str | Path, cell_width: int, cell_height: int, max_patches: int) -> Path:
    """Build cache path from the data directory, cell dimensions, and patch count."""
    resolved = Path(data_dir).resolve()
    encoded = str(resolved).replace("/", "-").lstrip("-")
    return Path.home() / ".cache" / "asciipic" / f"{encoded}_{cell_width}x{cell_height}_{max_patches}_bin.npy"


def _build_dataset(data_dir: str | Path, cell_width: int, cell_height: int, max_patches: int) -> np.ndarray:
    """Extract random 3x3-cell patches from all images in a directory.

    Each patch is converted to grayscale and binarized using the mean brightness
    of its center cell as threshold. Duplicate binary patches are discarded.

    Returns array of shape (N, 3*cell_h, 3*cell_w) uint8, values 0 or 1.
    """
    data_dir = Path(data_dir)
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    target_pixel_width = TARGET_COLS * cell_width
    patch_w = CONTEXT * cell_width
    patch_h = CONTEXT * cell_height
    rng = np.random.default_rng(42)

    # Distribute patches evenly: ceil(max_patches / num_images) per image
    per_image = max(1, -(-max_patches // len(paths)))

    result = np.empty((max_patches, patch_h, patch_w), dtype=np.uint8)
    seen: set[bytes] = set()
    count = 0

    print(f"Building dataset from {len(paths)} images ({per_image} patches each, max {max_patches})...")
    for i, path in enumerate(paths):
        if count >= max_patches:
            break
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)} images, {count} patches so far")
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

        max_y = arr.shape[0] - patch_h
        max_x = arr.shape[1] - patch_w
        n = min(per_image, max_patches - count)
        for _ in range(n):
            y = rng.integers(max_y + 1)
            x = rng.integers(max_x + 1)
            patch_rgb = arr[y : y + patch_h, x : x + patch_w]

            # Grayscale → center cell threshold → binary
            gray = patch_rgb.mean(axis=-1)
            center = gray[cell_height : 2 * cell_height, cell_width : 2 * cell_width]
            threshold = center.mean()
            binary = (gray > threshold).astype(np.uint8)

            # Deduplicate by center cell content
            center_bin = binary[cell_height : 2 * cell_height, cell_width : 2 * cell_width]
            digest = hashlib.md5(center_bin.tobytes()).digest()
            if digest in seen:
                continue
            seen.add(digest)

            result[count] = binary
            count += 1
            if count >= max_patches:
                break

    result = result[:count]
    rng.shuffle(result)
    print(f"  Done: {count} unique patches from {i + 1} images")
    return result


def load_or_build_dataset(data_dir: str | Path, cell_width: int, cell_height: int, max_patches: int) -> np.ndarray:
    """Load cached dataset or build and cache it."""
    cache = _cache_path(data_dir, cell_width, cell_height, max_patches)
    if cache.exists():
        print(f"Loading cached dataset: {cache}")
        return np.load(cache)

    dataset = _build_dataset(data_dir, cell_width, cell_height, max_patches)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, dataset)
    print(f"Cached dataset to: {cache}")
    return dataset


class NeuralModel(nn.Module):
    """CNN + MLP that maps a 3x3 cell patch to per-position predictions.

    The conv frontend pools to (batch, 256, 3, 3) — one 256-dim feature vector
    per cell position. The shared layer and heads process each position
    independently with shared weights, predicting all 9 cells.
    """

    def __init__(self, num_chars: int, cell_height: int, cell_width: int, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.shared = nn.Linear(256, hidden)
        self.char_head = nn.Linear(hidden, num_chars)
        self.invert_head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting all 9 cell positions.

        Args:
            x: (batch, 3*cell_h, 3*cell_w) binary float tensor

        Returns:
            char_logits: (batch, 9, num_chars)
            invert_logits: (batch, 9, 2) — class 0 = normal, class 1 = inverted
        """
        batch = x.shape[0]
        x = x.unsqueeze(1)  # (batch, 1, H, W)
        x = self.conv(x)  # (batch, 256, 3, 3)
        # Reshape to per-position vectors: (batch*9, 256)
        x = x.permute(0, 2, 3, 1).reshape(batch * 9, 256)
        h = F.relu(self.shared(x))  # (batch*9, hidden)
        char_logits = self.char_head(h).reshape(batch, 9, -1)
        invert_logits = self.invert_head(h).reshape(batch, 9, 2)
        return char_logits, invert_logits


def _haar_kernels(device: torch.device) -> torch.Tensor:
    """Build the 4 Haar wavelet kernels: LL, LH, HL, HH. Shape: (4, 1, 2, 2)."""
    return (
        torch.tensor(
            [
                [[1, 1], [1, 1]],  # LL — average
                [[1, 1], [-1, -1]],  # LH — horizontal edges
                [[1, -1], [1, -1]],  # HL — vertical edges
                [[1, -1], [-1, 1]],  # HH — diagonal edges
            ],
            dtype=torch.float32,
            device=device,
        ).reshape(4, 1, 2, 2)
        * 0.5
    )


def _haar_decompose(x: torch.Tensor, kernels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One level of Haar wavelet decomposition.

    Returns (ll, detail) where ll is the coarse approximation and
    detail is the (LH, HL, HH) coefficients concatenated along channel dim.
    """
    # Pad to even dimensions if needed
    _, _, h, w = x.shape
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2))
    coeffs = F.conv2d(x, kernels, stride=2)  # (B, 4, H/2, W/2)
    return coeffs[:, :1], coeffs[:, 1:]


def _haar_loss(a: torch.Tensor, b: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """MSE on Haar wavelet coefficients at multiple scales.

    Compares horizontal, vertical, and diagonal edge structure at each level,
    plus the coarse residual.
    """
    kernels = _haar_kernels(a.device)
    loss = torch.zeros(1, device=a.device)
    for _ in range(levels - 1):
        a_ll, a_detail = _haar_decompose(a, kernels)
        b_ll, b_detail = _haar_decompose(b, kernels)
        loss = loss + F.mse_loss(a_detail, b_detail)
        a, b = a_ll, b_ll
    # Coarsest level
    loss = loss + F.mse_loss(a, b)
    return loss


def render_grid(
    char_logits: torch.Tensor,
    invert_logits: torch.Tensor,
    atlas: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable rendering of a 3x3 glyph grid.

    Uses softmax for smooth gradient flow during training. Each cell blends
    glyphs weighted by character probabilities, with soft invert. Inference
    uses argmax for hard selection.

    Args:
        char_logits: (batch, 9, num_chars)
        invert_logits: (batch, 9, 2) — class 0 = normal, class 1 = inverted
        atlas: (num_chars, cell_h, cell_w) binary masks
        temperature: softmax temperature (lower = sharper, 1.0 = normal)

    Returns:
        rendered: (batch, 1, 3*cell_h, 3*cell_w)
    """
    batch = char_logits.shape[0]
    num_chars, cell_h, cell_w = atlas.shape
    flat_atlas = atlas.reshape(num_chars, -1)

    # Soft character selection
    weights = F.softmax(char_logits.reshape(batch * 9, -1) / temperature, dim=1)
    glyph = weights @ flat_atlas  # (batch*9, cell_h*cell_w)
    glyph = glyph.reshape(batch * 9, cell_h, cell_w)

    # Soft invert selection
    inv_probs = F.softmax(invert_logits.reshape(batch * 9, 2) / temperature, dim=1)
    inv = inv_probs[:, 1].reshape(batch * 9, 1, 1)  # P(inverted)

    # Render: mask XOR invert (soft version)
    cells = glyph * (1 - inv) + (1 - glyph) * inv

    # Arrange into 3x3 grid
    cells = cells.reshape(batch, 3, 3, cell_h, cell_w)
    grid = cells.permute(0, 1, 3, 2, 4).reshape(batch, 3 * cell_h, 3 * cell_w)
    return grid.unsqueeze(1)  # (batch, 1, H, W)


def _vis_grid(target: torch.Tensor, rendered: torch.Tensor) -> torch.Tensor:
    """Compose a 6x4 grid of target/rendered pairs.

    Layout: columns alternate target, rendered (3 pairs per row, 4 rows = 12 pairs).

    Args:
        target: (N, 1, H, W) binary patches
        rendered: (N, 1, H, W) soft-rendered patches

    Returns:
        (3, grid_H, grid_W) RGB tensor suitable for writer.add_image
    """
    ncols, nrows = 6, 4
    n_pairs = min(target.shape[0], (ncols // 2) * nrows)
    _, _, H, W = target.shape
    gap = 1

    grid_h = nrows * H + (nrows + 1) * gap
    grid_w = ncols * W + (ncols + 1) * gap
    grid = torch.full((3, grid_h, grid_w), 0.25, device=target.device)

    for idx in range(n_pairs):
        row, pair_col = divmod(idx, ncols // 2)
        for side, images in enumerate([target, rendered]):
            col = pair_col * 2 + side
            y0 = row * (H + gap) + gap
            x0 = col * (W + gap) + gap
            grid[:, y0 : y0 + H, x0 : x0 + W] = images[idx].expand(3, H, W)

    return grid


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
    model = NeuralModel(num_chars, cell_height, cell_width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.from_numpy(patches))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    writer = SummaryWriter(log_dir=log_dir) if log_dir else SummaryWriter()

    # Fixed visual samples for tensorboard (same patches every epoch)
    vis_samples = torch.from_numpy(patches[:12]).to(device, dtype=torch.float32)

    print(f"Training on {device}, {num_chars} characters, cell {cell_width}x{cell_height}")
    print(f"Logging to: {writer.log_dir}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        ep_loss = 0.0
        num_batches = 0
        for (batch,) in loader:
            batch = batch.to(device, dtype=torch.float32)  # uint8 → float32

            target = batch.unsqueeze(1)  # (B, 1, H, W)
            char_logits, invert_logits = model(batch)
            rendered = render_grid(char_logits, invert_logits, atlas)

            loss = _haar_loss(rendered, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar("loss/haar", loss.item(), global_step)

        avg_loss = ep_loss / num_batches
        print(f"epoch {epoch}/{epochs}  loss={avg_loss:.4f}" f"  lr={lr:.2e}  ({num_batches} batches)")
        writer.add_scalar("loss/epoch_haar", avg_loss, epoch)

        with torch.no_grad():
            vis_logits, vis_inv_logits = model(vis_samples)
            vis_rendered = render_grid(vis_logits, vis_inv_logits, atlas)
            writer.add_image("vis/comparison", _vis_grid(vis_samples.unsqueeze(1), vis_rendered), epoch)

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
    }
    for key, tensor in state.items():
        save_dict[f"weights.{key}"] = tensor.cpu().numpy()
    tmp_path = output_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **save_dict)
    tmp_path.rename(output_path)


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

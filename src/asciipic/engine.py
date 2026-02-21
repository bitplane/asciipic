from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass
class CellGrid:
    chars: list[str]  # one string per row
    colours: np.ndarray | None  # (rows, cols, 2, 3) uint8, or None


class Engine(Protocol):
    cell_width: int
    cell_height: int

    def render(self, image: Image.Image) -> CellGrid:
        """Convert an RGB image to a grid of characters with optional colours."""
        ...

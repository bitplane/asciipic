from PIL import Image

from asciipic.engine import CellGrid
from asciipic.model import FontModel
from asciipic.sampling import enhance_contrast, sample_colours, sample_grid


class SamplerEngine:
    """Rendering engine that maps image cells to characters via circle sampling."""

    def __init__(self, model: FontModel, exponent: float | None = None):
        self.model = model
        self.exponent = exponent
        self.cell_width = model.cell_width
        self.cell_height = model.cell_height

    def render(self, image: Image.Image) -> CellGrid:
        gray = image.convert("L")
        grid = sample_grid(gray, self.cell_width, self.cell_height) / 255.0
        if self.exponent is not None:
            grid = enhance_contrast(grid, self.exponent)
        chars = self.model.find_nearest_grid(grid)
        colours = sample_colours(image, self.cell_width, self.cell_height)
        return CellGrid(chars=chars, colours=colours)

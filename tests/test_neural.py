import numpy as np
from PIL import Image

from asciipic.neural import NeuralEngine
from tests.conftest import FONT_PATH, needs_font

CONTEXT = 3


def _make_random_weights(tmp_path, font_path, font_size=16, characters=" #@"):
    """Create a weights file with random CNN+MLP parameters."""
    from asciipic.glyph_atlas import build_atlas

    char_list, masks, cw, ch = build_atlas(font_path, font_size, characters)
    num_chars = len(char_list)
    hidden = 256

    rng = np.random.default_rng(42)
    weights_path = tmp_path / "weights.npz"
    np.savez(
        weights_path,
        char_list=np.array(char_list),
        cell_width=np.array(cw),
        cell_height=np.array(ch),
        font_path=np.array(font_path),
        font_size=np.array(font_size),
        **{
            "weights.conv.0.weight": rng.standard_normal((64, 1, 3, 3)).astype(np.float32) * 0.01,
            "weights.conv.0.bias": np.zeros(64, dtype=np.float32),
            "weights.conv.2.weight": rng.standard_normal((256, 64, 3, 3)).astype(np.float32) * 0.01,
            "weights.conv.2.bias": np.zeros(256, dtype=np.float32),
            "weights.shared.weight": rng.standard_normal((hidden, 256)).astype(np.float32) * 0.01,
            "weights.shared.bias": np.zeros(hidden, dtype=np.float32),
            "weights.char_head.weight": rng.standard_normal((num_chars, hidden)).astype(np.float32) * 0.01,
            "weights.char_head.bias": np.zeros(num_chars, dtype=np.float32),
            "weights.invert_head.weight": rng.standard_normal((2, hidden)).astype(np.float32) * 0.01,
            "weights.invert_head.bias": np.zeros(2, dtype=np.float32),
        },
    )
    return weights_path


@needs_font
def test_neural_engine_produces_cellgrid(tmp_path):
    weights_path = _make_random_weights(tmp_path, FONT_PATH)
    engine = NeuralEngine(weights_path)
    img = Image.new("RGB", (engine.cell_width * 5, engine.cell_height * 3), (128, 64, 200))
    result = engine.render(img)
    assert len(result.chars) == 3
    assert all(len(row) == 5 for row in result.chars)
    assert result.colours.shape == (3, 5, 2, 3)


@needs_font
def test_neural_engine_colours_in_range(tmp_path):
    weights_path = _make_random_weights(tmp_path, FONT_PATH)
    engine = NeuralEngine(weights_path)
    img = Image.new("RGB", (engine.cell_width * 3, engine.cell_height * 2), (255, 0, 0))
    result = engine.render(img)
    assert result.colours.dtype == np.uint8
    assert result.colours.min() >= 0
    assert result.colours.max() <= 255


@needs_font
def test_neural_engine_cell_dimensions(tmp_path):
    weights_path = _make_random_weights(tmp_path, FONT_PATH)
    engine = NeuralEngine(weights_path)
    assert engine.cell_width > 0
    assert engine.cell_height > 0

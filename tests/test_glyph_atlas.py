import numpy as np

from asciipic.glyph_atlas import build_atlas
from tests.conftest import FONT_PATH, needs_font


@needs_font
def test_atlas_shape():
    chars = " #@"
    char_list, masks, cw, ch = build_atlas(FONT_PATH, 16, chars)
    assert len(char_list) == 3
    assert masks.shape == (3, ch, cw)
    assert masks.dtype == np.float32


@needs_font
def test_atlas_values_in_range():
    _, masks, _, _ = build_atlas(FONT_PATH, 16, " #@ABCxyz")
    assert masks.min() >= 0.0
    assert masks.max() <= 1.0


@needs_font
def test_space_is_blank():
    char_list, masks, _, _ = build_atlas(FONT_PATH, 16, " #@")
    space_idx = char_list.index(" ")
    assert masks[space_idx].sum() == 0.0


@needs_font
def test_dense_char_has_ink():
    char_list, masks, _, _ = build_atlas(FONT_PATH, 16, " @")
    at_idx = char_list.index("@")
    assert masks[at_idx].sum() > 0.0


@needs_font
def test_cell_dimensions_positive():
    _, _, cw, ch = build_atlas(FONT_PATH, 16, "A")
    assert cw > 0
    assert ch > 0

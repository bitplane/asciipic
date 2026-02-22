import pytest

from asciipic.charsets import ASCII_PRINTABLE
from asciipic.generator import generate_font_model
from asciipic.sampling import NUM_SAMPLES

# FONT_PATH and needs_font are defined in conftest.py and loaded by pytest
from tests.conftest import FONT_PATH, needs_font


@needs_font
def test_generate_font_model_produces_vectors():
    model = generate_font_model(FONT_PATH, "AB #")
    assert len(model.characters) == 4
    for vector in model.characters.values():
        assert len(vector) == NUM_SAMPLES
        assert all(0.0 <= v <= 1.0 for v in vector)


@needs_font
def test_generate_font_model_normalization():
    model = generate_font_model(FONT_PATH, ASCII_PRINTABLE)
    # At least one dimension should have a 1.0 value (the max)
    for dim in range(NUM_SAMPLES):
        values = [v[dim] for v in model.characters.values()]
        assert max(values) == pytest.approx(1.0)


@needs_font
def test_space_has_zero_vector():
    model = generate_font_model(FONT_PATH, " #@MW.")
    space_vec = model.characters[" "]
    assert all(v == 0.0 for v in space_vec)


@needs_font
def test_cell_dimensions_positive():
    model = generate_font_model(FONT_PATH, "A")
    assert model.cell_width > 0
    assert model.cell_height > 0


@needs_font
def test_dense_char_has_high_values():
    model = generate_font_model(FONT_PATH, " @")
    at_vec = model.characters["@"]
    # @ should be fairly filled in across all regions
    assert sum(at_vec) / NUM_SAMPLES > 0.3

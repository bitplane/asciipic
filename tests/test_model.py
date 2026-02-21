import math

import pytest

from asciipic.model import FontModel
from asciipic.sampling import NUM_SAMPLES


def _vec(*values):
    """Repeat last value to fill NUM_SAMPLES slots."""
    return values + (values[-1],) * (NUM_SAMPLES - len(values))


def make_model():
    return FontModel(
        font_name="TestFont",
        font_size=16,
        cell_width=8,
        cell_height=16,
        characters={
            " ": (0.0,) * NUM_SAMPLES,
            "#": (1.0,) * NUM_SAMPLES,
            "L": _vec(0.5, 0.0, 0.5, 0.0, 0.5, 0.5),
        },
    )


def test_save_load_roundtrip(tmp_path):
    model = make_model()
    path = tmp_path / "test.apic"
    model.save(path)
    loaded = FontModel.load(path)

    assert loaded.font_name == model.font_name
    assert loaded.font_size == model.font_size
    assert loaded.cell_width == model.cell_width
    assert loaded.cell_height == model.cell_height
    assert loaded.characters.keys() == model.characters.keys()
    for char in model.characters:
        for a, b in zip(model.characters[char], loaded.characters[char]):
            assert math.isclose(a, b, rel_tol=1e-6)


def test_find_nearest_exact():
    model = make_model()
    assert model.find_nearest((0.0,) * NUM_SAMPLES) == " "
    assert model.find_nearest((1.0,) * NUM_SAMPLES) == "#"
    assert model.find_nearest(_vec(0.5, 0.0, 0.5, 0.0, 0.5, 0.5)) == "L"


def test_find_nearest_approx():
    model = make_model()
    # Close to all-ones should give #
    assert model.find_nearest((0.9,) * NUM_SAMPLES) == "#"
    # Close to all-zeros should give space
    assert model.find_nearest((0.1,) * NUM_SAMPLES) == " "


def test_unicode_character_roundtrip(tmp_path):
    model = FontModel(
        font_name="Unicode",
        font_size=12,
        cell_width=8,
        cell_height=16,
        characters={"\u2588": (1.0,) * NUM_SAMPLES},
    )
    path = tmp_path / "unicode.apic"
    model.save(path)
    loaded = FontModel.load(path)
    assert "\u2588" in loaded.characters
    assert loaded.characters["\u2588"] == model.characters["\u2588"]


def test_load_bad_magic(tmp_path):
    path = tmp_path / "bad.apic"
    path.write_bytes(b"NOPE" + b"\x00" * 50)
    with pytest.raises(ValueError, match="Not an APIC file"):
        FontModel.load(path)


def test_load_bad_version(tmp_path):
    path = tmp_path / "badver.apic"
    path.write_bytes(b"APIC" + b"\xff" + b"\x00" * 50)
    with pytest.raises(ValueError, match="Unsupported format version"):
        FontModel.load(path)

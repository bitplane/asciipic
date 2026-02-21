from PIL import Image

from asciipic.converter import image_to_ascii
from asciipic.model import FontModel
from asciipic.sampling import NUM_SAMPLES


def make_model(cell_width=10, cell_height=20, characters=None):
    """Build a FontModel with known vectors for testing."""
    if characters is None:
        characters = {
            " ": (0.0,) * NUM_SAMPLES,
            "#": (1.0,) * NUM_SAMPLES,
        }
    return FontModel(
        font_name="test",
        font_size=16,
        cell_width=cell_width,
        cell_height=cell_height,
        characters=characters,
    )


def test_solid_white_maps_to_densest():
    model = make_model()
    img = Image.new("L", (30, 40), 255)
    result = image_to_ascii(img, model)
    assert result == "###\n###"


def test_solid_black_maps_to_space():
    model = make_model()
    img = Image.new("L", (30, 40), 0)
    result = image_to_ascii(img, model)
    assert result == "   \n   "


def test_output_dimensions():
    model = make_model(cell_width=10, cell_height=20)
    img = Image.new("L", (50, 60), 128)
    result = image_to_ascii(img, model)
    lines = result.split("\n")
    assert len(lines) == 3  # 60 // 20
    assert all(len(line) == 5 for line in lines)  # 50 // 10


def test_truncation_of_remainder():
    model = make_model(cell_width=10, cell_height=20)
    img = Image.new("L", (35, 55), 128)  # 5px and 15px remainder
    result = image_to_ascii(img, model)
    lines = result.split("\n")
    assert len(lines) == 2  # 55 // 20 = 2
    assert all(len(line) == 3 for line in lines)  # 35 // 10 = 3


def test_width_parameter():
    model = make_model(cell_width=10, cell_height=20)
    img = Image.new("L", (100, 200), 128)
    result = image_to_ascii(img, model, width=5)
    lines = result.split("\n")
    assert all(len(line) == 5 for line in lines)


def test_accepts_file_path(tmp_path):
    model = make_model()
    img = Image.new("L", (20, 20), 255)
    path = tmp_path / "test.png"
    img.save(path)
    result = image_to_ascii(path, model)
    assert len(result) > 0


def test_accepts_rgb_image():
    model = make_model()
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    result = image_to_ascii(img, model)
    assert "#" in result


def test_gradient_produces_varying_characters():
    model = make_model()
    cw, ch = 10, 20
    img = Image.new("L", (cw * 2, ch))
    pixels = img.load()
    # Left cell: black, right cell: white
    for y in range(ch):
        for x in range(cw):
            pixels[x, y] = 0
        for x in range(cw, cw * 2):
            pixels[x, y] = 255
    result = image_to_ascii(img, model)
    assert result == " #"


def test_image_smaller_than_one_cell():
    model = make_model(cell_width=10, cell_height=20)
    img = Image.new("L", (5, 10), 128)
    result = image_to_ascii(img, model)
    assert result == ""


def test_contrast_enhancement_preserves_gradient():
    model = make_model()
    cw, ch = 10, 20
    img = Image.new("L", (cw * 2, ch))
    pixels = img.load()
    for y in range(ch):
        for x in range(cw):
            pixels[x, y] = 0
        for x in range(cw, cw * 2):
            pixels[x, y] = 255
    result = image_to_ascii(img, model, exponent=2.0)
    assert result == " #"


def test_colour_output_contains_ansi_escapes():
    model = make_model()
    img = Image.new("RGB", (20, 20), (255, 0, 0))
    result = image_to_ascii(img, model, colour=True)
    assert "\033[38;2;" in result
    assert "\033[48;2;" in result
    assert "\033[0m" in result


def test_colour_false_has_no_escapes():
    model = make_model()
    img = Image.new("RGB", (20, 20), (255, 0, 0))
    result = image_to_ascii(img, model, colour=False)
    assert "\033" not in result

import argparse
import sys
from pathlib import Path

from asciipic.converter import image_to_ascii
from asciipic.model import FontModel
from asciipic.terminal import get_terminal_size

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MODEL = DATA_DIR / "ascii"


def main():
    parser = argparse.ArgumentParser(description="Render an image as ASCII art")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "-s", "--size", type=int, default=None, help="Output width in columns (default: terminal width)"
    )
    parser.add_argument(
        "-e",
        "--exponent",
        type=float,
        default=10.0,
        help="Contrast enhancement exponent (default: 10.0). Higher values increase contrast at cell boundaries.",
    )
    args = parser.parse_args()

    width = args.size if args.size is not None else get_terminal_size()[0]

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    model = FontModel.load(DEFAULT_MODEL)
    print(image_to_ascii(image_path, model, width=width, exponent=args.exponent))

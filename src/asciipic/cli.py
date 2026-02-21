import argparse
import sys
from pathlib import Path

from asciipic.converter import image_to_ascii
from asciipic.model import FontModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MODEL = DATA_DIR / "ascii"


def main():
    parser = argparse.ArgumentParser(description="Render an image as ASCII art")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-w", "--width", type=int, default=80, help="Output width in columns (default: 80)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    model = FontModel.load(DEFAULT_MODEL)
    print(image_to_ascii(image_path, model, width=args.width))

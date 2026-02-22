import shutil
import subprocess

import pytest

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
]


def _find_monospace_font():
    """Find a monospace font on the system."""
    for path in _FONT_CANDIDATES:
        if shutil.os.path.exists(path):
            return path
    result = shutil.which("fc-match")
    if result:
        out = subprocess.run(["fc-match", "-f", "%{file}", "monospace"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


FONT_PATH = _find_monospace_font()
needs_font = pytest.mark.skipif(FONT_PATH is None, reason="No monospace font found on system")

import os
import sys


def get_terminal_size() -> tuple[int, int]:
    """Return (columns, rows) of the terminal, or (80, 24) if not a tty."""
    if not sys.stdout.isatty():
        return (80, 24)
    size = os.get_terminal_size()
    return (size.columns, size.lines)

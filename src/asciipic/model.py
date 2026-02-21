import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"APIC"
FORMAT_VERSION = 1


@dataclass
class FontModel:
    font_name: str
    font_size: int
    cell_width: int
    cell_height: int
    characters: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("B", FORMAT_VERSION))
            name_bytes = self.font_name.encode("utf-8")
            f.write(struct.pack(">H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack(">HHH", self.font_size, self.cell_width, self.cell_height))
            f.write(struct.pack(">I", len(self.characters)))
            for char, vector in self.characters.items():
                char_bytes = char.encode("utf-8")
                f.write(struct.pack("B", len(char_bytes)))
                f.write(char_bytes)
                f.write(struct.pack(">6f", *vector))

    @classmethod
    def load(cls, path: str | Path) -> "FontModel":
        path = Path(path)
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Not an APIC file: {magic!r}")
            (version,) = struct.unpack("B", f.read(1))
            if version != FORMAT_VERSION:
                raise ValueError(f"Unsupported format version: {version}")
            (name_len,) = struct.unpack(">H", f.read(2))
            font_name = f.read(name_len).decode("utf-8")
            font_size, cell_width, cell_height = struct.unpack(">HHH", f.read(6))
            (char_count,) = struct.unpack(">I", f.read(4))
            characters: dict[str, tuple[float, ...]] = {}
            for _ in range(char_count):
                (char_len,) = struct.unpack("B", f.read(1))
                char = f.read(char_len).decode("utf-8")
                vector = struct.unpack(">6f", f.read(24))
                characters[char] = vector
            return cls(
                font_name=font_name,
                font_size=font_size,
                cell_width=cell_width,
                cell_height=cell_height,
                characters=characters,
            )

    def find_nearest(self, vector: tuple[float, ...]) -> str:
        best_char = ""
        best_dist = math.inf
        for char, char_vector in self.characters.items():
            dist = sum((a - b) ** 2 for a, b in zip(vector, char_vector))
            if dist < best_dist:
                best_dist = dist
                best_char = char
        return best_char

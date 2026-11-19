"""Erzeugt die Testbilder der Capability-Suite (stdlib-only, reproduzierbar).

Ohne PIL bleibt nur der eigene PNG-Weg: ein RGB-Raster wird pixelweise gefuellt
und per zlib als PNG codiert. Damit entstehen zwei deterministische_assets:

- diagram.png: drei Balken, der linke ist am hoechsten (diagram_reasoning)
- ocr_label.png: der Text "AION 4711" als 5x7-Bitmap-Schrift (ocr)

Aufruf:  python scripts/make_suite_assets.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "evaluations" / "capability_suite" / "assets"

WEISS = (255, 255, 255)
SCHWARZ = (20, 20, 20)
BLAU = (40, 90, 200)
GRAU = (120, 120, 120)

# 5x7-Bitmap-Schrift, nur die fuer das Label benoetigten Zeichen.
GLYPHS = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "11001", "10101", "10011", "10011", "10001"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


class Raster:
    def __init__(self, breite: int, hoehe: int, farbe=WEISS):
        self.breite = breite
        self.hoehe = hoehe
        self.pixel = [[farbe for _ in range(breite)] for _ in range(hoehe)]

    def rechteck(self, x: int, y: int, b: int, h: int, farbe):
        for zeile in range(y, min(y + h, self.hoehe)):
            for spalte in range(x, min(x + b, self.breite)):
                self.pixel[zeile][spalte] = farbe

    def zeichen(self, glyph, x: int, y: int, farbe, skala: int = 1):
        for zeile, muster in enumerate(glyph):
            for spalte, bit in enumerate(muster):
                if bit == "1":
                    self.rechteck(x + spalte * skala, y + zeile * skala, skala, skala, farbe)

    def text(self, text: int | str, x: int, y: int, farbe=SCHWARZ, skala: int = 1):
        cursor = x
        for zeichen in str(text).upper():
            glyph = GLYPHS.get(zeichen)
            if glyph is None:
                cursor += 6 * skala
                continue
            self.zeichen(glyph, cursor, y, farbe, skala)
            cursor += 6 * skala
        return cursor

    def zeile(self, y: int, farbe=SCHWARZ, start: int = 0, ende: int | None = None):
        ende = self.breite if ende is None else ende
        for spalte in range(start, ende):
            self.pixel[y][spalte] = farbe

    def spalte(self, x: int, farbe=SCHWARZ, start: int = 0, ende: int | None = None):
        ende = self.hoehe if ende is None else ende
        for zeile in range(start, ende):
            self.pixel[zeile][x] = farbe

    def schreiben(self, pfad: Path):
        rohdaten = bytearray()
        for zeile in self.pixel:
            rohdaten.append(0)                      # PNG-Filter: None
            for r, g, b in zeile:
                rohdaten += bytes((r, g, b))
        block = zlib.compress(bytes(rohdaten), 9)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", struct.pack(">IIBBBBB", self.breite, self.hoehe, 8, 2, 0, 0, 0))
        png += _chunk(b"IDAT", block)
        png += _chunk(b"IEND", b"")
        pfad.parent.mkdir(parents=True, exist_ok=True)
        pfad.write_bytes(png)


def _chunk(tag: bytes, daten: bytes) -> bytes:
    return (struct.pack(">I", len(daten)) + tag + daten
            + struct.pack(">I", zlib.crc32(tag + daten) & 0xFFFFFFFF))


def diagramm() -> Raster:
    """Drei Balken: links am hoechsten, mitte mittel, rechts am niedrigsten."""
    bild = Raster(240, 160)
    bild.linie = None  # Attribut nur zur Klarheit; Achsen kommen unten.
    for x in range(16, 224):
        bild.pixel[140][x] = GRAU
    for y in range(20, 141):
        bild.pixel[y][16] = GRAU
    bild.rechteck(40, 40, 30, 100, BLAU)     # links: hoechster Balken
    bild.rechteck(105, 80, 30, 60, BLAU)     # mitte
    bild.rechteck(170, 110, 30, 30, BLAU)    # rechts: niedrigster
    return bild


def ocr_label() -> Raster:
    """Text 'AION 4711' als Bitmap-Schrift, schwarz auf weiss."""
    bild = Raster(240, 40)
    bild.text("AION 4711", 24, 12, SCHWARZ, skala=3)
    return bild


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    diagramm().schreiben(ASSETS / "diagram.png")
    ocr_label().schreiben(ASSETS / "ocr_label.png")
    for name in ("diagram.png", "ocr_label.png"):
        pfad = ASSETS / name
        print(f"{name}: {pfad.stat().st_size} Bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

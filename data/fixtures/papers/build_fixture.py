"""Generate fixture.pdf from fixture.txt with the standard library only.

Pages are separated by a line containing ``---page---``. Every line becomes
one text-showing operator on its own baseline so text extractors preserve
line breaks. The output is deterministic: no timestamps or IDs are embedded.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fixture.txt"
TARGET = HERE / "fixture.pdf"

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
MARGIN_LEFT, TOP, LEADING, FONT_SIZE = 72, 720, 14, 11


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(lines: list[str]) -> bytes:
    parts = [f"BT /F1 {FONT_SIZE} Tf {LEADING} TL {MARGIN_LEFT} {TOP} Td"]
    for index, line in enumerate(lines):
        if index:
            parts.append("T*")
        parts.append(f"({_escape(line)}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build(source: Path = SOURCE, target: Path = TARGET) -> Path:
    pages: list[list[str]] = [[]]
    for raw in source.read_text(encoding="utf-8").splitlines():
        if raw.strip() == "---page---":
            pages.append([])
        else:
            pages[-1].append(raw)
    pages = [page for page in pages if page]

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_placeholder = add(b"")  # filled in once page object numbers are known
    page_numbers: list[int] = []
    for lines in pages:
        stream = _content_stream(lines)
        contents = add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page = add(
            (
                f"<< /Type /Page /Parent {pages_placeholder} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] /Contents {contents} 0 R "
                f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
            ).encode()
        )
        page_numbers.append(page)
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects[pages_placeholder - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode()
    )
    catalog = add(f"<< /Type /Catalog /Pages {pages_placeholder} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    target.write_bytes(bytes(out))
    return target


if __name__ == "__main__":
    print(build())

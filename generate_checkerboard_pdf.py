"""Generate an exact-size OpenCV camera-calibration checkerboard PDF."""

from __future__ import annotations

import argparse
from pathlib import Path


MM_TO_POINTS = 72.0 / 25.4
PAGE_WIDTH_MM = 297.0
PAGE_HEIGHT_MM = 210.0
BOARD_COLUMNS = 10
BOARD_ROWS = 7
SQUARE_SIZE_MM = 25.0


def mm(value: float) -> float:
    return value * MM_TO_POINTS


def build_page_content() -> bytes:
    board_width_mm = BOARD_COLUMNS * SQUARE_SIZE_MM
    board_height_mm = BOARD_ROWS * SQUARE_SIZE_MM
    origin_x_mm = (PAGE_WIDTH_MM - board_width_mm) * 0.5
    origin_y_mm = (PAGE_HEIGHT_MM - board_height_mm) * 0.5
    square_points = mm(SQUARE_SIZE_MM)

    commands = ["0 0 0 rg"]
    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS):
            if (row + column) % 2 != 0:
                continue
            x = mm(origin_x_mm + column * SQUARE_SIZE_MM)
            y = mm(origin_y_mm + row * SQUARE_SIZE_MM)
            commands.append(
                f"{x:.6f} {y:.6f} {square_points:.6f} {square_points:.6f} re f"
            )
    return ("\n".join(commands) + "\n").encode("ascii")


def build_pdf() -> bytes:
    page_width = mm(PAGE_WIDTH_MM)
    page_height = mm(PAGE_HEIGHT_MM)
    content = build_page_content()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /ViewerPreferences << /PrintScaling /None >> >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {page_width:.6f} {page_height:.6f}] "
            f"/TrimBox [0 0 {page_width:.6f} {page_height:.6f}] "
            "/Resources << >> /Contents 4 0 R >>"
        ).encode("ascii"),
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
        + content
        + b"endstream",
    ]

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(
            "camera_calibration_checkerboard_A4_10x7_25mm.pdf"
        ),
    )
    args = parser.parse_args()
    output_path = args.output.resolve()
    output_path.write_bytes(build_pdf())
    print(f"Created: {output_path}")
    print("Page: A4 landscape (297 x 210 mm)")
    print("Board: 10 x 7 squares, 9 x 6 inner corners")
    print("Square: 25.0 mm; print at Actual size / 100%")


if __name__ == "__main__":
    main()

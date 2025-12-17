"""
Overlay visual markers on an image given pixel coordinates.

Supported point formats (either via --point or a --points-file):
- "x,y"
- "x y"
- "label: x,y" or "label x y" (label is optional)

Usage examples:
python utils/mark_points_on_image.py \
  --image downloads/google_maps/images/01_Berkeley_Stadium.png \
  --points-file points.txt \
  --out downloads/google_maps/images/01_Berkeley_Stadium_marked.png

python utils/mark_points_on_image.py \
  --image downloads/google_maps/images/01_Berkeley_Stadium.png \
  --point 120,200 --point "Goal: 330 410" --label-coordinates
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


Point = Tuple[float, float, Optional[str]]


def parse_point_entry(entry: str) -> Point:
    """
    Parse a point string into (x, y, label).
    Accepts "x,y", "x y", "label: x,y", or "label x y".
    """
    raw = entry.strip()
    label: Optional[str] = None
    if ":" in raw:
        maybe_label, coords = raw.split(":", 1)
        label = maybe_label.strip() or None
        raw = coords.strip()

    coord_parts = raw.replace(",", " ").split()
    if len(coord_parts) < 2:
        raise ValueError(f"Cannot parse point '{entry}'. Expected 'x y' or 'x,y'.")
    x, y = float(coord_parts[0]), float(coord_parts[1])
    return x, y, label


def load_points(points_file: Path) -> List[Point]:
    """Load points from a text file, ignoring blank lines and comments."""
    points: List[Point] = []
    with points_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            points.append(parse_point_entry(line))
    return points


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, color: str) -> None:
    """Draw a small crosshair + circle marker."""
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    draw.line((x - radius * 1.4, y, x + radius * 1.4, y), fill=color, width=2)
    draw.line((x, y - radius * 1.4, x, y + radius * 1.4), fill=color, width=2)


def annotate_image(
    image_path: Path,
    points: List[Point],
    out_path: Path,
    radius: int,
    color: str,
    label_coords: bool,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for idx, (x, y, label) in enumerate(points, start=1):
        draw_marker(draw, x, y, radius, color)

        text_parts: List[str] = []
        if label:
            text_parts.append(label)
        if label_coords:
            text_parts.append(f"{int(round(x))},{int(round(y))}")
        if not text_parts:
            text_parts.append(str(idx))

        text = " ".join(text_parts)
        # Offset text slightly so it does not cover the marker.
        draw.text((x + radius + 2, y - radius - 2), text, fill=color, font=font, stroke_width=2, stroke_fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Wrote annotated image to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw pixel-coordinate markers on an image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--points-file", type=Path, help="Text file with one point per line.")
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Point specified inline, e.g. '120,200' or 'Entrance: 50 80'. Can be used multiple times.",
    )
    parser.add_argument("--out", type=Path, help="Output image path. Defaults to <image>_marked.<ext>.")
    parser.add_argument("--radius", type=int, default=6, help="Marker radius in pixels.")
    parser.add_argument("--color", type=str, default="red", help="Marker/text color.")
    parser.add_argument(
        "--label-coordinates",
        action="store_true",
        help="Include 'x,y' text next to each marker (in addition to any labels).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_points: List[Point] = []
    if args.points_file:
        all_points.extend(load_points(args.points_file))
    all_points.extend(parse_point_entry(p) for p in args.point)

    if not all_points:
        raise RuntimeError("Provide at least one point via --point or --points-file.")

    out_path = args.out
    if out_path is None:
        out_path = args.image.with_name(f"{args.image.stem}_marked{args.image.suffix}")

    annotate_image(
        image_path=args.image,
        points=all_points,
        out_path=out_path,
        radius=args.radius,
        color=args.color,
        label_coords=args.label_coordinates,
    )


if __name__ == "__main__":
    main()

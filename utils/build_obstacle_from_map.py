"""
Generate binary obstacle maps (black/white) from Google Maps tiles using
the provided legend in utils/map_legend.json.

How it works:
- Reads the legend entries and their approx_hex_code + walkable flag.
- For each pixel in the map image, finds the closest legend color and
  marks it white if walkable, black otherwise.

Usage examples:
python utils/build_obstacle_from_map.py \
  --images downloads/google_maps/images \
  --out-dir downloads/google_maps/obstacles

python utils/build_obstacle_from_map.py \
  --images downloads/google_maps/images/00_eth_zurich.png \
  --out-dir downloads/google_maps/obstacles
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = hex_code.strip().lstrip("#")
    if len(hex_code) != 6:
        raise ValueError(f"Invalid hex code: {hex_code}")
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return r, g, b


def load_legend(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return legend colors (N,3) uint8 array and walkable mask (N,) bool.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("map_legend", [])
    if not entries:
        raise ValueError(f"No legend entries found in {path}")

    colors: List[Tuple[int, int, int]] = []
    walkable: List[bool] = []
    for entry in entries:
        hex_code = entry.get("approx_hex_code")
        if not hex_code:
            raise ValueError(f"Legend entry missing approx_hex_code: {entry}")
        colors.append(hex_to_rgb(hex_code))
        walkable.append(bool(entry.get("walkable", False)))

    return np.array(colors, dtype=np.uint8), np.array(walkable, dtype=bool)


def compute_obstacle_mask(image: Image.Image, legend_colors: np.ndarray, walkable_mask: np.ndarray) -> Image.Image:
    """
    Map each pixel to the nearest legend color; white if walkable, black otherwise.
    """
    img_arr = np.array(image.convert("RGB"), dtype=np.int16)  # int16 to avoid overflow in subtraction
    # distances shape: (H, W, N)
    diff = img_arr[:, :, None, :] - legend_colors[None, None, :, :].astype(np.int16)
    dist2 = np.sum(diff * diff, axis=3)
    nearest_idx = np.argmin(dist2, axis=2)
    walkable_pixels = walkable_mask[nearest_idx]
    obstacle = np.where(walkable_pixels, 255, 0).astype(np.uint8)
    return Image.fromarray(obstacle, mode="L")


def process_image_file(path: Path, out_dir: Path, legend_colors: np.ndarray, walkable_mask: np.ndarray) -> None:
    image = Image.open(path)
    obstacle = compute_obstacle_mask(image, legend_colors, walkable_mask)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_obstacle.png"
    obstacle.save(out_path)
    print(f"Wrote obstacle map: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate obstacle maps from Google Maps tiles using the map legend.")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to an image file or directory of map tiles (png/jpg).",
    )
    parser.add_argument(
        "--legend",
        type=Path,
        default=Path("utils/map_legend.json"),
        help="Path to map_legend.json with approx_hex_code and walkable fields.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write obstacle maps to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legend_colors, walkable_mask = load_legend(args.legend)

    if args.images.is_dir():
        image_paths = sorted(
            [p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
    else:
        image_paths = [args.images]

    if not image_paths:
        raise ValueError(f"No image files found at {args.images}")

    for img_path in image_paths:
        process_image_file(img_path, args.out_dir, legend_colors, walkable_mask)


if __name__ == "__main__":
    main()

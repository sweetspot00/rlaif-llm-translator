"""
Extract simple line obstacles from a binary obstacle PNG by converting each
connected obstacle blob into its convex hull and exporting the resulting line
segments in PySocialForce format (x1, x2, y1, y2) anchored in meters.

Also saves a visualization where hull edges are rasterized as 1s on a blank map.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from convert_obstacle_to_meter import load_homography, pixel_to_meter_factory


def boundary_segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Enumerate pixel-edge boundary segments between obstacle (True) and background.
    Returns segments as (x0, y0, x1, y1) in pixel units.
    """
    h, w = mask.shape
    segments: set[Tuple[int, int, int, int]] = set()
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]:
                segments.add((x, y, x + 1, y))  # top
            if y == h - 1 or not mask[y + 1, x]:
                segments.add((x, y + 1, x + 1, y + 1))  # bottom
            if x == 0 or not mask[y, x - 1]:
                segments.add((x, y, x, y + 1))  # left
            if x == w - 1 or not mask[y, x + 1]:
                segments.add((x + 1, y, x + 1, y + 1))  # right
    return list(segments)


def _normalize_segment(seg: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = seg
    if x0 == x1 and y0 <= y1:
        return x0, y0, x1, y1
    if y0 == y1 and x0 <= x1:
        return x0, y0, x1, y1
    return x1, y1, x0, y0


def merge_collinear(segments: Sequence[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Merge adjacent collinear axis-aligned segments to reduce count while preserving shape.
    """
    horiz: dict[int, List[Tuple[int, int]]] = {}
    vert: dict[int, List[Tuple[int, int]]] = {}
    for seg in segments:
        x0, y0, x1, y1 = _normalize_segment(seg)
        if y0 == y1:  # horizontal
            horiz.setdefault(y0, []).append((x0, x1))
        elif x0 == x1:  # vertical
            vert.setdefault(x0, []).append((y0, y1))

    merged: List[Tuple[int, int, int, int]] = []

    for y, spans in horiz.items():
        spans.sort()
        cur_start, cur_end = spans[0]
        for s, e in spans[1:]:
            if s <= cur_end:  # contiguous or overlapping
                cur_end = max(cur_end, e)
            elif s == cur_end:  # touching
                cur_end = e
            else:
                merged.append((cur_start, y, cur_end, y))
                cur_start, cur_end = s, e
        merged.append((cur_start, y, cur_end, y))

    for x, spans in vert.items():
        spans.sort()
        cur_start, cur_end = spans[0]
        for s, e in spans[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            elif s == cur_end:
                cur_end = e
            else:
                merged.append((x, cur_start, x, cur_end))
                cur_start, cur_end = s, e
        merged.append((x, cur_start, x, cur_end))

    return merged


def segments_to_psf_meters(
    segments_px: Iterable[Tuple[int, int, int, int]],
    px_to_m: callable,
) -> np.ndarray:
    seg_list = [tuple(map(float, s)) for s in segments_px]
    if not seg_list:
        return np.zeros((0, 4), dtype=np.float64)
    pts_px = np.array(seg_list, dtype=np.float64).reshape(-1, 2)
    pts_m = px_to_m(pts_px).reshape(-1, 2, 2)
    return np.stack(
        [pts_m[:, 0, 0], pts_m[:, 1, 0], pts_m[:, 0, 1], pts_m[:, 1, 1]],
        axis=1,
    )


def rasterize_segments(shape: Tuple[int, int], segments: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]) -> np.ndarray:
    """
    Create an integer mask with 1s along provided segments.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0), (x1, y1) in segments:
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, steps)
        ys = np.linspace(y0, y1, steps)
        xi = np.clip(xs.round().astype(int), 0, w - 1)
        yi = np.clip(ys.round().astype(int), 0, h - 1)
        out[yi, xi] = 1
    return out


def process(
    obstacle_png: Path,
    homography_path: Path,
    out_npz: Path,
    out_plot: Path,
    threshold: int = 127,
) -> np.ndarray:
    image = Image.open(obstacle_png).convert("L")
    mask = np.array(image) <= threshold  # True where obstacle
    segments_px = boundary_segments_from_mask(mask)
    merged_segments_px = merge_collinear(segments_px)

    H = load_homography(homography_path)
    origin_px = (0.0, float(image.height))  # bottom-left anchored at (0,0) meters
    px_to_m = pixel_to_meter_factory(H, origin_px)
    obstacles_m = segments_to_psf_meters(merged_segments_px, px_to_m)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, obstacles=obstacles_m)

    plt.figure(figsize=(8, 7))
    for x0, y0, x1, y1 in merged_segments_px:
        plt.plot([x0, x1], [y0, y1], color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(f"Line obstacles from {obstacle_png.name} (1 = edge)")
    plt.axis("off")
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, bbox_inches="tight", dpi=200)
    plt.close()

    return obstacles_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PySocialForce line obstacles from a simplified obstacle PNG.")
    parser.add_argument("--obstacle-png", type=Path, required=True, help="Path to binary obstacle PNG.")
    parser.add_argument("--homography", type=Path, required=True, help="Path to 3x3 homography txt.")
    parser.add_argument("--out-npz", type=Path, required=True, help="Path to write npz with 'obstacles' array.")
    parser.add_argument("--out-plot", type=Path, required=True, help="Path to write visualization PNG.")
    parser.add_argument("--threshold", type=int, default=127, help="Pixels <= threshold treated as obstacle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obstacles_m = process(
        obstacle_png=args.obstacle_png,
        homography_path=args.homography,
        out_npz=args.out_npz,
        out_plot=args.out_plot,
        threshold=args.threshold,
    )
    print(f"Saved {args.out_npz} with {obstacles_m.shape[0]} segments; plot at {args.out_plot}")


if __name__ == "__main__":
    main()

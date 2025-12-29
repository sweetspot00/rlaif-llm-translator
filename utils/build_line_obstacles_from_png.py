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

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from convert_obstacle_to_meter import load_homography, pixel_to_meter_factory, find_homography as find_homography_exact


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
    out_npz: Path,
    out_plot: Path,
    homography_path: Path | None,
    homographies_dir: Path | None = None,
    threshold: int = 127,
) -> np.ndarray:
    image = Image.open(obstacle_png).convert("L")
    mask = np.array(image) <= threshold  # True where obstacle
    segments_px = boundary_segments_from_mask(mask)
    merged_segments_px = merge_collinear(segments_px)

    if homography_path is None:
        # Try exact match helper first, then prefix, then fuzzy contains match.
        if homographies_dir is None:
            homographies_dir = obstacle_png.parent.parent / "homographies"
        h_exact = find_homography_exact(obstacle_png, homographies_dir)
        if h_exact is not None:
            homography_path = h_exact
        else:
            stem = obstacle_png.stem.lower()
            base_stem = stem.replace("_simplified_obstacle", "").replace("_obstacle", "")
            prefix = obstacle_png.name.split("_", 1)[0].lower()
            candidates = []
            for h in homographies_dir.glob("*.txt"):
                hstem = h.stem.lower()
                if hstem.startswith(prefix + "_") or hstem == prefix:
                    candidates.append(h)
                    continue
                if stem in hstem or hstem in stem or base_stem in hstem or hstem in base_stem:
                    candidates.append(h)
            homography_path = sorted(candidates)[0] if candidates else None
        if homography_path is None:
            raise FileNotFoundError(f"No homography found for {obstacle_png.name} in {homographies_dir}")

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
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--obstacle-png", type=Path, help="Path to a single binary obstacle PNG.")
    target.add_argument(
        "--obstacle-dir",
        type=Path,
        help="Directory of obstacle PNGs to batch process. Defaults to downloads/maps/simplified_obstacles.",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=None,
        help="Path to 3x3 homography txt. If omitted, attempts fuzzy match in --homographies-dir.",
    )
    parser.add_argument(
        "--homographies-dir",
        type=Path,
        default=None,
        help="Directory to search for homography txt when --homography is not provided. Defaults to a sibling 'homographies' folder next to the obstacle PNG.",
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Path to write npz with 'obstacles' array. Defaults to <stem>_anchored.npz beside --out-plot or obstacle.",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Path to write visualization PNG. Defaults to <stem>_anchored.png beside out-npz.",
    )
    parser.add_argument("--threshold", type=int, default=127, help="Pixels <= threshold treated as obstacle.")
    parser.add_argument(
        "--skip-existing-prefixes",
        action="store_true",
        help="When using --obstacle-dir, skip PNGs whose numeric prefix already has an NPZ in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_npz_dir = Path("preprocess/pysfm_obstacles_meter_close_shape")
    default_plot_dir = default_npz_dir / "line_obstacle_img"

    if args.obstacle_dir:
        obstacle_dir = args.obstacle_dir or Path("downloads/maps/simplified_obstacles")
        pngs = sorted(obstacle_dir.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"No PNGs found in {obstacle_dir}")
        if args.out_npz or args.out_plot:
            raise ValueError("--out-npz/--out-plot cannot be used with --obstacle-dir; outputs are derived per file.")

        existing_prefixes = {
            p.name.split("_", 1)[0] for p in default_npz_dir.glob("*.npz")
        } if args.skip_existing_prefixes else set()

        successes = 0
        skipped = 0
        failures: list[tuple[str, Exception]] = []
        for png in pngs:
            prefix = png.name.split("_", 1)[0]
            if prefix in existing_prefixes:
                print(f"Skipping {png.name} (prefix {prefix} already exists in {default_npz_dir})")
                skipped += 1
                continue

            stem = png.stem
            out_npz = default_npz_dir / f"{stem}_anchored.npz"
            out_plot = default_plot_dir / f"{stem}_anchored.png"
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            out_plot.parent.mkdir(parents=True, exist_ok=True)
            try:
                obstacles_m = process(
                    obstacle_png=png,
                    homography_path=args.homography,
                    homographies_dir=args.homographies_dir,
                    out_npz=out_npz,
                    out_plot=out_plot,
                    threshold=args.threshold,
                )
                successes += 1
                print(f"Saved {out_npz} with {obstacles_m.shape[0]} segments; plot at {out_plot}")
            except Exception as exc:  # pylint: disable=broad-except
                failures.append((png.name, exc))
                print(f"Failed {png.name}: {exc}")

        print(
            f"Done. {successes} saved, {skipped} skipped"
            + (f", {len(failures)} failed" if failures else "")
        )
        if failures:
            for name, exc in failures:
                print(f" - {name}: {exc}")
    else:
        stem = args.obstacle_png.stem
        base_out = args.out_npz if args.out_npz else default_npz_dir / f"{stem}_anchored.npz"
        out_npz = base_out
        out_plot = args.out_plot if args.out_plot else default_plot_dir / f"{stem}_anchored.png"

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        obstacles_m = process(
            obstacle_png=args.obstacle_png,
            homography_path=args.homography,
            homographies_dir=args.homographies_dir,
            out_npz=out_npz,
            out_plot=out_plot,
            threshold=args.threshold,
        )
        print(f"Saved {out_npz} with {obstacles_m.shape[0]} segments; plot at {out_plot}")


if __name__ == "__main__":
    main()

"""
Build PySocialForce line obstacles from the GT dataset assets (segmentation +
navmesh walkable area + homography), exporting the obstacles in meters and a
plot overlay for sanity checking.

The pipeline:
- Load walkable masks from navmesh/`*_walkable_area.png` (preferred) and
  segmentation/`*_seg.png` (fallback or intersection).
- Combine them into a single walkable mask; everything else is treated as
  obstacle.
- Extract axis-aligned boundary segments, merge collinear pieces, and convert
  pixel coordinates to meters via the GT homography while anchoring the
  bottom-left pixel at (0, 0).
- Save the obstacles to NPZ (`obstacles` key) and a visualization PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.build_line_obstacles_from_png import (
    boundary_segments_from_mask,
    merge_collinear,
    segments_to_psf_meters,
)
from utils.convert_obstacle_to_meter import load_homography, pixel_to_meter_factory


def _first_match(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _load_walkable_mask(img_path: Path, threshold: int = 127, invert: bool = False) -> np.ndarray:
    """
    Return boolean mask where True means walkable.
    The PNGs in GT are binary-ish; we threshold grayscale values to keep it robust.
    """
    gray = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    mask = gray > threshold
    return ~mask if invert else mask


def _otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def _density_obstacle_mask(density_path: Path, threshold: int | None = None) -> tuple[np.ndarray, int]:
    """
    Build obstacle mask from a grayscale population density map.
    Obstacles = pixels below or equal to threshold (dark regions).
    If threshold is None, use Otsu to auto-pick.
    """
    gray = np.array(Image.open(density_path).convert("L"), dtype=np.uint8)
    thr = threshold if threshold is not None else _otsu_threshold(gray)
    mask = gray <= thr
    return mask, thr


def _discover_assets(seq_dir: Path) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    seg_dir = seq_dir / "segmentation"
    navmesh_dir = seq_dir / "navmesh"
    info_dir = seq_dir / "information"
    homo_dir = seq_dir / "homography"
    img_dir = seq_dir / "image"
    bg_dir = seq_dir / "image_terrain"
    gt_dir = seq_dir / "gt"

    for key, directory, pattern in [
        ("segmentation", seg_dir, "*_seg.png"),
        ("walkable_area", navmesh_dir, "*_walkable_area.png"),
        ("navmesh_json", navmesh_dir, "*_navmesh.json"),
        ("information", info_dir, "*.json"),
        ("homography", homo_dir, "*_H.txt"),
        ("image", img_dir, "*_image.png"),
        ("background", bg_dir, "*_bg.png"),
        ("density", gt_dir, "*_population_density.png"),
    ]:
        found = _first_match(directory, pattern) if directory.exists() else None
        if found:
            assets[key] = found
    return assets


def build_gt_line_obstacle(
    seq_dir: Path,
    out_npz: Path,
    out_plot: Path,
    seg_threshold: int = 127,
    walkable_threshold: int = 127,
    invert_seg: bool = False,
    invert_walkable: bool = False,
    density_threshold: int | None = None,
    out_obstacle_png: Path | None = None,
) -> np.ndarray:
    assets = _discover_assets(seq_dir)
    if "homography" not in assets:
        raise FileNotFoundError(f"No homography file under {seq_dir}/homography")
    if "information" not in assets:
        raise FileNotFoundError(f"No information JSON under {seq_dir}/information")
    if "segmentation" not in assets and "walkable_area" not in assets and "density" not in assets:
        raise FileNotFoundError(f"Need density, segmentation, or walkable_area PNG under {seq_dir}")

    with assets["information"].open("r", encoding="utf-8") as f:
        info = json.load(f)
    height = info.get("height")
    width = info.get("width")
    if height is None or width is None:
        raise ValueError(f"'height'/'width' missing in {assets['information']}")

    # First choice: derive obstacle directly from population density map.
    if "density" in assets:
        obstacle_mask, used_thr = _density_obstacle_mask(assets["density"], threshold=density_threshold)
        if out_obstacle_png:
            out_obstacle_png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((~obstacle_mask).astype(np.uint8) * 255).save(out_obstacle_png)
        print(f"Using density map {assets['density'].name} with threshold {used_thr} for obstacles.")
    else:
        # Fallback: Prefer explicit walkable_area; fall back to segmentation.
        walkable_masks: list[np.ndarray] = []
        if "walkable_area" in assets:
            walkable_masks.append(
                _load_walkable_mask(assets["walkable_area"], threshold=walkable_threshold, invert=invert_walkable)
            )
        if not walkable_masks and "segmentation" in assets:
            walkable_masks.append(
                _load_walkable_mask(assets["segmentation"], threshold=seg_threshold, invert=invert_seg)
            )

        walkable = np.logical_and.reduce(walkable_masks)
        obstacle_mask = ~walkable

    mask_h, mask_w = obstacle_mask.shape
    if mask_h != height or mask_w != width:
        raise ValueError(
            f"Mask shape ({mask_w}x{mask_h}) does not match info ({width}x{height}); check GT assets in {seq_dir}"
        )

    segments_px = boundary_segments_from_mask(obstacle_mask)
    merged_segments_px = merge_collinear(segments_px)

    H = load_homography(assets["homography"])
    origin_px = (0.0, float(height))  # bottom-left anchor
    px_to_m = pixel_to_meter_factory(H, origin_px)
    obstacles_m = segments_to_psf_meters(merged_segments_px, px_to_m)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, obstacles=obstacles_m)

    # Plot: plain line obstacles in pixel coordinates to match the obstacle PNG orientation.
    plt.figure(figsize=(7, 6))
    for x0, y0, x1, y1 in merged_segments_px:
        plt.plot([x0, x1], [y0, y1], color="black", linewidth=0.8)
    plt.gca().invert_yaxis()  # align with image coordinate system (origin top-left)
    plt.axis("equal")
    plt.axis("off")
    plt.title(f"Line obstacles from {seq_dir.name}")

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, bbox_inches="tight", dpi=200)
    plt.close()

    return obstacles_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build line obstacles in meters from GT dataset assets (segmentation + navmesh walkable)."
    )
    parser.add_argument(
        "--seq-dir",
        type=Path,
        required=True,
        help="Sequence folder under downloads/gt (e.g., downloads/gt/eth).",
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Path to write npz with 'obstacles'. Defaults to preprocess/gt_line_obstacles/<seq_name>.npz",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Path to write visualization PNG. Defaults beside out-npz with .png suffix.",
    )
    parser.add_argument(
        "--seg-threshold",
        type=int,
        default=127,
        help="Pixels > threshold in segmentation are treated as walkable.",
    )
    parser.add_argument(
        "--walkable-threshold",
        type=int,
        default=127,
        help="Pixels > threshold in navmesh walkable_area are treated as walkable.",
    )
    parser.add_argument(
        "--invert-seg",
        action="store_true",
        help="Invert segmentation mask (treat dark as walkable).",
    )
    parser.add_argument(
        "--invert-walkable",
        action="store_true",
        help="Invert walkable_area mask if dataset encodes walkable as dark.",
    )
    parser.add_argument(
        "--density-threshold",
        type=int,
        default=None,
        help="Optional fixed threshold for density map (obstacle if gray <= threshold); defaults to Otsu.",
    )
    parser.add_argument(
        "--out-obstacle-png",
        type=Path,
        default=None,
        help="Optional path to save binary obstacle PNG (0 obstacle, 255 free) when using density.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq_dir = args.seq_dir
    stem = seq_dir.name
    default_npz = Path("preprocess/gt_line_obstacles") / f"{stem}_line_obstacles.npz"
    out_npz = args.out_npz if args.out_npz else default_npz
    out_plot = args.out_plot if args.out_plot else out_npz.with_suffix(".png")

    obstacles = build_gt_line_obstacle(
        seq_dir=seq_dir,
        out_npz=out_npz,
        out_plot=out_plot,
        seg_threshold=args.seg_threshold,
        walkable_threshold=args.walkable_threshold,
        invert_seg=args.invert_seg,
        invert_walkable=args.invert_walkable,
        density_threshold=args.density_threshold,
        out_obstacle_png=args.out_obstacle_png,
    )
    print(f"Saved {out_npz} with {obstacles.shape[0]} segments; plot at {out_plot}")


if __name__ == "__main__":
    main()

"""
Convert a binary obstacle PNG into PySocialForce-ready obstacle segments in meters,
re-anchored so the bottom-left pixel maps to (0, 0).

Also exposes a pixel->meter conversion helper that applies the homography and
subtracts the chosen origin.
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Callable

import numpy as np
from PIL import Image


def load_homography(path: Path) -> np.ndarray:
    H = np.loadtxt(path)
    if H.shape != (3, 3):
        raise ValueError(f"Homography at {path} has shape {H.shape}, expected (3, 3).")
    return H


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography to Nx2 points, returning Nx2 points."""
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.hstack([points, ones])
    mapped = (H @ homo.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def pixel_to_meter_factory(H: np.ndarray, origin_px: Tuple[float, float]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a function that maps pixel xy -> meter xy relative to origin_px (which maps to (0,0)).
    origin_px is (x, y) in pixel coordinates.
    """
    origin_m = apply_homography(np.array([origin_px], dtype=np.float64), H)[0]

    def _convert(px_points: np.ndarray) -> np.ndarray:
        m = apply_homography(px_points, H)
        anchored = m - origin_m
        anchored[anchored > -1e-6] = np.maximum(anchored[anchored > -1e-6], 0.0)
        return anchored

    return _convert


def meter_to_pixel_factory(H: np.ndarray, origin_px: Tuple[float, float]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a function that maps anchored meter xy -> pixel xy using the same origin convention.
    The inverse homography is applied after shifting by the origin expressed in meters.
    """
    H_inv = np.linalg.inv(H)
    origin_m = apply_homography(np.array([origin_px], dtype=np.float64), H)[0]

    def _convert(m_points: np.ndarray) -> np.ndarray:
        pts = np.asarray(m_points, dtype=float)
        orig_shape = pts.shape
        pts_flat = pts.reshape(-1, 2)
        world_m = pts_flat + origin_m
        homo = np.hstack([world_m, np.ones((world_m.shape[0], 1), dtype=np.float64)])
        mapped = (H_inv @ homo.T).T
        px = mapped[:, :2] / mapped[:, 2:3]
        return px.reshape(orig_shape)

    return _convert


def obstacle_mask(image: Image.Image, threshold: int) -> np.ndarray:
    """Return boolean mask where True means obstacle pixel."""
    gray = np.array(image.convert("L"))
    return gray <= threshold


def boundary_segments(mask: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Extract boundary edges between obstacle pixels and walkable pixels.
    Each segment is defined in pixel coordinates.
    """
    height, width = mask.shape
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]:
                segments.append(((x, y), (x + 1, y)))
            if y == height - 1 or not mask[y + 1, x]:
                segments.append(((x, y + 1), (x + 1, y + 1)))
            if x == 0 or not mask[y, x - 1]:
                segments.append(((x, y), (x, y + 1)))
            if x == width - 1 or not mask[y, x + 1]:
                segments.append(((x + 1, y), (x + 1, y + 1)))

    return segments


def find_homography(obstacle_path: Path, homographies_dir: Path) -> Path | None:
    stem = obstacle_path.stem
    if stem.endswith("_obstacle"):
        stem = stem[: -len("_obstacle")]
    candidate = homographies_dir / f"{stem}.txt"
    if candidate.exists():
        return candidate
    return None


def segments_to_psf(
    segments_px: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]],
    px_to_m: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    seg_arr = np.array(list(segments_px), dtype=np.float64)
    if seg_arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    points_px = seg_arr.reshape(-1, 2)
    points_m = px_to_m(points_px).reshape(-1, 2, 2)
    return np.stack(
        [points_m[:, 0, 0], points_m[:, 1, 0], points_m[:, 0, 1], points_m[:, 1, 1]],
        axis=1,
    )


def process_image(
    obstacle_path: Path,
    homographies_dir: Path,
    out_dir: Path,
    threshold: int,
    stride: int,
) -> Path | None:
    homography_path = find_homography(obstacle_path, homographies_dir)
    if homography_path is None:
        print(f"Skipping {obstacle_path.name}: no matching homography found.")
        return None

    image = Image.open(obstacle_path)
    mask = obstacle_mask(image, threshold)
    segments_px = boundary_segments(mask)
    if stride > 1:
        segments_px = segments_px[::stride]

    H = load_homography(homography_path)
    origin_px = (0.0, float(image.height))  # bottom-left edge anchors to (0,0)
    px_to_m = pixel_to_meter_factory(H, origin_px)
    psf_segments = segments_to_psf(segments_px, px_to_m)

    if np.any(psf_segments < -1e-9):
        min_val = float(psf_segments.min())
        raise ValueError(f"Anchored obstacle contains negative coordinates (min={min_val}) for {obstacle_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{homography_path.stem}_anchored.npy"
    np.save(out_path, psf_segments)
    print(f"Wrote {out_path} ({psf_segments.shape[0]} segments)")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert obstacle PNG to meter-anchored PySocialForce obstacle segments."
    )
    parser.add_argument(
        "--obstacles-dir",
        type=Path,
        default=Path("downloads/google_maps/obstacles"),
        help="Directory of binary obstacle maps (PNG).",
    )
    parser.add_argument(
        "--homographies-dir",
        type=Path,
        default=Path("downloads/google_maps/homographies"),
        help="Directory of 3x3 pixel->meter homography TXT files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preprocessing/pysocialforce_obstacles_meter"),
        help="Directory to write meter-anchored obstacle segments.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Grayscale threshold; pixels <= threshold are treated as obstacles.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth boundary segment to reduce density (1 keeps all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obstacle_paths = sorted(p for p in args.obstacles_dir.glob("*.png"))
    if not obstacle_paths:
        raise ValueError(f"No obstacle PNGs found in {args.obstacles_dir}")

    for path in obstacle_paths:
        process_image(
            obstacle_path=path,
            homographies_dir=args.homographies_dir,
            out_dir=args.out_dir,
            threshold=args.threshold,
            stride=max(1, args.stride),
        )


if __name__ == "__main__":
    main()

import io
import math
import os
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import requests
from PIL import Image

# Constants for Web Mercator (math behind Google Maps)
EARTH_RADIUS = 6378137.0


def latlon_to_meters(lat: float, lon: float) -> Tuple[float, float]:
    """Convert WGS84 lat/lon to Web Mercator meters."""
    mx = lon * (math.pi * EARTH_RADIUS / 180.0)
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * (math.pi * EARTH_RADIUS / 180.0)
    return mx, my


def get_meters_per_pixel(zoom: int, lat: float, scale: int = 1) -> float:
    """Meters represented by a single pixel at a given zoom/latitude."""
    # scale=2 doubles the pixels for the same ground coverage, so the
    # ground resolution halves.
    return 156543.03392 * math.cos(lat * math.pi / 180) / (2 ** zoom * scale)


def build_affine_matrix(
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    scale: int,
) -> np.ndarray:
    """
    Construct a 3x3 affine matrix mapping pixel coordinates [x, y, 1]
    to Web Mercator meters.
    """
    scale_x = get_meters_per_pixel(zoom, center_lat, scale)
    scale_y = -scale_x  # Y increases downward in images.

    center_mx, center_my = latlon_to_meters(center_lat, center_lon)
    top_left_mx = center_mx - (width / 2 * scale_x)
    top_left_my = center_my - (height / 2 * scale_y)

    return np.array(
        [
            [scale_x, 0.0, top_left_mx],
            [0.0, scale_y, top_left_my],
            [0.0, 0.0, 1.0],
        ]
    )


def fetch_map_image(
    api_key: str,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int = 640,
    height: int = 640,
    maptype: str = "roadmap",
    scale: int = 1,
    hide_labels: bool = True,
    crop_attribution_px: int = 40,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Download a Google Maps Static API image and return it with the
    pixel->meter affine matrix. Optionally crop the bottom attribution bar.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{center_lat},{center_lon}",
        "zoom": zoom,
        "size": f"{width}x{height}",
        "maptype": maptype,  # "roadmap" is the default (non-satellite)
        "key": api_key,
        "scale": scale,
    }
    if hide_labels:
        # Remove text labels/markers for a cleaner map.
        params["style"] = "element:labels|visibility:off"

    response = requests.get(base_url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Static Maps error {response.status_code}: {response.text}")

    image = Image.open(io.BytesIO(response.content))
    if crop_attribution_px > 0 and image.height > crop_attribution_px:
        # Remove Google attribution/footer text. Homography remains valid for the
        # retained pixels because origin/scale are unchanged.
        new_height = image.height - crop_attribution_px
        image = image.crop((0, 0, image.width, new_height))
        height = new_height

    homography = build_affine_matrix(center_lat, center_lon, zoom, width, height, scale)
    return image, homography


def _slug(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "location"


def download_coordinate_list(
    api_key: str,
    coords: Iterable[Tuple[str, float, float]],
    zoom: int,
    images_dir: Path,
    homographies_dir: Path,
    width: int = 640,
    height: int = 640,
    maptype: str = "roadmap",
    scale: int = 1,
    hide_labels: bool = True,
    crop_attribution_px: int = 40,
) -> List[Dict[str, object]]:
    """
    Fetch a default Google Maps image for each (name, lat, lon) triple,
    save it, and return metadata with the homography matrices. Each image
    gets a matching TXT file containing its 3x3 homography matrix.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    homographies_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []

    for idx, (name, lat, lon) in enumerate(coords):
        image, H = fetch_map_image(
            api_key,
            center_lat=lat,
            center_lon=lon,
            zoom=zoom,
            width=width,
            height=height,
            maptype=maptype,
            scale=scale,
            hide_labels=hide_labels,
            crop_attribution_px=crop_attribution_px,
        )
        filename = images_dir / f"{idx:02d}_{_slug(name)}.png"
        image.save(filename)
        homography_path = homographies_dir / f"{idx:02d}_{_slug(name)}.txt"
        np.savetxt(homography_path, H, fmt="%.12f")

        results.append(
            {
                "name": name,
                "lat": lat,
                "lon": lon,
                "image_path": filename,
                "homography_path": homography_path,
                "homography": H,
            }
        )

    return results


def read_coordinates_file(path: Path, max_count: int | None = None) -> List[Tuple[str, float, float]]:
    """
    Read coordinates from a TXT file with lines formatted:
        name, lat, lon
    Lines starting with # or blank lines are ignored.
    """
    coords: List[Tuple[str, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                raise ValueError(f"Line {idx} in {path} is malformed: '{line}'")
            name = ",".join(parts[:-2]).strip() if len(parts) > 3 else parts[0]
            try:
                lat = float(parts[-2])
                lon = float(parts[-1])
            except ValueError as exc:
                raise ValueError(f"Line {idx} in {path} has invalid lat/lon: '{line}'") from exc
            coords.append((name, lat, lon))
            if max_count is not None and len(coords) >= max_count:
                break
    if not coords:
        raise ValueError(f"No coordinates found in {path}")
    return coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Google Maps tiles and homographies for a list of coordinates.")
    parser.add_argument("--api-key", type=str, default=os.getenv("GOOGLE_MAPS_API_KEY"), help="Google Maps API key.")
    parser.add_argument("--coords-file", type=Path, required=True, help="TXT file with 'name, lat, lon' per line.")
    parser.add_argument("--zoom", type=int, default=18, help="Google Maps zoom level.")
    parser.add_argument("--maptype", type=str, default="roadmap", choices=["roadmap", "satellite", "hybrid", "terrain"])
    parser.add_argument("--scale", type=int, default=1, choices=[1, 2], help="Static Maps scale (1 or 2).")
    parser.add_argument("--width", type=int, default=640, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=640, help="Image height in pixels (before cropping).")
    parser.add_argument("--hide-labels", action="store_true", default=True, help="Hide labels for cleaner maps.")
    parser.add_argument("--show-labels", dest="hide_labels", action="store_false", help="Keep labels.")
    parser.add_argument("--crop-attribution-px", type=int, default=40, help="Pixels to crop from bottom to remove attribution.")
    parser.add_argument("--images-dir", type=Path, default=Path("downloads/google_maps/images"))
    parser.add_argument("--homographies-dir", type=Path, default=Path("downloads/google_maps/homographies"))
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Only download the first N entries from the coords file (e.g., 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("Provide --api-key or set GOOGLE_MAPS_API_KEY.")

    coords = read_coordinates_file(args.coords_file, max_count=args.max_count)

    tiles = download_coordinate_list(
        api_key=args.api_key,
        coords=coords,
        zoom=args.zoom,
        images_dir=args.images_dir,
        homographies_dir=args.homographies_dir,
        width=args.width,
        height=args.height,
        maptype=args.maptype,
        scale=args.scale,
        hide_labels=args.hide_labels,
        crop_attribution_px=args.crop_attribution_px,
    )

    print(f"Downloaded {len(tiles)} tiles to {args.images_dir} and homographies to {args.homographies_dir}")

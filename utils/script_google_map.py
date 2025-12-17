import io
import math
import os
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
) -> Tuple[Image.Image, np.ndarray]:
    """
    Download a Google Maps Static API image and return it with the
    pixel->meter affine matrix.
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


if __name__ == "__main__":
    # Example usage; set GOOGLE_MAPS_API_KEY or replace the placeholder.
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_API_KEY")
    locations = [
        ("eth_zurich", 47.3763, 8.5477),
        ("times_square", 40.758, -73.9855),
    ]
    base_out = Path("downloads/google_maps")

    try:
        tiles = download_coordinate_list(
            api_key=api_key,
            coords=locations,
            zoom=18,
            images_dir=base_out / "images",
            homographies_dir=base_out / "homographies",
            maptype="roadmap",  # Use "satellite" if you need satellite imagery.
            scale=1,
            hide_labels=True,
        )

        for tile in tiles:
            print(f"{tile['name']}: saved to {tile['image_path']}")
            print("Homography (pixel -> Web Mercator meters):")
            print(tile["homography"])
            print()

    except Exception as exc:
        print(f"Failed to download maps: {exc}")

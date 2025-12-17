"""
Geocode place names to coordinates and write them to a text file in
`name, lat, lon` format.

Requires a Google Geocoding API key.

Example:
python utils/geocode_places.py \
  --api-key "$GOOGLE_MAPS_API_KEY" \
  --names-file names_to_geocode.txt \
  --out-file downloads/world_crowd_locations.txt \
  --append
"""

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import requests


def read_names(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    if not names:
        raise ValueError(f"No place names found in {path}")
    return names


def geocode(api_key: str, name: str, region: str | None = None) -> Tuple[float, float] | None:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": name, "key": api_key}
    if region:
        params["region"] = region
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status")
    if status == "ZERO_RESULTS":
        # Nothing found; skip gracefully.
        return None
    if status != "OK":
        raise RuntimeError(f"Geocoding failed for '{name}': {status} ({data.get('error_message','')})")
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])


def write_rows(path: Path, rows: Iterable[Tuple[str, float, float]], append: bool) -> None:
    mode = "a" if append else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for name, lat, lon in rows:
            f.write(f"{name}, {lat:.8f}, {lon:.8f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geocode a list of place names to lat/lon.")
    parser.add_argument("--api-key", type=str, default=os.getenv("GOOGLE_MAPS_API_KEY"), required=False)
    parser.add_argument("--names-file", type=Path, required=True, help="Text file with one place name per line.")
    parser.add_argument("--out-file", type=Path, required=True, help="Output file for 'name, lat, lon' rows.")
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwrite.")
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region bias (ccTLD, e.g., 'us', 'uk', 'jp') to disambiguate names.",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between requests to respect quotas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("Provide --api-key or set GOOGLE_MAPS_API_KEY")

    names = read_names(args.names_file)
    rows: List[Tuple[str, float, float]] = []
    for name in names:
        result = geocode(args.api_key, name, region=args.region)
        if result is None:
            print(f"Skip (no results): {name}")
            continue
        lat, lon = result
        rows.append((name, lat, lon))
        time.sleep(args.sleep)

    write_rows(args.out_file, rows, append=args.append)
    print(f"Wrote {len(rows)} locations to {args.out_file} ({'appended' if args.append else 'overwrote'})")


if __name__ == "__main__":
    main()

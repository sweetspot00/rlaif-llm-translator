"""
Generate a list of crowd-heavy, human-centered locations worldwide using the
Google Places API and save them to a text file as:
    Place Name, lat, lon

Categories sampled include campuses, sightseeing spots, plazas, transit hubs,
stadiums, museums, malls, parks, etc. Roads/vehicle-only features are avoided.

Usage:
python utils/generate_world_crowd_locations.py \
  --api-key YOUR_API_KEY \
  --output downloads/world_crowd_locations.txt \
  --target-count 500

Notes:
- Requires Google Places API access.
- Respects pagination; may take a few seconds due to API next_page_token delays.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import requests


SEED_LOCATIONS: List[Tuple[str, float, float, int]] = [
    # name, lat, lon, radius_m
    ("New York", 40.7580, -73.9855, 3000),
    ("Boston", 42.3601, -71.0589, 2500),
    ("Chicago", 41.8781, -87.6298, 3000),
    ("San Francisco", 37.7749, -122.4194, 3000),
    ("Los Angeles", 34.0522, -118.2437, 3000),
    ("Washington DC", 38.9072, -77.0369, 3000),
    ("Miami", 25.7617, -80.1918, 2500),
    ("Toronto", 43.6532, -79.3832, 2500),
    ("Mexico City", 19.4326, -99.1332, 3000),
    ("Rio de Janeiro", -22.9068, -43.1729, 3000),
    ("Buenos Aires", -34.6037, -58.3816, 3000),
    ("London", 51.5074, -0.1278, 3000),
    ("Paris", 48.8566, 2.3522, 3000),
    ("Berlin", 52.5200, 13.4050, 3000),
    ("Amsterdam", 52.3676, 4.9041, 2500),
    ("Madrid", 40.4168, -3.7038, 3000),
    ("Barcelona", 41.3851, 2.1734, 3000),
    ("Rome", 41.9028, 12.4964, 3000),
    ("Vienna", 48.2082, 16.3738, 2500),
    ("Prague", 50.0755, 14.4378, 2500),
    ("Athens", 37.9838, 23.7275, 2500),
    ("Istanbul", 41.0082, 28.9784, 3000),
    ("Dubai", 25.2048, 55.2708, 3000),
    ("Cairo", 30.0444, 31.2357, 3000),
    ("Nairobi", -1.2921, 36.8219, 2500),
    ("Lagos", 6.5244, 3.3792, 3000),
    ("Johannesburg", -26.2041, 28.0473, 3000),
    ("Cape Town", -33.9249, 18.4241, 2500),
    ("Moscow", 55.7558, 37.6173, 3000),
    ("Mumbai", 19.0760, 72.8777, 3000),
    ("Delhi", 28.6139, 77.2090, 3000),
    ("Bengaluru", 12.9716, 77.5946, 3000),
    ("Singapore", 1.3521, 103.8198, 2500),
    ("Bangkok", 13.7563, 100.5018, 3000),
    ("Kuala Lumpur", 3.1390, 101.6869, 3000),
    ("Hong Kong", 22.3193, 114.1694, 2500),
    ("Seoul", 37.5665, 126.9780, 3000),
    ("Tokyo", 35.6762, 139.6503, 3000),
    ("Osaka", 34.6937, 135.5023, 2500),
    ("Kyoto", 35.0116, 135.7681, 2500),
    ("Taipei", 25.0330, 121.5654, 3000),
    ("Sydney", -33.8688, 151.2093, 3000),
    ("Melbourne", -37.8136, 144.9631, 3000),
    ("Auckland", -36.8485, 174.7633, 2500),
    ("Santiago", -33.4489, -70.6693, 3000),
    ("Lima", -12.0464, -77.0428, 3000),
    ("Stockholm", 59.3293, 18.0686, 2500),
    ("Copenhagen", 55.6761, 12.5683, 2500),
    ("Helsinki", 60.1699, 24.9384, 2500),
    ("Oslo", 59.9139, 10.7522, 2500),
    ("Dublin", 53.3498, -6.2603, 2500),
    ("Edinburgh", 55.9533, -3.1883, 2500),
    ("Zurich", 47.3769, 8.5417, 2500),
    ("Munich", 48.1351, 11.5820, 3000),
    ("Lisbon", 38.7223, -9.1393, 2500),
    ("Brussels", 50.8503, 4.3517, 2500),
    ("Warsaw", 52.2297, 21.0122, 3000),
    ("Budapest", 47.4979, 19.0402, 3000),
    ("Mexico Guadalajara", 20.6597, -103.3496, 3000),
    ("Saopaulo", -23.5505, -46.6333, 3000),
    ("Seville", 37.3891, -5.9845, 2500),
    ("Valencia", 39.4699, -0.3763, 2500),
]

PLACE_TYPES: List[str] = [
    "university",
    "tourist_attraction",
    "train_station",
    "transit_station",
    "subway_station",
    "light_rail_station",
    "bus_station",
    "museum",
    "stadium",
    "park",
    "shopping_mall",
    "place_of_worship",
    "city_hall",
    "library",
    "night_club",
    "zoo",
    "amusement_park",
]


def fetch_places(
    api_key: str,
    location: Tuple[float, float],
    radius_m: int,
    place_types: Iterable[str],
    max_results_per_seed: int = 80,
    sleep_between_pages: float = 2.0,
) -> List[Dict]:
    """Fetch nearby places for a given location across multiple types."""
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    results: List[Dict] = []
    for place_type in place_types:
        params = {
            "key": api_key,
            "location": f"{location[0]},{location[1]}",
            "radius": radius_m,
            "type": place_type,
        }
        page_token = None
        fetched_for_type = 0
        while True:
            if page_token:
                params["pagetoken"] = page_token
                time.sleep(sleep_between_pages)
            resp = requests.get(base_url, params=params, timeout=20)
            if resp.status_code != 200:
                break
            payload = resp.json()
            if payload.get("status") not in {"OK", "ZERO_RESULTS"}:
                break
            results.extend(payload.get("results", []))
            fetched_for_type += len(payload.get("results", []))
            if fetched_for_type >= max_results_per_seed:
                break
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    return results


def collect_locations(
    api_key: str,
    target_count: int,
    seeds: List[Tuple[str, float, float, int]],
    place_types: List[str],
) -> List[Tuple[str, float, float]]:
    seen_ids: Set[str] = set()
    collected: List[Tuple[str, float, float]] = []
    for seed_name, lat, lon, radius in seeds:
        places = fetch_places(api_key, (lat, lon), radius, place_types)
        for p in places:
            place_id = p.get("place_id")
            if not place_id or place_id in seen_ids:
                continue
            geometry = p.get("geometry", {}).get("location", {})
            plat = geometry.get("lat")
            plon = geometry.get("lng")
            if plat is None or plon is None:
                continue
            name = p.get("name", "Unknown")
            collected.append((name, float(plat), float(plon)))
            seen_ids.add(place_id)
            if len(collected) >= target_count:
                return collected
    return collected


def write_locations(out_path: Path, rows: List[Tuple[str, float, float]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for name, lat, lon in rows:
            f.write(f"{name}, {lat:.6f}, {lon:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate crowd-heavy location coordinates via Google Places API.")
    parser.add_argument("--api-key", type=str, default=os.getenv("GOOGLE_MAPS_API_KEY"), required=False)
    parser.add_argument("--output", type=Path, default=Path("downloads/world_crowd_locations.txt"))
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--max-per-seed", type=int, default=80, help="Max results per seed city per place type.")
    parser.add_argument("--radius", type=int, default=None, help="Override radius (meters) for all seeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Provide --api-key or set GOOGLE_MAPS_API_KEY")

    seeds = SEED_LOCATIONS
    if args.radius:
        seeds = [(name, lat, lon, args.radius) for name, lat, lon, _ in SEED_LOCATIONS]

    locations = collect_locations(
        api_key=args.api_key,
        target_count=args.target_count,
        seeds=seeds,
        place_types=PLACE_TYPES,
    )
    if not locations:
        raise RuntimeError("No locations collected; check API key/permissions.")

    write_locations(args.output, locations)
    print(f"Wrote {len(locations)} locations to {args.output}")


if __name__ == "__main__":
    main()

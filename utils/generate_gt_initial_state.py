"""
Generate PySocialForce initial_state arrays from a GT origin/goal CSV.

Input CSV format (columns):
scene, agent_id, agent_type, frame_origin, origin_x, origin_y,
frame_goal, goal_x, goal_y, preferred_speed

Output: N x 6 NumPy array with rows (mx, my, vx, vy, gx_m, gy_m) in meters,
anchored so bottom-left pixel is (0, 0). Velocities default to zero or can be
pointed toward the goal using preferred_speed (interpreted in meters/sec).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from convert_obstacle_to_meter import load_homography, pixel_to_meter_factory


def _convert_px_to_m(px_to_m, points_px: np.ndarray) -> np.ndarray:
    """
    Anchor pixel coordinates so bottom-left is (0,0) and convert to meters.
    """
    points_px = np.asarray(points_px, dtype=float)
    if points_px.ndim == 1:
        points_px = points_px[None, :]
    anchored = px_to_m(points_px)
    anchored[anchored > -1e-6] = np.maximum(anchored[anchored > -1e-6], 0.0)
    return anchored


def load_origin_goal(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # deterministic order by agent_id then frame_origin
    rows.sort(key=lambda r: (int(r["agent_id"]), int(r["frame_origin"])))
    return rows


def to_initial_state(
    rows: Iterable[dict[str, str]],
    use_goal_velocity: bool = False,
    px_to_m=None,
    eps: float = 1e-8,
) -> np.ndarray:
    state_rows: list[list[float]] = []
    for row in rows:
        px = float(row["origin_x"])
        py = float(row["origin_y"])
        gx = float(row["goal_x"])
        gy = float(row["goal_y"])
        if px_to_m is not None:
            anchored = _convert_px_to_m(px_to_m, np.array([[px, py], [gx, gy]]))
            (px, py), (gx, gy) = anchored
        vx = 0.0
        vy = 0.0

        if use_goal_velocity:
            preferred_speed = float(row.get("preferred_speed") or 0.0)
            dx = gx - px
            dy = gy - py
            norm = (dx * dx + dy * dy) ** 0.5
            if preferred_speed > 0.0 and norm > eps:
                scale = preferred_speed / norm
                vx = dx * scale
                vy = dy * scale

        state_rows.append([px, py, vx, vy, gx, gy])

    return np.array(state_rows, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GT origin/goal CSV into a PySocialForce initial_state array."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        help="Path to origin_goal CSV (e.g., seq_eth_origin_goal.csv).",
    )
    parser.add_argument(
        "--csv-path",
        "--csv_path",
        dest="csv_path_kw",
        type=Path,
        help="Alternative way to pass the origin_goal CSV path.",
    )
    parser.add_argument(
        "--out",
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Directory to write outputs. Saves <prefix>_initial_state.npy inside.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for outputs. Defaults to the CSV stem without '_origin_goal' if present.",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        required=True,
        help="Homography TXT path for px->m conversion (anchored at bottom-left).",
    )
    parser.add_argument(
        "--info-json",
        type=Path,
        required=True,
        help="Information JSON containing image height (used for anchoring bottom-left to (0,0) in meters).",
    )
    parser.add_argument(
        "--use-goal-velocity",
        action="store_true",
        help="Initialize velocity toward the goal using preferred_speed.",
    )
    args = parser.parse_args()
    csv_path = args.csv_path or args.csv_path_kw
    if csv_path is None:
        parser.error("csv_path is required (positional or --csv-path).")
    args.csv_path = csv_path
    return args


def main() -> None:
    args = parse_args()
    with args.info_json.open("r", encoding="utf-8") as f:
        info = json.load(f)
    height = info.get("height")
    if height is None:
        raise ValueError(f"'height' missing in {args.info_json}")
    H = load_homography(args.homography)
    px_to_m = pixel_to_meter_factory(H, origin_px=(0.0, float(height)))

    rows = load_origin_goal(args.csv_path)
    initial_state = to_initial_state(rows, use_goal_velocity=args.use_goal_velocity, px_to_m=px_to_m)

    if args.out_dir:
        prefix = args.prefix
        if prefix is None:
            stem = args.csv_path.stem
            prefix = stem.removesuffix("_origin_goal")
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{prefix}_initial_state.npy"
        np.save(npy_path, initial_state)
        print(f"Saved initial_state array with shape {initial_state.shape} to {npy_path}")
    else:
        print("initial_state shape:", initial_state.shape)
        print(initial_state)


if __name__ == "__main__":
    main()

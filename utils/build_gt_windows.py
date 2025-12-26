"""
Slice GT dense trajectories into start-frame windows and emit separate preprocessed JSONs.

For each window:
- select agents whose first frame lies in [window_start, window_end]
- take their first position in the window as (px, py) and their last position in the window as (gx, gy)
- velocities are initialized to zero
- write a scene JSON (compatible with sim.py) with window-specific scene_id

Window sizing:
- You provide a base window size (--window-size, default 300 frames).
- Windows are built greedily from the sorted start frames; if a window has fewer than
  --min-agents (default 20), it will expand forward to include more start frames until
  the threshold is met or no more agents remain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from convert_obstacle_to_meter import load_homography, pixel_to_meter_factory


def load_prompt(prompt_file: Path, scene_key: str) -> str:
    text = prompt_file.read_text(encoding="utf-8").strip()
    # accept comma-separated prompt file too
    if "," in text:
        for line in text.splitlines():
            if not line.strip() or "," not in line:
                continue
            key, desc = line.split(",", 1)
            if key.strip() == scene_key:
                return desc.strip()
    return text


def _convert_px_to_m(px_to_m, points_px: np.ndarray) -> np.ndarray:
    points_px = np.asarray(points_px, dtype=float)
    if points_px.ndim == 1:
        points_px = points_px[None, :]
    anchored = px_to_m(points_px)
    anchored[anchored > -1e-6] = np.maximum(anchored[anchored > -1e-6], 0.0)
    return anchored


def read_trajectories(csv_path: Path) -> dict[str, list[tuple[int, float, float]]]:
    agents: dict[str, list[tuple[int, float, float]]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row["agent_id"]
            frame = int(row["frame"])
            x, y = float(row["x"]), float(row["y"])
            agents.setdefault(aid, []).append((frame, x, y))
    # sort per agent
    for v in agents.values():
        v.sort(key=lambda t: t[0])
    return agents


def build_windows(
    agents: dict[str, list[tuple[int, float, float]]], base_size: int, min_agents: int
) -> list[tuple[int, int, list[str]]]:
    """
    Returns list of (start, end, agent_ids) windows.
    Windows are built over the start frames.
    """
    starts = sorted((min(fr for fr, _, _ in traj), aid) for aid, traj in agents.items())
    idx = 0
    windows: list[tuple[int, int, list[str]]] = []
    n = len(starts)
    while idx < n:
        start_frame = starts[idx][0]
        end_frame = start_frame + base_size - 1
        collected: list[str] = []
        j = idx
        while j < n and starts[j][0] <= end_frame:
            collected.append(starts[j][1])
            j += 1
        # if not enough, expand to include more start frames
        while len(collected) < min_agents and j < n:
            end_frame = starts[j][0] + base_size - 1
            while j < n and starts[j][0] <= end_frame:
                collected.append(starts[j][1])
                j += 1
        windows.append((start_frame, end_frame, collected))
        idx = j
    return windows


def initial_state_for_window(
    agent_ids: list[str],
    agents: dict[str, list[tuple[int, float, float]]],
    frame_start: int,
    frame_end: int,
    px_to_m,
) -> np.ndarray:
    rows: list[list[float]] = []
    for aid in agent_ids:
        traj = agents[aid]
        # filter frames inside window
        in_window = [(f, x, y) for f, x, y in traj if frame_start <= f <= frame_end]
        if not in_window:
            continue
        f0, x0, y0 = in_window[0]
        f_last, x1, y1 = in_window[-1]
        (mx0, my0), (mx1, my1) = _convert_px_to_m(px_to_m, np.array([[x0, y0], [x1, y1]]))
        rows.append([mx0, my0, 0.0, 0.0, mx1, my1])
    return np.asarray(rows, dtype=float)


def make_scene_payload(
    scene_key: str,
    window_idx: int,
    frame_start: int,
    frame_end: int,
    initial_state: np.ndarray,
    prompt: str,
    anchored_obstacles: Path,
    homography: Path,
) -> dict:
    scene_id = f"{scene_key}_win{window_idx}_{frame_start}_{frame_end}"
    return {
        "scene_id": scene_id,
        "scene_index": window_idx,
        "scenario": prompt,
        "category": "GT",
        "crowd_size_label": str(initial_state.shape[0]),
        "crowd_size": int(initial_state.shape[0]),
        "ungrouped_agents": int(initial_state.shape[0]),
        "event_center_px": [],
        "event_center_m": [],
        "goals_px": [],
        "goals_m": [],
        "initial_state": initial_state.tolist(),
        "groups": [],
        "assets": {
            "anchored_obstacles": str(anchored_obstacles),
            "homography": str(homography),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-window preprocessed scenes from dense GT trajectories.")
    p.add_argument("--scene", default="eth", help="Scene key (used in ids and prompt lookup).")
    p.add_argument(
        "--trajectory",
        type=Path,
        default=Path("downloads/gt/eth/trajectory_dense/seq_eth_trajectory_dense.csv"),
        help="Dense trajectory CSV path.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("preprocess/gt/gt_scene/eth_prompt.txt"),
        help="Prompt text file (single prompt or keyed CSV-style).",
    )
    p.add_argument(
        "--homography",
        type=Path,
        default=Path("downloads/gt/eth/homography/seq_eth_H.txt"),
        help="Homography TXT path.",
    )
    p.add_argument(
        "--info-json",
        type=Path,
        default=Path("downloads/gt/eth/information/seq_eth_info.json"),
        help="Information JSON with height/width.",
    )
    p.add_argument(
        "--anchored-obstacles",
        type=Path,
        default=Path("preprocess/gt/gt_line_obstacles/eth_line_obstacles.npz"),
        help="Anchored obstacles NPZ path.",
    )
    p.add_argument("--window-size", type=int, default=300, help="Base window size in frames.")
    p.add_argument("--min-agents", type=int, default=20, help="Minimum agents per window.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preprocess/preprocessed_scene"),
        help="Directory to write windowed scene JSONs.",
    )
    p.add_argument(
        "--gt-split-dir",
        type=Path,
        default=Path("downloads/gt/eth/trajectory_dense_windows"),
        help="Directory to write per-window dense GT CSV slices (frames shifted to window start).",
    )
    p.add_argument(
        "--no-gt-splits",
        action="store_true",
        help="Skip writing per-window GT CSV slices.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompt = load_prompt(args.prompt_file, args.scene)

    with args.info_json.open("r", encoding="utf-8") as f:
        info = json.load(f)
    height = info.get("height")
    if height is None:
        raise ValueError(f"'height' missing in {args.info_json}")
    H = load_homography(args.homography)
    px_to_m = pixel_to_meter_factory(H, origin_px=(0.0, float(height)))

    agents = read_trajectories(args.trajectory)
    windows = build_windows(agents, args.window_size, args.min_agents)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_gt_splits:
        args.gt_split_dir.mkdir(parents=True, exist_ok=True)
    for idx, (f_start, f_end, agent_ids) in enumerate(windows):
        init_state = initial_state_for_window(agent_ids, agents, f_start, f_end, px_to_m)
        if init_state.shape[0] == 0:
            continue
        payload = make_scene_payload(
            scene_key=args.scene,
            window_idx=idx,
            frame_start=f_start,
            frame_end=f_end,
            initial_state=init_state,
            prompt=prompt,
            anchored_obstacles=args.anchored_obstacles,
            homography=args.homography,
        )
        out_path = args.out_dir / f"{args.scene}_win{idx:03d}_{f_start}_{f_end}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} (agents={init_state.shape[0]})")
        if not args.no_gt_splits:
            gt_csv = args.gt_split_dir / f"{args.scene}_win{idx:03d}_{f_start}_{f_end}.csv"
            with gt_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scene", "agent_id", "agent_type", "frame", "x", "y"])
                for aid in agent_ids:
                    for frame, x, y in agents[aid]:
                        if frame < f_start or frame > f_end:
                            continue
                        writer.writerow(["seq_eth", aid, "0", frame - f_start, x, y])
            print(f"Wrote GT slice {gt_csv}")


if __name__ == "__main__":
    main()

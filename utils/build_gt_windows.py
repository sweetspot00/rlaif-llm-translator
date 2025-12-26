"""
Slice GT trajectories into start-frame windows and emit separate preprocessed JSONs.

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
from typing import Dict, List, Tuple, Callable

import numpy as np


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


def read_trajectories_obsmat(obsmat_path: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """
    obsmat.txt format: time/frame, ped_id, world_x, ?, world_y, vel_y, acc_x, acc_y (meters).
    We round the time column to int frames for windowing. Positions use col 2 (x) and col 4 (y),
    velocities use col 5 (vx) and col 7 (vy).
    """
    agents: dict[str, list[tuple[int, float, float, float, float]]] = {}
    with obsmat_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            t = int(round(float(parts[0])))
            pid = parts[1]
            x = float(parts[2])
            y = float(parts[4])
            vx = float(parts[5])
            vy = float(parts[7])
            agents.setdefault(pid, []).append((t, x, y, vx, vy))
    for v in agents.values():
        v.sort(key=lambda t: t[0])
    return agents


def build_windows_greedy(
    agents: dict[str, list[tuple[int, float, float, float, float]]],
    base_size: int,
    min_agents: int,
    synchronized: bool = False,
) -> list[tuple[int, int, list[str]]]:
    """
    Greedy windows over agent start frames. When synchronized=True, group agents with identical
    start frames; otherwise expand to include starts until min_agents.
    """
    starts = sorted((min(item[0] for item in traj), aid) for aid, traj in agents.items())
    if synchronized:
        by_start: dict[int, list[str]] = {}
        for sf, aid in starts:
            by_start.setdefault(sf, []).append(aid)
        windows: list[tuple[int, int, list[str]]] = []
        for sf in sorted(by_start.keys()):
            aids = by_start[sf]
            if len(aids) < min_agents:
                continue
            windows.append((sf, sf + base_size - 1, aids))
        return windows

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


def build_windows_sliding(
    agents: dict[str, list[tuple[int, float, float, float, float]]],
    total_len: int,
    stride: int,
    min_agents: int,
) -> list[tuple[int, int, list[str]]]:
    """
    Sliding fixed-length windows over time. Collect agents that have observations for every frame in the window.
    total_len = T_obs + T_pred. stride can be 1 for fully overlapping windows.
    """
    # derive frame step from data (min positive delta)
    frame_maps: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    all_frames: list[int] = []
    min_step = None
    for aid, traj in agents.items():
        fm: dict[int, tuple[float, float, float, float]] = {}
        last = None
        for fr, x, y, vx, vy in traj:
            fr_i = int(round(fr))
            fm[fr_i] = (x, y, vx, vy)
            if last is not None:
                delta = fr_i - last
                if delta > 0:
                    min_step = delta if min_step is None else min(min_step, delta)
            last = fr_i
            all_frames.append(fr_i)
        frame_maps[aid] = fm
    if not all_frames:
        return []
    if min_step is None or min_step <= 0:
        min_step = 1
    min_fr, max_fr = min(all_frames), max(all_frames)
    windows: list[tuple[int, int, list[str]]] = []
    start = min_fr
    stride_frames = stride * min_step
    while start + (total_len - 1) * min_step <= max_fr:
        end = start + (total_len - 1) * min_step
        agents_full: list[str] = []
        required_frames = [start + k * min_step for k in range(total_len)]
        for aid, fm in frame_maps.items():
            if all(fr in fm for fr in required_frames):
                agents_full.append(aid)
        if len(agents_full) >= min_agents:
            windows.append((start, end, agents_full))
        start += stride_frames
    return windows


def initial_state_for_window(
    agent_ids: list[str],
    agents: dict[str, list[tuple[int, float, float, float, float]]],
    frame_start: int,
    frame_end: int,
    px_to_m,
) -> np.ndarray:
    rows: list[list[float]] = []
    for aid in agent_ids:
        traj = agents[aid]
        # filter frames inside window
        in_window = [(f, x, y, vx, vy) for f, x, y, vx, vy in traj if frame_start <= f <= frame_end]
        if not in_window:
            continue
        f0, x0, y0, vx0, vy0 = in_window[0]
        f_last, x1, y1, _, _ = in_window[-1]
        (mx0, my0), (mx1, my1) = _convert_px_to_m(px_to_m, np.array([[x0, y0], [x1, y1]]))
        rows.append([mx0, my0, vx0, vy0, mx1, my1])
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
    p = argparse.ArgumentParser(description="Build per-window preprocessed scenes from GT trajectories.(obsmat.txt)")
    p.add_argument("--scene", default="eth", help="Scene key (used in ids and prompt lookup).")
    p.add_argument("--prompt-file", type=Path, default=None, help="Prompt text file; defaults to preprocess/gt/gt_scene/<scene>_prompt.txt")
    p.add_argument("--homography", type=Path, default=None, help="Homography TXT path; defaults to downloads/gt/<scene>/obstacle/H.txt")
    p.add_argument("--anchored-obstacles", type=Path, default=None, help="Anchored obstacles NPZ path; defaults to preprocess/gt/gt_line_obstacles/<scene>_line_obstacles.npz")
    p.add_argument("--window-size", type=int, default=300, help="Base window size in frames.")
    p.add_argument("--min-agents", type=int, default=20, help="Minimum agents per window.")
    p.add_argument(
        "--synchronized-starts",
        action="store_true",
        help="If set, each window includes only agents sharing the same start frame.",
    )
    p.add_argument(
        "--sliding-window",
        action="store_true",
        help="Use sliding windows over time (agents must be present for obs+pred frames).",
    )
    p.add_argument("--total-frames", type=int, default=20, help="Total frames per sliding window (obs+pred).")
    p.add_argument("--window-stride", type=int, default=1, help="Stride for sliding windows.")
    p.add_argument(
        "--write-gt",
        action="store_true",
        help="Also write per-window GT slices (CSV with frame shifted to window start).",
    )
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Directory for GT slices; defaults to downloads/gt/<scene>/trajectory_windows.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preprocess/preprocessed_scene"),
        help="Directory to write windowed scene JSONs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_key = args.scene
    prompt_path = args.prompt_file or Path(f"downloads/gt/{scene_key}/obstacle/prompt.txt")
    homography_path = args.homography or Path(f"downloads/gt/{scene_key}/obstacle/H.txt")
    anchored_path = args.anchored_obstacles or Path(f"preprocess/gt/gt_line_obstacles/{scene_key}_line_obstacles.npz")
    obsmat_path = Path(f"downloads/gt/{scene_key}/obstacle/obsmat.txt")
    gt_dir = args.gt_dir or Path(f"downloads/gt/{scene_key}/trajectory_windows")

    prompt = load_prompt(prompt_path, scene_key)

    if not obsmat_path.exists():
        raise FileNotFoundError(f"obsmat not found at {obsmat_path}")
    agents = read_trajectories_obsmat(obsmat_path)
    px_to_m: Callable[[np.ndarray], np.ndarray] = lambda pts: np.asarray(pts, dtype=float)
    if args.sliding_window:
        total_len = args.total_frames
        windows = build_windows_sliding(agents, total_len, args.window_stride, args.min_agents)
    else:
        windows = build_windows_greedy(agents, args.window_size, args.min_agents, synchronized=args.synchronized_starts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_gt:
        gt_dir.mkdir(parents=True, exist_ok=True)
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
            anchored_obstacles=anchored_path,
            homography=homography_path,
        )
        out_path = args.out_dir / f"{args.scene}_win{idx:03d}_{f_start}_{f_end}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} (agents={init_state.shape[0]})")
        if args.write_gt:
            gt_csv = gt_dir / f"{args.scene}_win{idx:03d}_{f_start}_{f_end}.csv"
            with gt_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scene", "agent_id", "agent_type", "frame", "x", "y"])
                for aid in agent_ids:
                    for frame, x, y, _, _ in agents[aid]:
                        if frame < f_start or frame > f_end:
                            continue
                        writer.writerow([args.scene, aid, "0", frame - f_start, x, y])
            print(f"Wrote GT slice {gt_csv}")


if __name__ == "__main__":
    main()

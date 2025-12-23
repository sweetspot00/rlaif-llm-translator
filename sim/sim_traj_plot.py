from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection


def _load_scene(scene_path: Optional[Path], embedded_scene: Optional[dict]) -> dict:
    if scene_path:
        return json.loads(Path(scene_path).read_text(encoding="utf-8"))
    if embedded_scene:
        return embedded_scene
    raise ValueError("Scene data is required via --scene or embedded in the simulation file.")


def _load_simulation(sim_path: Path) -> tuple[np.ndarray, Optional[dict]]:
    data = np.load(sim_path, allow_pickle=True)
    states = data["states"]
    scene = data["scene"].item() if "scene" in data else None
    return states, scene


def _resolve_obstacle(scene: dict, obstacle_path: Optional[Path]) -> Path:
    assets = scene.get("assets", {})
    obstacle_str = (
        str(obstacle_path)
        if obstacle_path
        else assets.get("anchored_obstacles") or assets.get("obstacle_npz")
    )
    if not obstacle_str:
        raise FileNotFoundError("Obstacle map not provided and missing in scene assets.")
    obstacle = Path(obstacle_str)
    if not obstacle.exists():
        raise FileNotFoundError(f"Obstacle map not found: {obstacle}")
    return obstacle


def _load_obstacle_segments(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        key = "obstacles" if "obstacles" in data.files else data.files[0]
        segments = data[key]
    else:
        segments = data
    arr = np.asarray(segments, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=float)
    return arr.reshape(-1, 4)


def _grid_step(coords: np.ndarray) -> float:
    diffs = np.diff(np.unique(coords))
    diffs = diffs[diffs > 1e-6]
    return float(diffs.min()) if diffs.size else 1.0


def _segments_to_polygons(segments: np.ndarray) -> list[np.ndarray]:
    """
    Assemble closed polygons from axis-aligned obstacle boundary segments.
    Segments are expected to be (x0, x1, y0, y1) in meters.
    """
    if segments.size == 0:
        return []

    xs = np.concatenate([segments[:, 0], segments[:, 1]])
    ys = np.concatenate([segments[:, 2], segments[:, 3]])
    step = _grid_step(np.concatenate([xs, ys]))
    if step <= 0:
        step = 1.0

    def quantize(v: float) -> int:
        return int(round(v / step))

    # Build undirected edge set on quantized grid
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x0, x1, y0, y1 in segments:
        p0 = (quantize(x0), quantize(y0))
        p1 = (quantize(x1), quantize(y1))
        edge = (p0, p1) if p0 <= p1 else (p1, p0)
        if edge in edges:
            continue
        edges.add(edge)
        adjacency.setdefault(p0, []).append(p1)
        adjacency.setdefault(p1, []).append(p0)

    polygons: list[np.ndarray] = []
    while edges:
        edge = edges.pop()
        p_start, p_next = edge
        loop = [p_start, p_next]
        current = p_next
        prev = p_start
        while current != p_start:
            neighbors = adjacency.get(current, [])
            nxt = None
            for cand in neighbors:
                if cand != prev and ((current, cand) in edges or (cand, current) in edges):
                    nxt = cand
                    break
            if nxt is None:
                # dead-end, abort this loop
                break
            edge_key = (current, nxt) if current <= nxt else (nxt, current)
            edges.discard(edge_key)
            loop.append(nxt)
            prev, current = current, nxt
        if len(loop) >= 4 and loop[0] == loop[-1]:
            poly = np.array(loop, dtype=float) * step
            polygons.append(poly)

    return polygons


def _goals_m(scene: dict) -> Optional[np.ndarray]:
    if "goals_m" in scene:
        goals = np.asarray(scene["goals_m"], dtype=float)
    elif "goals_px" in scene:
        goals = np.asarray(scene["goals_px"], dtype=float)
    else:
        return None
    return goals if goals.ndim == 2 else goals.reshape(-1, 2)


def plot_trajectory(
    sim_path: Path,
    scene_path: Optional[Path],
    obstacle_path: Optional[Path],
    out_path: Path,
    dpi: int = 150,
) -> Path:
    states, scene_embedded = _load_simulation(sim_path)
    scene = _load_scene(scene_path, scene_embedded)
    obstacle_path = _resolve_obstacle(scene, obstacle_path)
    obstacles = _load_obstacle_segments(obstacle_path)
    polygons = _segments_to_polygons(obstacles)

    positions_m = np.asarray(states, dtype=float)[..., :2]
    goals_m = _goals_m(scene)

    x_bounds: list[float] = []
    y_bounds: list[float] = []
    for poly in polygons:
        x_bounds.extend([float(poly[:, 0].min()), float(poly[:, 0].max())])
        y_bounds.extend([float(poly[:, 1].min()), float(poly[:, 1].max())])
    if obstacles.size:
        x_bounds.extend([float(obstacles[:, :2].min()), float(obstacles[:, :2].max())])
        y_bounds.extend([float(obstacles[:, 2:].min()), float(obstacles[:, 2:].max())])

    flat_pos = positions_m.reshape(-1, 2)
    valid_pos = np.all(np.isfinite(flat_pos), axis=1)
    if np.any(valid_pos):
        pos_valid = flat_pos[valid_pos]
        x_bounds.extend([float(pos_valid[:, 0].min()), float(pos_valid[:, 0].max())])
        y_bounds.extend([float(pos_valid[:, 1].min()), float(pos_valid[:, 1].max())])

    if goals_m is not None and len(goals_m) > 0:
        x_bounds.extend([float(goals_m[:, 0].min()), float(goals_m[:, 0].max())])
        y_bounds.extend([float(goals_m[:, 1].min()), float(goals_m[:, 1].max())])

    if x_bounds and y_bounds:
        x_min, x_max = min(x_bounds), max(x_bounds)
        y_min, y_max = min(y_bounds), max(y_bounds)
    else:
        x_min = y_min = 0.0
        x_max = y_max = 1.0
    span_x = max(1e-6, x_max - x_min)
    span_y = max(1e-6, y_max - y_min)
    pad = max(1.0, 0.02 * max(span_x, span_y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.set_facecolor("white")

    for poly in polygons:
        ax.fill(poly[:, 0], poly[:, 1], color="black", alpha=1.0, zorder=0)

    if obstacles.size:
        lines = np.stack(
            [
                np.stack([obstacles[:, 0], obstacles[:, 2]], axis=1),
                np.stack([obstacles[:, 1], obstacles[:, 3]], axis=1),
            ],
            axis=1,
        )
        lc = LineCollection(lines, colors="black", linewidths=0.6, alpha=0.3, zorder=1)
        ax.add_collection(lc)

    n_agents = positions_m.shape[1]
    cmap = plt.cm.get_cmap("tab20", n_agents if n_agents > 0 else 1)
    for idx in range(n_agents):
        traj = positions_m[:, idx, :]
        valid = np.all(np.isfinite(traj), axis=1)
        traj = traj[valid]
        if traj.size == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], color=cmap(idx), linewidth=1.6, alpha=0.85, zorder=2)
        ax.scatter(traj[0, 0], traj[0, 1], color=cmap(idx), s=12, zorder=3)

    if goals_m is not None and len(goals_m) > 0:
        ax.scatter(
            goals_m[:, 0],
            goals_m[:, 1],
            marker="*",
            s=90,
            c="#e74c3c",
            edgecolors="white",
            linewidths=0.8,
            label="goals",
            zorder=4,
        )
        ax.legend(loc="upper right")

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = scene.get("scene_id") or sim_path.stem
    ax.set_title(f"Trajectories: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def _gather_sim_paths(sim_inputs: Iterable[Path], sim_dir: Optional[Path], pattern: str) -> list[Path]:
    paths: list[Path] = []
    paths.extend(sim_inputs)
    if sim_dir:
        paths.extend(sorted(sim_dir.glob(pattern)))
    unique = []
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    if not unique:
        raise ValueError("No simulation files provided. Use --sim or --sim-dir.")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulated trajectories over the obstacle map.")
    parser.add_argument("--sim", type=Path, nargs="*", default=[], help="Simulation .npz file(s) to plot.")
    parser.add_argument("--sim-dir", type=Path, help="Directory containing simulation .npz files.")
    parser.add_argument("--glob", default="sim_*.npz", help="Glob pattern for --sim-dir (default: sim_*.npz).")
    parser.add_argument("--scene", type=Path, help="Optional preprocessed scene JSON; otherwise use embedded scene.")
    parser.add_argument(
        "--obstacle",
        type=Path,
        help="Anchored obstacle segments (.npy/.npz) in meters. Defaults to scene assets if omitted.",
    )
    parser.add_argument("--out", type=Path, help="Output path for a single simulation plot.")
    parser.add_argument("--out-dir", type=Path, default=Path("sim/results/plots"), help="Directory to write batch plots.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    sim_paths = _gather_sim_paths(args.sim, args.sim_dir, args.glob)

    if len(sim_paths) == 1:
        out_path = args.out or (args.out_dir / f"{sim_paths[0].stem}.png")
        plot_trajectory(
            sim_path=sim_paths[0],
            scene_path=args.scene,
            obstacle_path=args.obstacle,
            out_path=out_path,
            dpi=args.dpi,
        )
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sim_path in sim_paths:
        out_path = args.out_dir / f"{sim_path.stem}.png"
        plot_trajectory(
            sim_path=sim_path,
            scene_path=args.scene,
            obstacle_path=args.obstacle,
            out_path=out_path,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

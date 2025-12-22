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
from PIL import Image

from utils.convert_obstacle_to_meter import load_homography, meter_to_pixel_factory


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


def _resolve_assets(scene: dict, obstacle_path: Optional[Path], homography_path: Optional[Path]) -> tuple[Path, Path]:
    assets = scene.get("assets", {})
    obstacle_str = str(obstacle_path) if obstacle_path else assets.get("obstacle_png")
    homography_str = str(homography_path) if homography_path else assets.get("homography")
    if not obstacle_str:
        raise FileNotFoundError("Obstacle map not provided and missing in scene assets.")
    if not homography_str:
        raise FileNotFoundError("Homography not provided and missing in scene assets.")
    obstacle = Path(obstacle_str)
    homography = Path(homography_str)
    if not obstacle.exists():
        raise FileNotFoundError(f"Obstacle map not found: {obstacle}")
    if not homography.exists():
        raise FileNotFoundError(f"Homography not found: {homography}")
    return obstacle, homography


def _meter_to_pixel_converter(obstacle_path: Path, homography_path: Path):
    img = Image.open(obstacle_path)
    H = load_homography(homography_path)
    origin_px = (0.0, float(img.height))  # bottom-left anchors to (0,0) in meter space
    return meter_to_pixel_factory(H, origin_px), img


def _goals_px(scene: dict, m_to_px) -> Optional[np.ndarray]:
    if "goals_px" in scene:
        goals = np.asarray(scene["goals_px"], dtype=float)
    elif "goals_m" in scene:
        goals = m_to_px(np.asarray(scene["goals_m"], dtype=float))
    else:
        return None
    return goals if goals.ndim == 2 else goals.reshape(-1, 2)


def plot_trajectory(
    sim_path: Path,
    scene_path: Optional[Path],
    obstacle_path: Optional[Path],
    homography_path: Optional[Path],
    out_path: Path,
    dpi: int = 150,
) -> Path:
    states, scene_embedded = _load_simulation(sim_path)
    scene = _load_scene(scene_path, scene_embedded)
    obstacle_path, homography_path = _resolve_assets(scene, obstacle_path, homography_path)
    to_px, obstacle_img = _meter_to_pixel_converter(obstacle_path, homography_path)

    positions_m = np.asarray(states, dtype=float)[..., :2]
    positions_px = to_px(positions_m)
    goals_px = _goals_px(scene, to_px)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = obstacle_img.width / 100, obstacle_img.height / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(obstacle_img.convert("RGB"), origin="upper")

    n_agents = positions_px.shape[1]
    cmap = plt.cm.get_cmap("tab20", n_agents if n_agents > 0 else 1)
    for idx in range(n_agents):
        traj = positions_px[:, idx, :]
        valid = np.all(np.isfinite(traj), axis=1)
        traj = traj[valid]
        if traj.size == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], color=cmap(idx), linewidth=1.6, alpha=0.85)
        ax.scatter(traj[0, 0], traj[0, 1], color=cmap(idx), s=12, zorder=3)

    if goals_px is not None and len(goals_px) > 0:
        ax.scatter(
            goals_px[:, 0],
            goals_px[:, 1],
            marker="*",
            s=90,
            c="#e74c3c",
            edgecolors="white",
            linewidths=0.8,
            label="goals",
            zorder=4,
        )
        ax.legend(loc="upper right")

    ax.set_xlim(0, obstacle_img.width)
    ax.set_ylim(obstacle_img.height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
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
    parser.add_argument("--obstacle", type=Path, help="Obstacle PNG. Defaults to scene assets if omitted.")
    parser.add_argument("--homography", type=Path, help="Homography TXT. Defaults to scene assets if omitted.")
    parser.add_argument("--out", type=Path, help="Output path for a single simulation plot.")
    parser.add_argument("--out-dir", type=Path, default=Path("sim/results/plots"), help="Directory to write batch plots.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_paths = _gather_sim_paths(args.sim, args.sim_dir, args.glob)

    if len(sim_paths) == 1:
        out_path = args.out or (args.out_dir / f"{sim_paths[0].stem}.png")
        plot_trajectory(
            sim_path=sim_paths[0],
            scene_path=args.scene,
            obstacle_path=args.obstacle,
            homography_path=args.homography,
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
            homography_path=args.homography,
            out_path=out_path,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

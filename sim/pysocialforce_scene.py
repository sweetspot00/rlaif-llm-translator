"""
Typed container for pysocialforce simulation inputs.

The dataclass bundles the pieces needed to spin up a ``psf.Simulator``:
    - obstacles: linear segments in meters, typically loaded from the
      ``preprocess/pysocialforce_obstacles_meter`` directory.
    - map_mask: optional occupancy map for clipping sampled positions.
    - initial_state: (N, 6) array of [px, py, vx, vy, gx, gy].
    - groups: social grouping metadata.
    - config_path: TOML config for the simulator.
It also includes helpers for constructing scenarios from the context
dataset (``datasets/context.jsonl``), which provides an image name,
event center, and goal hint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


def _parse_goal_count(goal_location: Optional[str]) -> int:
    """Extract the number inside strings like ``Random (4)``."""
    if not goal_location:
        return 1
    match = re.search(r"(\\d+)", goal_location)
    return int(match.group(1)) if match else 1


def _crowd_size_to_agents(label: Optional[str], default: int) -> int:
    """Convert a crowd-size label like ``100-500`` into a usable count."""
    if not label:
        return default
    range_match = re.match(r"(\\d+)\\s*[-–]\\s*(\\d+)", label)
    if range_match:
        low, high = map(int, range_match.groups())
        return int((low + high) / 2)
    digit_match = re.search(r"(\\d+)", label)
    if digit_match:
        return int(digit_match.group(1))
    return default


def _resolve_obstacle_path(scene_id: str, obstacles_root: Path) -> Path:
    path = obstacles_root / f"{scene_id}_anchored.npy"
    if not path.exists():
        raise FileNotFoundError(f"Obstacle file not found: {path}")
    return path


def _load_obstacles(path: Path) -> np.ndarray:
    obstacles = np.load(path)
    if obstacles.ndim != 2 or obstacles.shape[1] != 4:
        raise ValueError(f"Obstacle file {path} must have shape (N, 4), got {obstacles.shape}.")
    return obstacles


def _sample_goals(
    rng: np.random.Generator,
    *,
    n_agents: int,
    n_clusters: int,
    center: np.ndarray,
    spread_m: float,
) -> np.ndarray:
    """Sample goal centers around ``center`` and assign each agent to one."""
    goal_centers = center + rng.uniform(-spread_m, spread_m, size=(n_clusters, 2))
    assignments = rng.integers(0, n_clusters, size=n_agents)
    return goal_centers[assignments]


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    with path.open("rb") as f:
        return tomllib.load(f)


def _flatten_dict(data: dict, parent_key: str = "scene.config", sep: str = ".") -> dict:
    """Flatten nested dictionaries into dot-delimited keys."""
    items: dict[str, object] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


@dataclass
class PysocialforceScene:
    scene_id: str
    obstacles: np.ndarray
    map_mask: Optional[np.ndarray]
    initial_state: np.ndarray
    groups: list[list[int]]
    config_path: Path
    config: Optional[dict] = None
    config_flat: dict = field(default_factory=dict, init=False)
    metadata: dict = field(default_factory=dict)

    @property
    def map(self) -> Optional[np.ndarray]:
        """Alias to match the requested field name."""
        return self.map_mask

    def simulator_kwargs(self) -> dict:
        """Return kwargs ready to pass to ``psf.Simulator``."""
        return {
            "initial_state": self.initial_state,
            "groups": self.groups or None,
            "obstacles": self.obstacles.tolist() if isinstance(self.obstacles, np.ndarray) else self.obstacles,
            "config_file": str(self.config_path),
        }
        if self.config is None and self.config_path.exists():
            self.config = _load_toml(self.config_path)
        if not self.config_flat and self.config is not None:
            self.config_flat = _flatten_dict(self.config)
        return self.simulator_kwargs_raw()

    def simulator_kwargs_raw(self) -> dict:
        """Raw kwargs without auto-loading/flattening for custom use."""
        return {
            "initial_state": self.initial_state,
            "groups": self.groups or None,
            "obstacles": self.obstacles.tolist() if isinstance(self.obstacles, np.ndarray) else self.obstacles,
            "config_file": str(self.config_path),
        }

    @classmethod
    def from_context(
        cls,
        context: dict,
        *,
        obstacles_root: Path = Path("preprocess/pysocialforce_obstacles_meter"),
        config_path: Path = Path("PySocialForce/pysocialforce/config/default.toml"),
        n_agents: Optional[int] = None,
        spawn_std_m: float = 3.0,
        goal_spread_m: float = 8.0,
        desired_speed: float = 1.2,
        pixel_to_meter: float = 1.0,
        random_state: Optional[int] = None,
        map_mask: Optional[np.ndarray] = None,
        load_config: bool = True,
    ) -> "PysocialforceScene":
        """
        Build a scene from a single context JSON entry.

        Args:
            context: Row from ``datasets/context.jsonl``.
            obstacles_root: Directory containing ``*_anchored.npy`` files.
            config_path: Path to TOML config for pysocialforce.
            n_agents: Override the number of agents; falls back to the crowd_size label.
            spawn_std_m: Stddev (meters) for sampling starts around the event center.
            goal_spread_m: Range (meters) for sampling goal centers around the event center.
            desired_speed: Seed speed used for the initial velocity direction.
            pixel_to_meter: Scale factor to convert image pixels to meters.
            random_state: Optional seed for reproducibility.
            map_mask: Optional occupancy mask for clipping sampled positions.
        """
        rng = np.random.default_rng(random_state)

        scene_id = Path(str(context.get("image", "scene"))).stem
        obstacle_path = _resolve_obstacle_path(scene_id, obstacles_root)
        obstacles = _load_obstacles(obstacle_path)

        # Derive agent count and anchors.
        agent_count = n_agents or _crowd_size_to_agents(context.get("crowd_size"), default=20)
        center_px = np.asarray(context.get("event_center", (0.0, 0.0)), dtype=float)
        center_m = center_px * pixel_to_meter

        # Start positions clustered around the event center.
        start_positions = rng.normal(loc=center_m, scale=spawn_std_m, size=(agent_count, 2))
        if map_mask is not None and map_mask.size > 0:
            height, width = map_mask.shape
            start_positions[:, 0] = np.clip(start_positions[:, 0], 0.0, width)
            start_positions[:, 1] = np.clip(start_positions[:, 1], 0.0, height)

        # Goal sampling.
        n_goal_clusters = max(1, _parse_goal_count(context.get("goal_location")))
        goals = _sample_goals(
            rng,
            n_agents=agent_count,
            n_clusters=n_goal_clusters,
            center=center_m,
            spread_m=goal_spread_m,
        )

        # Velocities point toward each agent's assigned goal.
        directions = goals - start_positions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms > 0)
        velocities = directions * desired_speed

        initial_state = np.hstack([start_positions, velocities, goals])

        metadata = {
            "scenario": context.get("scenario"),
            "category": context.get("category"),
            "source_image": context.get("image"),
            "crowd_size": context.get("crowd_size"),
            "goal_location": context.get("goal_location"),
        }

        cfg = _load_toml(config_path) if load_config and config_path.exists() else None
        cfg_flat = _flatten_dict(cfg) if cfg else {}

        return cls(
            scene_id=scene_id,
            obstacles=obstacles,
            map_mask=map_mask,
            initial_state=initial_state,
            groups=[],
            config_path=config_path,
            config=cfg,
            config_flat=cfg_flat,
            metadata=metadata,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        *,
        obstacles_root: Path = Path("preprocess/pysocialforce_obstacles_meter"),
        config_path: Path = Path("PySocialForce/pysocialforce/config/default.toml"),
        n_agents: Optional[int] = None,
        spawn_std_m: float = 3.0,
        goal_spread_m: float = 8.0,
        desired_speed: float = 1.2,
        pixel_to_meter: float = 1.0,
        random_state: Optional[int] = None,
        map_mask: Optional[np.ndarray] = None,
        load_config: bool = True,
    ) -> list["PysocialforceScene"]:
        """Load all scenes from a JSONL file and return instantiated objects."""
        scenes: list[PysocialforceScene] = []
        rng = np.random.default_rng(random_state)
        jsonl_path = Path(path)
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                context = json.loads(line)
                scenes.append(
                    cls.from_context(
                        context,
                        obstacles_root=obstacles_root,
                        config_path=config_path,
                        n_agents=n_agents,
                        spawn_std_m=spawn_std_m,
                        goal_spread_m=goal_spread_m,
                        desired_speed=desired_speed,
                        pixel_to_meter=pixel_to_meter,
                        random_state=rng.integers(0, 2**32 - 1),
                        map_mask=map_mask,
                        load_config=load_config,
                    )
                )
        return scenes

    @classmethod
    def from_manual(
        cls,
        *,
        scene_id: str,
        initial_state: np.ndarray,
        obstacles: Iterable[Sequence[float]] | np.ndarray,
        groups: Optional[list[list[int]]] = None,
        map_mask: Optional[np.ndarray] = None,
        config_path: Path = Path("PySocialForce/pysocialforce/config/default.toml"),
        metadata: Optional[dict] = None,
        load_config: bool = True,
    ) -> "PysocialforceScene":
        """Create a scene from precomputed arrays (e.g., during preprocessing)."""
        obstacles_arr = np.asarray(obstacles, dtype=float)
        if obstacles_arr.ndim != 2 or obstacles_arr.shape[1] != 4:
            raise ValueError(f"obstacles must have shape (N, 4), got {obstacles_arr.shape}")
        if initial_state.ndim != 2 or initial_state.shape[1] != 6:
            raise ValueError(f"initial_state must have shape (N, 6), got {initial_state.shape}")
        cfg = _load_toml(config_path) if load_config and config_path.exists() else None
        cfg_flat = _flatten_dict(cfg) if cfg else {}
        return cls(
            scene_id=scene_id,
            obstacles=obstacles_arr,
            map_mask=map_mask,
            initial_state=initial_state,
            groups=groups or [],
            config_path=config_path,
            config=cfg,
            config_flat=cfg_flat,
            metadata=metadata or {},
        )

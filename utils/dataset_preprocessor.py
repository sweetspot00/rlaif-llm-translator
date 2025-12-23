"""
Dataset preprocessor that turns context.jsonl rows + map assets into
PySocialForce-ready scene data.

Inputs expected per scene:
- context row with image name, event_center, goal_location, crowd_size.
- obstacle PNG under downloads/google_maps/simplified_obstacles/{scene_id}_obstacle_simplified_obstacle.png
  (falls back to {scene_id}_obstacle.png).
- homography under downloads/google_maps/homographies/{scene_id}.txt.
- anchored obstacle segments are not required; sampling uses the obstacle PNG directly.

Outputs:
- One JSON file per scene under preprocess/preprocessed_scene containing:
    * scene metadata (id, index, scenario text, etc.)
    * anchored event center and goals
    * chosen crowd size
    * initial_state array (list-of-lists) [px, py, vx, vy, gx, gy]
    * groups (indices)
    * paths to supporting assets
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image
import logging

from convert_obstacle_to_meter import load_homography, pixel_to_meter_factory

# Configure a simple logger
logger = logging.getLogger("dataset_preprocessor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def _load_jsonl(path: Path) -> list[dict]:
    logger.info("Loading context JSONL from %s", path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _crowd_size_to_range(label: str, rng: Optional[np.random.Generator] = None) -> tuple[int, int, int]:
    """
    Parse a crowd-size label and return (low, high, sampled_value).

    Sampling uses the provided RNG (or a new default generator), so callers can
    pass a seeded RNG for reproducibility.
    """
    nums = [int(n) for n in label.replace("–", "-").replace("+", "-").split("-") if n.strip().isdigit()]
    if len(nums) == 2:
        low, high = min(nums), max(nums)
    elif len(nums) == 1:
        low = high = nums[0]
    else:
        low, high = 10, 20
    generator = rng or np.random.default_rng()
    sampled = int(generator.integers(low, high + 1))
    return low, high, sampled


def _is_walkable(mask: np.ndarray, pt_px: np.ndarray) -> bool:
    x, y = int(round(pt_px[0])), int(round(pt_px[1]))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def _nearest_walkable(mask: np.ndarray, walkable_points: np.ndarray, target_px: np.ndarray) -> np.ndarray:
    if not _is_walkable(mask, target_px):
        diff = walkable_points - target_px
        idx = int(np.argmin(np.sum(diff * diff, axis=1)))
        return walkable_points[idx]
    return target_px


def _parse_goal_spec(goal_location: str) -> tuple[str, int]:
    text = goal_location.lower().strip()
    if text.startswith("random"):
        count = 1
        if "(" in text and ")" in text:
            try:
                count = int(text[text.find("(") + 1 : text.find(")")])
            except Exception:
                count = 1
        return "random", max(1, count)
    if "gaussian" in text:
        count = 1
        if "(" in text and ")" in text:
            try:
                count = int(text[text.find("(") + 1 : text.find(")")])
            except Exception:
                count = 1
        return "gaussian", max(1, count)
    return "fixed", 1


def _sample_truncated_gaussian(
    rng: np.random.Generator,
    walkable_points: np.ndarray,
    mask: np.ndarray,
    mean_px: np.ndarray,
    std_px: float,
    n: int,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    while len(samples) < n:
        candidate = rng.normal(loc=mean_px, scale=std_px, size=2)
        if _is_walkable(mask, candidate):
            samples.append(candidate)
        # fallback to uniform if stuck
        if len(samples) == 0 and len(samples) < n and rng.random() < 0.01:
            idx = rng.integers(0, len(walkable_points))
            samples.append(walkable_points[idx])
    return np.vstack(samples)


def _assign_groups(rng: np.random.Generator, n_agents: int, prob: float = 0.25, max_size: int = 4) -> list[list[int]]:
    groups: list[list[int]] = []
    if n_agents < 2 or prob <= 0 or max_size < 2:
        return groups
    indices = np.arange(n_agents)
    rng.shuffle(indices)
    remaining = list(indices)
    while len(remaining) >= 2:
        if rng.random() < prob:
            size = int(rng.integers(2, min(max_size, len(remaining)) + 1))
            members = [remaining.pop() for _ in range(size)]
            if len(members) > 1:
                groups.append(sorted(members))
        else:
            remaining.pop()
    return groups


def _validate_group_coverage(n_agents: int, groups: list[list[int]]) -> int:
    """
    Ensure grouped agent indices are unique and within range.

    Returns the number of ungrouped individuals (should satisfy grouped + ungrouped = crowd size).
    """
    seen: set[int] = set()
    for grp in groups:
        for member in grp:
            m_int = int(member)
            if m_int < 0 or m_int >= n_agents:
                raise ValueError(f"Group member index {m_int} out of range for {n_agents} agents.")
            if m_int in seen:
                raise ValueError(f"Agent {m_int} appears in multiple groups.")
            seen.add(m_int)
    return n_agents - len(seen)


@dataclass
class ScenePreprocessor:
    context_path: Path = Path("datasets/context_simplified_test.jsonl")
    obstacles_png_dir: Path = Path("downloads/google_maps/simplified_obstacles")
    homography_dir: Path = Path("downloads/google_maps/homographies")
    output_dir: Path = Path("preprocess/preprocessed_scene")
    # desired_speed: float = 1.2 get from context.jsonl
    spawn_std_px: float = 30.0
    goal_std_px: float = 40.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    anchored_obstacles_dir: Path = Path("preprocess/pysfm_obstacles_meter_close_shape")

    def _load_assets(self, scene_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
        # Normalize scene id so we can match both raw names and stripped suffixes.
        base_id = scene_id
        for suffix in ["_obstacle_simplified_obstacle", "_obstacle_simplified", "_obstacle"]:
            if base_id.endswith(suffix):
                base_id = base_id[: -len(suffix)]
                break

        png_candidates = [
            self.obstacles_png_dir / f"{scene_id}.png",
            self.obstacles_png_dir / f"{base_id}_obstacle_simplified_obstacle.png",
            self.obstacles_png_dir / f"{base_id}_obstacle.png",
            self.obstacles_png_dir / f"{base_id}.png",
        ]
        png_path = next((p for p in png_candidates if p.exists()), None)

        homography_candidates = [
            self.homography_dir / f"{scene_id}.txt",
            self.homography_dir / f"{base_id}_obstacle_simplified.txt",
            self.homography_dir / f"{base_id}.txt",
        ]
        homography_path = next((h for h in homography_candidates if h.exists()), None)

        if png_path is None:
            raise FileNotFoundError(
                f"No obstacle PNG found for {scene_id} (checked: {', '.join(map(str, png_candidates))})"
            )
        if homography_path is None:
            raise FileNotFoundError(
                f"No homography found for {scene_id} (checked: {', '.join(map(str, homography_candidates))})"
            )

        img = Image.open(png_path).convert("L")
        mask = (np.array(img, dtype=np.uint8) > 0)  # True = walkable
        walkable_points = np.argwhere(mask)[:, ::-1].astype(float)  # (x, y)

        H = load_homography(homography_path)
        origin_px = (0.0, float(img.height))  # bottom-left anchors to (0,0)
        px_to_m = pixel_to_meter_factory(H, origin_px)
        
        self.anchored_obstacles_dir = Path("preprocess/pysfm_obstacles_meter_close_shape")

        logger.info(
            "Assets loaded for %s (walkable points: %d, obstacle_png: %s, homography: %s)",
            scene_id,
            len(walkable_points),
            png_path.name,
            homography_path.name,
        )
        return mask, walkable_points, px_to_m, png_path

    def _convert_px_to_m(self, px_to_m, points_px: np.ndarray) -> np.ndarray:
        points_px = np.asarray(points_px, dtype=float)
        if points_px.ndim == 1:
            points_px = points_px[None, :]
        anchored = px_to_m(points_px)
        anchored[anchored > -1e-6] = np.maximum(anchored[anchored > -1e-6], 0.0)
        return anchored

    def _pick_crowd_size(self, label: str) -> int:
        _, _, sample = _crowd_size_to_range(label, rng=self.rng)
        return sample

    def _goals_px(
        self,
        goal_location: str,
        event_center_px: np.ndarray,
        mask: np.ndarray,
        walkable_points: np.ndarray,
    ) -> np.ndarray:
        if isinstance(goal_location, (list, tuple)):
            px = np.asarray(goal_location, dtype=float)
            px = _nearest_walkable(mask, walkable_points, px)
            return np.vstack([px])

        mode, count = _parse_goal_spec(str(goal_location))
        if mode == "random":
            idx = self.rng.choice(len(walkable_points), size=count, replace=False if len(walkable_points) >= count else True)
            return walkable_points[idx]
        if mode == "gaussian":
            return _sample_truncated_gaussian(
                self.rng, walkable_points, mask, mean_px=event_center_px, std_px=self.goal_std_px, n=count
            )
        # fixed location string not parsed -> fall back to event center
        return np.vstack([_nearest_walkable(mask, walkable_points, event_center_px)] * count)

    def _assign_goals(
        self,
        starts_m: np.ndarray,
        goals_m: np.ndarray,
        strategy: str = "nearest",
    ) -> np.ndarray:
        """
        Assign each agent a goal using either:
        - "nearest": pick the closest goal (default)
        - "random": random assignment (with wrap if fewer goals than agents)
        """
        n_agents = starts_m.shape[0]
        if strategy == "random":
            idx = np.arange(n_agents) % goals_m.shape[0]
            self.rng.shuffle(idx)
            return goals_m[idx]

        # nearest assignment (one-to-one if possible; otherwise reuse nearest)
        distances = np.linalg.norm(starts_m[:, None, :] - goals_m[None, :, :], axis=2)
        assigned = np.empty((n_agents, 2), dtype=float)
        used_goals: set[int] = set()
        for i in range(n_agents):
            goal_order = np.argsort(distances[i])
            chosen = next((g for g in goal_order if g not in used_goals), goal_order[0])
            used_goals.add(chosen)
            assigned[i] = goals_m[chosen]
        return assigned

    def _initial_state(
        self,
        starts_m: np.ndarray,
        goals_m: np.ndarray,
        goal_assignment: str = "nearest",
    ) -> np.ndarray:
        goals_use = self._assign_goals(starts_m, goals_m, strategy=goal_assignment)
        directions = goals_use - starts_m
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms > 0)
        velocities = directions * self.desired_speed
        return np.hstack([starts_m, velocities, goals_use])

    def process_scene(self, context: dict, index: int) -> tuple[Path, dict]:
        scene_id = Path(str(context["image"])).stem
        mask, walkable_points, px_to_m, obstacle_png_path = self._load_assets(scene_id)
        self.desired_speed = context.get("desired_speed", 1.2)
        event_center_raw = context.get("event_center", (0, 0))
        if isinstance(event_center_raw, str) and "gaussian" in event_center_raw.lower():
            mean_px = walkable_points.mean(axis=0)
            event_center_px = _sample_truncated_gaussian(
                self.rng, walkable_points, mask, mean_px=mean_px, std_px=self.spawn_std_px, n=1
            )[0]
        else:
            event_center_px = np.asarray(event_center_raw, dtype=float)
            if not _is_walkable(mask, event_center_px):
                event_center_px = _nearest_walkable(mask, walkable_points, event_center_px)
        event_center_m = self._convert_px_to_m(px_to_m, event_center_px)[0]
        logger.info("Scene %s: event center px=%s m=%s", scene_id, event_center_px, event_center_m)

        goals_px = self._goals_px(context.get("goal_location", "Random (1)"), event_center_px, mask, walkable_points)
        goals_m = self._convert_px_to_m(px_to_m, goals_px)

        n_agents = self._pick_crowd_size(context.get("crowd_size", "10-20"))
        replace = len(walkable_points) < n_agents
        idx = self.rng.choice(len(walkable_points), size=n_agents, replace=replace) # random sample agents by map walkable places
        # TODO: sample based on event center; agents maybe around the event center
        starts_px = walkable_points[idx]
        starts_m = self._convert_px_to_m(px_to_m, starts_px)

        initial_state = self._initial_state(starts_m, goals_m)
        groups = _assign_groups(self.rng, n_agents)
        ungrouped = _validate_group_coverage(n_agents, groups)
        logger.info(
            "Scene %s: crowd=%d (ungrouped=%d, groups=%d), goals=%d, starts=%d",
            scene_id,
            n_agents,
            ungrouped,
            len(groups),
            goals_m.shape[0],
            starts_m.shape[0],
        )

        out = {
            "scene_id": scene_id,
            "scene_index": int(index),
            "scenario": context.get("scenario"),
            "category": context.get("category"),
            "crowd_size_label": context.get("crowd_size"),
            "crowd_size": int(n_agents),
            "ungrouped_agents": int(ungrouped),
            "event_center_px": event_center_px.tolist(),
            "event_center_m": event_center_m.tolist(),
            "goal_location_raw": context.get("goal_location"),
            "goals_px": goals_px.tolist(),
            "goals_m": goals_m.tolist(),
            "initial_state": initial_state.tolist(),
            "groups": [list(map(int, g)) for g in groups],
            "assets": {
                "obstacle_png": str(obstacle_png_path),
                "homography": str(self.homography_dir / f"{scene_id}.txt"),
                "anchored_obstacles": str(self.anchored_obstacles_dir / f"{scene_id}_anchored.npz"),
            },
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{index:04d}_{scene_id}.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_path)
        return out_path, out

    def process_all(self) -> list[Path]:
        contexts = _load_jsonl(self.context_path)
        outputs: list[Path] = []
        all_records: list[dict] = []
        for idx, row in enumerate(contexts):
            path, data = self.process_scene(row, idx)
            outputs.append(path)
            all_records.append(data)
        jsonl_path = self.output_dir / "preprocessed.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec) + "\n")
        return outputs

    def process_one(self, name: Optional[str] = None, *, context_index: Optional[int] = None) -> Optional[Path]:
        """
        Preprocess a single scene identified by image filename/stem or by line number in context.jsonl.

        Returns the output path if found, otherwise None.
        """
        if name is None and context_index is None:
            raise ValueError("Provide either a scene name or context_index.")

        contexts = _load_jsonl(self.context_path)

        if context_index is not None:
            if context_index < 0 or context_index >= len(contexts):
                raise IndexError(f"context_index {context_index} out of range (0-{len(contexts)-1})")
            row = contexts[context_index]
            path, _ = self.process_scene(row, context_index)
            return path

        target_stem = Path(name).stem
        for idx, row in enumerate(contexts):
            scene_id = Path(str(row.get("image", ""))).stem
            if scene_id == target_stem:
                path, _ = self.process_scene(row, idx)
                return path
        return None

# test
if __name__ == "__main__":
    preprocessor = ScenePreprocessor()
    # Example usages:
    # preprocessor.process_one("00_Zurich_HB")
    preprocessor.process_one(context_index=30)

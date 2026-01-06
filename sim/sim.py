"""
LLM-SFM translator + Simulation

1. Load preprocessed data e.g. preprocess/preprocessed_scene/0000_00_Zurich_HB.json
2. Call llm to generat config file of SFM parameters -> also save to sim/results/configs. Name: <input_scene_id_name>_<timestamp>.toml
3. Run simulation with generated config file and the preprocessed data
4. Save simulation results (trajectories) to sim/results/simulations. Name: sim_<input_scene_id_name>_<timestamp>.npz<input_scene_id_name>_<timestamp>
5. Save simulation result (metrics and plots) to sim/results/metrics. Name: metrics_<input_scene_id_name>_<timestamp>.json/png

Support simple test with single input file.

"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import csv

import numpy as np
import openai  
from pathlib import Path
import pysocialforce as psf
import toml
from evaluation import TrajectoryEvaluator  
from sim_traj_plot import plot_trajectory
from utils.convert_obstacle_to_meter import load_homography, pixel_to_meter_factory
from tqdm import tqdm

logger = logging.getLogger(__name__)
# Reduce noisy debug logging from dependencies.
logging.getLogger("pysocialforce").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Default to INFO; attach a basic handler if none exists.
root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="[{levelname}] {message}", style="{")
root_logger.setLevel(logging.INFO)

llm_param_translator_prompt =  """
You are a structured scene-to-physics translator for pysocialforce simulations. You will receive a natural language description of a real world scenario, 
and You'll need to think about how to simulate crowd using social force model under the scenario.
Your task is to generate a valid TOML configuration file containing SFM's parameters that accurately reflects the described scenario.
Respond ONLY with a JSON object containing:
{
    "config_file": "TOML string with the config parameters. The last section is the reason why you choose those parameters based on the scenario.",
    "min_distance": "Suggested minimum distance between agents in meters.ONLY GIVE NUMBER",
}
  
Ensure TOML validity and no extra commentary.
Here's an example TOML file for reference:

THE RESOLUTION OF THE SCENE IS 1 METER PER UNIT.
 
title = "Social Force Default Config File"

[scene]
enable_group = true
agent_radius = 0.35
step_width = 0.4 # seconds per simulation step
max_speed_multiplier = 1.3 # max speed = multiplier * desired speed
tau = 0.5
resolution = 10

[goal_attractive_force]
factor = 1

[ped_repulsive_force]
factor = 1.5
v0 = 2.1
sigma = 0.3
# fov params
fov_phi = 100.0
fov_factor = 0.5 # out of view factor

[space_repulsive_force]
factor = 1
u0 = 10
r = 0.2

[group_coherence_force]
factor = 3.0

[group_repulsive_force]
factor = 1.0
threshold = 0.55

[group_gaze_force]
factor = 4.0
# fov params
fov_phi = 90.0

[desired_force]
factor = 1.0
relaxation_time = 0.5
goal_threshold = 0.2

[social_force]
factor = 5.1
lambda_importance = 2.0
gamma = 0.35
n = 2
n_prime = 3

[obstacle_force]
factor = 10.0
sigma = 0.2
threshold = 3.0

[along_wall_force]

[explanation]
Why you choose these parameters based on the scenario.

"""


def _timestamp() -> int:
    return int(time.time())


def _coerce_float(value: object, *, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return float(stripped)
    return default


def _safe_tag(text: str) -> str:
    """Sanitize text for filesystem-friendly tags."""
    return re.sub(r"[\\/:]+", "-", text.strip())


def _sanitize_toml(config_text: str) -> tuple[str, list[str]]:
    """
    Attempt to make loosely formatted TOML from the LLM parseable by:
    - Commenting free text under an [explanation] section.
    Returns the cleaned text and a list of applied fixes.
    """
    lines = config_text.splitlines()
    cleaned: list[str] = []
    notes: set[str] = set()
    in_explanation = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_explanation = stripped.lower() == "[explanation]"
            cleaned.append(line)
            continue

        if in_explanation:
            # Explanation is often free-form; TOML expects key/value. Comment it out.
            bare = stripped.split("#", 1)[0].strip()
            if bare and "=" not in bare:
                cleaned.append(f"# {stripped}")
                notes.add("commented free text in [explanation]")
                continue

        cleaned.append(line)

    return "\n".join(cleaned), sorted(notes)


def _process_gt_data(gt_path: Path, *, apply_homography: bool = False) -> np.ndarray:
    """
    Load a windowed GT CSV (scene,agent_id,agent_type,frame,x,y) and return an
    array shaped (T, N, 2). Missing observations
    are filled with NaN so downstream filtering can drop incomplete timesteps.

    GT from ETH/UCY obsmat-derived windows is already in world meters, so
    homography is skipped by default. Set apply_homography=True only if your
    GT is in pixel space and needs conversion.
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_path}")

    px_to_m = None
    if apply_homography:
        homography_path = gt_path.parent.parent / "obstacle" / "H.txt"
        ref_mask_path = gt_path.parent.parent / "obstacle" / "reference_mask.png"
        if homography_path.exists() and ref_mask_path.exists():
            try:
                from PIL import Image

                H = load_homography(homography_path)
                img = Image.open(ref_mask_path)
                origin_px = (0.0, float(img.height))  # bottom-left pixel anchors (0, 0)
                px_to_m = pixel_to_meter_factory(H, origin_px)
                logger.info("Applying homography for GT using %s", homography_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falling back to raw GT coords (homography error: %s)", exc)
                px_to_m = None
        else:
            logger.warning("Homography or reference mask missing for GT; using raw coords.")

    agents: dict[str, list[tuple[int, float, float]]] = {}
    max_frame = 0
    with gt_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(float(row["frame"]))
                aid_raw = row["agent_id"]
                aid = str(int(float(aid_raw))) if aid_raw else "unknown"
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, ValueError, TypeError) as exc:  # noqa: BLE001
                logger.warning("Skipping malformed GT row %s (%s)", row, exc)
                continue

            if px_to_m is not None:
                xy_m = px_to_m(np.array([[x, y]], dtype=np.float64))[0]
                x, y = float(xy_m[0]), float(xy_m[1])

            agents.setdefault(aid, []).append((frame, x, y))
            max_frame = max(max_frame, frame)

    if not agents:
        raise ValueError(f"No GT trajectories found in {gt_path}")

    def _agent_sort_key(agent_id: str) -> tuple[int, str]:
        try:
            return (0, int(float(agent_id)))
        except Exception:  # noqa: BLE001
            return (1, agent_id)

    agent_ids = sorted(agents.keys(), key=_agent_sort_key)
    traj = np.full((max_frame + 1, len(agent_ids), 2), np.nan, dtype=np.float64)

    for j, aid in enumerate(agent_ids):
        for frame, x, y in sorted(agents[aid], key=lambda t: t[0]):
            if frame < 0:
                continue
            traj[frame, j, 0] = x
            traj[frame, j, 1] = y

    return traj


@dataclass
class SimulationRunner:
    """Pipeline orchestrator for LLM -> config -> simulation -> metrics."""

    results_root: Path = Path("sim/results")
    provider: str = "litellm"
    base_url: str = "https://aikey-gateway.ivia.ch"
    model: str = "azure/gpt-5"
    api_key: str = os.getenv("OPENAI_KEY", "")
    llm_client: Callable[[str], dict] | None = None
    config: dict = None
    min_distance: Optional[float] = None
    step_width: float = 0.4  # seconds per simulation step
    if_need_traj_plot: bool = True

    def __post_init__(self) -> None:
        self.config_dir = self.results_root / "configs"
        self.sim_dir = self.results_root / "simulations"
        self.metrics_dir = self.results_root / "metrics"
        for d in (self.config_dir, self.sim_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._init_client()
        logger.info(f"SimulationRunner initialized (results_root={self.results_root})")

    def _init_client(self) -> None:
        self.llm_client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_KEY"),
                base_url="https://aikey-gateway.ivia.ch",
            )
        logger.info(f"LLM client initialized (provider={self.provider}, model={self.model})")

    def _chat_completion(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": llm_param_translator_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.llm_client.chat.completions.create(model=self.model, messages=messages)
        print(resp)
        return resp.choices[0].message.content or ""

    # ---- LLM -> config ----
    def generate_config(self, scene: dict, *, model_name: str = "gpt-5") -> Path:
        """
        Call the provided LLM client to produce a TOML config.

        The client must return a dict with a "config_file" key containing TOML.
        """
        user_prompt = scene.get("scenario") or json.dumps(scene, ensure_ascii=False)

        content = self._chat_completion(user_prompt)
        try:
            response = json.loads(content)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("LLM response is not valid JSON.") from exc
        if not isinstance(response, dict):
            raise ValueError("LLM client must return a dict.")
        
        config_text = response.get("config_file") or response.get("config")
        if not config_text:
            raise ValueError("LLM response missing 'config_file'.")
        
        # make it a dict
        try:
            config_dict = toml.loads(config_text)
        except toml.TomlDecodeError as exc:
            cleaned, notes = _sanitize_toml(config_text)
            try:
                config_dict = toml.loads(cleaned)
                if notes:
                    logger.warning("Sanitized TOML from LLM (%s)", "; ".join(notes))
                config_text = cleaned
            except Exception as exc2:  # noqa: BLE001
                raise ValueError(f"LLM returned invalid TOML (original error: {exc})") from exc2

        self.config = config_dict
        self.min_distance = _coerce_float(response.get("min_distance"))
        if scene.get("step_width") is not None: # gt already has step_width
            self.step_width = _coerce_float(scene.get("step_width"))
        else:
            self.step_width = _coerce_float(self.config.get("scene", {}).get("step_width"), default=0.4)

        scene_id = scene.get("scene_id", "scene")
        scene_idx = scene.get("scene_index")
        scene_tag = f"{scene_idx:04d}_{scene_id}" if isinstance(scene_idx, int) else scene_id
        safe_model = _safe_tag(model_name)  # azure/gpt-5 -> azure-gpt-5 to avoid filesystem issues
        out_path = self.config_dir / f"{scene_tag}_{safe_model}_{_timestamp()}.toml"
        if not config_text.endswith("\n"):
            config_text += "\n"
        out_path.write_text(config_text, encoding="utf-8")

        return out_path

    # ---- simulation ----
    def _load_obstacles(self, scene: dict) -> np.ndarray:
        path = scene.get("assets", {}).get("anchored_obstacles")
        if not path:
            raise FileNotFoundError("anchored_obstacles path missing in scene assets.")
        npz_path = Path(path)
        if not npz_path.exists():
            # Common mismatch: file stored as *_anchored.npz but path omits suffix.
            alt = npz_path.with_name(f"{npz_path.stem}_anchored{npz_path.suffix}")
            if alt.exists():
                logger.warning("anchored_obstacles not found at %s; using %s", npz_path, alt)
                npz_path = alt
        loaded = np.load(npz_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "obstacles" in loaded.files:
                return loaded["obstacles"]
            if loaded.files:
                return loaded[loaded.files[0]]
            raise ValueError(f"No arrays found in obstacle npz: {path}")
        obstacles = np.asarray(loaded)
        logger.info(f"Loaded obstacles: {npz_path} (shape={obstacles.shape})")
        return obstacles

    def _build_simulator(self, scene: dict, config_path: Path) -> psf.Simulator:
        initial_state = np.asarray(scene["initial_state"], dtype=float)
        groups = scene.get("groups") or None
        obstacles = self._load_obstacles(scene).tolist()
        logger.info(
            f"Building simulator: scene_id={scene.get('scene_id', 'scene')}, "
            f"agents={initial_state.shape[0]}, obstacles={len(obstacles)}"
            f"goals={len(scene.get('goals_m', []))}"
        )
        return psf.Simulator(
            state=initial_state,
            groups=groups,
            obstacles=obstacles,
            config_file=str(config_path),
        )

    def run_simulation(self, scene: dict, config_path: Path, *, steps: int = 150) -> Path:
        sim = self._build_simulator(scene, config_path)
        logger.info(f"Running simulation: scene_id={scene.get('scene_id', 'scene')}, steps={steps}")
        logger.info(f"Simulation step width: {self.step_width}, steps: {steps}")
        # Manual stepping with progress bar to show simulation progress.
        for _ in tqdm(range(steps), desc=f"Simulating {scene.get('scene_id', 'scene')}"):
            sim.step()
        states, _ = sim.get_states()
        scene_id = scene.get("scene_id", "scene")
        out_path = self.sim_dir / f"sim_{scene_id}_{_timestamp()}.npz"
        np.savez_compressed(out_path, states=states, scene=scene)
        logger.info(f"Simulation complete: saved states to {out_path}")
        return out_path

    # ---- metrics ----
    def evaluate(self, scene: dict, states_path: Path, model_name: str, gt_path: Path = None) -> dict:
        data = np.load(states_path, allow_pickle=True)
        states = data["states"]
        scene_id = scene.get("scene_id", "scene")
        logger.info(
            f"Evaluating simulation: scene_id={scene_id}, frames={states.shape[0]}, agents={states.shape[1]}"
        )
        if gt_path is not None:
            logger.info(f"Loading GT data from {gt_path}")
            gt_traj = _process_gt_data(gt_path, apply_homography=False)
            T_common = min(states.shape[0], gt_traj.shape[0])
            N_common = min(states.shape[1], gt_traj.shape[1])
            states_eval = states[:T_common, :N_common, :2]
            gt_eval = gt_traj[:T_common, :N_common, :]
            valid_mask = np.isfinite(gt_eval).all(axis=(1, 2))
            if not valid_mask.any():
                logger.warning("GT data has no fully valid timesteps; skipping GT metrics.")
                gt_eval = None
            else:
                states_eval = states_eval[valid_mask]
                gt_eval = gt_eval[valid_mask]
        else:
            gt_eval = None
            states_eval = states[:, :, :2]
        evaluator = TrajectoryEvaluator(
            collision_distance=_coerce_float(self.config["scene"]["agent_radius"] * 2) or 0.7,
            min_distance=self.min_distance or 0.35,
            dt = self.step_width,
        )
        goals_ctx = None
        goal_raw = scene.get("goal_location_raw", scene.get("goal_location"))
        goal_disabled = False
        if goal_raw is None or (isinstance(goal_raw, str) and goal_raw.strip().lower() == "none"):
            goal_disabled = True
        if isinstance(goal_raw, dict) and str(goal_raw.get("type", "")).lower() == "none":
            goal_disabled = True

        if not goal_disabled:
            # Prefer per-agent goals embedded in the initial state (columns 4:6).
            initial_state = scene.get("initial_state")
            if initial_state is not None:
                try:
                    init_arr = np.asarray(initial_state, dtype=float)
                    if init_arr.ndim == 2 and init_arr.shape[1] >= 6:
                        goals_ctx = init_arr[: states_eval.shape[1], 4:6]
                except Exception:  # noqa: BLE001
                    goals_ctx = None
            if goals_ctx is None:
                goals = np.asarray(scene.get("goals_m", []), dtype=float)
                if goals.size > 0:
                    goals_ctx = goals
        groups = scene.get("groups")
        event_center = scene.get("event_center_m") or scene.get("event_center")
        event_center_ctx = None
        if event_center is not None:
            try:
                event_center_arr = np.asarray(event_center, dtype=float).reshape(-1)
                if event_center_arr.size >= 2:
                    event_center_ctx = event_center_arr[:2]
            except Exception:  # noqa: BLE001
                event_center_ctx = None
        towards_event = scene.get("towards_event")
        if towards_event is None and "should_be_away_from_event_center" in scene:
            legacy_away = scene.get("should_be_away_from_event_center")
            if legacy_away is True:
                towards_event = False
            elif legacy_away is False:
                towards_event = "random"
        if event_center_ctx is None:
            towards_event = "random" if towards_event not in (None, "random") else towards_event

        metrics = evaluator.evaluate(
            states_eval,
            gt=gt_eval,
            goals=goals_ctx,
            bounds=None,
            towards_event=towards_event,
            groups=groups,
            event_center=event_center_ctx,
        )
        ts = _timestamp()
        scene_idx = scene.get("scene_index")
        scene_tag = f"{scene_idx:04d}_{scene_id}" if isinstance(scene_idx, int) else scene_id

        safe_model = _safe_tag(model_name)
        traj_path = None
        if self.if_need_traj_plot:
            traj_path = self.metrics_dir / f"traj_{scene_tag}_{safe_model}_{ts}.png"
            try:
                plot_trajectory(
                    sim_path=states_path,
                    scene_path=None,  # uses embedded scene
                    obstacle_path=None,
                    out_path=traj_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to plot trajectories: {exc}")
                traj_path = None

        flow_path = self.metrics_dir / f"flow_{scene_tag}_{safe_model}_{ts}.png"
        density_path = self.metrics_dir / f"density_{scene_tag}_{safe_model}_{ts}.png"
        try:
            evaluator.draw_flow_heatmap(states[:, :, :2], save_path=str(flow_path))
            evaluator.draw_density_map(states[:, :, :2], save_path=str(density_path))
        except Exception as exc:  # noqa: BLE001
            flow_path = None
            density_path = None
            metrics["PlotError"] = str(exc)

        out = {
            "metrics": metrics,
            "states_path": str(states_path),
            "flow_heatmap_path": str(flow_path) if flow_path else None,
            "density_map_path": str(density_path) if density_path else None,
            "trajectory_plot_path": str(traj_path) if traj_path else None,
        }
        logger.info("Evaluation metrics: %s", metrics)
        out_path = self.metrics_dir / f"metrics_{scene_tag}_{safe_model}_{ts}.json"
        # if the value is NaN, convert to string
        for k, v in out["metrics"].items():
            if isinstance(v, float) and (v != v):  # NaN check
                out["metrics"][k] = "NaN"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    
    # ---- orchestration ----
    def load_scene(self, path: Path) -> dict:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data

    def run_scene(self, scene_path: Path, *, 
                  config_path: Path = None, 
                  steps: int = 150, 
                  model_name: str = "gpt-5", 
                  gt_path: Path = None) -> tuple[Path, Path, dict]:
        scene = self.load_scene(scene_path)
        if not config_path:
            config_path = self.generate_config(scene, model_name=model_name)
        else:
            self.config = toml.load(config_path)
            
        if scene.get("steps") is not None:  
            steps = int(scene.get("steps"))
        elif scene.get("step_width") is not None:
            step_width = _coerce_float(scene.get("step_width"), default=0.4)
            total_time = 600.0  # default to 10 mins
            steps = int(total_time / step_width)
        sim_path = self.run_simulation(scene, config_path, steps=steps)
        metrics = self.evaluate(scene, sim_path, model_name, gt_path)
        return config_path, sim_path, metrics

    def run_all_preprocessed(
        self,
        preprocessed_dir: Path,
        *,
        steps: int = 500,
        model_name: str = "azure/gpt-5",
        max_workers: int = 4,
    ) -> list[tuple[Path, Path, dict]]:
        """
        Run simulations for all preprocessed scenes in a directory using threading to parallelize work.

        If `preprocessed_dir` contains subfolders (e.g., preprocess/preprocessed_scene/00_zurich_hb),
        results are written under corresponding subfolders of `results_root`
        (e.g., sim/results/00_zurich_hb/{configs,simulations,metrics}).
        """
        preprocessed_dir = Path(preprocessed_dir)
        files = sorted(p for p in preprocessed_dir.rglob("*.json"))
        if not files:
            raise FileNotFoundError(f"No preprocessed scenes found in {preprocessed_dir}")

        results: list[tuple[Path, Path, dict]] = []

        # Group files by dataset (first path component under preprocessed_dir)
        grouped: dict[str, list[Path]] = {}
        for p in files:
            rel_parts = p.relative_to(preprocessed_dir).parts
            dataset = rel_parts[0] if len(rel_parts) > 1 else preprocessed_dir.name
            grouped.setdefault(dataset, []).append(p)

        for dataset, dataset_files in grouped.items():
            dataset_root = self.results_root / dataset
            logger.info("Processing dataset %s (%d scenes) -> %s", dataset, len(dataset_files), dataset_root)

            def _worker(scene_path: Path):
                # Fresh runner per thread to avoid shared mutable state.
                runner = SimulationRunner(
                    results_root=dataset_root,
                    provider=self.provider,
                    base_url=self.base_url,
                    model=self.model,
                    api_key=self.api_key,
                    llm_client=None,
                    if_need_traj_plot=self.if_need_traj_plot,
                )
                return runner.run_scene(scene_path, steps=steps, model_name=model_name)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(_worker, p): p for p in dataset_files}
                for future in tqdm(as_completed(future_to_path), total=len(dataset_files), desc=f"Scenes ({dataset})"):
                    path = future_to_path[future]
                    try:
                        cfg, sim, metrics = future.result()
                        results.append((cfg, sim, metrics))
                        logger.info(f"Completed simulation for {dataset}/{path.name}")
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"Failed simulation for {dataset}/{path.name}: {exc}")

        return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM -> pysocialforce simulation pipeline.")
    parser.add_argument(
        "--scene",
        type=Path,
        nargs="?",
        help="Path to preprocessed scene JSON. Optional if --context-index is provided.",
    )
    parser.add_argument(
        "--context-index",
        type=int,
        help="0-based line number in context.jsonl to select a preprocessed scene (uses filename <idx>_<scene_id>.json).",
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        default=Path("datasets/context.jsonl"),
        help="Path to context.jsonl (used with --context-index).",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("preprocess/preprocessed_scene"),
        help="Directory containing preprocessed scene JSON files.",
    )
    # gt if given
    parser.add_argument(
        "--gt-path",
        type=Path, # downloads/gt/eth/trajectory_dense/seq_eth_trajectory_dense.csv
        help="GT if run the gt scene",
    )
    parser.add_argument("--steps", type=int, default=1500, help="Simulation steps.") # sim 10 mins
    parser.add_argument("--model-name", default="azure/gpt-5", help="Label to embed in config filename.")
    parser.add_argument("--run-all", action="store_true", help="Simulate all scenes under --preprocessed-dir.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers when using --run-all.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("sim/results"),
        help="Directory to store generated configs, simulations, and metrics.",
    )
    return parser.parse_args()


def _scene_path_from_context_index(index: int, context_path: Path, preprocessed_dir: Path) -> Path:
    """
    Resolve a preprocessed scene path from a context.jsonl line number.

    Files are expected to be named <index:04d>_<scene_id>.json.
    """
    if index < 0:
        raise ValueError("context index must be non-negative")
    with context_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                record = json.loads(line)
                scene_id = Path(str(record.get("image", ""))).stem
                if not scene_id:
                    raise ValueError(f"Missing image field in context line {index}")
                path = preprocessed_dir / f"{index:04d}_{scene_id}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Preprocessed scene not found: {path}")
                return path
    raise IndexError(f"context index {index} out of range for {context_path}")


def main() -> None:
    args = _parse_args()
    runner = SimulationRunner(results_root=args.results_root)
    # If neither scene nor context-index is provided, default to run_all.
    if args.run_all or (args.scene is None and args.context_index is None):
        runner.run_all_preprocessed(
            preprocessed_dir=args.preprocessed_dir,
            steps=args.steps,
            model_name=args.model_name,
            max_workers=args.max_workers,
        )
        return

    if args.context_index is not None:
        scene_path = _scene_path_from_context_index(args.context_index, args.context_path, args.preprocessed_dir)
    elif args.scene is not None:
        scene_path = args.scene
    else:
        raise ValueError("Provide either a scene path or --context-index.")
    gt_path = None
    if args.gt_path is not None:
        gt_path = args.gt_path

    runner.run_scene(scene_path, steps=args.steps, model_name=args.model_name, gt_path=gt_path)


if __name__ == "__main__":
    main()
    #test
    # runner = SimulationRunner()
    # test_scene = _scene_path_from_context_index(
    #     index=30,
    #     context_path=Path("datasets/context_simplified_test.jsonl"),
    #     preprocessed_dir=Path("preprocess/preprocessed_scene"),
    # )
    # runner.run_scene(test_scene, 
    #                  steps=400) # if giving config_path, it will skip llm generation

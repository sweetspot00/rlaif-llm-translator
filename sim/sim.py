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
    "config_file": "TOML string with the config parameters.",
    "explanation": "Why you choose these parameters. ", 
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

def _process_gt_data(gt_path: Path) -> np.ndarray:
    """
    Load GT dense trajectories and return array of shape (T, N, 2) in meters.
    - Uses homography + image height to anchor bottom-left pixel as (0, 0).
    - Frames are 1-indexed in the CSV; output is indexed by frame number (0..max_frame).
    - Missing frames for an agent remain NaN.
    """
    base_dir = gt_path.parent.parent  # e.g., downloads/gt/eth

    def _first_match(directory: Path, pattern: str) -> Path | None:
        matches = sorted(directory.glob(pattern))
        return matches[0] if matches else None

    homo_path = _first_match(base_dir / "homography", "*_H.txt")
    info_path = _first_match(base_dir / "information", "*_info.json")
    if homo_path is None or info_path is None:
        raise FileNotFoundError(f"Missing homography/info under {base_dir}")
    info = json.loads(info_path.read_text())
    height = info.get("height")
    if height is None:
        raise ValueError(f"'height' missing in {info_path}")
    H = load_homography(homo_path)
    px_to_m = pixel_to_meter_factory(H, origin_px=(0.0, float(height)))

    agents: dict[str, list[tuple[int, float, float]]] = {}
    max_frame = 0
    with gt_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            x = float(row["x"])
            y = float(row["y"])
            agent = row["agent_id"]
            ax, ay = px_to_m((x, y))
            if agent not in agents:
                agents[agent] = []
            agents[agent].append((frame, ax, ay))
            max_frame = max(max_frame, frame)

    agent_ids = sorted(agents.keys(), key=lambda a: int(a))
    T = max_frame + 1
    N = len(agent_ids)
    traj = np.full((T, N, 2), np.nan, dtype=float)
    for j, aid in enumerate(agent_ids):
        for frame, ax, ay in agents[aid]:
            traj[frame, j, 0] = ax
            traj[frame, j, 1] = ay
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
        config_dict = toml.loads(config_text)
        self.config = config_dict
        self.min_distance = _coerce_float(response.get("min_distance"))
        if scene.get().get("step_width") is not None: # gt already has step_width
            self.step_width = _coerce_float(scene.get().get("step_width"))
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
        loaded = np.load(Path(path), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "obstacles" in loaded.files:
                return loaded["obstacles"]
            if loaded.files:
                return loaded[loaded.files[0]]
            raise ValueError(f"No arrays found in obstacle npz: {path}")
        obstacles = np.asarray(loaded)
        logger.info(f"Loaded obstacles: {path} (shape={obstacles.shape})")
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
        sim.step(n=steps)
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
            gt_traj = _process_gt_data(gt_path)
            T_common = min(states.shape[0], gt_traj.shape[0])
            N_common = min(states.shape[1], gt_traj.shape[1])
            states_eval = states[:T_common, :N_common, :2]
            gt_eval = gt_traj[:T_common, :N_common, :]
            # keep only timesteps where all agents have finite GT
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
        goals = np.asarray(scene["goals_m"], dtype=float)
        # pad goals to match agents
        if goals.shape[0] < states_eval.shape[1]:
            idx = np.arange(states_eval.shape[1]) % goals.shape[0]
            goals = goals[idx]
        metrics = evaluator.evaluate(states_eval, gt=gt_eval, goals=goals)
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
        """
        files = sorted(p for p in preprocessed_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No preprocessed scenes found in {preprocessed_dir}")

        results: list[tuple[Path, Path, dict]] = []

        def _worker(scene_path: Path):
            # Fresh runner per thread to avoid shared mutable state.
            runner = SimulationRunner(
                results_root=self.results_root,
                provider=self.provider,
                base_url=self.base_url,
                model=self.model,
                api_key=self.api_key,
                llm_client=None,
                if_need_traj_plot=self.if_need_traj_plot,
            )
            return runner.run_scene(scene_path, steps=steps, model_name=model_name)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_worker, p): p for p in files}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    cfg, sim, metrics = future.result()
                    results.append((cfg, sim, metrics))
                    logger.info(f"Completed simulation for {path.name}")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed simulation for {path.name}: {exc}")

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
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps.")
    parser.add_argument("--model-name", default="azure/gpt-5", help="Label to embed in config filename.")
    parser.add_argument("--run-all", action="store_true", help="Simulate all scenes under --preprocessed-dir.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers when using --run-all.")
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
    runner = SimulationRunner()
    if args.run_all:
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

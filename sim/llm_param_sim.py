from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from utils import call_model_with_platform, call_model_with_yunwu
from vanila_pysfm.vanila_pysocialforce import render_visuals
from utils.build_obstacles_map import build_obstacle_map_from_text_llm

SYSTEM_PROMPT = """
You are a structured scene-to-physics translator for pysocialforce simulations. You will receive a natural language description of a real world scenario, 
and You'll need to think about how to simulate crowd using social force model under the scenario.
Your task is to generate a valid TOML configuration file containing SFM's parameters that accurately reflects the described scenario.
Respond ONLY with a JSON object containing:
{
    "config_file": "TOML string with the config parameters.",
    "explanation": "Why you choose these parameters. ", 
    "n_agents": "Suggested number of agents for the scenario.ONLY GIVE NUMBER",
    "min_distance": "Suggested minimum distance between agents in meters.ONLY GIVE NUMBER",
    "desired_speed": "Suggested desired starting speed of agents in m/s. ONLY GIVE NUMBER",
    "need_random_targets": "true/false, whether random targets are needed for this scenario. If false, then will use the exits as targets."
}
  
Ensure TOML validity and no extra commentary.

THE RESOLUTION IS 1 METER = 1 UNIT IN SIMULATION. YOUR CONFIG MUST REFLECT THIS SCALE.

Here's an example TOML file for reference:

title = "Social Force Config File"

[scene]
enable_group = true
agent_radius = 0.35
# the maximum speed doesn't exceed 1.3x initial speed
max_speed_multiplier = 1.3

[desired_force]
factor = 1.0
# The relaxation distance of the goal
goal_threshold = 0.2
# How long the relaxation process would take
relaxation_time = 0.5


[social_force]
factor = 5.1
# Moussaid-Helbing 2009
# relative importance of position vs velocity vector
lambda_importance = 2.0
# define speed interaction
gamma = 0.35
n = 2
# define angular interaction
n_prime = 3

[obstacle_force]
factor = 10.0
# the standard deviation of obstacle force
sigma = 0.2
# threshold to trigger this force
threshold = 3.0

[group_coherence_force]
factor = 3.0

[group_repulsive_force]
factor = 1.0
# threshold to trigger this force
threshold = 0.55

[group_gaze_force]
factor = 4.0
# fielf of view
fov_phi = 90.0

[along_wall_force]
factor = 0.6
"""

TMP_DIR = Path("vanila_pysfm/configs")


def _normalize_label(text: str) -> str:
    return text.lower().replace("/", "_")


def _prepare_model_identifier(platform: str, model: str) -> str:
    if "/" in model:
        return model
    if platform in {"azure", "openai"}:
        return f"{platform}/{model}"
    return model


def _auto_temperature(model_identifier: str) -> Optional[float]:
    return 1.0 if "gpt-5" in model_identifier else 0.2


def _load_prompt_text(prompt_path: Path) -> str:
    if not prompt_path.exists():
        prompt_path = prompt_path.with_suffix(".txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}.")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file '{prompt_path}' is empty.")
    return text


def generate_config(
    prompt_path: str,
    *,
    platform: str = "openai",
    model: str = "gpt-4o",
    tmp_dir: Optional[Path] = None,
) -> Path:
    """Call an LLM to generate a TOML config and save it under tmp/."""
    prompt_path = Path(prompt_path)
    user_prompt = _load_prompt_text(prompt_path)

    platform_lower = platform.lower()
    if platform_lower in {"openai", "azure"}:
        model_identifier = _prepare_model_identifier(platform_lower, model)
        temperature = _auto_temperature(model_identifier)
        response_data = call_model_with_platform(
            platform_lower,
            model_identifier,
            SYSTEM_PROMPT,
            user_prompt,
            temperature=temperature,
        )
    elif platform in {"gemini", "yunwu"}:
        response_data = call_model_with_yunwu(SYSTEM_PROMPT, user_prompt)
    else:
        raise ValueError(f"Unsupported LLM platform '{platform}'.")

    if not response_data:
        raise RuntimeError("LLM returned no data.")


    config_text = response_data.get("config_file") or response_data.get("config")
    if not config_text:
        raise ValueError("LLM response missing 'config_file'.")

    output_dir = tmp_dir or TMP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    sanitized_label = _normalize_label(model)
    out_path = output_dir / f"{prompt_path.stem}_{sanitized_label}_{timestamp}.toml"
    if not config_text.endswith("\n"):
        config_text += "\n"
    out_path.write_text(config_text, encoding="utf-8")
    
    # load others
    n_agents = response_data.get("n_agents")
    min_distance = response_data.get("min_distance")
    desired_speed = response_data.get("desired_speed")
    need_random_targets = response_data.get("need_random_targets")
    print(f"Generated config saved to: {out_path}")
    print(f"Suggested n_agents: {n_agents}, min_distance: {min_distance}, desired_speed: {desired_speed}, need_random_targets: {need_random_targets}")
    return out_path, n_agents, min_distance, desired_speed, need_random_targets

def simulate_with_llm_config(prompt_path: str, platform: str = "azure", model: str = "azure/gpt-4o-mini"):
    from vanila_pysfm.vanila_pysocialforce import run_sim

    config_path, n_agents, min_distance, desired_speed, need_random_targets = generate_config(
        prompt_path, platform=platform, model=model
    )
    n_agents = int(n_agents) if n_agents else 10
    min_distance = float(min_distance) if min_distance else 0.2
    desired_speed = float(desired_speed) if desired_speed else 1.2
    print(f"155: Need ramdom targets: {need_random_targets}")

    run_sim(
        n_agents=n_agents,
        min_distance=min_distance,
        desired_speed=desired_speed,
        config_path=config_path,
        need_random_targets=need_random_targets,
        save_file_name=f"llm_sim_{Path(prompt_path).stem}_{model}_{int(time.time())}",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pysocialforce configs via LLM.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Path to prompt text (relative or absolute).",
    )
    parser.add_argument("--platform", default="azure", help="LLM platform (azure/openai/gemini/yunwu).")
    parser.add_argument("--model", default="gpt-4o", help="Model name for the chosen platform.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="If set, run the simulation after generating the config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt_path = args.prompt
    cfg_path, n_agents, min_distance, desired_speed, need_random_targets = generate_config(
        prompt_path, model=args.model, platform=args.platform
    )
    text_description = _load_prompt_text(Path(prompt_path))
    if args.simulate:
        n_agents = int(n_agents) if n_agents else 10
        min_distance = float(min_distance) if min_distance else 0.2
        desired_speed = float(desired_speed) if desired_speed else 1.2
        from vanila_pysfm.vanila_pysocialforce import run_sim
        
        # dynamic obstacle map from llm
        obstacle_map, target_positions = build_obstacle_map_from_text_llm(description=text_description, 
                                                                          model=args.model, 
                                                                          platform=args.platform)
        metris, states, groups, visual_path = run_sim(
            n_agents=n_agents,
            min_distance=min_distance,
            desired_speed=desired_speed,
            need_random_targets=need_random_targets,
            config_path=cfg_path,
            obstacle_map=obstacle_map,
            target_positions=target_positions,
            use_default_map=False, # use llm map
            save_file_name=f"llm_sim_{Path(prompt_path).stem}_{_normalize_label(args.model)}_{int(time.time())}",
        )
        # TODO: llm as a judge
        
        # # render visuals
        render_visuals(
            obstacle_map,
            target_positions,
            states,
            visual_path,
            groups=groups,
        )

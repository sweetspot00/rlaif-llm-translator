"""
Build a preprocessed GT scene JSON (similar to preprocess/preprocessed_scene/*).

Inputs:
- Prompt text file with lines: "<scene_key>, <scenario description>"
- Initial state NPY (px, py, vx, vy, gx, gy) already in pixel coordinates.
- Anchored obstacles NPZ (from preprocess/gt/gt_line_obstacles).
- Homography TXT (downloads/gt/<scene>/homography/*_H.txt).

Outputs:
- JSON written to preprocess/preprocessed_scene/<scene>_gt.json (or --out).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_prompts(prompt_file: Path, scene_hint: str | None = None) -> Dict[str, str]:
    """
    Load prompts from either:
    - keyed lines: "<scene_key>, <scenario description>"
    - single prompt file (e.g., eth_prompt.txt) containing only the text.
    """
    prompts: Dict[str, str] = {}
    with prompt_file.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        if "," in line:
            key, desc = line.split(",", 1)
            if key.strip():
                prompts[key.strip()] = desc.strip()

    if scene_hint and scene_hint not in prompts:
        # Fallback: treat entire file as one prompt.
        text = "\n".join(lines).strip()
        key = scene_hint or prompt_file.stem
        if key.endswith("_prompt"):
            key = key[: -len("_prompt")]
        prompts[key] = text
    elif not prompts:
        # No keyed entries at all; still fallback.
        text = "\n".join(lines).strip()
        key = scene_hint or prompt_file.stem
        if key.endswith("_prompt"):
            key = key[: -len("_prompt")]
        prompts[key] = text

    return prompts


@dataclass
class ScenePayload:
    scene_id: str
    scene_index: int
    scenario: str
    category: str
    crowd_size_label: str
    crowd_size: int
    ungrouped_agents: int
    event_center_px: List[float]
    event_center_m: List[float]
    goals_px: List[List[float]]
    goals_m: List[List[float]]
    initial_state: List[List[float]]
    groups: List[List[int]]
    assets: dict

    def to_json(self) -> dict:
        data = asdict(self)
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GT preprocessed scene JSON.")
    parser.add_argument("--scene", default="eth", help="Scene key matching prompts.txt prefix (e.g., eth).")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("preprocess/gt/gt_scene/prompts.txt"),
        help="CSV-style prompt file.",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=Path("preprocess/gt/gt_scene/seq_eth_initial_state.npy"),
        help="NPY file with initial_state in pixel coordinates.",
    )
    parser.add_argument(
        "--anchored-obstacles",
        type=Path,
        default=Path("preprocess/gt/gt_line_obstacles/eth_line_obstacles.npz"),
        help="Anchored obstacles NPZ path.",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=Path("downloads/gt/eth/homography/seq_eth_H.txt"),
        help="Homography TXT path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Default: preprocess/preprocessed_scene/<scene>_gt.json",
    )
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index to record.")
    parser.add_argument("--category", type=str, default="GT", help="Scene category label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompt_file, scene_hint=args.scene)
    scenario = prompts.get(args.scene)
    if not scenario:
        raise ValueError(f"No scenario text found for scene '{args.scene}' in {args.prompt_file}")

    init_state = np.load(args.initial_state)
    initial_state_list = init_state.tolist()

    payload = ScenePayload(
        scene_id=args.scene,
        scene_index=args.scene_index,
        scenario=scenario,
        category=args.category,
        crowd_size_label=str(len(initial_state_list)),
        crowd_size=len(initial_state_list),
        ungrouped_agents=len(initial_state_list),
        event_center_px=[],
        event_center_m=[],
        goals_px=[],
        goals_m=[],
        initial_state=initial_state_list,
        groups=[],
        assets={
            "anchored_obstacles": str(args.anchored_obstacles),
            "homography": str(args.homography),
        },
    )

    out_path = args.out
    if out_path is None:
        out_path = Path("preprocess/preprocessed_scene") / f"{args.scene}_gt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload.to_json(), indent=2))
    print(f"Wrote {out_path} (agents={len(initial_state_list)})")


if __name__ == "__main__":
    main()

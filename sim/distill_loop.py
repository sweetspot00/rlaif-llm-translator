# distill_loop.py
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import toml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import openai  # your gateway usage
from sim import SimulationRunner, _coerce_float, _safe_tag, _timestamp  # reuse your existing code


# ----------------------------
# Student: Qwen local generator
# ----------------------------
class QwenStudent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        # robustly extract the first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Student output has no JSON object:\n{text[:500]}")
        return json.loads(m.group(0))

    def generate(self, system_prompt: str, user_prompt: str, *, max_new_tokens: int = 1800) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return self._extract_json(decoded)


# --------------------------------
# Teacher feedback schema + helpers
# --------------------------------
TEACHER_SCHEMA_EXAMPLE = {
    "verdict": "FAIL",
    "edits": [{"path": "ped_repulsive_force.factor", "op": "set", "value": 1.6}],
    "stop_recommendation": False,
    "rationale": ["Increase ped repulsion to reduce collisions / bunching."]
}

def _read_image_as_data_url(path: Path) -> str:
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _teacher_prompt(scene: Dict[str, Any], toml_text: str, metrics: Dict[str, Any]) -> str:
    # Keep this short + structured. GPT sees images separately.
    # Important: tell teacher to output ONLY JSON that matches schema.
    return f"""
You are a simulation critic for pysocialforce SFM parameters.

Return ONLY a JSON object with fields:
- verdict: "PASS" or "FAIL"
- edits: list of edits. Each edit is {{ "path": "dot.path", "op": "set", "value": number|bool|string }}
- stop_recommendation: boolean
- rationale: list of short bullet strings explaining the edits

Rules:
- Only propose edits that modify the TOML parameters.
- Keep edits minimal. Prefer changing 1-5 parameters per iteration.
- If metrics already satisfy the target behavior, output verdict="PASS" and edits=[].

SCENE_DESCRIPTION:
{scene.get("scenario","")}

CATEGORY:
{scene.get("category","")}

TOML:
{toml_text}

METRICS_JSON:
{json.dumps(metrics, ensure_ascii=False, sort_keys=True)}
""".strip()

def _parse_teacher_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Teacher output has no JSON object:\n{text[:500]}")
    obj = json.loads(m.group(0))
    # minimal validation
    for k in ["verdict", "edits", "stop_recommendation"]:
        if k not in obj:
            raise ValueError(f"Teacher JSON missing '{k}': {obj}")
    if not isinstance(obj["edits"], list):
        raise ValueError("Teacher 'edits' must be a list.")
    return obj

def _set_by_dotted_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def apply_teacher_edits(prev_toml_text: str, feedback: Dict[str, Any]) -> str:
    cfg = toml.loads(prev_toml_text)
    for e in feedback.get("edits", []):
        if e.get("op") != "set":
            continue
        _set_by_dotted_path(cfg, e["path"], e["value"])

    # keep explanation consistent (optional but helps learning)
    rationale = feedback.get("rationale") or []
    if "explanation" not in cfg:
        cfg["explanation"] = {}
    if isinstance(cfg["explanation"], dict):
        cfg["explanation"]["text"] = " ".join(str(x) for x in rationale)[:1500] or "Auto-updated based on teacher edits."

    return toml.dumps(cfg)


# ----------------------------
# Stop rules (metric + teacher)
# ----------------------------
@dataclass
class StopConfig:
    max_iters: int = 6
    collision_max: float = 0.03
    goalrate_min: float = 0.85
    sdv_max: float = 0.02  # SocialDistanceViolations
    plateau_patience: int = 2
    min_improve: float = 1e-4

def score_metrics(m: Dict[str, Any]) -> float:
    # Higher is better.
    # Clamp NaNs (your code stores "NaN" as string sometimes)
    def num(x, default=0.0):
        try:
            if x is None: return default
            if isinstance(x, str) and x.strip().lower() == "nan": return default
            return float(x)
        except Exception:
            return default

    collision = num(m.get("CollisionRate"), 1.0)
    goal = num(m.get("GoalRate"), 0.0)
    sdv = num(m.get("SocialDistanceViolations"), 1.0)

    # simple weighted score
    return (goal * 2.0) - (collision * 3.0) - (sdv * 1.0)

def metrics_pass(m: Dict[str, Any], s: StopConfig) -> bool:
    def num(x, default=0.0):
        try:
            if x is None: return default
            if isinstance(x, str) and x.strip().lower() == "nan": return default
            return float(x)
        except Exception:
            return default

    return (
        num(m.get("CollisionRate"), 1.0) <= s.collision_max and
        num(m.get("GoalRate"), 0.0) >= s.goalrate_min and
        num(m.get("SocialDistanceViolations"), 1.0) <= s.sdv_max
    )


# ----------------------------
# Main distillation runner
# ----------------------------
class DistillRunner:
    def __init__(
        self,
        *,
        student: QwenStudent,
        teacher_model: str = "gpt-4o",  # or your azure/gpt-5 if it supports vision in your gateway
        results_root: Path = Path("sim/results"),
        distill_out: Path = Path("distill_runs.jsonl"),
    ):
        self.student = student
        self.teacher_model = teacher_model
        self.distill_out = distill_out

        self.sim_runner = SimulationRunner(results_root=results_root)
        self.teacher_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_KEY"),
            base_url="https://aikey-gateway.ivia.ch",
        )

        # use your existing system prompt (import from sim.py if you prefer)
        from sim import llm_param_translator_prompt
        self.student_system_prompt = llm_param_translator_prompt

    def call_teacher_with_images(
        self,
        *,
        scene: Dict[str, Any],
        toml_text: str,
        metrics: Dict[str, Any],
        flow_path: Optional[Path],
        density_path: Optional[Path],
        traj_path: Optional[Path],
    ) -> Dict[str, Any]:
        prompt = _teacher_prompt(scene, toml_text, metrics)

        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for p in [flow_path, density_path, traj_path]:
            if p and p.exists():
                content_blocks.append({"type": "image_url", "image_url": {"url": _read_image_as_data_url(p)}})

        resp = self.teacher_client.chat.completions.create(
            model=self.teacher_model,
            messages=[{"role": "user", "content": content_blocks}],
        )
        text = resp.choices[0].message.content or ""
        return _parse_teacher_json(text)

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        self.distill_out.parent.mkdir(parents=True, exist_ok=True)
        with self.distill_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_scene_distill(self, scene_path: Path, *, stop_cfg: StopConfig) -> None:
        scene = self.sim_runner.load_scene(scene_path)

        prev_toml_text: Optional[str] = None
        prev_min_distance: Optional[float] = None

        best_score = -1e18
        best_toml = None
        best_min_dist = None
        no_improve = 0

        for it in range(stop_cfg.max_iters):
            # 1) Student generates (first time from scenario, later from feedback+metrics)
            if it == 0:
                user_prompt = scene.get("scenario") or json.dumps(scene, ensure_ascii=False)
            else:
                # summarize last run for student
                user_prompt = f"""
                        SCENARIO:
                        {scene.get("scenario","")}

                        PREVIOUS_TOML:
                        {prev_toml_text}

                        PREVIOUS_METRICS_JSON:
                        {json.dumps(last_metrics, ensure_ascii=False, sort_keys=True)}

                        TEACHER_FEEDBACK_JSON:
                        {json.dumps(last_teacher, ensure_ascii=False, sort_keys=True)}

                        TASK:
                        Regenerate a corrected TOML + min_distance as JSON.
                        """.strip()

            student_out = self.student.generate(self.student_system_prompt, user_prompt)
            toml_text = student_out["config_file"]
            min_dist = _coerce_float(student_out.get("min_distance"), default=0.35)

            # 2) Simulate
            # write TOML to disk using existing runner for consistency
            scene_id = scene.get("scene_id", "scene")
            safe_model = _safe_tag("qwen2p5-3b")
            cfg_path = self.sim_runner.config_dir / f"{scene_id}_{safe_model}_{_timestamp()}.toml"
            cfg_path.write_text(toml_text if toml_text.endswith("\n") else toml_text + "\n", encoding="utf-8")

            # set runner state for evaluator thresholds
            self.sim_runner.config = toml.loads(toml_text)
            self.sim_runner.min_distance = min_dist

            sim_path = self.sim_runner.run_simulation(scene, cfg_path, steps=int(scene.get("steps", 1500)))
            out = self.sim_runner.evaluate(scene, sim_path, model_name="qwen2p5-3b", gt_path=None)

            last_metrics = out["metrics"]
            flow_path = Path(out["flow_heatmap_path"]) if out.get("flow_heatmap_path") else None
            density_path = Path(out["density_map_path"]) if out.get("density_map_path") else None
            traj_path = Path(out["trajectory_plot_path"]) if out.get("trajectory_plot_path") else None

            # track best
            sc = score_metrics(last_metrics)
            if sc > best_score + stop_cfg.min_improve:
                best_score = sc
                best_toml = toml_text
                best_min_dist = min_dist
                no_improve = 0
            else:
                no_improve += 1

            # 3) Teacher feedback (with images)
            last_teacher = self.call_teacher_with_images(
                scene=scene,
                toml_text=toml_text,
                metrics=last_metrics,
                flow_path=flow_path,
                density_path=density_path,
                traj_path=traj_path,
            )

            # 4) Log a distillation training pair for SFT:
            # input: (scenario + prev_toml + metrics + teacher_feedback) -> target: next_toml
            # We can set "target_toml" as:
            # - if teacher says PASS, target is current toml
            # - else apply teacher edits deterministically (gives a clean target)
            if last_teacher["verdict"] == "PASS" or not last_teacher.get("edits"):
                target_toml = toml_text
                target_min = min_dist
            else:
                target_toml = apply_teacher_edits(toml_text, last_teacher)
                target_min = min_dist  # keep same unless you include min_distance edits in teacher schema

            record = {
                "scenario": scene.get("scenario", ""),
                "category": scene.get("category", ""),
                "resolution_m_per_unit": 1,
                "prev_toml": toml_text,
                "metrics": last_metrics,
                "teacher_feedback": last_teacher,
                "target_min_distance": float(target_min),
                "target_toml": target_toml,
            }
            self._write_jsonl(record)

            # Stop checks
            if last_teacher.get("stop_recommendation") and metrics_pass(last_metrics, stop_cfg):
                break
            if last_teacher.get("verdict") == "PASS" and metrics_pass(last_metrics, stop_cfg):
                break
            if no_improve >= stop_cfg.plateau_patience:
                break

            prev_toml_text = toml_text
            prev_min_distance = min_dist

        # (optional) also log “best” as a final SFT example
        if best_toml is not None:
            self._write_jsonl({
                "scenario": scene.get("scenario", ""),
                "category": scene.get("category", ""),
                "resolution_m_per_unit": 1,
                "prev_toml": "",
                "metrics": {},
                "teacher_feedback": {"verdict": "PASS", "edits": [], "stop_recommendation": True, "rationale": ["Best-of-run selection."]},
                "target_min_distance": float(best_min_dist or 0.35),
                "target_toml": best_toml,
            })


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--distill-out", type=Path, default=Path("distill_runs.jsonl"))
    ap.add_argument("--teacher-model", type=str, default="gpt-4o")
    ap.add_argument("--max-iters", type=int, default=6)
    args = ap.parse_args()

    student = QwenStudent("Qwen/Qwen2.5-3B-Instruct", device="cuda:0")
    runner = DistillRunner(
        student=student,
        teacher_model=args.teacher_model,
        distill_out=args.distill_out,
    )
    stop = StopConfig(max_iters=args.max_iters)
    runner.run_scene_distill(args.scene, stop_cfg=stop)

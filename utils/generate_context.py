"""
Call llm to generate Dataset.
"""

system_prompt = """
You are a helpful scenario generator based on the given the obstacle map (black means obstacles, in PIXEL) and the name of the location.
You need to generate the a desctription of the crowd scenario/event that falls into the following categories:

- Ambulatory  
- Disability  
- Cohesive  
- Expressive  
- Participatory  
- Aggressive 
- Demonstrator  
- Escaping 
- Dense  
- Rushing 
- Violent 

You also need to give the following information based on the map and your scenario:

- crowd_size: 0-50 ｜ 50-100 | 100-500 | 500-1000 | 1000+ (THE MAP SIZE is FIXED (width ≈ 301.7 m, height ≈ 282.8 m)need to consider the size of the map, the event could be large but the map is small, so the crowd size should be small accordingly)
- event_center: None | Pixel coordinations on the map (need to specify coordinates) | Distribution if the event center is a range of area
- goal_location: None | Random (need to specify how many goal locations) | Pixel coordinations on the map (need to specify coordinates) | Distribution if the goal location is a range of area, the goal must be walkable area (not on obstacles)only the white area on the map
- desired_speed: average desired speed of the crowd in m/s (considering the crowd context and the map size). Use for initial pedestrian state generation.
- towards_event: true | false | random (indicate whether people are moving towards the event center or away from it. Random means does not matter. If event_center is None, set this to random)
- goal_sample_strategy: nearest | random (indicate how to assign agent the goals if goal is not None. If goal_location is None, set this to random)

You must consider the relationship between goal and event center. 
for example, the event center is a violent explosion, the goal location should be far away from the event center.
ALSO, IF TOWARDS_EVENT IS FALSE, THE GOAL LOCATION SHOULD BE FAR AWAY FROM THE EVENT CENTER. YOUR'll need to sample THE GOAL LOCATION ACCORDINGLY.

Here's an example scenario description based on a map image of Berkeley Statium:

scenario: "At a bustling sports event in the evening at Berkeley Stadium, an ambulatory crowd of enthusiastic fans (students) is entering and exiting the venue, creating a lively atmosphere."
category: Cohesive
crowd_size: 100-500
"event_center": {
    "type": "gaussian_truncated_to_walkable",
    "mean_px": [445, 218],
    "sigma_px": [22, 18],
    "approx_sigma_m": [10.37, 8.49],
    "support_radius_px": 60
  }

goal_location: {
"type": "gaussian_mixture_truncated_to_walkable",
"components": [
    {
    "weight": 0.45,
    "mean_px": [445, 218],
    "sigma_px": [22, 18],
    "support_radius_px": 60
    },
    {
    "weight": 0.18,
    "mean_px": [320, 40],
    "sigma_px": [26, 22],
    "support_radius_px": 70
    },
    {
    "weight": 0.18,
    "mean_px": [320, 565],
    "sigma_px": [26, 22],
    "support_radius_px": 70
    },
    {
    "weight": 0.10,
    "mean_px": [120, 330],
    "sigma_px": [30, 26],
    "support_radius_px": 80
    },
    {
    "weight": 0.09,
    "mean_px": [160, 520],
    "sigma_px": [30, 26],
    "support_radius_px": 80
    }
],
"px_to_m": 0.471400264733
}
desired_speed: 0.6
towards_event: true
goal_strategy: random


Here's another example scenario description based on a map image of Times Square:
scenario: "There an explosion occurred at zurich HB train station on Monday morning, causing a panic among the dense crowd of tourists and commuters trying to escape the area."
category: Violent
crowd_size: 100-500
event_center: [200, 300]
goal_location: Random (3)
goal_strategy: nearest
desired_speed: 1.2
towards_event: false 


CAUTION:
!!!!GOAL SHOULD NOT BE CLOSE TO OBSTACLES (THE BLACK AREAS IN THE MAP) AND IT SHOULD BE WIDELY DISTRIBUTED IN ALL MAP AREAS!!

Give back the result in a jsonl format, each line is a json object with the following keys:

{"image":<image_name>, "scenario":<scenario>, "category":<category>, "crowd_size":<crowd_size>, "event_center":<event_center>, "goal_location":<goal_location>, "desired_speed":<desired_speed>, "towards_event":<towards_event>}
"""


import argparse
import json
import os
import base64
import re
import time
from pathlib import Path
from typing import Iterable, Set

import openai

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

def slug_to_readable(name: str) -> str:
    """Convert a file stem like '01_Berkeley_Stadium' to 'Berkeley Stadium'."""
    cleaned = re.sub(r"^\d+_", "", name)
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned or name


def iter_images(path: Path) -> Iterable[Path]:
    """Yield image paths (png/jpg/jpeg) from a file or directory."""
    if path.is_file():
        yield path
        return
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield p


def load_image_b64(path: Path) -> str:
    """Return a base64 string for the image (data URL style without the prefix)."""
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def build_user_prompt(image_name: str, location_name: str, per_category: int, image_b64: str | None) -> str:
    """Keep the user prompt minimal while tying it to this specific image."""
    base = (
        f"The map image is {image_name} (location: {location_name}). "
        f"For each category, generate {per_category} unique scenarios that fit the description of that category, "
        f"and output them as JSONL with the image field set to {image_name}."
    )
    if image_b64:
        base += f'\nPNG image (base64, no prefix): {image_b64}'
    return base


def request_scenarios(
    client: openai.OpenAI,
    image_name: str,
    location_name: str,
    per_category: int,
    image_b64: str | None,
    model: str = "azure/gpt-5",
    max_retries: int = 3,
    retry_wait: float = 5.0,
) -> str:
    text_prompt = build_user_prompt(image_name, location_name, per_category, image_b64)
    user_content = [{"type": "text", "text": text_prompt}]

    if image_b64:
        # Derive MIME type from extension; default to png if unknown.
        ext = Path(image_name).suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if ext in {"jpg", "jpeg"} else "png"
        data_url = f"data:image/{mime};base64,{image_b64}"
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt >= max_retries:
                break
            wait = retry_wait * attempt
            print(f"Request failed (attempt {attempt}/{max_retries}): {err}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise last_err or RuntimeError("Unknown error during request")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate crowd scenario JSONL with an LLM.")
    parser.add_argument("--images", type=Path, required=True, help="Path to an image file or directory of images.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the JSONL dataset.")
    parser.add_argument("--model", type=str, default="azure/gpt-5", help="Model name to request from the proxy.")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API key for the LiteLLM proxy.")
    parser.add_argument("--per-category", type=int, default=100, help="Number of scenarios per category per image.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per LLM request.")
    parser.add_argument("--retry-wait", type=float, default=5.0, help="Base seconds to wait between retries (multiplies by attempt).")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only; do not call the API.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: limit to 5 images and 5 scenarios per category.",
    )
    return parser.parse_args()


def load_processed_images(path: Path) -> Set[str]:
    """Return a set of image names already present in the JSONL output."""
    processed: Set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                img = obj.get("image")
                if img:
                    processed.add(str(img))
            except json.JSONDecodeError:
                continue
    return processed


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("API key is required (pass --api-key or set OPENAI_API_KEY).")

    client = openai.OpenAI(
        api_key=args.api_key,
        base_url="https://aikey-gateway.ivia.ch",
    )

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_images = load_processed_images(out_path)

    per_category = 5 if args.test else args.per_category
    max_images = 5 if args.test else None

    image_list = list(iter_images(args.images))
    if max_images is not None:
        image_list = image_list[:max_images]

    iterator = tqdm(image_list, desc="Images") if tqdm else image_list

    for image_path in iterator:
        image_name = image_path.name
        if image_name in processed_images:
            if tqdm:
                iterator.set_postfix_str(f"skip {image_name}")
            continue
        location_name = slug_to_readable(image_path.stem)
        image_b64 = load_image_b64(image_path)
        user_prompt = build_user_prompt(image_name, location_name, per_category, image_b64)

        if args.dry_run:
            print(f"--- DRY RUN for {image_name} ---")
            print(user_prompt)
            continue

        print(f"Requesting scenarios for {image_name} ...")
        content = request_scenarios(
            client=client,
            image_name=image_name,
            location_name=location_name,
            per_category=per_category,
            image_b64=image_b64,
            model=args.model,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )

        with out_path.open("a", encoding="utf-8") as f:
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "image" not in obj:
                        obj["image"] = image_name
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    # If the model returned non-JSONL lines, write them verbatim to inspect/fix later.
                    f.write(line.rstrip("\n") + "\n")

    if args.dry_run:
        print("Dry run complete. No API calls made.")
    else:
        print(f"Finished writing dataset to {out_path}")


if __name__ == "__main__":
    main()

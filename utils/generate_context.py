"""
Call llm to generate Dataset.
"""

system_prompt = """
You are a helpful scenario generator based on the given map image and the name of the location.
You need to generate the a desctription of the crowd scenario/event that falls into the following categories:

- Ambulatory crowd — people moving through a site (entering/exiting, to/from parking, circulating for facilities). 
- Disability crowd — people restricted in mobility (e.g., cannot fully walk/see/hear/speak). 
- Cohesive crowd — people watching an event (planned or discovered on-site). 
- Expressive crowd — people doing emotional release (singing/cheering/chanting/celebrating/moving together). 
- Participatory crowd — people taking part in the activity (performers, athletes, or audience brought into participation). 
- Aggressive crowd — becomes abusive/threatening/boisterous, may be unlawful and ignores officials’ instructions. 
- Demonstrator crowd — organised for a cause (often with a leader): picket, march, chant, etc. 
- Escaping crowd — trying to escape real or perceived danger (organised evacuation or chaotic pushing/shoving). 
- Dense crowd — movement collapses due to extreme density; people compressed/swept along → suffocation risk. 
- Rushing crowd — crowd’s main aim is to obtain/steal something (best seats, autographs, theft), causing damage/injury risk. 
- Violent crowd — attacking/rioting/terrorising with no regard for law or others’ rights.

You also need to give the following information based on the map and your scenario:

- crowd_size: 0-100 | 100-500 | 500-1000 | 1000+ (need to consider the size of the map, the event could be large but the map is small, so the crowd size should be small accordingly)
- event_center: None | Pixel coordinations on the map (need to specify coordinates) | Distribution if the event center is a range of area
- goal_location: None | Random (need to specify how many goal locations) | Pixel coordinations on the map (need to specify coordinates) | Distribution if the goal location is a range of area

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
    "name": "entering_to_event_center",
    "weight": 0.45,
    "mean_px": [445, 218],
    "sigma_px": [22, 18],
    "support_radius_px": 60
    },
    {
    "name": "north_crosswalk",
    "weight": 0.18,
    "mean_px": [320, 40],
    "sigma_px": [26, 22],
    "support_radius_px": 70
    },
    {
    "name": "south_crosswalk",
    "weight": 0.18,
    "mean_px": [320, 565],
    "sigma_px": [26, 22],
    "support_radius_px": 70
    },
    {
    "name": "west_parking_campus_dispersal",
    "weight": 0.10,
    "mean_px": [120, 330],
    "sigma_px": [30, 26],
    "support_radius_px": 80
    },
    {
    "name": "west_south_path_junction",
    "weight": 0.09,
    "mean_px": [160, 520],
    "sigma_px": [30, 26],
    "support_radius_px": 80
    }
],
"px_to_m": 0.471400264733
}

Here's another example scenario description based on a map image of Times Square:
scenario: "There an explosion occurred at zurich HB train station on Monday morning, causing a panic among the dense crowd of tourists and commuters trying to escape the area."
category: Violent
crowd_size: 100-500
event_center: [200, 300]
goal_location: Random (3)

Give back the result in a jsonl format, each line is a json object with the following keys:

{"image":<image_name>, "scenario":<scenario>, "category":<category>, "crowd_size":<crowd_size>, "event_center":<event_center>, "goal_location":<goal_location>}
"""



import argparse
import json
import os
import base64
import re
from pathlib import Path
from typing import Iterable

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
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate crowd scenario JSONL with an LLM.")
    parser.add_argument("--images", type=Path, required=True, help="Path to an image file or directory of images.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the JSONL dataset.")
    parser.add_argument("--model", type=str, default="azure/gpt-5", help="Model name to request from the proxy.")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API key for the LiteLLM proxy.")
    parser.add_argument("--per-category", type=int, default=100, help="Number of scenarios per category per image.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only; do not call the API.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: limit to 5 images and 5 scenarios per category.",
    )
    return parser.parse_args()


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

    per_category = 5 if args.test else args.per_category
    max_images = 5 if args.test else None

    image_list = list(iter_images(args.images))
    if max_images is not None:
        image_list = image_list[:max_images]

    iterator = tqdm(image_list, desc="Images") if tqdm else image_list

    for image_path in iterator:
        image_name = image_path.name
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

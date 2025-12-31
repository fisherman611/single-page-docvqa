import os
import sys
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import *  # expects: image_file_to_data_url, clean_json_output, enforce_schema, call_with_retry, etc.

with open("augment_data/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
IMAGE_FOLDER = Path(config["image_folder"])
DATA_PATH = Path(config["data_path"])
OUTPUT_PATH = Path(config["caption_output_path"])
NUM_SAMPLES = config.get("num_samples", None)

MAX_RPM = config["max_rpm"]
MAX_TPM = config["max_tpm"]
AVG_TOKENS_PER_CALL = config["avg_tokens_per_call"]

SLEEP_BETWEEN_CALLS = 60.0 / max(MAX_RPM, 1)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("Missing NVIDIA_API_KEY in environment.")

# Prompt for caption-only generation
with open("augment_data/caption_system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# Your original pipeline expects a helper that extracts image paths from a data file
images = extract_image_paths_from_file(DATA_PATH)
total_images = len(images)
limit = min(total_images, NUM_SAMPLES) if isinstance(NUM_SAMPLES, int) else total_images

print(f"Starting caption augmentation on {limit}/{total_images} images...")
results = []

token_bucket = 0
minute_window_start = time.time()
request_count = 0

for i in tqdm(range(limit)):
    # TPM window reset
    now = time.time()
    if now - minute_window_start >= 60:
        token_bucket = 0
        minute_window_start = now

    # TPM throttle
    if token_bucket + AVG_TOKENS_PER_CALL > MAX_TPM:
        wait_time = 60 - (now - minute_window_start)
        if wait_time > 0:
            print(f"[TPM] Waiting {wait_time:.1f}s…")
            time.sleep(wait_time)
        token_bucket = 0
        minute_window_start = time.time()

    image = images[i]
    image_path = IMAGE_FOLDER / image

    if not image_path.exists():
        print(f"[{i+1}] Missing image → skipping: {image_path}")
        continue

    print(f"\n[{i+1}/{limit}] Processing: {image_path.name}")

    image_caption = ""

    try:
        image_data_url = image_file_to_data_url(image_path)
        user_text = "Document image:"

        def _do_call():
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
                temperature=0.2,
                top_p=1.0,
                max_tokens=1024,
                stream=False,
            )

        response = call_with_retry(_do_call, max_retries=100, base_backoff=2.0)

        raw_text = (response.choices[0].message.content or "").strip()
        parsed = clean_json_output(raw_text)

        # If you have a schema enforcer, keep it consistent with your other script.
        # This assumes enforce_schema can handle caption-only outputs too.
        try:
            parsed = enforce_schema(parsed)
        except Exception:
            # If enforce_schema is strict for multi-field outputs, fall back gracefully.
            if not isinstance(parsed, dict):
                parsed = {}
            parsed = {"image_caption": parsed.get("image_caption", "")}

        image_caption = (parsed.get("image_caption") or "").strip()

        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1

        print(f"Generated: {len(image_caption)} chars (caption).")

    except Exception as e:
        print(f"Error on image {image_path.name}: {e}")

    results.append(
        {
            "image": str(image),
            "image_caption": image_caption,
        }
    )

    # Save progress every 10 iterations (by index), plus last
    if ((i + 1) % 10 == 0) or (i == limit - 1):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{limit}).")

    time.sleep(SLEEP_BETWEEN_CALLS)

print(f"\nCompleted {len(results)} images.")
print(f"Results saved to: {OUTPUT_PATH}")
print(f"Total API calls: {request_count}")

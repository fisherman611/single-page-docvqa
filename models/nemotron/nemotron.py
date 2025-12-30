from openai import OpenAI
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from utils.helpers import image_file_to_data_url
from dotenv import load_dotenv
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

with open("models/nemotron/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
MODEL_NAME = config["model_name"]
NUM_SAMPLES = config["num_samples"]
IMAGE_FOLDER = Path(config["image_folder"])
DATA_PATH = Path(config["data_path"])
OUTPUT_PATH = Path(config["output_path"])
MAX_RPM = config["max_rpm"]
MAX_TPM = config["max_tpm"]
AVG_TOKENS_PER_CALL = config["avg_tokens_per_call"]
SLEEP_BETWEEN_CALLS = 60.0 / MAX_RPM  # seconds between requests
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open("models/nemotron/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
start_idx = 0
token_bucket = 0
minute_window_start = time.time()

request_count = 0
total_samples = len(data["data"])

print(f"Starting DocVQA inference on {total_samples} samples...")

for i in range(start_idx, min(total_samples, NUM_SAMPLES)):
    now = time.time()
    if now - minute_window_start >= 60:
        token_bucket = 0
        minute_window_start = now

    if token_bucket + AVG_TOKENS_PER_CALL > MAX_TPM:
        sleep_time = 60 - (now - minute_window_start)
        print(f"Waiting {sleep_time:.1f}s to respect TPM limit...")
        time.sleep(max(sleep_time, 0))
        token_bucket = 0
        minute_window_start = time.time()
        
    sample = data["data"][i]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name
    question = sample["question"]

    if not image_path.exists():
        print(f"Missing image, skipping: {image_path}")
        continue

    print(f"\n[{i+1}] Processing: {image_path.name}")
    
    try:
        image_data_url = image_file_to_data_url(image_path)

        # --- NVIDIA multimodal chat completion ---
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": question},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=256,
            stream=False,   # batch mode like Gemini; set True if you want streaming
        )

        pred_answer = (resp.choices[0].message.content or "").strip()

        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1
        
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        pred_answer = ""

    print("\nQuestion:", question)
    print("Image:", image_path)
    print(f"Model Answer: {pred_answer}\n")
    
    results.append(
        {
            "questionId": sample.get("questionId"),
            "question": question,
            "image": str(image_path),
            "predicted_answer": pred_answer,
            "ground_truth": sample.get("answers"),
        }
    )
    
    if (i + 1) % 10 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{total_samples})")

    time.sleep(SLEEP_BETWEEN_CALLS)

print(f"\nCompleted {len(results)} samples.")
print(f"Results saved to: {OUTPUT_PATH}")
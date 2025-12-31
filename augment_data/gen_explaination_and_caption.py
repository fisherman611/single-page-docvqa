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
OUTPUT_PATH = Path(config["output_path"])
NUM_SAMPLES = config.get("num_samples", None)

MAX_RPM = config["max_rpm"]
MAX_TPM = config["max_tpm"]
AVG_TOKENS_PER_CALL = config["avg_tokens_per_call"]

SLEEP_BETWEEN_CALLS = 60.0 / max(MAX_RPM, 1)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("Missing NVIDIA_API_KEY in environment.")

# Single prompt that produces BOTH image_caption (image-only) + answer_explanation (image+question+GT)
with open("augment_data/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["data"]
total_samples = len(samples)
limit = min(total_samples, NUM_SAMPLES) if isinstance(NUM_SAMPLES, int) else total_samples

print(f"Starting augmentation (caption + explanation) on {limit}/{total_samples} samples...")
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

    sample = samples[i]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name

    if not image_path.exists():
        print(f"[{i+1}] Missing image → skipping: {image_path}")
        continue

    question = sample.get("question", "")
    answers = sample.get("answers", [""])
    gt_answer = answers[0] if answers else ""

    print(f"\n[{i+1}/{limit}] Processing: {image_path.name}")
    print(f"Question: {question[:120]}")
    print(f"GT Answer: {gt_answer}")

    image_caption = ""
    answer_explanation = ""

    try:
        image_data_url = image_file_to_data_url(image_path)

        # NOTE: We still pass question + GT for the explanation,
        # but the prompt must force image_caption to be image-only.
        user_text = (
            f"Question: {question}\n"
            f"Ground truth answer: {gt_answer}\n"
            f"Document image:"
        )

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

        response = call_with_retry(_do_call, max_retries=5, base_backoff=2.0)

        raw_text = (response.choices[0].message.content or "").strip()
        parsed = clean_json_output(raw_text)
        parsed = enforce_schema(parsed)

        image_caption = parsed.get("image_caption", "")
        answer_explanation = parsed.get("answer_explanation", "")

        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1

        print(
            f"Generated: {len(image_caption)} chars (caption), "
            f"{len(answer_explanation)} chars (expl)"
        )

    except Exception as e:
        print(f"Error on sample {i}: {e}")

    results.append(
        {
            "questionId": sample.get("questionId"),
            "question": question,
            "question_types": sample.get("question_types"),
            "image": str(image_path),
            "docId": sample.get("docId"),
            "ucsf_document_id": sample.get("ucsf_document_id"),
            "ucsf_document_page_no": sample.get("ucsf_document_page_no"),
            "answers": answers,
            "image_caption": image_caption,
            "answer_explanation": answer_explanation,
        }
    )

    # Save progress every 10 iterations (by index), plus last
    if ((i + 1) % 10 == 0) or (i == limit - 1):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{limit}).")

    time.sleep(SLEEP_BETWEEN_CALLS)

print(f"\nCompleted {len(results)} samples.")
print(f"Results saved to: {OUTPUT_PATH}")
print(f"Total API calls: {request_count}")

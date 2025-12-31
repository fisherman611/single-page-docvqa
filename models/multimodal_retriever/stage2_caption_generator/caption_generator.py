import os
import sys
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from utils.helpers import *

with open("models/multimodal_retriever/stage2_caption_generator/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
MODEL_NAME = config["model_name"]
IMAGE_FOLDER = Path(config["image_folder"])
MAX_RPM = config["max_rpm"]
MAX_TPM = config["max_tpm"]
AVG_TOKENS_PER_CALL = config["avg_tokens_per_call"]

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("Missing NVIDIA_API_KEY in environment.")

with open("models/multimodal_retriever/stage2_caption_generator/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()
    
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

def caption_generator(image_path: Path) -> dict:
    if not image_path.exists():
        return {
            "image_caption": "",
            "success": False,
            "error": f"Image not found: {image_path}"
        }
    
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

        response = call_with_retry(_do_call, max_retries=5, base_backoff=2.0)

        raw_text = (response.choices[0].message.content or "").strip()
        parsed = clean_json_output(raw_text)

        image_caption = parsed.get("image_caption", "")
        print(f"Generated: {len(image_caption)} chars (caption).")

        return {
            "image_caption": image_caption,
            "success": True,
            "error": ""
        }

    except Exception as e:
        return {
            "image_caption": "",
            "success": False,
            "error": f"Error generating caption: {e}"
        }

# if __name__ == "__main__":
#     test_image_path = IMAGE_FOLDER / "nkbl0226_1.png"

#     result = caption_generator(test_image_path)
#     if result["success"]:
#         print("Image Caption:", result["image_caption"])
#     else:
#         print("Error:", result["error"])

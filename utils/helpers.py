import os
import sys
from pathlib import Path
import re
import json
import base64
import mimetypes
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def image_file_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def clean_json_output(text: str) -> dict:
    """Remove markdown fences (if any) and parse JSON object."""
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    cleaned = re.sub(r"^json\s*\n", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        # Try to recover the first JSON object if the model added extra text
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}
    
    
def enforce_schema(parsed: dict) -> dict:
    """
    Ensure output matches the required schema:
    {
      "image_description": "...",
      "answer_explanation": "..."
    }
    """
    if not isinstance(parsed, dict):
        return {"image_description": "", "answer_explanation": ""}

    return {
        "image_description": str(parsed.get("image_description", "") or ""),
        "answer_explanation": str(parsed.get("answer_explanation", "") or ""),
    }
    
def call_with_retry(payload_fn, max_retries: int = 5, base_backoff: float = 2.0):
    """
    Retry wrapper with exponential backoff.
    payload_fn: function that performs the API call and returns response.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return payload_fn()
        except Exception as e:
            last_err = e
            wait = base_backoff * (2 ** (attempt - 1))
            # Cap wait so it doesn't explode
            wait = min(wait, 30.0)
            print(f"[Retry {attempt}/{max_retries}] API error: {e} | Sleeping {wait:.1f}s")
            time.sleep(wait)
    raise last_err

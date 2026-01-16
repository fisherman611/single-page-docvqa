import os
import sys
from pathlib import Path
import re
import json
import base64
import mimetypes
import time
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List, Any,Tuple, Optional
import string
from collections import Counter

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
      "image_caption": "...",
      "answer_explanation": "..."
    }
    """
    if not isinstance(parsed, dict):
        return {"image_caption": "", "answer_explanation": ""}

    return {
        "image_caption": str(parsed.get("image_caption", "") or ""),
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

def load_best_model(model, checkpoint_path="checkpoints/best_model.pt", device="cuda"):
    """
    Load the best model checkpoint.
    
    Args:
        model: The model architecture to load weights into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Model with loaded weights
        checkpoint: Dictionary containing metrics and other info
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Metrics: Jaccard={checkpoint['best_jaccard']:.4f}, "
          f"Macro Label Recall={checkpoint['macro_label_recall']:.4f}")
    
    return model, checkpoint

def subset_recall(pred_vec, gt_vec):
    """
    pred_vec, gt_vec: arrays of shape [num_classes] containing 0/1
    returns 1 or 0
    """
    gt_indices = np.where(gt_vec == 1)[0]
    for idx in gt_indices:
        if pred_vec[idx] != 1:
            return 0.0
    return 1.0

def subset_recall_macro(y_true, y_pred):
    scores = []
    for gt, pred in zip(y_true, y_pred):
        scores.append(subset_recall(pred, gt))
    return np.mean(scores)

def label_recall_vector(pred_vec, gt_vec):
    gt_idx = np.where(gt_vec == 1)[0]
    if len(gt_idx) == 0:
        return 1.0
    correct = sum(pred_vec[i] == 1 for i in gt_idx)
    return correct / len(gt_idx)

def label_recall_macro(Y_true, Y_pred):
    return np.mean([label_recall_vector(p, g) for p, g in zip(Y_pred, Y_true)])

def extract_image_paths_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    images = []
    data_list = data["data"]
    for i in range(len(data_list)):
        sample = data_list[i]
        image = Path(sample["image"]).name
        if image not in images:
            images.append(image)
    
    return images

def load_image(path: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    max_side = 1024  # try 768 if you still get OOM
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    return image

def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def build_example_template(
    ex: Dict[str, Any],
    idx: int
) -> str:
    question = ex["question"]
    img_caption = ex.get("image_caption", "")
    answers = ex.get("answers", [])
    answer_explanation = ex.get("answer_explanation", "")
    
    block = []
    block.append(f"### Example {idx}\n")
    block.append(f"<image_{idx}>\n")
    if img_caption:
        block.append(f"[IMAGE_DESCRIPTION]: {img_caption}\n")
    
    block.append(f"[QUESTION]: {question}\n")
    if answer_explanation:
        block.append(f"[CHAIN OF THOUGHT]: {answer_explanation}\n")
    
    block.append(f"[FINAL_ANSWER]:")
    for answer in answers:
        block.append(answer)
    block.append("")
    
    return "\n".join(block)
    
def build_query_template(
    query_ex: Dict[str, Any],
    idx: int
) -> str:
    question = query_ex["question"]
    img_caption = query_ex.get("image_caption", "")
    
    block = []
    block.append("### Query Example")
    block.append(f"<image_{idx}>\n")
    if img_caption:
        block.append(f"[IMAGE_CAPTION]\n{img_caption}\n")

    block.append("[QUESTION]")
    block.append(question + "\n")

    # Leave CoT blank for the model to fill
    block.append("[CHAIN_OF_THOUGHT]")
    block.append("")  # model will generate here

    block.append("[FINAL_ANSWER]")
    block.append("")  # model will generate here

    return "\n".join(block)

def extract_final_answer(text: str) -> str:
    m = re.search(r"\[FINAL ANSWER\](.*)", text, flags=re.DOTALL|re.IGNORECASE)
    if not m:
        return text.strip()
    
    tail = m.group(1).strip()
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    return lines[0] if lines else tail

def normalize_answer(ans: str) -> str:
    ans = ans.strip().lower()
    ans = ans.translate(str.maketrans("", "", string.punctuation))
    ans = " ".join(ans.split())
    return ans

def majority_vote(answers: List[str]) -> Tuple[str, Dict[str, int]]:
    counter = Counter(answers)
    return max(counter.keys(), key=lambda x: counter.get(x))
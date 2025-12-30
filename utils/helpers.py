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
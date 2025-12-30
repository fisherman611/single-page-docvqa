import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
login(token=os.getenv("HF_READ_TOKEN"))

with open("models/qwen2_vl/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
MODEL_NAME = "Qwen/Qwen2-VL-2B"
# MODEL_NAME = "Qwen/Qwen2-VL-7B"

DATA_PATH = Path(config["data_path"])
IMAGE_FOLDER = Path(config["image_folder"])
OUTPUT_PATH = Path("results/qwen2_vl_7b_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("models/qwen2_vl/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()
    
print(f"Running on device: {DEVICE}")

print(f"Loading {MODEL_NAME} and processor...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

if hasattr(model, "generation_config"):
    model.generation_config.use_cache = False
    
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results = []

total_samples = len(data["data"])
print(f"Starting DocVQA inference on {total_samples} samples...")

for i in range(min(total_samples, 10)):
    sample = data["data"][i]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name
    question = sample["question"]
    
    if not image_path.exists():
        print(f"Missing image, skipping: {image_path}")
        continue

    print(f"\n[{i+1}] Processing: {image_path.name}")

    image = Image.open(image_path).convert("RGB")
    max_side = 1024  # try 768 if you still get OOM
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    
    prompt = (
        f"System prompt: {SYSTEM_PROMPT}"
        f"<|vision_start|><|image_pad|><|vision_end|>"
        f"Question: {question}"
    )
    
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.3)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            pred_answer = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        
        pred_answer[0] = pred_answer[0].replace("\nAnswer:", "").strip()
        print("\nQuestion:", question)
        print("Image:", image_path)
        print(f"Model Answer: {pred_answer[0]}\n")
        
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        pred_answer = [""]
        
    results.append(
        {
            "questionId": sample.get("questionId"),
            "question": question,
            "image": str(image_path),
            "predicted_answer": pred_answer[0],
            "ground_truth": sample.get("answers"),
        }
    )
    
    if (i + 1) % 10 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{total_samples})")
        
print(f"\nCompleted {len(results)} samples.")
print(f"Results saved to: {OUTPUT_PATH}")

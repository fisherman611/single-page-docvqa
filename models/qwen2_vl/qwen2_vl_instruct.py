import torch
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
from unsloth import FastVisionModel

print("Unsloth installed correctly!!")

with open("models/qwen2_vl/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
# MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"

DATA_PATH = Path(config["data_path"])
IMAGE_FOLDER = Path(config["image_folder"])
OUTPUT_PATH = Path(config["output_path"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("models/qwen2_vl/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

print(f"Running on device: {DEVICE}")
print(f"Loading {MODEL_NAME} and processor...")

model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    device=DEVICE,
    load_in_4bit=True,
)

FastVisionModel.for_inference(model)

print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results = []

total_samples = len(data["data"])
print(f"Starting DocVQA inference on {total_samples} samples...")

for i in range(min(total_samples, 1000)):
    sample = data["data"][i]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name
    question = sample["question"]

    if not image_path.exists():
        print(f"Image not found: {image_path}, skipping sample {i+1}/{total_samples}")
        continue

    print(f"\n[{i+1}] Processing: {image_path.name}")
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temparature=0.3,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        pred_answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        print("\nQuestion:", question)
        print("Image:", image_path)
        print(f"Model Answer: {pred_answer}\n")

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
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{total_samples})")

print(f"\nCompleted {len(results)} samples.")
print(f"Results saved to: {OUTPUT_PATH}")
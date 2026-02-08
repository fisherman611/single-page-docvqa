import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor
import numpy as np
from tqdm.auto import tqdm
import json

with open("models/multimodal_retriever/stage1_multimodal_classifier/clip/config.json", "r") as f:
    config = json.load(f)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = config['model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

class CLIPDocVQAMultimodalDataset(Dataset):
    def __init__(self, data_source, img_root, processor, label2id, max_len=MAX_LEN) -> None:
        if isinstance(data_source, str) or isinstance(data_source, Path):
            with open(data_source, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = data_source
        self.questions = [self.data['data'][i]['question'].strip() for i in range(len(self.data['data']))]
        # self.questions = [self.data['data'][i]['question'].strip() for i in range(1)]
        self.image_paths = [Path(self.data['data'][i]['image']).name for i in range(len(self.data['data']))]
        # self.image_paths = [Path(self.data['data'][i]['image']).name for i in range(1)]
        self.labels = []
        for x in tqdm(self.data['data']):
        # for x in tqdm(self.data['data'][:1]):
            vec = np.zeros(len(label2id), dtype=np.float32)
            for lbl in x['question_types']:
                vec[label2id[lbl]] = 1.0
            self.labels.append(vec)
            
        self.processor = processor
        self.img_root = Path(img_root)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        img_path = self.img_root / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        enc = self.processor(
            text=q,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True, 
        )
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": enc["pixel_values"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        
# if __name__ == "__main__":
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

#     label_list = ["handwritten", "form", "layout", "table/list", "others", "free_text", "Image/Photo", "figure/diagram", "Yes/No"]
#     label2id = {lbl: i for i, lbl in enumerate(label_list)}

#     train_ds = CLIPDocVQAMultimodalDataset("data/spdocvqa_qas/val_v1.0_withQT.json", "data/spdocvqa_images", processor=processor, label2id=label2id)
#     print(len(train_ds))
#     print(train_ds[0]) 
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model import *
from train_eval import *
from dataset import *
from utils.helper import load_config
from transformers import CLIPProcessor
from torch.utils.data import DataLoader

with open("models/multimodal_retriever/stage1_multimodal_classifier/clip/config.json", "r") as f:
    config = json.load(f)

EPOCHS = config["epoch"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["lr"]
MODEL_NAME = config["model"]
FREEZE_CLIP = config.get("freeze_clip", True)
USE_ATTENTION_FUSION = config.get("use_attention_fusion", True)
NUM_ATTENTION_HEADS = config.get("num_attention_heads", 8)
DROPOUT = config.get("dropout", 0.2)

print("="*60)
print("CLIP Multimodal Classifier Training")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Freeze CLIP: {FREEZE_CLIP}")
print(f"Attention Fusion: {USE_ATTENTION_FUSION}")
print(f"Dropout: {DROPOUT}")
print("="*60 + "\n")

clip_proc = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)

# Label mapping
label_list = config['classes']
label2id = {lbl: i for i, lbl in enumerate(label_list)}
id2label = {i: lbl for lbl, i in label2id.items()}

print(f"Classes ({len(label_list)}):")
for i, label in enumerate(label_list):
    print(f"  {i}: {label}")
print()

# Load datasets
print("Loading datasets...")
train_ds = CLIPDocVQAMultimodalDataset(
    "data/spdocvqa_qas/train_v1.0_withQT.json", 
    "data/spdocvqa_images", 
    clip_proc, 
    label2id
)
val_ds = CLIPDocVQAMultimodalDataset(
    "data/spdocvqa_qas/val_v1.0_withQT.json", 
    "data/spdocvqa_images", 
    clip_proc, 
    label2id
)

print(f"Train dataset size: {len(train_ds)}")
print(f"Val dataset size: {len(val_ds)}\n")

# Create dataloaders
train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,  # Adjust based on your system
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds, 
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# Initialize model
print("Initializing model...")
model = CLIPMultimodalClassifier(
    model_name=MODEL_NAME,
    num_labels=NUM_CLASSES,
    freeze_clip=FREEZE_CLIP,
    dropout=DROPOUT,
    use_attention_fusion=USE_ATTENTION_FUSION,
    num_attention_heads=NUM_ATTENTION_HEADS
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

# Train model
best_model_path = train_model(
    model, 
    train_loader, 
    val_loader, 
    label2id,
    epochs=EPOCHS,
    lr=LEARNING_RATE
)

print(f"\nTraining complete! Best model saved at: {best_model_path}")
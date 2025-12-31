import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, 
    precision_score, recall_score, hamming_loss, jaccard_score
)
from tqdm.auto import tqdm
import numpy as np
import json
from datetime import datetime
from utils.helpers import label_recall_macro

with open("models/multimodal_retriever/stage1_multimodal_classifier/clip/config.json", "r") as f:
    config = json.load(f)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = config['model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

def train_model(
    model, 
    train_loader, 
    val_loader, 
    label2id,
    epochs=None,
    lr=None,
    save_dir=None,
    warmup_steps=None,
    weight_decay=None,
    max_grad_norm=None,
    early_stopping_patience=None,
    use_mixed_precision=None
):
    """
    Enhanced training function with:
    - Learning rate scheduler with warmup
    - Gradient clipping
    - Early stopping
    - Mixed precision training
    - Better checkpoint management
    """
    # Load config values with defaults
    epochs = epochs or config.get('epoch', 10)
    lr = lr or config.get('lr', 2e-5)
    save_dir = save_dir or config.get('save_dir', 'checkpoints/clip_classifier')
    warmup_steps = warmup_steps or config.get('warmup_steps', 100)
    weight_decay = weight_decay or config.get('weight_decay', 0.01)
    max_grad_norm = max_grad_norm or config.get('max_grad_norm', 1.0)
    early_stopping_patience = early_stopping_patience or config.get('early_stopping_patience', 3)
    use_mixed_precision = use_mixed_precision if use_mixed_precision is not None else config.get('use_mixed_precision', True)
    history_path = os.path.join(save_dir, "training_history.json")
    
    # Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model.to(DEVICE)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_mixed_precision and DEVICE == 'cuda' else None
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")
    last_model_path = os.path.join(save_dir, "last_model.pt")
    
    # Tracking
    best_jaccard = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    print(f"{'='*60}")
    print(f"Training Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {train_loader.batch_size}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Max Grad Norm: {max_grad_norm}")
    print(f"  Early Stopping Patience: {early_stopping_patience}")
    print(f"  Mixed Precision: {use_mixed_precision and DEVICE == 'cuda'}")
    print(f"  Save Directory: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # ==================== Training ====================
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            imgs = batch["pixel_values"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # Mixed precision forward pass
            if scaler is not None:
                with autocast():
                    logits = model(ids, mask, imgs)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids, mask, imgs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        
        # ==================== Validation ====================
        print(f"\n[Epoch {epoch+1}] Evaluating...")
        metrics = evaluate(model, val_loader, label2id, detailed=True)
        
        # Log results
        print(f"\n[Epoch {epoch+1}] Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Macro Label Recall: {metrics['macro_label_recall']:.4f}")
        print(f"  Val Jaccard: {metrics['jaccard']:.4f}")
        print(f"  Val Hamming Loss: {metrics['hamming']:.4f}")
        print(f"  Val Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Val Macro F1: {metrics['macro_f1']:.4f}")
        
        # Save training history
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_metrics': metrics,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        training_history.append(epoch_history)
        
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
            
        # Save last model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': config
        }, last_model_path)
        
        # Save best model and early stopping
        current_jaccard = metrics['jaccard']
        if current_jaccard > best_jaccard:
            best_jaccard = current_jaccard
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_jaccard': best_jaccard,
                'metrics': metrics,
                'config': config
            }, best_model_path)
            print(f"Best model saved! (Jaccard: {best_jaccard:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        print(f"{'-'*60}\n")
    
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Jaccard Score: {best_jaccard:.4f}")
    print(f"  Best Model: {best_model_path}")
    print(f"  Training History: {history_path}")
    print(f"{'='*60}\n")
    
    return best_model_path


@torch.no_grad()
def evaluate(model, val_loader, label2id, threshold=None, detailed=False):
    """
    Enhanced evaluation with more metrics and optional detailed output.
    """
    threshold = threshold or config.get('threshold', 0.5)
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    eval_pbar = tqdm(val_loader, desc="Evaluating")
    for batch in eval_pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        imgs = batch["pixel_values"].to(DEVICE)
        lbls = batch["label"].cpu().numpy()

        logits = model(ids, mask, imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(int)

        all_preds.append(preds)
        all_labels.append(lbls)
        all_probs.append(probs)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_probs = np.vstack(all_probs)

    # Calculate metrics
    macro_label_recall = label_recall_macro(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'macro_label_recall': float(macro_label_recall),
        'jaccard': float(jaccard),
        'hamming': float(hamming),
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1)
    }
    
    if detailed:
        # Per-class metrics
        id2label = {v: k for k, v in label2id.items()}
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {
            id2label[i]: {
                'f1': float(per_class_f1[i]),
                'recall': float(per_class_recall[i]),
                'precision': float(per_class_precision[i])
            }
            for i in range(len(id2label))
        }
    
    if not detailed:
        print(f"Macro Label Recall={macro_label_recall:.4f} | Jaccard={jaccard:.4f}")
    
    return metrics if detailed else (macro_label_recall, jaccard)
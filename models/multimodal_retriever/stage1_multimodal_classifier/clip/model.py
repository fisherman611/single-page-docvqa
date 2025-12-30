import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm
from PIL import Image

with open("models/multimodal_retriever/stage1_multimodal_classifier/clip/config.json", "r") as f:
    config = json.load(f)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = config['model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

class MultimodalAttentionFusion(nn.Module):
    """
    Cross-attention based fusion mechanism for multimodal embeddings.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_embeds, image_embeds):
        # Add sequence dimension for attention
        text = text_embeds.unsqueeze(1)  # [B, 1, D]
        image = image_embeds.unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention: text attends to image
        attn_out, _ = self.multihead_attn(text, image, image)
        text = self.norm1(text + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(text)
        text = self.norm2(text + ffn_out)
        
        return text.squeeze(1)  # [B, D]
    
class CLIPMultimodalClassifier(nn.Module):
    """
    Enhanced CLIP-based multimodal classifier with attention fusion and improved architecture.
    """
    def __init__(
        self, 
        model_name=MODEL, 
        num_labels=NUM_CLASSES, 
        freeze_clip=True,
        dropout=0.2,
        use_attention_fusion=True,
        num_attention_heads=8
    ) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        emb_dim = self.clip.config.projection_dim
        self.use_attention_fusion = use_attention_fusion

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
        
        # Fusion mechanism
        if use_attention_fusion:
            self.fusion = MultimodalAttentionFusion(
                emb_dim, num_heads=num_attention_heads, dropout=dropout
            )
            classifier_input_dim = emb_dim
        else:
            self.fusion = None
            classifier_input_dim = emb_dim * 2
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # Get normalized embeddings
        image_embeds = outputs.image_embeds  # [B, D]
        text_embeds = outputs.text_embeds    # [B, D]
        
        # Fusion
        if self.use_attention_fusion:
            # Use attention-based fusion
            fused = self.fusion(text_embeds, image_embeds)
        else:
            # Simple concatenation
            fused = torch.cat([image_embeds, text_embeds], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        return logits
    
    def get_embeddings(self, input_ids, attention_mask, pixel_values):
        """Extract multimodal embeddings without classification."""
        with torch.no_grad():
            outputs = self.clip(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            if self.use_attention_fusion:
                fused = self.fusion(text_embeds, image_embeds)
            else:
                fused = torch.cat([image_embeds, text_embeds], dim=1)
            
            return fused
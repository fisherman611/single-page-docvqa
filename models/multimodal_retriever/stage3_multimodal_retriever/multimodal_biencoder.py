import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

import json
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import faiss
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from utils.helpers import *
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HF_READ_TOKEN"))

with open("models/multimodal_retriever/stage3_multimodal_retriever/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TEXT_MODEL = config["text_model_embedding"]
IMAGE_MODEL = config["image_model_embedding"]
PROJ_DIM = config["proj_dim"]

class MultimodalBiEncoder:
    """
    Multimodal Bi-encoder for (image, question, image caption)
    """
    def __init__(
        self,
        text_model_name: str=TEXT_MODEL,
        image_model_name: str=IMAGE_MODEL,
        proj_dim: int=PROJ_DIM,
        device: str=None
     ) -> None:
        super().__init__()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Image encoder (CLIP)
        self.image_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name, use_fast=True)
        image_dim = self.image_model.config.projection_dim
        
        # Text encoder
        self.text_encoder_type = self._infer_text_encoder_type(text_model_name)
        if self.text_encoder_type == "sentence_transformers":
            self.text_model = SentenceTransformer(text_model_name).to(self.device)
            text_dim = self.text_model.get_sentence_embedding_dimension()
            self.text_processor = None
        
        elif self.text_encoder_type == "clip":
            self.text_model = CLIPModel.from_pretrained(text_model_name).to(self.device)
            self.text_processor = CLIPProcessor.from_pretrained(text_model_name, use_fast=True)
            text_dim = self.text_model.config.projection_dim
        
        else:
            raise ValueError(
                f"Unsupported text encoder for model '{text_model_name}'. "
                f"Use a SentenceTransformer (e.g. 'sentence-transformers/...') "
                f"or a CLIP checkpoint (e.g. 'openai/clip-vit-base-patch32')."
            )
        
        self.proj = nn.Linear(image_dim + text_dim, proj_dim).to(self.device)
    
    
    def _infer_text_encoder_type(self, text_model_name: str) -> str:
        name = (text_model_name or "").lower()
        if name.startswith("sentence-transformers/"):
            return "sentence_transformers"
        if "clip" in name:
            return "clip"
        
        return "unknown"
    
    
    def build_text_input(self, example: Dict) -> str:
        question = example.get("question", "")
        image_caption = example.get("image_caption", "")
        return f"Question: {question} [SEP] Image caption: {image_caption}"
    
    
    def build_image_input(self, example: Dict) -> Image.Image:
        return load_image(example.get("image", ""))
    
    
    def encode_text(self, texts: List[str], no_grad: bool=True) -> torch.Tensor:
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            if self.text_encoder_type == "sentence_transformers":
                emb = self.text_model.encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                )
            
            else:
                inputs = self.text_processor(
                    text=texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                emb = self.text_model.get_text_features(**inputs)
            
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb
        
    
    def encode_image(self, images: List[Image.Image], no_grad: bool=True) -> torch.Tensor:
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
            emb = self.image_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb
    
    
    def encode_pair(
        self,
        images: List[Image.Image],
        texts: List[str],
        no_grad: bool=True
    ) -> torch.Tensor:
        """
        Returns fused embedding: [B, proj_dim], normalized
        """
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            img_emb = self.encode_image(images, no_grad=no_grad)
            txt_emb = self.encode_text(texts, no_grad=no_grad)
            concat = torch.cat([img_emb, txt_emb], dim=-1)
            proj = self.proj(concat)
            proj = proj / proj.norm(dim=-1, keepdim=True)
            return proj


# def run_one_test(text_model_name: str, image_model_name: str, proj_dim: int):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("\n==============================")
#     print("TEXT:", text_model_name)
#     print("IMG :", image_model_name)
#     print("PROJ:", proj_dim)
#     print("DEVICE:", device)

#     model = MultimodalBiEncoder(
#         text_model_name=text_model_name,
#         image_model_name=image_model_name,
#         proj_dim=proj_dim,
#         device=device,
#     )

#     # Two simple images + texts
#     img1 = load_image_from_url("https://picsum.photos/id/237/512/512")  # dog photo (random service)
#     img2 = load_image_from_url("https://picsum.photos/id/1003/512/512") # landscape-ish

#     examples = [
#         {"question": "What animal is in the image?", "image_description": "A photo with an animal."},
#         {"question": "What is the scene?", "image_description": "Outdoor landscape view."},
#     ]
#     texts = [model.build_text_input(ex) for ex in examples]
#     images = [img1, img2]

#     with torch.no_grad():
#         txt_emb = model.encode_text(texts, no_grad=True)
#         img_emb = model.encode_image(images, no_grad=True)
#         pair_emb = model.encode_pair(images, texts, no_grad=True)

#     print("text_emb shape:", tuple(txt_emb.shape))
#     print("image_emb shape:", tuple(img_emb.shape))
#     print("pair_emb shape:", tuple(pair_emb.shape))

#     # cosine similarity between fused pair embeddings (2x2)
#     sim = pair_emb @ pair_emb.t()
#     print("pair_emb self-similarity matrix:\n", sim.cpu())


# if __name__ == "__main__":
#     # -------- Test A: SentenceTransformer text branch --------
#     run_one_test(
#         text_model_name="sentence-transformers/all-mpnet-base-v2",
#         image_model_name="openai/clip-vit-base-patch32",
#         proj_dim=512,
#     )

#     # -------- Test B: CLIP text branch --------
#     run_one_test(
#         text_model_name="openai/clip-vit-base-patch32",
#         image_model_name="openai/clip-vit-base-patch32",
#         proj_dim=512,
#     )

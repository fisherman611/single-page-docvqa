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
import random
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from utils.helpers import *
from dotenv import load_dotenv
from models.multimodal_retriever.stage3_multimodal_retriever import faiss_utils


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
    
    
    def train(self):
        """Set model to training mode."""
        self.image_model.train()
        if self.text_encoder_type == "clip":
            self.text_model.train()
        self.proj.train()
    
    
    def eval(self):
        """Set model to evaluation mode."""
        self.image_model.eval()
        if self.text_encoder_type == "clip":
            self.text_model.eval()
        self.proj.eval()
    
    
    def _infer_text_encoder_type(self, text_model_name: str) -> str:
        name = (text_model_name or "").lower()
        if name.startswith("sentence-transformers/"):
            return "sentence_transformers"
        if "clip" in name:
            return "clip"
        
        return "unknown"
    
    
    def build_text_input(self, example: Dict, drop_q: bool=False, drop_cap: bool=False) -> str:
        question = "" if drop_q else example.get("question", "")
        image_caption ="" if drop_cap else  example.get("image_caption", "")
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


    def encode_bundle(
        self,
        batch_examples: List[Dict],
        use_image_mask: List[bool],
        drop_q_mask: List[bool],
        drop_cap_mask: List[bool],
        no_grad: bool=True
    ) -> torch.Tensor:
        images = [self.build_image_input(example) for example in batch_examples]
        texts = [
            self.build_text_input(
                batch_examples[i],
                drop_q=drop_q_mask[i],
                drop_cap=drop_q_mask[i]
            )
            for i in range(len(batch_examples))
        ]
        
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            img_emb = self.encode_image(images, no_grad=no_grad)   # [B, image_dim]
            txt_emb = self.encode_text(texts, no_grad=no_grad)     # [B, text_dim]
            mask = torch.tensor(use_image_mask, device=self.device, dtype=torch.float32).unsqueeze(1)
            img_emb = img_emb * mask
            concat = torch.cat([img_emb, txt_emb], dim=-1)
            proj = self.proj(concat)
            proj = proj / proj.norm(dim=-1, keepdim=True)
            return proj
        
        
    @staticmethod
    def info_nce_loss(
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Standard InfoNCE loss for symmetric contrastive learning.
        query_emb: [B, D]
        doc_emb:   [B, D]
        """
        logits = query_emb @ doc_emb.t() / temperature  # [B, B]
        labels = torch.arange(query_emb.size(0), device=query_emb.device)
        loss_q = torch.nn.functional.cross_entropy(logits, labels)
        loss_d = torch.nn.functional.cross_entropy(logits.t(), labels)
        return (loss_q + loss_d) / 2.0
    
    
    @staticmethod
    def sample_view() -> Tuple[bool, bool, bool]:
        """
        Returns (use_image, drop_q, drop_cap)
        """
        r = random.random()
        if r < 0.25:
            return True, False, True     # image + question
        elif r < 0.50:
            return True, True, False     # image + caption
        elif r < 0.65:
            return False, False, False   # text-only (q+cap)
        else:
            return True, False, False    # full (image+q+cap)
    
    
    def two_view_contrastive_training_step(
        self,
        batch_examples: List[Dict],
        optimizer: torch.optim.Optimizer,
        temperature: float=0.07,
    ) -> float:
        """
        Self-retrieval training (query/doc from SAME example pool):
          - build view A and view B for each example
          - positive: A(i) ~ B(i)
          - negatives: A(i) ~ B(j != i) in-batch
        """
        self.train()

        use_a, dq_a, dc_a = [], [], []
        use_b, dq_b, dc_b = [], [], []

        for _ in batch_examples:
            ua, dqa, dca = self.sample_view()
            ub, dqb, dcb = self.sample_view()
            use_a.append(ua); dq_a.append(dqa); dc_a.append(dca)
            use_b.append(ub); dq_b.append(dqb); dc_b.append(dcb)

        z_a = self.encode_bundle(batch_examples, use_a, dq_a, dc_a, no_grad=False)
        z_b = self.encode_bundle(batch_examples, use_b, dq_b, dc_b, no_grad=False)

        loss = self.info_nce_loss(z_a, z_b, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        return float(loss.item())
    
    
    def build_faiss_index(
        self, 
        examples: List[Dict], 
        batch_size: int = 64,
        use_gpu: bool = False,
        index_type: str = "flat",
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8
    ) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build a FAISS index from document embeddings.
        
        Args:
            examples: List of examples with image, question, and image_caption fields
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU for FAISS index
            index_type: Type of FAISS index to build
                - "flat": exact search (IndexFlatIP)
                - "ivf": inverted file index for faster approximate search
                - "pq": product quantization for memory efficiency
                - "ivfpq": combination of IVF and PQ
            nlist: Number of clusters for IVF indices
            m: Number of subquantizers for PQ
            nbits: Number of bits per subquantizer for PQ
        
        Returns:
            Tuple of (FAISS index, embedding matrix)
        """
        self.eval()
        vecs = []

        # Doc embedding uses FULL view (image + question + caption)
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            use = [True] * len(batch)      # use image
            dq = [False] * len(batch)      # don't drop question
            dc = [False] * len(batch)      # don't drop caption

            with torch.no_grad():
                z = self.encode_bundle(batch, use, dq, dc, no_grad=True)
            vecs.append(z.cpu().numpy().astype("float32"))

        mat = np.concatenate(vecs, axis=0)
        
        # Use the build_index function from faiss_utils
        index = faiss_utils.build_index(
            embeddings=mat,
            use_gpu=use_gpu,
            index_type=index_type,
            nlist=nlist,
            m=m,
            nbits=nbits
        )
        
        return index, mat
    
    
    def mine_hard_negatives(
        self,
        index: faiss.Index,
        examples: List[Dict],
        topm: int = 50,
        query_view: str = "image+question",
        batch_size: int = 64,
    ) -> Dict[str, List[str]]:
        """
        Mine hard negatives using FAISS index.
        
        No true relevance labels needed:
        - hard negatives = nearest neighbors excluding itself.
        You can add extra filters (e.g., different answer) later if you want.
        
        Args:
            index: Pre-built FAISS index from build_faiss_index
            examples: List of examples (same as used to build index)
            topm: Number of top neighbors to retrieve
            query_view: Query view type:
                - "image+question": use image + question (drop caption)
                - "text-only": use question + caption only (no image)
                - "full": use image + question + caption
            batch_size: Batch size for encoding
        
        Returns:
            Dictionary mapping example ID to list of hard negative IDs
        """
        self.eval()

        ids = [ex["id"] for ex in examples]
        id_to_pos = {ex_id: i for i, ex_id in enumerate(ids)}

        hard_negs = {ex_id: [] for ex_id in ids}

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]

            # Configure view based on query_view parameter
            if query_view == "image+question":
                use = [True] * len(batch)      # use image
                dq = [False] * len(batch)      # don't drop question
                dc = [True] * len(batch)       # drop caption
            elif query_view == "text-only":
                use = [False] * len(batch)     # don't use image
                dq = [False] * len(batch)      # don't drop question
                dc = [False] * len(batch)      # don't drop caption
            else:  # full
                use = [True] * len(batch)      # use image
                dq = [False] * len(batch)      # don't drop question
                dc = [False] * len(batch)      # don't drop caption

            with torch.no_grad():
                q = self.encode_bundle(batch, use, dq, dc, no_grad=True).cpu().numpy().astype("float32")

            # Search for nearest neighbors
            scores, neigh_pos = index.search(q, topm)

            # Extract hard negatives (exclude self)
            for b_idx, ex in enumerate(batch):
                ex_id = ex["id"]
                self_pos = id_to_pos[ex_id]

                # Filter out the example itself
                candidates = [p for p in neigh_pos[b_idx].tolist() if p != self_pos]
                hard_negs[ex_id] = [examples[p]["id"] for p in candidates]

        return hard_negs
    


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

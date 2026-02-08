import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))

import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import transformers
import platform
from datetime import datetime
import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForImageTextRetrieval
from utils.helpers import load_image
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login

login(token=os.getenv("HF_READ_TOKEN"))

with open(
    "models/multimodal_retriever/stage3_multimodal_retriever/config.json",
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)

RERANK_MODEL = config.get("rerank_model", "Salesforce/blip2-itm-vit-g-coco")
BATCH_SIZE = config.get("batchsize", 32)


class MultimodalCrossEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        device: str = None,
        fp16: bool = False,
    ) -> None:
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.fp16 = fp16

        print(f"Loading cross-encoder model: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)

        if fp16 and self.device == "cuda":
            self.model = Blip2ForImageTextRetrieval.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
        else:
            self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()
        print(f"Cross-encoder loaded on device: {self.device}")

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    @staticmethod
    def build_text_input(example: Dict) -> str:
        question = example.get("question", "")
        img_caption = example.get("image_caption", "")
        return f"Question: {question} [SEP] Image caption: {img_caption}"

    @classmethod
    def build_pair_text(cls, query_example: Dict, cand_example: Dict) -> str:
        q_txt = cls.build_text_input(query_example)
        c_txt = cls.build_text_input(cand_example)

        return (
            f"[QUERY] {q_txt} "
            f"[SEP] [CANDIDATE] {c_txt} "
            f"[TASK] Does this candidate match the query?"
        )

    @staticmethod
    def build_image_input(example: Dict) -> Image.Image:
        image_path = example.get("image", "")
        return load_image(image_path)

    @torch.no_grad()
    def score_single_pair(
        self,
        query_example: Dict,
        candidate_example: Dict,
    ) -> float:
        self.model.eval()

        image = self.build_image_input(candidate_example)
        text = self.build_pair_text(query_example, candidate_example)

        inputs = self.processor(
            images=[image],
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)

        if hasattr(outputs, "itm_score"):
            logits = outputs.itm_score
        else:
            logits = outputs[0]

        logits = logits.view(-1)
        prob = torch.sigmoid(logits)

        return float(prob.item())

    @torch.no_grad()
    def score_candidates(
        self,
        query_example: Dict,
        candidate_examples: List[Dict],
        batch_size: int = BATCH_SIZE,
    ) -> List[float]:
        self.model.eval()

        all_scores: List[float] = []
        num_cands = len(candidate_examples)

        for start in range(0, num_cands, batch_size):
            end = min(start + batch_size, num_cands)
            batch = candidate_examples[start:end]

            # Use candidate images with query-candidate joint text
            images = [self.build_image_input(ex) for ex in batch]
            texts = [self.build_pair_text(query_example, ex) for ex in batch]

            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)

            if hasattr(outputs, "itm_score"):
                logits = outputs.itm_score  # [B, 1] or [B]
            else:
                logits = outputs[0]  # fallback: first element

            # Ensure shape [B]
            logits = logits.view(-1)

            # Convert logits -> probabilities with sigmoid
            probs = torch.sigmoid(logits)

            all_scores.extend(probs.detach().cpu().tolist())

        return all_scores

    @torch.no_grad()
    def rerank(
        self,
        query_example: Dict,
        candidate_examples: List[Dict],
        top_k: Optional[int] = None,
        batch_size: int = BATCH_SIZE,
    ) -> List[Tuple[Dict, float]]:
        scores = self.score_candidates(
            query_example=query_example,
            candidate_examples=candidate_examples,
            batch_size=batch_size,
        )

        ranked = sorted(
            zip(candidate_examples, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def compute_loss(
        self,
        query_examples: List[Dict],
        positive_examples: List[Dict],
        negative_examples: Optional[List[List[Dict]]] = None,
        margin: float = 0.2,
    ) -> torch.Tensor:
        self.model.train()

        batch_size = len(query_examples)
        total_loss = torch.tensor(0.0, device=self.device)

        for i in range(batch_size):
            query = query_examples[i]
            positive = positive_examples[i]

            # Positive pair
            pos_image = self.build_image_input(positive)
            pos_text = self.build_pair_text(query, positive)

            pos_inputs = self.processor(
                images=[pos_image],
                text=[pos_text],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            pos_outputs = self.model(**pos_inputs)

            if hasattr(pos_outputs, "itm_score"):
                pos_logits = pos_outputs.itm_score.view(-1)
            else:
                pos_logits = pos_outputs[0].view(-1)

            pos_score = torch.sigmoid(pos_logits)

            # Binary cross-entropy for positive (target = 1)
            pos_loss = -torch.log(pos_score + 1e-8)

            # Negative pairs
            if negative_examples is not None and len(negative_examples[i]) > 0:
                for negative in negative_examples[i]:
                    neg_image = self.build_image_input(negative)
                    neg_text = self.build_pair_text(query, negative)

                    neg_inputs = self.processor(
                        images=[neg_image],
                        text=[neg_text],
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)

                    neg_outputs = self.model(**neg_inputs)

                    if hasattr(neg_outputs, "itm_score"):
                        neg_logits = neg_outputs.itm_score.view(-1)
                    else:
                        neg_logits = neg_outputs[0].view(-1)

                    neg_score = torch.sigmoid(neg_logits)

                    # Binary cross-entropy for negative (target = 0)
                    neg_loss = -torch.log(1 - neg_score + 1e-8)

                    # Accumulate loss with margin triplet component
                    triplet_loss = torch.relu(neg_score - pos_score + margin)
                    total_loss += pos_loss + neg_loss + triplet_loss
            else:
                total_loss += pos_loss

        return total_loss / batch_size

    def training_step(
        self,
        query_examples: List[Dict],
        positive_examples: List[Dict],
        negative_examples: Optional[List[List[Dict]]],
        optimizer: torch.optim.Optimizer,
        margin: float = 0.2,
    ) -> float:
        self.model.train()

        loss = self.compute_loss(
            query_examples=query_examples,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            margin=margin,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return float(loss.item())

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: str = None,
    ) -> "MultimodalCrossEncoder":
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = payload.get("config", {})

        model = cls(
            model_name=cfg.get("model_name", RERANK_MODEL),
            device=device,
            fp16=cfg.get("fp16", False),
        )

        # Load fine-tuned weights if present
        if "model_state_dict" in payload:
            model.model.load_state_dict(payload["model_state_dict"])

        model.eval_mode()

        # Attach metadata for convenience
        model.ckpt_config = cfg
        model.ckpt_meta = payload.get("best", {})
        model.ckpt_env = payload.get("env", {})

        return model

    def save_checkpoint(
        self,
        save_path: str,
        config: Dict,
        epoch: int,
        loss: float,
    ) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "model_name": self.model_name,
                "fp16": self.fp16,
                **config,
            },
            "best": {
                "epoch": epoch,
                "loss": float(loss),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
        }

        torch.save(payload, save_path)
        print(f"Checkpoint saved to: {save_path}")
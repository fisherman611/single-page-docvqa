import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))

import json
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
import faiss
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login

login(token=os.getenv("HF_READ_TOKEN"))

# Import component models
from models.multimodal_retriever.stage3_multimodal_retriever.multimodal_biencoder import (
    MultimodalBiEncoder,
)
from models.multimodal_retriever.stage3_multimodal_retriever.multimodal_crossencoder import (
    MultimodalCrossEncoder,
)
from models.multimodal_retriever.stage3_multimodal_retriever import faiss_utils

# Load config
with open(
    "models/multimodal_retriever/stage3_multimodal_retriever/config.json",
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)

DEFAULT_TOP_K = config.get("top_k", 20)
DEFAULT_BATCH_SIZE = config.get("batchsize", 32)


class MultimodalRetriever:
    """
    Two-stage Multimodal Retriever Pipeline.
    
    Stage 1: BiEncoder for fast approximate retrieval via FAISS
    Stage 2: CrossEncoder for precise re-ranking (optional)
    
    The BiEncoder uses contrastive learning to encode (image, question, caption)
    tuples into a shared embedding space for efficient similarity search.
    
    The CrossEncoder uses BLIP-2 Image-Text Matching to compute fine-grained
    relevance scores between query-candidate pairs.
    """

    def __init__(
        self,
        biencoder: MultimodalBiEncoder,
        crossencoder: Optional[MultimodalCrossEncoder] = None,
        device: str = None,
    ) -> None:
        """
        Initialize the retriever with pre-loaded encoder models.
        
        Args:
            biencoder: Trained MultimodalBiEncoder for first-stage retrieval
            crossencoder: Optional trained MultimodalCrossEncoder for re-ranking
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.biencoder = biencoder
        self.crossencoder = crossencoder
        
        # FAISS index and corpus
        self.index: Optional[faiss.Index] = None
        self.corpus: Optional[List[Dict]] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.id_to_idx: Dict[str, int] = {}
        
        print(f"MultimodalRetriever initialized on device: {self.device}")
        print(f"  - BiEncoder: ✓ loaded")
        print(f"  - CrossEncoder: {'✓ loaded' if crossencoder else '✗ not loaded (retrieval only)'}")

    @classmethod
    def from_checkpoints(
        cls,
        biencoder_ckpt: str,
        crossencoder_ckpt: Optional[str] = None,
        device: str = None,
    ) -> "MultimodalRetriever":
        """
        Create retriever from saved model checkpoints.
        
        Args:
            biencoder_ckpt: Path to trained BiEncoder checkpoint (.pt file)
            crossencoder_ckpt: Optional path to trained CrossEncoder checkpoint
            device: Device to use
            
        Returns:
            Initialized MultimodalRetriever
        """
        print(f"Loading BiEncoder from: {biencoder_ckpt}")
        biencoder = MultimodalBiEncoder.from_checkpoint(biencoder_ckpt, device=device)
        
        crossencoder = None
        if crossencoder_ckpt is not None:
            print(f"Loading CrossEncoder from: {crossencoder_ckpt}")
            crossencoder = MultimodalCrossEncoder.from_checkpoint(
                crossencoder_ckpt, device=device
            )
        
        return cls(
            biencoder=biencoder,
            crossencoder=crossencoder,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        text_model_name: str = None,
        image_model_name: str = None,
        proj_dim: int = None,
        rerank_model_name: str = None,
        use_crossencoder: bool = True,
        device: str = None,
        fp16: bool = False,
    ) -> "MultimodalRetriever":
        """
        Create retriever with fresh (untrained) models from HuggingFace.
        
        Note: For production use, prefer from_checkpoints() with trained models.
        
        Args:
            text_model_name: Text encoder model name (default from config)
            image_model_name: Image encoder model name (default from config)
            proj_dim: Projection dimension (default from config)
            rerank_model_name: CrossEncoder model name (default from config)
            use_crossencoder: Whether to load CrossEncoder
            device: Device to use
            fp16: Use FP16 for CrossEncoder (GPU only)
            
        Returns:
            Initialized MultimodalRetriever
        """
        # Use config defaults if not specified
        text_model_name = text_model_name or config["text_model_embedding"]
        image_model_name = image_model_name or config["image_model_embedding"]
        proj_dim = proj_dim or config["proj_dim"]
        rerank_model_name = rerank_model_name or config.get(
            "rerank_model", "Salesforce/blip2-itm-vit-g-coco"
        )
        
        print("Initializing fresh models from HuggingFace...")
        biencoder = MultimodalBiEncoder(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            proj_dim=proj_dim,
            device=device,
        )
        
        crossencoder = None
        if use_crossencoder:
            crossencoder = MultimodalCrossEncoder(
                model_name=rerank_model_name,
                device=device,
                fp16=fp16,
            )
        
        return cls(
            biencoder=biencoder,
            crossencoder=crossencoder,
            device=device,
        )

    def build_index(
        self,
        corpus: List[Dict],
        batch_size: int = 64,
        use_gpu: bool = False,
        index_type: str = "flat",
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        show_progress: bool = True,
    ) -> None:
        """
        Build FAISS index from corpus documents.
        
        Each document should have:
            - "id": unique identifier
            - "image": path to image file
            - "question": question text
            - "image_caption": caption text
        
        Args:
            corpus: List of document dictionaries
            batch_size: Batch size for encoding
            use_gpu: Use GPU for FAISS index
            index_type: Type of index ('flat', 'ivf', 'pq', 'ivfpq')
            nlist: Number of clusters for IVF
            m: Number of subquantizers for PQ
            nbits: Bits per subquantizer for PQ
            show_progress: Show progress bar
        """
        print(f"\nBuilding FAISS index for {len(corpus)} documents...")
        
        self.corpus = corpus
        self.id_to_idx = {doc["id"]: i for i, doc in enumerate(corpus)}
        
        # Build index using BiEncoder
        self.index, self.corpus_embeddings = self.biencoder.build_faiss_index(
            examples=corpus,
            batch_size=batch_size,
            use_gpu=use_gpu,
            index_type=index_type,
            nlist=nlist,
            m=m,
            nbits=nbits,
        )
        
        print(f"Index built: {self.index.ntotal} vectors, dimension={self.index.d}")

    def save_index(
        self,
        save_path: str,
        save_corpus: bool = True,
    ) -> None:
        """
        Save FAISS index and corpus metadata to disk.
        
        Args:
            save_path: Path prefix for saving (without extension)
            save_corpus: Whether to save corpus metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")
        
        metadata = None
        if save_corpus and self.corpus is not None:
            metadata = {
                "corpus": self.corpus,
                "id_to_idx": self.id_to_idx,
            }
        
        faiss_utils.save_index(self.index, save_path, metadata=metadata)
        
        # Also save embeddings for potential future use
        if self.corpus_embeddings is not None:
            emb_path = f"{save_path}.embeddings.npy"
            np.save(emb_path, self.corpus_embeddings)
            print(f"Embeddings saved to: {emb_path}")

    def load_index(
        self,
        load_path: str,
        use_gpu: bool = False,
    ) -> None:
        """
        Load FAISS index and corpus metadata from disk.
        
        Args:
            load_path: Path prefix to load from (without extension)
            use_gpu: Move index to GPU after loading
        """
        self.index, metadata = faiss_utils.load_index(
            load_path, use_gpu=use_gpu, load_metadata=True
        )
        
        if metadata is not None:
            self.corpus = metadata.get("corpus")
            self.id_to_idx = metadata.get("id_to_idx", {})
        
        # Load embeddings if available
        emb_path = f"{load_path}.embeddings.npy"
        if os.path.exists(emb_path):
            self.corpus_embeddings = np.load(emb_path)
            print(f"Embeddings loaded from: {emb_path}")

    def _encode_query(
        self,
        query: Dict,
        query_view: str = "image+question",
    ) -> np.ndarray:
        """
        Encode a single query using the BiEncoder.
        
        Args:
            query: Query dictionary with image, question, image_caption
            query_view: View type for query encoding
                - "image+question": use image + question (drop caption)
                - "image+caption": use image + caption (drop question)
                - "text-only": use question + caption (no image)
                - "full": use all modalities
        
        Returns:
            Query embedding as numpy array [1, dim]
        """
        self.biencoder.eval()
        
        # Configure view
        if query_view == "image+question":
            use = [True]
            dq = [False]
            dc = [True]
        elif query_view == "image+caption":
            use = [True]
            dq = [True]
            dc = [False]
        elif query_view == "text-only":
            use = [False]
            dq = [False]
            dc = [False]
        else:  # full
            use = [True]
            dq = [False]
            dc = [False]
        
        with torch.no_grad():
            emb = self.biencoder.encode_bundle(
                [query], use, dq, dc, no_grad=True
            ).cpu().numpy().astype("float32")
        
        return emb

    def retrieve_biencoder(
        self,
        query: Dict,
        top_k: int = DEFAULT_TOP_K,
        query_view: str = "image+question",
    ) -> List[Tuple[Dict, float]]:
        """
        First-stage retrieval using BiEncoder + FAISS.
        
        Args:
            query: Query dictionary
            top_k: Number of results to return
            query_view: Query encoding view type
            
        Returns:
            List of (document, score) tuples sorted by descending score
        """
        if self.index is None or self.corpus is None:
            raise ValueError("No index available. Call build_index() or load_index() first.")
        
        # Encode query
        query_emb = self._encode_query(query, query_view)
        
        # Search FAISS index
        scores, indices = self.index.search(query_emb, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # valid index
                doc = self.corpus[idx]
                results.append((doc, float(score)))
        
        return results

    def rerank_crossencoder(
        self,
        query: Dict,
        candidates: List[Dict],
        top_k: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Tuple[Dict, float]]:
        """
        Second-stage re-ranking using CrossEncoder.
        
        Args:
            query: Query dictionary
            candidates: List of candidate documents from first stage
            top_k: Number of results to return (None = all)
            batch_size: Batch size for scoring
            
        Returns:
            List of (document, score) tuples sorted by descending score
        """
        if self.crossencoder is None:
            raise ValueError("CrossEncoder not loaded. Initialize with crossencoder_ckpt.")
        
        return self.crossencoder.rerank(
            query_example=query,
            candidate_examples=candidates,
            top_k=top_k,
            batch_size=batch_size,
        )

    def retrieve(
        self,
        query: Dict,
        top_k: int = 10,
        first_stage_k: Optional[int] = None,
        query_view: str = "image+question",
        use_reranking: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Tuple[Dict, float]]:
        """
        Full two-stage retrieval pipeline.
        
        Stage 1: BiEncoder retrieval (fast, approximate)
        Stage 2: CrossEncoder re-ranking (precise, slower)
        
        Args:
            query: Query dictionary with:
                - "image": path to query image
                - "question": query question
                - "image_caption": query image caption (optional)
            top_k: Final number of results to return
            first_stage_k: Number of candidates for re-ranking
                           (default: 5x top_k for re-ranking, or top_k otherwise)
            query_view: View for BiEncoder query encoding
            use_reranking: Whether to use CrossEncoder re-ranking
            batch_size: Batch size for CrossEncoder scoring
            
        Returns:
            List of (document, score) tuples sorted by descending relevance
            
        Example:
            query = {
                "image": "path/to/query/image.jpg",
                "question": "What is the invoice number?",
                "image_caption": "A scanned invoice document"
            }
            
            results = retriever.retrieve(query, top_k=5)
            for doc, score in results:
                print(f"Score: {score:.4f}, ID: {doc['id']}")
        """
        # Determine first stage k
        if first_stage_k is None:
            if use_reranking and self.crossencoder is not None:
                first_stage_k = min(top_k * 5, len(self.corpus) if self.corpus else 100)
            else:
                first_stage_k = top_k
        
        # Stage 1: BiEncoder retrieval
        candidates = self.retrieve_biencoder(
            query=query,
            top_k=first_stage_k,
            query_view=query_view,
        )
        
        # Stage 2: CrossEncoder re-ranking (if enabled and available)
        if use_reranking and self.crossencoder is not None:
            candidate_docs = [doc for doc, _ in candidates]
            results = self.rerank_crossencoder(
                query=query,
                candidates=candidate_docs,
                top_k=top_k,
                batch_size=batch_size,
            )
        else:
            results = candidates[:top_k]
        
        return results

    def batch_retrieve(
        self,
        queries: List[Dict],
        top_k: int = 10,
        first_stage_k: Optional[int] = None,
        query_view: str = "image+question",
        use_reranking: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> List[List[Tuple[Dict, float]]]:
        """
        Batch retrieval for multiple queries.
        
        Args:
            queries: List of query dictionaries
            top_k: Number of results per query
            first_stage_k: Candidates for re-ranking per query
            query_view: Query encoding view
            use_reranking: Use CrossEncoder re-ranking
            batch_size: CrossEncoder batch size
            show_progress: Show progress bar
            
        Returns:
            List of result lists, one per query
        """
        all_results = []
        
        iterator = tqdm(queries, desc="Retrieving") if show_progress else queries
        
        for query in iterator:
            results = self.retrieve(
                query=query,
                top_k=top_k,
                first_stage_k=first_stage_k,
                query_view=query_view,
                use_reranking=use_reranking,
                batch_size=batch_size,
            )
            all_results.append(results)
        
        return all_results

    def evaluate(
        self,
        queries: List[Dict],
        gold_ids: List[str],
        ks: List[int] = [1, 5, 10, 20],
        use_reranking: bool = True,
        first_stage_k: Optional[int] = None,
        query_view: str = "image+question",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance with Recall@K metrics.
        
        Args:
            queries: List of query dictionaries
            gold_ids: List of gold/correct document IDs for each query
            ks: List of K values for Recall@K
            use_reranking: Whether to use CrossEncoder
            first_stage_k: Candidates for re-ranking
            query_view: Query encoding view
            batch_size: CrossEncoder batch size
            
        Returns:
            Dictionary of metrics: {
                "recall@1": float,
                "recall@5": float,
                ...
                "mrr": float  # Mean Reciprocal Rank
            }
        """
        max_k = max(ks)
        
        # Batch retrieve
        all_results = self.batch_retrieve(
            queries=queries,
            top_k=max_k,
            first_stage_k=first_stage_k,
            query_view=query_view,
            use_reranking=use_reranking,
            batch_size=batch_size,
            show_progress=True,
        )
        
        # Compute metrics
        recalls = {k: 0.0 for k in ks}
        mrr_sum = 0.0
        
        for results, gold_id in zip(all_results, gold_ids):
            retrieved_ids = [doc["id"] for doc, _ in results]
            
            # Recall@K
            for k in ks:
                if gold_id in retrieved_ids[:k]:
                    recalls[k] += 1
            
            # MRR
            if gold_id in retrieved_ids:
                rank = retrieved_ids.index(gold_id) + 1
                mrr_sum += 1.0 / rank
        
        n = len(queries)
        metrics = {f"recall@{k}": recalls[k] / n for k in ks}
        metrics["mrr"] = mrr_sum / n
        
        return metrics

    def __repr__(self) -> str:
        index_info = f"index={self.index.ntotal} vectors" if self.index else "no index"
        corpus_info = f"corpus={len(self.corpus)} docs" if self.corpus else "no corpus"
        ce_info = "with CrossEncoder" if self.crossencoder else "BiEncoder only"
        return f"MultimodalRetriever({index_info}, {corpus_info}, {ce_info})"


# def demo():
#     """
#     Demonstration of the MultimodalRetriever pipeline.
#     """
#     print("\n" + "=" * 70)
#     print("  MultimodalRetriever Pipeline Demo")
#     print("=" * 70)
    
#     # Load from trained checkpoints
#     biencoder_ckpt = "multimodal_biencoder_checkpoints/best_model.pt"
#     crossencoder_ckpt = "multimodal_crossencoder_checkpoints/best_model.pt"  # Optional
    
#     # Check if checkpoints exist for demo
#     if not os.path.exists(biencoder_ckpt):
#         print(f"\n⚠ BiEncoder checkpoint not found at: {biencoder_ckpt}")
#         print("  To use the pipeline, first train the BiEncoder:")
#         print("  python train_multimodal_biencoder.py")
#         print("\n  Or initialize with fresh models (not recommended for production):")
#         print('  retriever = MultimodalRetriever.from_pretrained(use_crossencoder=False)')
#         return
    
#     # Initialize retriever
#     retriever = MultimodalRetriever.from_checkpoints(
#         biencoder_ckpt=biencoder_ckpt,
#         crossencoder_ckpt=crossencoder_ckpt if os.path.exists(crossencoder_ckpt) else None,
#     )
    
#     # Example corpus (replace with your actual data)
#     corpus = [
#         {
#             "id": "doc_001",
#             "image": "data/spdocvqa_images/ffbf0023_4.png",
#             "question": "What is the total amount?",
#             "image_caption": "Invoice document with financial details",
#         },
#         {
#             "id": "doc_002", 
#             "image": "data/spdocvqa_images/ffbf0023_6.png",
#             "question": "When was this issued?",
#             "image_caption": "Certificate with date information",
#         },
#         # ... more documents
#     ]
    
#     # Build index
#     print("\n1. Building FAISS index...")
#     retriever.build_index(corpus, batch_size=32)
    
#     # Save index for future use
#     print("\n2. Saving index...")
#     retriever.save_index("checkpoints/retriever_index")
    
#     # Example query
#     query = {
#         "image": "data/spdocvqa_images/ffbf0227_1.png",
#         "question": "What is the invoice number?",
#         "image_caption": "A document with invoice information",
#     }
    
#     # Retrieve
#     print("\n3. Retrieving...")
#     results = retriever.retrieve(query, top_k=5)
    
#     print("\nTop-5 Results:")
#     for i, (doc, score) in enumerate(results, 1):
#         print(f"  {i}. Score: {score:.4f}, ID: {doc['id']}")
    
#     print("\n" + "=" * 70)


# if __name__ == "__main__":
#     demo()

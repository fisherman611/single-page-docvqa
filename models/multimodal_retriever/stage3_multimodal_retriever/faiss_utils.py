import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import faiss


def build_index(
    embeddings: np.ndarray,
    use_gpu: bool = False,
    index_type: str = "flat",
    nlist: int = 100,
    m: int = 8,
    nbits: int = 8,
) -> faiss.Index:
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings: numpy array of shape [N, D] where N is number of vectors, D is dimension
        use_gpu: whether to use GPU for index (if available)
        index_type: type of index to build
            - "flat": exact search (IndexFlatIP for inner product)
            - "ivf": inverted file index for faster approximate search
            - "pq": product quantization for memory efficiency
            - "ivfpq": combination of IVF and PQ
        nlist: number of clusters for IVF indices
        m: number of subquantizers for PQ
        nbits: number of bits per subquantizer for PQ

    Returns:
        FAISS index ready for search
    """
    N, D = embeddings.shape
    embeddings = embeddings.astype(np.float32)

    # Normalize embeddings for cosine similarity (using inner product on normalized vectors)
    faiss.normalize_L2(embeddings)

    # Build index based on type
    if index_type == "flat":
        index = faiss.IndexFlatIP(
            D
        )  # Inner product for normalized vectors = cosine similarity

    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)

    elif index_type == "pq":
        index = faiss.IndexPQ(D, m, nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)

    elif index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFPQ(
            quantizer, D, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)

    else:
        raise ValueError(
            f"Unknown index_type: {index_type}. Choose from: flat, ivf, pq, ivfpq"
        )

    # Move to GPU if requested and available
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add vectors to index
    index.add(embeddings)

    return index


def save_index(
    index: faiss.Index, save_path: str, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save FAISS index to disk along with optional metadata.

    Args:
        index: FAISS index to save
        save_path: path to save the index (without extension)
        metadata: optional dictionary of metadata to save alongside index
                  (e.g., document IDs, configuration, etc.)
    """
    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # If index is on GPU, move to CPU before saving
    if hasattr(index, "getDevice"):
        index = faiss.index_gpu_to_cpu(index)

    # Save index
    index_file = f"{save_path}.index"
    faiss.write_index(index, index_file)
    print(f"Index saved to: {index_file}")

    # Save metadata if provided
    if metadata is not None:
        import pickle

        metadata_file = f"{save_path}.metadata"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to: {metadata_file}")


def load_index(
    load_path: str, use_gpu: bool = False, load_metadata: bool = True
) -> Tuple[faiss.Index, Optional[Dict[str, Any]]]:
    """
    Load FAISS index from disk along with optional metadata.

    Args:
        load_path: path to load the index from (without extension)
        use_gpu: whether to move index to GPU after loading
        load_metadata: whether to load metadata file if it exists

    Returns:
        Tuple of (index, metadata). metadata is None if not found or load_metadata=False
    """
    # Load index
    index_file = f"{load_path}.index"
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    index = faiss.read_index(index_file)
    print(f"Index loaded from: {index_file}")

    # Move to GPU if requested
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("Index moved to GPU")

    # Load metadata if requested
    metadata = None
    if load_metadata:
        metadata_file = f"{load_path}.metadata"
        if os.path.exists(metadata_file):
            import pickle

            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            print(f"Metadata loaded from: {metadata_file}")
        else:
            print(f"No metadata file found at: {metadata_file}")

    return index, metadata


# def test_faiss_utils():
#     """
#     Test the FAISS utility functions with sample data.
#     """
#     print("\n" + "=" * 60)
#     print("Testing FAISS Utilities")
#     print("=" * 60)

#     # Create sample embeddings (100 vectors of dimension 512)
#     np.random.seed(42)
#     num_vectors = 100
#     dim = 512
#     embeddings = np.random.randn(num_vectors, dim).astype(np.float32)

#     print(f"\n1. Created sample embeddings: shape={embeddings.shape}")

#     # Test 1: Build a flat index (exact search)
#     print("\n2. Building FLAT index...")
#     index_flat = build_index(embeddings, index_type="flat", use_gpu=False)
#     print(f"   Index built: ntotal={index_flat.ntotal}, dimension={index_flat.d}")

#     # Test 2: Build an IVF index (approximate search)
#     print("\n3. Building IVF index...")
#     index_ivf = build_index(embeddings, index_type="ivf", nlist=10, use_gpu=False)
#     print(f"   Index built: ntotal={index_ivf.ntotal}, dimension={index_ivf.d}")

#     # Test 3: Save index with metadata
#     print("\n4. Saving index with metadata...")
#     save_path = "test_faiss_index"
#     metadata = {
#         "num_vectors": num_vectors,
#         "dimension": dim,
#         "index_type": "flat",
#         "doc_ids": [f"doc_{i}" for i in range(num_vectors)],
#         "created_at": "2026-01-02",
#     }
#     save_index(index_flat, save_path, metadata=metadata)

#     # Test 4: Load index with metadata
#     print("\n5. Loading index with metadata...")
#     loaded_index, loaded_metadata = load_index(
#         save_path, use_gpu=False, load_metadata=True
#     )
#     print(f"   Index loaded: ntotal={loaded_index.ntotal}, dimension={loaded_index.d}")
#     print(f"   Metadata keys: {list(loaded_metadata.keys())}")
#     print(f"   Sample doc_ids: {loaded_metadata['doc_ids'][:5]}")

#     # Test 5: Perform similarity search
#     print("\n6. Testing similarity search...")
#     query = np.random.randn(1, dim).astype(np.float32)
#     faiss.normalize_L2(query)  # Normalize query

#     k = 5  # top-5 results
#     distances, indices = loaded_index.search(query, k)

#     print(f"   Query shape: {query.shape}")
#     print(f"   Top-{k} results:")
#     for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
#         doc_id = loaded_metadata["doc_ids"][idx]
#         print(f"      Rank {i+1}: doc_id={doc_id}, similarity={dist:.4f}")

#     # Cleanup test files
#     print("\n7. Cleaning up test files...")
#     import os

#     if os.path.exists(f"{save_path}.index"):
#         os.remove(f"{save_path}.index")
#         print(f"   Removed {save_path}.index")
#     if os.path.exists(f"{save_path}.metadata"):
#         os.remove(f"{save_path}.metadata")
#         print(f"   Removed {save_path}.metadata")

#     print("\n" + "=" * 60)
#     print("All tests passed successfully!")
#     print("=" * 60 + "\n")


# if __name__ == "__main__":
#     test_faiss_utils()

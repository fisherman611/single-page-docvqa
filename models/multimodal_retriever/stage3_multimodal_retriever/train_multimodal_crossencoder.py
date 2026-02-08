import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))

import json
import random
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import torch
from tqdm import tqdm
import platform
import transformers

from multimodal_crossencoder import MultimodalCrossEncoder
from multimodal_biencoder import MultimodalBiEncoder


def load_examples(path: str, max_examples: Optional[int] = None) -> List[Dict]:
    """
    Load training examples from JSON file.

    Args:
        path: Path to JSON file
        max_examples: Maximum number of examples to load (None = all)

    Returns:
        List of example dictionaries
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle different JSON structures
    if "data" in data:
        examples = data["data"]
    else:
        examples = data

    # Ensure required keys and fix image paths
    for i, ex in enumerate(examples):
        if "id" not in ex:
            ex["id"] = i
        if "image_caption" not in ex:
            ex["image_caption"] = f"Document image {i}"
        # Fix image path
        if "image" in ex:
            ex["image"] = Path(ex["image"])

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


def build_training_pairs(
    examples: List[Dict],
    hard_negs: Optional[Dict[str, List[str]]] = None,
    negatives_per_query: int = 3,
) -> List[Tuple[Dict, Dict, List[Dict]]]:
    """
    Build (query, positive, negatives) training tuples.

    For cross-encoder reranking training, we need:
    - Query example
    - Positive example (similar/relevant)
    - Negative examples (dissimilar/irrelevant)

    Since we don't have explicit relevance labels, we use:
    - Random in-batch negatives
    - Hard negatives from bi-encoder (if provided)

    Args:
        examples: List of all examples
        hard_negs: Optional hard negatives dict from bi-encoder mining
        negatives_per_query: Number of negatives per query

    Returns:
        List of (query, positive, negatives) tuples
    """
    id_to_ex = {ex["id"]: ex for ex in examples}
    training_tuples = []

    for ex in examples:
        query = ex
        # For self-supervised training, positive is the example itself
        # In practice, you'd have explicit positive pairs
        positive = ex

        negatives = []

        # Add hard negatives if available
        if hard_negs is not None:
            hn_ids = hard_negs.get(ex["id"], [])
            for neg_id in hn_ids[:negatives_per_query]:
                if neg_id in id_to_ex and neg_id != ex["id"]:
                    negatives.append(id_to_ex[neg_id])

        # Fill remaining with random negatives
        while len(negatives) < negatives_per_query:
            random_ex = random.choice(examples)
            if random_ex["id"] != ex["id"] and random_ex not in negatives:
                negatives.append(random_ex)

        training_tuples.append((query, positive, negatives))

    return training_tuples


def make_batch(
    training_tuples: List[Tuple[Dict, Dict, List[Dict]]],
    batch_size: int,
) -> Tuple[List[Dict], List[Dict], List[List[Dict]]]:
    """
    Sample a training batch.

    Args:
        training_tuples: List of (query, positive, negatives) tuples
        batch_size: Batch size

    Returns:
        Tuple of (queries, positives, negatives_list)
    """
    batch_tuples = random.sample(
        training_tuples, k=min(batch_size, len(training_tuples))
    )

    queries = [t[0] for t in batch_tuples]
    positives = [t[1] for t in batch_tuples]
    negatives = [t[2] for t in batch_tuples]

    return queries, positives, negatives


def save_checkpoint(
    model: MultimodalCrossEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"   Checkpoint saved to: {save_path}")


def load_checkpoint(
    model: MultimodalCrossEncoder,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> Tuple[int, int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


def save_best_model(
    model: MultimodalCrossEncoder,
    save_path: str,
    config: Dict,
    epoch: int,
    loss: float,
) -> None:
    """Save best model checkpoint with full metadata."""
    payload = {
        "model_state_dict": model.model.state_dict(),
        "config": {
            "model_name": config["model_name"],
            "fp16": config.get("fp16", False),
            "margin": config.get("margin", 0.2),
            "negatives_per_query": config.get("negatives_per_query", 3),
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
    print(f"   Best model saved to: {save_path}")


def mine_hard_negatives_with_biencoder(
    biencoder_ckpt: str,
    examples: List[Dict],
    top_m: int = 50,
    batch_size: int = 64,
) -> Dict[str, List[str]]:
    """
    Mine hard negatives using a pre-trained bi-encoder.

    Args:
        biencoder_ckpt: Path to bi-encoder checkpoint
        examples: List of examples
        top_m: Number of top neighbors to retrieve
        batch_size: Batch size for encoding

    Returns:
        Dictionary mapping example ID to list of hard negative IDs
    """
    print(f"Loading bi-encoder from: {biencoder_ckpt}")
    biencoder = MultimodalBiEncoder.from_checkpoint(biencoder_ckpt)

    print("Building FAISS index...")
    index, _ = biencoder.build_faiss_index(
        examples=examples,
        batch_size=batch_size,
        use_gpu=False,
        index_type="flat",
    )

    print(f"Mining hard negatives (top-{top_m})...")
    hard_negs = biencoder.mine_hard_negatives(
        index=index,
        examples=examples,
        topm=top_m,
        query_view="full",
        batch_size=batch_size,
    )

    print(f"Mined hard negatives for {len(hard_negs)} examples")
    return hard_negs


def main():
    parser = argparse.ArgumentParser(
        description="Train Multimodal Cross-Encoder for Reranking"
    )

    # Data arguments
    parser.add_argument("--data", required=True, help="Path to training data JSON")
    parser.add_argument(
        "--max_examples", type=int, default=None, help="Max examples to use"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        default="Salesforce/blip2-itm-vit-g-coco",
        help="BLIP-2 model name for cross-encoder",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )

    # Bi-encoder for hard negative mining
    parser.add_argument(
        "--biencoder_ckpt",
        default=None,
        help="Path to bi-encoder checkpoint for hard negative mining",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=200, help="Training steps per epoch"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--margin", type=float, default=0.2, help="Margin for triplet loss"
    )
    parser.add_argument(
        "--negatives_per_query",
        type=int,
        default=3,
        help="Number of negative examples per query",
    )
    parser.add_argument(
        "--hard_topm",
        type=int,
        default=50,
        help="Number of top neighbors for hard negative mining",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--save_dir",
        default="crossencoder_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint to resume from"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log loss every N steps"
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("Training Multimodal Cross-Encoder")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Model: {args.model_name}")
    print(f"FP16: {args.fp16}")
    print(f"Epochs: {args.epochs}")
    print(f"Steps/Epoch: {args.steps_per_epoch}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Margin: {args.margin}")
    print(f"Negatives per Query: {args.negatives_per_query}")
    if args.biencoder_ckpt:
        print(f"Bi-encoder for Hard Negs: {args.biencoder_ckpt}")
    print("=" * 70 + "\n")

    # Load data
    print("Loading training data...")
    examples = load_examples(args.data, max_examples=args.max_examples)
    print(f"Loaded {len(examples)} examples\n")

    # Mine hard negatives if bi-encoder checkpoint provided
    hard_negs = None
    if args.biencoder_ckpt:
        hard_negs = mine_hard_negatives_with_biencoder(
            biencoder_ckpt=args.biencoder_ckpt,
            examples=examples,
            top_m=args.hard_topm,
            batch_size=64,
        )

    # Build training pairs
    print("Building training pairs...")
    training_tuples = build_training_pairs(
        examples=examples,
        hard_negs=hard_negs,
        negatives_per_query=args.negatives_per_query,
    )
    print(f"Created {len(training_tuples)} training tuples\n")

    # Initialize model
    print("Initializing cross-encoder model...")
    model = MultimodalCrossEncoder(
        model_name=args.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=args.fp16,
    )
    print(f"Model initialized on device: {model.device}\n")

    # Initialize optimizer (only optimize specific layers for efficiency)
    # For BLIP-2, we can fine-tune the Q-Former and projection layers
    trainable_params = []
    for name, param in model.model.named_parameters():
        # Fine-tune Q-Former and text projection layers
        if "qformer" in name.lower() or "text_proj" in name.lower() or "itm_head" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    if len(trainable_params) == 0:
        # Fallback: train all parameters
        print("Warning: No specific layers found, training all parameters")
        trainable_params = model.model.parameters()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Prepare config for saving
    model_config = {
        "model_name": args.model_name,
        "fp16": args.fp16,
        "margin": args.margin,
        "negatives_per_query": args.negatives_per_query,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
    }

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        last_epoch, _, last_loss = load_checkpoint(model, optimizer, args.resume)
        start_epoch = last_epoch + 1
        print(f"Resuming from epoch {start_epoch}\n")

    # Best tracking
    best_loss = float("inf")
    best_epoch = 0
    best_path = os.path.join(args.save_dir, "best_crossencoder.pt")
    best_meta_path = os.path.join(args.save_dir, "best_meta.json")

    # Training loop
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 70}")

        model.train_mode()
        running_loss = 0.0
        epoch_loss = 0.0

        progress_bar = tqdm(
            range(1, args.steps_per_epoch + 1),
            desc=f"Epoch {epoch}",
        )

        for step in progress_bar:
            global_step += 1

            # Create batch
            queries, positives, negatives = make_batch(
                training_tuples=training_tuples,
                batch_size=args.batch_size,
            )

            # Training step
            loss = model.training_step(
                query_examples=queries,
                positive_examples=positives,
                negative_examples=negatives,
                optimizer=optimizer,
                margin=args.margin,
            )

            running_loss += loss
            epoch_loss += loss

            # Logging
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0.0

        avg_epoch_loss = epoch_loss / args.steps_per_epoch
        print(f"\nEpoch {epoch} completed | Average Loss: {avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch

            save_best_model(
                model=model,
                save_path=best_path,
                config=model_config,
                epoch=best_epoch,
                loss=best_loss,
            )

            with open(best_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_loss": best_loss,
                        "best_model_path": best_path,
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                        "config": model_config,
                    },
                    f,
                    indent=2,
                )

            print(f"   New best model! loss={best_loss:.4f} @ epoch {best_epoch}")

        # Save periodic checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f"checkpoint_epoch{epoch}.pt"
            )
            save_checkpoint(
                model, optimizer, epoch, global_step, avg_epoch_loss, checkpoint_path
            )

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Model: {best_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

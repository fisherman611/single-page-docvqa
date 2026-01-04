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
from typing import List, Dict, Optional
from datetime import datetime

import torch
from tqdm import tqdm

from multimodal_biencoder import MultimodalBiEncoder
from faiss_utils import save_index

IMAGE_ROOT = Path("data/spdocvqa_images")


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
            if "image_description" in ex:
                ex["image_caption"] = ex["image_description"]
            else:
                ex["image_caption"] = f"Document image {i}"
        # Fix image path
        if "image" in ex:
            ex["image"] = IMAGE_ROOT / Path(ex["image"]).name

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


def make_batch_with_hard_negs(
    examples: List[Dict],
    hard_negs: Optional[Dict[str, List[str]]],
    batch_size: int,
    hard_neg_k: int = 2,
) -> List[Dict]:
    """
    Create a batch with hard negatives mixed in.
    """
    id_to_ex = {ex["id"]: ex for ex in examples}

    # Sample anchor examples
    anchors = random.sample(examples, k=min(batch_size, len(examples)))
    batch: List[Dict] = []

    for ex in anchors:
        batch.append(ex)

        # Add hard negatives for this anchor
        if hard_negs is not None:
            hn = hard_negs.get(ex["id"], [])[:hard_neg_k]
            for neg_id in hn:
                if neg_id in id_to_ex and len(batch) < batch_size:
                    batch.append(id_to_ex[neg_id])

        if len(batch) >= batch_size:
            break

    # Fill remaining slots with random examples
    while len(batch) < batch_size:
        batch.append(random.choice(examples))

    return batch[:batch_size]


def save_checkpoint(
    model: MultimodalBiEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
):
    """Save training checkpoint (projection layer + optimizer)."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.proj.state_dict(),  # Only save projection layer
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"   Checkpoint saved to: {save_path}")


def load_checkpoint(
    model: MultimodalBiEncoder,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> tuple:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.proj.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


def save_best_model(
    model: MultimodalBiEncoder,
    save_path: str,
    config: Dict,
    epoch: int,
    loss: float,
):
    """Save the current best model (projection layer + config + metadata)."""
    payload = {
        "proj_state_dict": model.proj.state_dict(),
        "config": config,
        "best": {
            "epoch": epoch,
            "loss": loss,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    torch.save(payload, save_path)


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Bi-encoder")

    # Data arguments
    parser.add_argument("--data", required=True, help="Path to training data JSON")
    parser.add_argument(
        "--max_examples", type=int, default=None, help="Max examples to use"
    )

    # Model arguments
    parser.add_argument(
        "--text_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Text encoder model name",
    )
    parser.add_argument(
        "--image_model",
        default="openai/clip-vit-base-patch32",
        help="Image encoder model name",
    )
    parser.add_argument(
        "--proj_dim", type=int, default=512, help="Projection dimension"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=300, help="Training steps per epoch"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss",
    )

    # Hard negative mining arguments
    parser.add_argument(
        "--hard_mining_every",
        type=int,
        default=1,
        help="Mine hard negatives every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--hard_topm",
        type=int,
        default=50,
        help="Number of top neighbors to retrieve for hard negatives",
    )
    parser.add_argument(
        "--hard_neg_k",
        type=int,
        default=2,
        help="Number of hard negatives per anchor in batch",
    )
    parser.add_argument(
        "--query_view",
        default="image+question",
        choices=["image+question", "text-only", "full"],
        help="Query view for hard negative mining",
    )
    parser.add_argument(
        "--index_type",
        default="flat",
        choices=["flat", "ivf", "pq", "ivfpq"],
        help="FAISS index type",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--save_dir",
        default="multimodal_biencoder_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="if 1 is save every epoch, 0 is not save every",
    )
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint to resume from"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_every", type=int, default=20, help="Log loss every N steps"
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
    print("Training Multimodal Bi-encoder")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Text Model: {args.text_model}")
    print(f"Image Model: {args.image_model}")
    print(f"Projection Dim: {args.proj_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Steps/Epoch: {args.steps_per_epoch}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print(f"Hard Mining Every: {args.hard_mining_every} epochs")
    print(f"Query View: {args.query_view}")
    print(f"Index Type: {args.index_type}")
    print("=" * 70 + "\n")

    # Load data
    print("Loading training data...")
    examples = load_examples(args.data, max_examples=args.max_examples)
    print(f"Loaded {len(examples)} examples\n")

    # Initialize model
    print("Initializing model...")
    model = MultimodalBiEncoder(
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        proj_dim=args.proj_dim,
    )
    print(f"Model initialized on device: {model.device}\n")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.proj.parameters(), lr=args.lr)

    # Prepare config for saving
    model_config = {
        "text_model": args.text_model,
        "image_model": args.image_model,
        "proj_dim": args.proj_dim,
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
    best_path = os.path.join(args.save_dir, "best_model.pt")
    final_path = os.path.join(args.save_dir, "biencoder_final.pt")
    best_meta_path = os.path.join(args.save_dir, "best_meta.json")

    # Training loop
    hard_negs = None
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 70}")

        # Hard negative mining
        if args.hard_mining_every > 0 and (
            epoch == 1 or epoch % args.hard_mining_every == 0
        ):
            print("\nMining hard negatives...")
            print(f"  Building FAISS index (type={args.index_type})...")

            index, embeddings = model.build_faiss_index(
                examples=examples,
                batch_size=64,
                use_gpu=False,
                index_type=args.index_type,
                nlist=100 if args.index_type in ["ivf", "ivfpq"] else 100,
            )
            print(f"  Index built: {index.ntotal} vectors")

            print(
                f"  Mining hard negatives (query_view={args.query_view}, top-{args.hard_topm})..."
            )
            hard_negs = model.mine_hard_negatives(
                index=index,
                examples=examples,
                topm=args.hard_topm,
                query_view=args.query_view,
                batch_size=64,
            )
            print(f"  Mined hard negatives for {len(hard_negs)} examples")

            # Save index for later use
            index_path = os.path.join(args.save_dir, f"index_epoch{epoch}")
            save_index(
                index,
                index_path,
                metadata={
                    "epoch": epoch,
                    "num_examples": len(examples),
                    "dimension": embeddings.shape[1],
                },
            )

        # Training steps
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        progress_bar = tqdm(range(1, args.steps_per_epoch + 1), desc=f"Epoch {epoch}")

        for step in progress_bar:
            global_step += 1

            # Create batch with hard negatives
            batch = make_batch_with_hard_negs(
                examples=examples,
                hard_negs=hard_negs,
                batch_size=args.batch_size,
                hard_neg_k=args.hard_neg_k,
            )

            # Training step
            loss = model.two_view_contrastive_training_step(
                batch_examples=batch,
                optimizer=optimizer,
                temperature=args.temperature,
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

        # ---- Save BEST model (lowest avg_epoch_loss) ----
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

            print(f" New best model! loss={best_loss:.4f} @ epoch {best_epoch}")
            print(f"   Saved best to: {best_path}")
            print(f"   Meta: {best_meta_path}")

        # Save periodic checkpoint
        if args.save_every == 1:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pt")
            save_checkpoint(
                model, optimizer, epoch, global_step, avg_epoch_loss, checkpoint_path
            )


if __name__ == "__main__":
    main()

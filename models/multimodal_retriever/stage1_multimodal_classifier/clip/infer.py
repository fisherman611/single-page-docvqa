import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor
import numpy as np
import json
from typing import Union, List, Dict, Tuple
import argparse

from model import CLIPMultimodalClassifier

with open("models/multimodal_retriever/stage1_multimodal_classifier/clip/config.json", "r") as f:
    config = json.load(f)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = config['model']
NUM_CLASSES = len(config['classes'])
THRESHOLD = config.get('threshold', 0.5)

class CLIPMultimodalInference:
    """
    Inference wrapper for CLIP Multimodal Classifier.
    """
    def __init__(
        self, 
        checkpoint_path: str,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        threshold: float = THRESHOLD,
        config_dict: dict = None
    ):
        self.device = device
        self.threshold = threshold
        self.config = config_dict or config
        
        # Load label mappings
        self.label_list = self.config['classes']
        self.label2id = {lbl: i for i, lbl in enumerate(self.label_list)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}
        
        # Initialize processor
        print(f"Loading CLIP processor from {model_name}...")
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        
        # Initialize and load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path, model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Using threshold: {self.threshold}")
        print(f"Classes: {self.label_list}\n")
    
    def _load_model(self, checkpoint_path: str, model_name: str) -> CLIPMultimodalClassifier:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config from checkpoint if available
        ckpt_config = checkpoint.get('config', {})
        
        model = CLIPMultimodalClassifier(
            model_name=model_name,
            num_labels=NUM_CLASSES,
            freeze_clip=ckpt_config.get('freeze_clip', True),
            dropout=ckpt_config.get('dropout', 0.2),
            use_attention_fusion=ckpt_config.get('use_attention_fusion', True),
            num_attention_heads=ckpt_config.get('num_attention_heads', 8)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print checkpoint info
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'best_jaccard' in checkpoint:
            print(f"  Best Jaccard: {checkpoint['best_jaccard']:.4f}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            if isinstance(metrics, dict):
                print(f"  Metrics: {metrics}")
        
        return model
    
    @torch.no_grad()
    def predict(
        self, 
        image: Union[str, Path, Image.Image],
        question: str,
        return_probabilities: bool = False,
        top_k: int = None
    ) -> Dict:
        """
        Predict labels for a single image-question pair.
        
        Args:
            image: Path to image or PIL Image
            question: Question text
            return_probabilities: Whether to return probabilities
            top_k: Return top-k predictions only
            
        Returns:
            Dictionary with predictions and optional probabilities
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Process inputs
        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=config['max_len'],
            truncation=True
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, attention_mask, pixel_values)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > self.threshold).astype(int)
        
        # Get predicted labels
        predicted_labels = [self.id2label[i] for i, pred in enumerate(preds) if pred == 1]
        
        # Prepare result
        result = {
            'question': question,
            'predicted_labels': predicted_labels,
        }
        
        if return_probabilities:
            label_probs = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
            
            if top_k:
                # Sort by probability and take top-k
                sorted_probs = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
                label_probs = dict(sorted_probs[:top_k])
            
            result['probabilities'] = label_probs
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        questions: List[str],
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Predict labels for multiple image-question pairs.
        
        Args:
            images: List of image paths or PIL Images
            questions: List of question texts
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of dictionaries with predictions
        """
        results = []
        
        for image, question in zip(images, questions):
            result = self.predict(
                image, 
                question, 
                return_probabilities=return_probabilities
            )
            results.append(result)
        
        return results
    
    @torch.no_grad()
    def get_embeddings(
        self,
        image: Union[str, Path, Image.Image],
        question: str
    ) -> np.ndarray:
        """
        Extract multimodal embeddings for an image-question pair.
        
        Args:
            image: Path to image or PIL Image
            question: Question text
            
        Returns:
            Numpy array of embeddings
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Process inputs
        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=config['max_len'],
            truncation=True
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Extract embeddings
        embeddings = self.model.get_embeddings(input_ids, attention_mask, pixel_values)
        
        return embeddings.cpu().numpy()


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description="CLIP Multimodal Classifier Inference")
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--question', 
        type=str, 
        required=True,
        help='Question text'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=THRESHOLD,
        help=f'Prediction threshold (default: {THRESHOLD})'
    )
    parser.add_argument(
        '--show-probs', 
        action='store_true',
        help='Show prediction probabilities'
    )
    parser.add_argument(
        '--top-k', 
        type=int,
        help='Show top-k predictions by probability'
    )
    parser.add_argument(
        '--extract-embeddings', 
        action='store_true',
        help='Extract and save multimodal embeddings'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = CLIPMultimodalInference(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold
    )
    
    # Run inference
    if args.extract_embeddings:
        print("Extracting embeddings...")
        embeddings = inferencer.get_embeddings(args.image, args.question)
        print(f"Embeddings shape: {embeddings.shape}")
        
        if args.output:
            np.save(args.output.replace('.json', '.npy'), embeddings)
            print(f"Embeddings saved to: {args.output.replace('.json', '.npy')}")
    else:
        print("Running inference...")
        result = inferencer.predict(
            args.image,
            args.question,
            return_probabilities=args.show_probs,
            top_k=args.top_k
        )
        
        # Display results
        print("\n" + "="*60)
        print("Results:")
        print("="*60)
        print(f"Question: {result['question']}")
        print(f"Predicted Labels: {', '.join(result['predicted_labels']) if result['predicted_labels'] else 'None'}")
        
        if args.show_probs:
            print("\nProbabilities:")
            for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {prob:.4f}")
        
        print("="*60 + "\n")
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()


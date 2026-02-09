import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
from dataclasses import dataclass
from tqdm.auto import tqdm

from dotenv import load_dotenv

load_dotenv()


# ======================== Stage Imports ========================

# Stage 1: Multimodal Classifier
from models.multimodal_retriever.stage1_multimodal_classifier.clip.infer import (
    CLIPMultimodalInference,
)

# Stage 2: Caption Generator
from models.multimodal_retriever.stage2_caption_generator.caption_generator import (
    caption_generator,
)

# Stage 3: Multimodal Retriever
from models.multimodal_retriever.stage3_multimodal_retriever.multimodal_retriever import (
    MultimodalRetriever,
)

# Stage 4: CoT Prompt Builder
from models.multimodal_retriever.stage4_cot_builder.cot_prompt_builder import (
    CoTPromptBuilder,
)

# Stage 5: MLLM Inference
from models.multimodal_retriever.stage5_mllm_inference.mllm_inference import (
    MLLMInference,
)

# Helpers
from utils.helpers import extract_final_answer, normalize_answer, majority_vote


# ======================== Configuration ========================

@dataclass
class PipelineConfig:
    """Configuration for the multimodal retriever pipeline."""
    
    # Stage 1: Classifier
    classifier_threshold: float = 0.5
    
    # Stage 2: Caption Generator
    generate_caption: bool = True
    
    # Stage 3: Retriever
    top_k: int = 5  # Number of examples to retrieve for ICL
    first_stage_k: int = 50  # Candidates for re-ranking
    use_reranking: bool = True
    query_view: str = "image+question"  # Query encoding mode
    
    # Stage 4: CoT Builder
    max_examples: int = 5  # Max examples in prompt
    
    # Stage 5: MLLM Inference
    max_new_tokens: int = 512
    temperature: float = 0.1
    num_samples: int = 1  # For self-consistency (majority voting)
    
    # Device
    device: str = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================== Pipeline Result ========================

@dataclass
class PipelineResult:
    """Result from the multimodal retriever pipeline."""
    
    # Input
    image_path: str
    question: str
    
    # Stage 1: Classification
    predicted_labels: List[str] = None
    label_probabilities: Dict[str, float] = None
    
    # Stage 2: Caption
    image_caption: str = ""
    caption_success: bool = False
    
    # Stage 3: Retrieval
    retrieved_examples: List[Dict] = None
    retrieval_scores: List[float] = None
    
    # Stage 4: CoT Prompt
    prompt_text: str = ""
    prompt_length: int = 0
    
    # Stage 5: Final Answer
    raw_output: str = ""
    final_answer: str = ""
    normalized_answer: str = ""
    
    # Metadata
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "image_path": self.image_path,
            "question": self.question,
            "predicted_labels": self.predicted_labels,
            "label_probabilities": self.label_probabilities,
            "image_caption": self.image_caption,
            "caption_success": self.caption_success,
            "retrieved_examples": [
                {"id": ex.get("id", ex.get("questionId")), "question": ex.get("question")}
                for ex in (self.retrieved_examples or [])
            ],
            "retrieval_scores": self.retrieval_scores,
            "prompt_length": self.prompt_length,
            "raw_output": self.raw_output,
            "final_answer": self.final_answer,
            "normalized_answer": self.normalized_answer,
            "success": self.success,
            "error": self.error,
        }


# ======================== Main Pipeline ========================

class MultimodalRetrieverPipeline:
    """
    End-to-end Multimodal Retriever Pipeline for Document VQA.
    
    Integrates 5 stages:
        1. Multimodal Classifier: Classify document/question types
        2. Caption Generator: Generate image captions
        3. Multimodal Retriever: Retrieve similar examples via BiEncoder + CrossEncoder
        4. CoT Prompt Builder: Build chain-of-thought prompts with retrieved examples
        5. MLLM Inference: Generate final answer using a multimodal LLM
    """
    
    def __init__(
        self,
        classifier: Optional[CLIPMultimodalInference] = None,
        retriever: Optional[MultimodalRetriever] = None,
        mllm: Optional[MLLMInference] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """
        Initialize the pipeline with pre-loaded components.
        
        Args:
            classifier: CLIP multimodal classifier (Stage 1)
            retriever: Multimodal retriever with loaded index (Stage 3)
            mllm: MLLM inference engine (Stage 5)
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Store components
        self.classifier = classifier
        self.retriever = retriever
        self.mllm = mllm
        
        # Log initialization status
        print("\n" + "=" * 70)
        print("  Multimodal Retriever Pipeline Initialized")
        print("=" * 70)
        print(f"  Device: {self.config.device}")
        print(f"  Stage 1 (Classifier):  {'loaded' if classifier else 'not loaded'}")
        print(f"  Stage 2 (Caption):     {'enabled' if self.config.generate_caption else 'disabled'}")
        print(f"  Stage 3 (Retriever):   {'loaded' if retriever else 'not loaded'}")
        print(f"  Stage 4 (CoT Builder): available")
        print(f"  Stage 5 (MLLM):        {'loaded' if mllm else 'not loaded'}")
        print("=" * 70 + "\n")
    
    @classmethod
    def from_checkpoints(
        cls,
        classifier_ckpt: Optional[str] = None,
        biencoder_ckpt: Optional[str] = None,
        crossencoder_ckpt: Optional[str] = None,
        index_path: Optional[str] = None,
        mllm_model_name: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ) -> "MultimodalRetrieverPipeline":
        """
        Create pipeline from saved checkpoints.
        
        Args:
            classifier_ckpt: Path to classifier checkpoint (.pt file)
            biencoder_ckpt: Path to BiEncoder checkpoint (.pt file)
            crossencoder_ckpt: Path to CrossEncoder checkpoint (optional)
            index_path: Path to FAISS index (prefix without extension)
            mllm_model_name: MLLM model name (default from config)
            config: Pipeline configuration
            
        Returns:
            Initialized pipeline
        """
        config = config or PipelineConfig()
        
        # Stage 1: Load Classifier
        classifier = None
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            print(f"\n[Stage 1] Loading Classifier from: {classifier_ckpt}")
            classifier = CLIPMultimodalInference(
                checkpoint_path=classifier_ckpt,
                threshold=config.classifier_threshold,
                device=config.device,
            )
        else:
            print("[Stage 1] Classifier not loaded (checkpoint not provided)")
        
        # Stage 3: Load Retriever
        retriever = None
        if biencoder_ckpt and os.path.exists(biencoder_ckpt):
            print(f"\n[Stage 3] Loading Retriever...")
            retriever = MultimodalRetriever.from_checkpoints(
                biencoder_ckpt=biencoder_ckpt,
                crossencoder_ckpt=crossencoder_ckpt if crossencoder_ckpt and os.path.exists(crossencoder_ckpt) else None,
                device=config.device,
            )
            
            # Load FAISS index if provided
            if index_path:
                print(f"  Loading FAISS index from: {index_path}")
                retriever.load_index(index_path)
        else:
            print("[Stage 3] Retriever not loaded (checkpoint not provided)")
        
        # Stage 5: Load MLLM
        mllm = None
        try:
            print(f"\n[Stage 5] Loading MLLM...")
            mllm = MLLMInference(
                model_name=mllm_model_name,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                device=config.device,
            ) if mllm_model_name else MLLMInference(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
        except Exception as e:
            print(f"[Stage 5] MLLM not loaded: {e}")
        
        return cls(
            classifier=classifier,
            retriever=retriever,
            mllm=mllm,
            config=config,
        )
    
    def _stage1_classify(
        self,
        image_path: Union[str, Path],
        question: str,
        result: PipelineResult,
    ) -> None:
        """Stage 1: Classify document/question type."""
        if self.classifier is None:
            result.predicted_labels = []
            return
        
        try:
            pred = self.classifier.predict(
                image=image_path,
                question=question,
                return_probabilities=True,
            )
            result.predicted_labels = pred.get("predicted_labels", [])
            result.label_probabilities = pred.get("probabilities", {})
            print(f"  [Stage 1] Predicted labels: {result.predicted_labels}")
        except Exception as e:
            result.predicted_labels = []
            print(f"  [Stage 1] Classification failed: {e}")
    
    def _stage2_caption(
        self,
        image_path: Union[str, Path],
        result: PipelineResult,
    ) -> None:
        """Stage 2: Generate image caption."""
        if not self.config.generate_caption:
            result.image_caption = ""
            result.caption_success = False
            return
        
        try:
            caption_result = caption_generator(Path(image_path))
            result.image_caption = caption_result.get("image_caption", "")
            result.caption_success = caption_result.get("success", False)
            
            if result.caption_success:
                print(f"  [Stage 2] Caption generated: {len(result.image_caption)} chars")
            else:
                print(f"  [Stage 2] Caption generation failed: {caption_result.get('error', '')}")
        except Exception as e:
            result.image_caption = ""
            result.caption_success = False
            print(f"  [Stage 2] Caption generation failed: {e}")
    
    def _stage3_retrieve(
        self,
        image_path: Union[str, Path],
        question: str,
        image_caption: str,
        result: PipelineResult,
    ) -> None:
        """Stage 3: Retrieve similar examples."""
        if self.retriever is None:
            result.retrieved_examples = []
            result.retrieval_scores = []
            return
        
        try:
            # Build query
            query = {
                "image": str(image_path),
                "question": question,
                "image_caption": image_caption,
            }
            
            # Retrieve
            retrieved = self.retriever.retrieve(
                query=query,
                top_k=self.config.top_k,
                first_stage_k=self.config.first_stage_k,
                query_view=self.config.query_view,
                use_reranking=self.config.use_reranking,
            )
            
            result.retrieved_examples = [doc for doc, _ in retrieved]
            result.retrieval_scores = [float(score) for _, score in retrieved]
            print(f"  [Stage 3] Retrieved {len(result.retrieved_examples)} examples")
        except Exception as e:
            result.retrieved_examples = []
            result.retrieval_scores = []
            print(f"  [Stage 3] Retrieval failed: {e}")
    
    def _stage4_build_prompt(
        self,
        query_example: Dict[str, Any],
        retrieved_examples: List[Dict],
        result: PipelineResult,
    ) -> Tuple[str, List[Image.Image]]:
        """Stage 4: Build Chain-of-Thought prompt."""
        try:
            builder = CoTPromptBuilder(
                query_ex=query_example,
                retrieved_examples=retrieved_examples,
                max_examples=self.config.max_examples,
            )
            
            prompt_text, prompt_images = builder.build()
            result.prompt_text = prompt_text
            result.prompt_length = builder.length()
            
            print(f"  [Stage 4] Built prompt: {result.prompt_length} chars, {len(prompt_images)} images")
            return prompt_text, prompt_images
        except Exception as e:
            result.prompt_text = ""
            result.prompt_length = 0
            print(f"  [Stage 4] Prompt building failed: {e}")
            return "", []
    
    def _stage5_generate(
        self,
        prompt_text: str,
        prompt_images: List[Image.Image],
        result: PipelineResult,
    ) -> None:
        """Stage 5: Generate final answer using MLLM."""
        if self.mllm is None:
            result.raw_output = ""
            result.final_answer = ""
            result.normalized_answer = ""
            return
        
        if not prompt_text or not prompt_images:
            result.raw_output = ""
            result.final_answer = ""
            result.normalized_answer = ""
            print("  [Stage 5] Skipped (empty prompt)")
            return
        
        try:
            # Build messages
            messages = self.mllm.build_messages(prompt_text, prompt_images)
            
            # Prepare inputs
            inputs = self.mllm.prepare_inputs(messages)
            
            # Generate response
            if self.config.num_samples > 1:
                # Self-consistency: multiple samples + majority voting
                answers = []
                for _ in range(self.config.num_samples):
                    raw = self.mllm.single_generate(inputs)
                    extracted = extract_final_answer(raw)
                    answers.append(normalize_answer(extracted))
                
                result.normalized_answer = majority_vote(answers)
                result.final_answer = result.normalized_answer
                result.raw_output = raw  # Last raw output
            else:
                # Single sample
                result.raw_output = self.mllm.single_generate(inputs)
                result.final_answer = extract_final_answer(result.raw_output)
                result.normalized_answer = normalize_answer(result.final_answer)
            
            print(f"  [Stage 5] Generated answer: {result.final_answer[:100]}...")
        except Exception as e:
            result.raw_output = ""
            result.final_answer = ""
            result.normalized_answer = ""
            result.success = False
            result.error = str(e)
            print(f"  [Stage 5] Generation failed: {e}")
    
    def run(
        self,
        image_path: Union[str, Path],
        question: str,
        existing_caption: Optional[str] = None,
        skip_stages: Optional[List[int]] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline on a single query.
        
        Args:
            image_path: Path to query image
            question: Query question
            existing_caption: Pre-computed caption (skip Stage 2 if provided)
            skip_stages: List of stage numbers to skip (1-5)
            
        Returns:
            PipelineResult with all outputs
        """
        skip_stages = skip_stages or []
        
        print(f"\n{'='*70}")
        print(f"  Running Pipeline")
        print(f"  Image: {image_path}")
        print(f"  Question: {question}")
        print(f"{'='*70}")
        
        # Initialize result
        result = PipelineResult(
            image_path=str(image_path),
            question=question,
        )
        
        try:
            # Stage 1: Classification
            if 1 not in skip_stages:
                self._stage1_classify(image_path, question, result)
            
            # Stage 2: Caption Generation
            if existing_caption:
                result.image_caption = existing_caption
                result.caption_success = True
                print(f"  [Stage 2] Using provided caption: {len(existing_caption)} chars")
            elif 2 not in skip_stages:
                self._stage2_caption(image_path, result)
            
            # Stage 3: Retrieval
            if 3 not in skip_stages:
                self._stage3_retrieve(
                    image_path, question, result.image_caption, result
                )
            
            # Stage 4: Build CoT Prompt
            query_example = {
                "image": str(image_path),
                "question": question,
                "image_caption": result.image_caption,
                "question_types": result.predicted_labels or [],
            }
            
            prompt_text, prompt_images = "", []
            if 4 not in skip_stages and result.retrieved_examples:
                prompt_text, prompt_images = self._stage4_build_prompt(
                    query_example, result.retrieved_examples, result
                )
            
            # Stage 5: MLLM Inference
            if 5 not in skip_stages:
                self._stage5_generate(prompt_text, prompt_images, result)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            print(f"  Pipeline error: {e}")
        
        print(f"{'='*70}")
        print(f"  Pipeline Complete")
        print(f"  Final Answer: {result.final_answer}")
        print(f"{'='*70}\n")
        
        return result
    
    def batch_run(
        self,
        examples: List[Dict[str, Any]],
        skip_stages: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> List[PipelineResult]:
        """
        Run pipeline on multiple examples.
        
        Args:
            examples: List of example dictionaries with 'image' and 'question' keys
            skip_stages: Stages to skip
            show_progress: Show progress bar
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        iterator = tqdm(examples, desc="Running Pipeline") if show_progress else examples
        
        for ex in iterator:
            result = self.run(
                image_path=ex["image"],
                question=ex["question"],
                existing_caption=ex.get("image_caption"),
                skip_stages=skip_stages,
            )
            results.append(result)
        
        return results
    
    def evaluate(
        self,
        examples: List[Dict[str, Any]],
        skip_stages: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate pipeline on examples with ground truth answers.
        
        Args:
            examples: List with 'image', 'question', and 'answers' keys
            skip_stages: Stages to skip
            
        Returns:
            Evaluation metrics (accuracy, ANLS, etc.)
        """
        results = self.batch_run(examples, skip_stages=skip_stages)
        
        correct = 0
        total = len(examples)
        
        for ex, result in zip(examples, results):
            gold_answers = ex.get("answers", [])
            if not gold_answers:
                continue
            
            # Normalize all answers for comparison
            normalized_golds = [normalize_answer(ans) for ans in gold_answers]
            pred = result.normalized_answer
            
            if pred in normalized_golds:
                correct += 1
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }


# ======================== Main Entry Point ========================

def demo():
    """Demonstration of the pipeline."""
    print("\n" + "=" * 70)
    print("  Multimodal Retriever Pipeline Demo")
    print("=" * 70)
    
    # Configuration
    config = PipelineConfig(
        top_k=3,
        max_examples=3,
        generate_caption=True,
        use_reranking=True,
    )
    
    # Initialize pipeline
    # Checkpoint paths from trained models
    classifier_ckpt = "checkpoints/clip_classifier/last_model.pt"
    biencoder_ckpt = "multimodal_biencoder_checkpoints/best_model.pt"
    crossencoder_ckpt = "multimodal_crossencoder_checkpoints/best_model.pt"
    index_path = "multimodal_biencoder_checkpoints/index_epoch20"
    
    # Check if checkpoints exist
    missing_ckpts = []
    if not os.path.exists(classifier_ckpt):
        missing_ckpts.append(f"Classifier: {classifier_ckpt}")
    if not os.path.exists(biencoder_ckpt):
        missing_ckpts.append(f"BiEncoder: {biencoder_ckpt}")
    
    if missing_ckpts:
        print("\n⚠ Missing checkpoints:")
        for ckpt in missing_ckpts:
            print(f"  - {ckpt}")
        print("\nTo use the pipeline, train the required models first.")
        print("See the TRAINING_GUIDE.md in each stage folder.")
        return
    
    # Initialize
    pipeline = MultimodalRetrieverPipeline.from_checkpoints(
        classifier_ckpt=classifier_ckpt,
        biencoder_ckpt=biencoder_ckpt,
        crossencoder_ckpt=crossencoder_ckpt,
        index_path=index_path,
        config=config,
    )
    
    # Example query
    result = pipeline.run(
        image_path="data/spdocvqa_images/ffbf0023_4.png",
        question="What is the total amount on this invoice?",
    )
    
    # Print result
    print("\nPipeline Result:")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    demo()

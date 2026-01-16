import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

from PIL import Image
import json
from typing import List, Dict, Any, Optional
from utils.helpers import (
    load_image,
    build_example_template, 
    build_query_template
)

class CoTPromptBuilder:
    def __init__(
        self,
        query_ex: Dict[str, Any],
        retrieved_examples: List[Dict[str, Any]],
        max_examples: int=5,
    ) -> None:
        self.query_ex = query_ex
        self.retrieved_examples = retrieved_examples
        self.max_examples = max_examples
        self.full_prompt = ""
                
    def build(self):
        prompt_blocks = []
        prompt_images = []
        
        for idx, ex in enumerate(self.retrieved_examples[:self.max_examples], start=1):
            # Load image
            img = Image.open(ex["image"]).convert("RGB")
            prompt_images.append(img)

            # Build text block
            block_text = build_example_template(ex, idx)
            prompt_blocks.append(block_text)
        
        query_image_slot = len(prompt_images) + 1

        q_img = Image.open(self.query_ex["image"]).convert("RGB")
        prompt_images.append(q_img)

        query_text = build_query_template(self.query_ex, idx=query_image_slot)

        # ----- Build final prompt text -----
        self.full_prompt = (
            "\n".join(prompt_blocks)
            + "\nNow answer the query example:\n"
            + query_text
        )
        
        return self.full_prompt, prompt_images
        
    def length(self):
        return len(self.full_prompt)
    
if __name__ == "__main__":
    ex = {
        "questionId": 337,
        "question": "what is the date mentioned in this letter?",
        "question_types": [
            "handwritten",
            "form"
        ],
        "image": "data/spdocvqa_images/xnbl0037_1.png",
        "docId": 279,
        "ucsf_document_id": "xnbl0037",
        "ucsf_document_page_no": "1",
        "answers": [
            "1/8/93"
        ],
        "answer_explanation": "Step 1: The date is mentioned in the 'DATE:' field at the top-left of the document image, which reads '1/8/93'. This is the date associated with the document.\nStep 2: The same date '1/8/93' is also found in the 'Date' column of the routing list, next to 'Peggy Carter', reinforcing that '1/8/93' is the date mentioned in the letter.",
        "image_caption": "A confidential document with a routing list, dated 11/8/93, with subject 'Rj gdfs'. The list includes names, initials, and dates. It is marked 'RJRT PR APPROVAL' at the top and has a return address to Peggy Carter, PR, 16 Reynolds Building."
    }
    cot_prompt_builder = CoTPromptBuilder(query_ex=ex, retrieved_examples=[ex, ex, ex])
    print(*cot_prompt_builder.build(), end="\n\n")
    print(cot_prompt_builder.length())
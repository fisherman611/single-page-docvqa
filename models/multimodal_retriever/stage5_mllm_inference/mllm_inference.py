import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")
sys.path.append(PROJECT_ROOT / "models/multimodal_retriever/stage4_cot_builder")

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
from typing import List, Dict, Any, Optional
from utils.helpers import extract_final_answer, normalize_answer, majority_vote
from dotenv import load_dotenv
from models.multimodal_retriever.stage4_cot_builder.cot_prompt_builder import CoTPromptBuilder

load_dotenv()

### Login to Hugging Face Hub
from huggingface_hub import login

login(token=os.getenv("HF_READ_TOKEN"))

with open("models/multimodal_retriever/stage5_mllm_inference/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
with open("models/multimodal_retriever/stage5_mllm_inference/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

MAX_INPUT_TOKENS = config["max_input_tokens"]
MAX_NEW_TOKENS = config["max_new_tokens"]
TEMPERATURE = config["temperature"]

class MLLMInference:
    def __init__(
        self,
        model_name: str=MODEL_NAME,
        system_prompt: str=SYSTEM_PROMPT,
        device: str = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_new_tokens: int = MAX_NEW_TOKENS,   
        temperature: float = TEMPERATURE,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        ).eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = False

        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def build_messages(self, prompt_text, images) -> List[Dict[str, Any]]:
        user_content = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return messages
        
    
    def prepare_inputs(self, messages):
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        print("max_input_tokens:", self.max_input_tokens)
        print("text_prompt chars:", len(text_prompt))
        print("input_ids shape:", inputs["input_ids"].shape)

        return inputs

    def single_generate(self, inputs) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens
            )

        trimmed_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        pred = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return pred
    
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
            "image_caption": "A confidential document with a routing list, dated 1/8/93, with subject 'Rj gdfs'. The list includes names, initials, and dates. It is marked 'RJRT PR APPROVAL' at the top and has a return address to Peggy Carter, PR, 16 Reynolds Building."
        }
    
    retrieved_examples = [
        {
            "questionId": 338,
            "question": "what is the contact person name mentioned in letter?",
            "question_types": [
                "handwritten",
                "form"
            ],
            "image": "data/spdocvqa_images/xnbl0037_1.png",
            "docId": 279,
            "ucsf_document_id": "xnbl0037",
            "ucsf_document_page_no": "1",
            "answers": [
                "P. Carter",
                "p. carter"
            ],
            "answer_explanation": "Step 1: The document image contains a section labeled 'CONTACT:' with the text 'P. CARTER' written next to it, located in the middle-left part of the document. This indicates that P. Carter is the contact person mentioned in the letter.\nStep 2: The name 'Peggy Carter' is listed under the 'ROUTE TO:' section, found below the 'CONTACT:' section, further confirming that P. Carter refers to Peggy Carter.",
            "image_caption": "A confidential document with a routing list, dated 11/8/93, with subject 'Rj gdfs'. The list includes names, initials, and dates. It is marked 'RJRT PR APPROVAL' at the top and has a return address to Peggy Carter, PR, 16 Reynolds Building."
        },
        {
            "questionId": 339,
            "question": "Which corporation's letterhead is this?",
            "question_types": [
                "layout"
            ],
            "image": "data/spdocvqa_images/mxcj0037_1.png",
            "docId": 280,
            "ucsf_document_id": "mxcj0037",
            "ucsf_document_page_no": "1",
            "answers": [
                "Brown & Williamson Tobacco Corporation"
            ],
            "answer_explanation": "Step 1: The document image displays a letterhead with the text 'BROWN & WILLIAMSON TOBACCO CORPORATION' at the top-center. This text is clearly visible and indicates the corporation associated with the document.\nStep 2: The presence of 'BROWN & WILLIAMSON TOBACCO CORPORATION' on the letterhead directly answers the question about which corporation's letterhead this is, as it is the name of the corporation printed on the document.",
            "image_caption": "A memo from Brown & Williamson Tobacco Corporation Research & Development, dated May 8, 1995, titled 'Review of Existing Brainstorming Ideas/483'. The document includes sections for 'TO', 'CC', 'FROM', 'DATE', and 'SUBJECT', followed by a detailed discussion on novel cigarette constructions and product innovation ideas."
        }
    ]
    cot_prompt_builder = CoTPromptBuilder(query_ex=ex, retrieved_examples=retrieved_examples)
    prompt_text, images = cot_prompt_builder.build()
    
    mllm_inference = MLLMInference()
    messages = mllm_inference.build_messages(prompt_text=prompt_text, images=images)
    print("Done step 1")
    inputs = mllm_inference.prepare_inputs(messages)
    print("Done step 2")
    result = mllm_inference.single_generate(inputs)
    print(result)
    


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
from utils.helper import extract_final_answer, normalize_answer, majority_vote
from dotenv import load_dotenv
from models.multimodal_retriever.stage4_cot_builder.cot_prompt_builder import CoTPromptBuilder

load_dotenv()

### Login to Hugging Face Hub
from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

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
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        ).to(self.device)

        print("Input tokens:", int(inputs["input_ids"].shape[-1]))
        
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
        "image": "data\\spdocvqa_images\\xnbl0037_1.png",
        "docId": 279,
        "ucsf_document_id": "xnbl0037",
        "ucsf_document_page_no": "1",
        "ocr": "xnbl0037_1.json",
        "answers": [
            "1/8/93"
        ],
        "image_description": "The document is a form or letter titled \"Confidential\" and \"RJRT PR APPROVAL\" at the top. It contains several structured fields in the top-left section, including 'DATE:', 'SUBJECT:', 'PROPOSED RELEASE DATE:', 'FOR RELEASE TO:', and 'CONTACT:'. A handwritten date, '1/8/93', is visible next to the 'DATE:' label. Further down, there is a list of names under a 'ROUTE TO:' section, and a table-like structure on the right with headers 'Initials' and 'Date', under which '1/8/93' is also visibly written.",
        "answer_explanation": "Step 1: Located the label 'DATE :' with bounding box [254,295,343,294,344,322,255,323] in the top-left section of the document. Step 2: Identified the date value next to this label. The OCR text for this value is '1/8/13' with bounding box [396,262,561,262,559,333,398,327]. Visually, the date written is '1/8/93'.",
        "reasoning_type": "single-hop"
    }
    
    retrieved_examples = [
        {
        "questionId": 338,
        "question": "what is the contact person name mentioned in letter?",
        "question_types": [
            "handwritten",
            "form"
        ],
        "image": "data\\spdocvqa_images\\xnbl0037_1.png",
        "docId": 279,
        "ucsf_document_id": "xnbl0037",
        "ucsf_document_page_no": "1",
        "ocr": "xnbl0037_1.json",
        "answers": [
            "P. Carter",
            "p. carter"
        ],
        "image_description": "The document is a memo or internal form with a 'Confidential' label at the top. It has several fields arranged vertically on the left side, including 'DATE:', 'PROPOSED RELEASE DATE:', 'FOR RELEASE TO:', and 'CONTACT:'. The 'CONTACT:' field is located in the middle-left section of the document. Below these fields, there is a section titled 'ROUTE TO' with a list of names. A 'Return to' instruction is present towards the bottom-middle.",
        "answer_explanation": "Step 1: Locate the label \"CONTACT:\" in the middle-left section of the document with bounding box [252,529,411,530,410,565,251,564]. Step 2: Identify the text immediately to the right of this label. The text is \"P. CARTER\" with bounding box [429,521,663,511,666,568,432,578].",
        "reasoning_type": "single-hop"
    },
    {
        "questionId": 339,
        "question": "Which corporation's letterhead is this?",
        "question_types": [
            "layout"
        ],
        "image": "data\\spdocvqa_images\\mxcj0037_1.png",
        "docId": 280,
        "ucsf_document_id": "mxcj0037",
        "ucsf_document_page_no": "1",
        "ocr": "mxcj0037_1.json",
        "answers": [
            "Brown & Williamson Tobacco Corporation"
        ],
        "image_description": "",
        "answer_explanation": "",
        "reasoning_type": ""
    }
    ]
    cot_prompt_builder = CoTPromptBuilder(query_ex=ex, retrieved_examples=retrieved_examples, ocr_root="data/spdocvqa_ocr")
    prompt_text, images = cot_prompt_builder.build()
    
    mllm_inference = MLLMInference()
    messages = mllm_inference.build_messages(prompt_text=prompt_text, images=images)
    print("Done step 1")
    inputs = mllm_inference.prepare_inputs(messages)
    print("Done step 2")
    result = mllm_inference.single_generate(inputs)
    print(result)
    


"""Model loading and generation utilities."""

import logging
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from typing import List
from vllm import LLM, SamplingParams

from utils.config import ModelConfig

logger = logging.getLogger(__name__)


class QwenVLModel:
    """Processor for loading models and generating answers."""
    
    def __init__(self, model_config: ModelConfig, device: str, system_prompt: str = None):
        """
        Initialize the image processor.
        
        Args:
            model_config: Model configuration
            device: Device to use (e.g., "cuda:0")
        """
        self.device = device
        self.system_prompt = system_prompt
        self.model_config = model_config
        self.model = self._load_model()
        self.model.eval()
        self.processor = self._load_processor()


    def _load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        """Load the model."""
        try:
            # support multi-GPU via device_map
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_config.model_name,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _load_processor(self) -> AutoProcessor:
        """Load the processor."""
        try:
            return AutoProcessor.from_pretrained(self.model_config.processor_name)
        except Exception as e:
            logger.error(f"Failed to load processor: {str(e)}")
            raise

    def generate_answer(self, image_url: str, instruction: str, n: int = 1) -> List[str]:
        """
        Generate one or multiple answers for the given image and instruction.
        
        Args:
            image_url: URL or path to the image, or list of images
            instruction: The instruction/question text
            n: Number of answers to generate (default: 1)
            
        Returns:
            List of generated answer strings
        """

        if not self.system_prompt:
            SYSTEM_PROMPT = """You are a helpful AI Assistant."""
        else:
            SYSTEM_PROMPT = self.system_prompt
        

        if isinstance(image_url, list):
            content = []
            content += [
                {"type": "image", "image": url} for url in image_url
            ]
            content += [{"type": "text", "text": instruction + "\n\nYour final answer MUST BE put in \\boxed{}"}]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "text", "text": instruction + "\n\nYour final answer MUST BE put in \\boxed{}"},
                    ],
                }
            ]
        # batch inference
        messages_batch = [messages] * 1
        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)


        # Repeat the same text n times
        # Process batch inputs - processor will handle the batch
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Use torch.no_grad() to disable gradient computation and save memory
        with torch.no_grad():
            # Generate all n answers in one batch (this is the main performance gain)
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=self.model_config.max_new_tokens,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
                temperature=self.model_config.temperature,
                repetition_penalty=self.model_config.repetition_penalty,
                num_return_sequences=n,
            )
            generated_ids_trimmed1 = [
                out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids
            ]
            decoded_texts = self.processor.batch_decode(
                generated_ids_trimmed1,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        
        return decoded_texts if isinstance(decoded_texts, list) else [decoded_texts]

        # # Trim input tokens from output
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
    
        # # Decode all generated texts
        # decoded_texts = self.processor.batch_decode(
        #     generated_ids_trimmed,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False
        # )
        
        # # return decoded_texts if isinstance(decoded_texts, list) else [decoded_texts]
        # return self.processor.batch_decode(
        #     generated_ids_trimmed1,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False
        # )


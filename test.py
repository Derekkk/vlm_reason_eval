import os
import logging
import argparse
import torch
from tqdm import tqdm
from mathruler.grader import extract_boxed_content

from utils.config import DatasetType, ModelConfig, get_dataset_config
from utils.data_utils import load_image_dataset, save_descriptions, process_response, format_instruction
from utils.model_utils import QwenVLModel
from utils.metric_utils import evaluate_single_answer, calculate_accuracy_metrics


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

SYSTEM_PROMPT = """You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""

if __name__ == "__main__":
    model_path = ""
    model_config = ModelConfig(
        model_name=model_path,
        processor_name=model_path
    )

    model = QwenVLModel(model_config, use_vllm=False, device="cuda", system_prompt=SYSTEM_PROMPT)

    # Load dataset
    dataset_config = get_dataset_config(DatasetType.EMMA_CHEM)
    data = load_image_dataset(dataset_config)
    data = data[1:2]

    print(data)
    # Process dataset
    for item in data:
        formatted_instruction = format_instruction(item['instruction'], item.get('options'))
        print(formatted_instruction)
        answer_list = model.generate_answer(item['image_url'], formatted_instruction, n=3)
        print(answer_list)
        print("--------------------------------")

        processed_response = process_response(
            item['response'],
            item.get('choices'),
            item.get('options')
        )

        # Evaluate each answer
        correct_answers = []
        all_answers = []
        all_reasonings = []
        
        for answer_text in answer_list:
            if not answer_text:
                all_answers.append("Failed to generate.")
                all_reasonings.append("")
                correct_answers.append(False)
                continue
            
            # Extract reasoning
            if "<think>" in answer_text:
                reasoning = answer_text.split("<think>")[-1].split("</think>")[0].strip()
            else:
                reasoning = answer_text
            
            # Extract answer
            if "</answer>" in answer_text:
                answer = answer_text.split("<answer>")[-1].split("</answer>")[0].strip()
            else:
                answer = extract_boxed_content(answer_text)
                if not answer:
                    answer = answer_text
            
            all_answers.append(answer)
            all_reasonings.append(reasoning)
            
            # Check if this answer is correct
            is_correct = evaluate_single_answer(answer_text, processed_response, None)
            correct_answers.append(is_correct)
        
        print(correct_answers)
        print(all_answers)

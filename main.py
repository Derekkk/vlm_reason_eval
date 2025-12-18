"""Main evaluation script for Qwen model on various math datasets."""

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_single_dataset(
    dataset_type: DatasetType,
    model_config: ModelConfig,
    processor: QwenVLModel,
    data_subset: int = -1,
    n: int = 1,
    gen_times: int = 1
) -> dict:
    """
    Evaluate a single dataset and return results.
    
    Args:
        dataset_type: The dataset type to evaluate
        model_config: Model configuration
        device: Device to use
        processor: Image processor instance
        n: Number of answers to generate per question (default: 1)
        gen_times: Number of times to generate answers per question (default: 1)
            total_answers = n * gen_times, this is for GPU memory consideration
        
    Returns:
        Dictionary containing evaluation results
    """
    dataset_config = get_dataset_config(dataset_type)
    output_file = f"./outputs/{dataset_type.value}_{model_config.model_name.split('/')[-1]}.json"
    
    # Load dataset
    logger.info(f"Loading dataset {dataset_config.name}")
    data = load_image_dataset(dataset_config)
    
    # Limit to first data_subset samples
    if data_subset!= -1 and len(data) > data_subset:
        logger.info(f"Limiting dataset to first {data_subset} samples (total: {len(data)})")
        data = data[:data_subset]
    
    descriptions = []
    all_correct_answers = []  # Store all correct_answers lists for metric calculation
    
    # For SFTSEED dataset, track accuracy per source
    source_correct = {}
    source_total = {}

    # Process each image
    for i, item in tqdm(enumerate(data), total=len(data), desc=f"Processing {dataset_type.value}"):
        # Format instruction based on dataset type
        if dataset_type in [DatasetType.MATHVISION, DatasetType.EMMA_MATH, DatasetType.EMMA_CHEM,
                           DatasetType.EMMA_CODE, DatasetType.EMMA_PHYSICS, DatasetType.MMMU_PRO_10,
                           DatasetType.MMMU_PRO_4]:
            formatted_instruction = format_instruction(item['instruction'], item.get('options'))
        elif dataset_type == DatasetType.HALLUSIONBENCH:
            formatted_instruction = format_instruction(item['instruction'], yes_no=True)
        elif dataset_type == DatasetType.MMMU_PRO_VISION:
            formatted_instruction = format_instruction(item['instruction'], item.get('options'), vision=True)
        else:
            formatted_instruction = item['instruction']
        if i == 0:
            logger.info(f"item {item}")
            logger.info(f"formatted_instruction {formatted_instruction}\n")
        # Generate n answers
        answer_list = []
        for _ in range(gen_times):
            cur_answer_list = processor.generate_answer(item['image_url'], formatted_instruction, n=n)
            answer_list.extend(cur_answer_list)

        # Process ground truth response
        if dataset_type in [DatasetType.MMMU_PRO_VISION, DatasetType.MMMU_PRO_4, DatasetType.MMMU_PRO_10]:
            processed_response = item['response']
        else:
            processed_response = process_response(
                item['response'],
                item.get('choices'),
                item.get('options')
            )
        if dataset_type == DatasetType.HALLUSIONBENCH:
            processed_response = "Yes" if processed_response == "1" else "No"
        
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
            is_correct = evaluate_single_answer(answer_text, processed_response, dataset_type)
            correct_answers.append(is_correct) # [True, True, True]
        
        # logger.info(f"answer_list {all_answers}\n")
        # Store correct_answers for metric calculation
        all_correct_answers.append(correct_answers) 
        
        # Use the first answer for accuracy calculation (backward compatibility)
        correct_flag = 1 if correct_answers[0] else 0
        if correct_flag:
            if dataset_type == DatasetType.SFTSEED and 'source' in item:
                source = item['source']
                if source not in source_correct:
                    source_correct[source] = 0
                    source_total[source] = 0
                source_correct[source] += 1

        if dataset_type == DatasetType.SFTSEED and 'source' in item:
            source = item['source']
            if source not in source_total:
                source_total[source] = 0
            source_total[source] += 1

        description = {
            'instruction': item['instruction'],
            'response': item['response'],
            'reasoning': all_reasonings[0] if all_reasonings else "",
            'answer': all_answers[0] if all_answers else "Failed to generate.",
            'correct': correct_flag,
            'n_answers': len(answer_list),
            'all_answers': all_answers,
            'all_reasonings': all_reasonings,
            'all_correct': correct_answers
        }
        
        if dataset_type == DatasetType.SFTSEED and 'source' in item:
            description['source'] = item['source']
            
        descriptions.append(description)
        
        # Save periodically
        if (i + 1) % 10 == 0:
            save_descriptions(descriptions, output_file)
    
    # Final save
    save_descriptions(descriptions, output_file)
    
    # Calculate metrics using the metric_utils functions
    metrics = calculate_accuracy_metrics(all_correct_answers, n)
    
    accuracy = metrics['first_answer_accuracy']
    mean_accuracy = metrics['mean_accuracy']
    correct = metrics['correct']
    total = metrics['total']
    pass_at_k_accuracy = metrics['pass_at_k_accuracy']
    majority_vote_accuracy = metrics['majority_vote_accuracy']
    
    logger.info(f"Completed {dataset_type.value}!")
    logger.info(f"All metrics: {metrics}")
    logger.info(f"===" * 10)

    result = {
        'dataset': dataset_type.value,
        'first_answer_accuracy': accuracy,
        'correct': correct,
        'total': total,
        'mean_accuracy': mean_accuracy,
        'output_file': output_file,
        'pass_at_k': pass_at_k_accuracy,
        'majority_vote': majority_vote_accuracy
    }

    if dataset_type == DatasetType.SFTSEED and source_correct:
        logger.info("Accuracy per source:")
        source_accuracies = {}
        for source in sorted(source_correct.keys()):
            source_accuracy = source_correct[source] / source_total[source]
            source_accuracies[source] = {
                'accuracy': source_accuracy,
                'correct': source_correct[source],
                'total': source_total[source]
            }
            logger.info(f"  {source}: {source_accuracy:.4f} ({source_correct[source]}/{source_total[source]})")
        result['source_accuracies'] = source_accuracies

    return result


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate Qwen model on various math datasets')
    parser.add_argument('--model_path', type=str, help='Path to the model', 
                      default="/data3/huzhe/workspace/model_cards/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--dataset', type=str, nargs='+', 
                      choices=['mathvista', 'mathverse', 'mathvision', 'sftseed', 'hallusionbench',
                               'emma-math', 'emma-chem', 'emma-code', 'emma-physics',
                               'mmmu-pro-10', 'mmmu-pro-4', 'mmmu-pro-vision'],
                      default=['mathvista'], help='Dataset(s) to evaluate on (can specify multiple)')
    parser.add_argument('--n', type=int, default=10,
                       help='Number of answers to generate per question (for pass@k calculation)')
    parser.add_argument('--gen_times', type=int, default=1, help='Number of times to generate answers per question')
    parser.add_argument('--data_subset', type=int, default=100, help='Number of data samples to evaluate on')
    parser.add_argument('--is_reason', type=bool, default=True, help='Number of data samples to evaluate on')
    args = parser.parse_args()
    
    # Override args for testing (remove these lines in production)
    # args.dataset = ['mathvista', 'mathverse', 'mathvision', 'hallusionbench', 'emma-math', 'emma-chem', 'mmmu-pro-vision', 'emma-physics', 'mmmu-pro-10', 'mmmu-pro-4']
    # args.dataset = ['mathvista']

    # args.model_path = "/data3/huzhe/workspace/evaluations/models/qwen25_vl_7b_guru_mixed_grpo_run4_step150"
    # args.model_path = "/data3/huzhe/workspace/evaluations/models/qwen25_vl_7b_guru_mixed_dapo_run2_step110"

    if args.is_reason:
        args.SYSTEM_PROMPT = """You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""
    else:
        args.SYSTEM_PROMPT = None
    
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"args: {args}")
    logger.info(f"Using device: {device}")
    logger.info(f"Generating {args.n} answer(s) per question")

    # Configuration
    model_config = ModelConfig(
        model_name=args.model_path,
        processor_name=args.model_path
    )
    
    # Initialize processor and model (shared across all datasets)
    logger.info(f"Loading model {model_config.model_name}")
    processor = QwenVLModel(
        model_config, 
        device="cuda", 
        system_prompt=args.SYSTEM_PROMPT,
        use_vllm=True
    )

    # Process each dataset
    all_results = []
    for dataset_name in args.dataset:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        dataset_type = DatasetType(dataset_name)

        result = evaluate_single_dataset(dataset_type, model_config, processor, data_subset=args.data_subset, n=args.n, gen_times=args.gen_times)
        all_results.append(result)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY OF ALL DATASETS")
    logger.info(f"{'='*60}")
    for result in all_results:
        if 'error' in result:
            logger.info(f"{result['dataset']}: ERROR - {result['error']}")
        else:
            logger.info(f"{result['dataset']}: First Answer Accuracy: {result['first_answer_accuracy']:.4f} ({result['correct']}/{result['total']})")
            if 'mean_accuracy' in result:
                logger.info(f" Mean Accuracy: {result['mean_accuracy']:.4f}")
            if 'pass_at_k' in result:
                logger.info(f" pass@{args.n}: {result['pass_at_k']:.4f}")
            if 'majority_vote' in result:
                logger.info(f" Majority Vote Accuracy: {result['majority_vote']:.4f}")
    
    # Calculate overall statistics if all datasets succeeded
    successful_results = [r for r in all_results if 'error' not in r]
    if successful_results:
        total_correct = sum(r['correct'] for r in successful_results)
        total_samples = sum(r['total'] for r in successful_results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        logger.info(f"\nOverall accuracy across all datasets: {overall_accuracy:.4f} ({total_correct}/{total_samples})")


if __name__ == "__main__":
    main()


"""Data loading and processing utilities."""

import logging
from typing import List, Dict, Optional
from datasets import load_dataset
from pathlib import Path
import json

from utils.config import DatasetConfig, DatasetType

logger = logging.getLogger(__name__)


def load_image_dataset(dataset_config: DatasetConfig) -> List[Dict]:
    """
    Load dataset from Hugging Face and extract image URLs and metadata.
    
    Args:
        dataset_config: Configuration for the dataset to load
        
    Returns:
        List of dataset items with image_url, instruction, and response fields
    """
    try:
        if dataset_config.subset:
            data = load_dataset(dataset_config.name, dataset_config.subset, split=dataset_config.split)
        else:
            data = load_dataset(dataset_config.name, split=dataset_config.split)
        items = []
        for item in data:
            if isinstance(dataset_config.image_field, list):
                dataset_item = {
                    'image_url': [item.get(x) for x in dataset_config.image_field if item.get(x) is not None],
                    'instruction': item.get(dataset_config.instruction_field, ''),
                    'response': item.get(dataset_config.response_field, ''),
                }
            else:
                dataset_item = {
                    'image_url': item[dataset_config.image_field],
                    'instruction': item.get(dataset_config.instruction_field, ''),
                    'response': item.get(dataset_config.response_field, ''),
                }
            if dataset_config.choices_field:
                dataset_item['choices'] = item.get(dataset_config.choices_field)
            if dataset_config.options_field:
                dataset_item['options'] = item.get(dataset_config.options_field, [])
            if dataset_config.source_field:
                dataset_item['source'] = item.get(dataset_config.source_field, '')
            items.append(dataset_item)
        return items
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


def save_descriptions(descriptions: List[Dict], output_file: str) -> None:
    """
    Save generated descriptions to a JSON file.
    
    Args:
        descriptions: List of description dictionaries to save
        output_file: Path to output JSON file
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
        logger.info(f"Saved {len(descriptions)} descriptions to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save descriptions: {str(e)}")
        raise


def process_response(response: str, choices: Optional[List[str]], options: Optional[List[str]] = None) -> str:
    """
    Process response to convert to option letter if applicable.
    
    Args:
        response: The response string
        choices: Optional list of choice strings
        options: Optional list of option strings
        
    Returns:
        Processed response (option letter if applicable, otherwise original response)
    """
    if choices is not None:
        try:
            response_index = choices.index(response)
            return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][response_index]
        except ValueError:
            pass
    if options is not None and len(options) > 0:
        try:
            response_index = options.index(response)
            return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][response_index]
        except ValueError:
            pass
    return response


def format_instruction(instruction: str, options: Optional[List[str]] = None, yes_no: bool = False, vision: bool = False) -> str:
    """
    Format instruction with optional choices/options.
    
    Args:
        instruction: The instruction/question text
        options: Optional list of options
        yes_no: Whether this is a yes/no question
        vision: Whether this is a vision-only question
        
    Returns:
        Formatted instruction string
    """
    options = eval(options) if isinstance(options, str) else options
    if vision:
        prompt_hint = "Hint: Please answer the question shown in the image."
        if options and len(options) > 0:
            prompt_hint += " Provide the correct option letter, e.g., A, B, C, D, E, at the end."
            choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
            return f"{prompt_hint}\nChoices:\n{choice_list}"
        return prompt_hint
    elif yes_no:
        prompt_hint = "Hint: Please answer the question requiring an answer of yes or no."
        return f"{prompt_hint}\nQuestion: {instruction}"
    elif options and len(options) > 0:
        prompt_hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end."
        choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"{prompt_hint}\nQuestion: {instruction}\nChoices:\n{choice_list}"
    else:
        prompt_hint = "Hint: Please answer the question requiring an answer."
        return f"{prompt_hint}\nQuestion: {instruction}"


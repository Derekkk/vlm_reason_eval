"""Metric calculation utilities for accuracy and pass@k."""

import logging
from typing import List, Dict
from mathruler.grader import grade_answer, extract_boxed_content

from utils.config import DatasetType

logger = logging.getLogger(__name__)


def evaluate_single_answer(answer_text: str, processed_response: str, dataset_type: DatasetType) -> bool:
    """
    Evaluate if a single answer is correct.
    
    Args:
        answer_text: The generated answer text
        processed_response: The ground truth response
        dataset_type: The dataset type
        
    Returns:
        True if the answer is correct, False otherwise
    """
    if not answer_text:
        return False
    
    # Extract answer from <answer> tags if present
    if "</answer>" in answer_text:
        answer = answer_text.split("<answer>")[-1].split("</answer>")[0].strip()
        if processed_response.lower() == answer.lower() or grade_answer(processed_response, answer):
            return True
    else:
        # Try to extract from boxed content
        answer = extract_boxed_content(answer_text)
        direct_answer = answer_text.lower().split("Answer:")[-1].strip() if "Answer:" in answer_text.lower() else ""
        
        if processed_response.lower() == answer.lower() or grade_answer(processed_response, answer):
            return True
        if direct_answer and (processed_response.lower() == direct_answer.lower() or grade_answer(processed_response, direct_answer)):
            return True
    
    return False


def calculate_pass_at_k(correct_answers: List[bool], k: int) -> bool:
    """
    Calculate pass@k for a single question.
    
    Pass@k means: generate k outputs for one input, and if at least one output is correct,
    consider this question as correct for pass@k.
    
    Args:
        correct_answers: List of boolean values indicating if each generated answer is correct
        k: Number of answers to consider (pass@k)
        
    Returns:
        True if at least one of the first k answers is correct, False otherwise
    """
    if k <= 0 or len(correct_answers) == 0:
        return False
    
    # Take the first k answers (or all if we have fewer than k)
    answers_to_check = correct_answers[:k]
    
    # Return True if at least one is correct
    return any(answers_to_check)


def calculate_majority_vote(correct_answers: List[bool]) -> bool:
    """
    Calculate majority vote for a single question.
    
    Returns:
        True if the majority of the answers are correct, False otherwise
    """
    return sum(correct_answers) > len(correct_answers) / 2


def calculate_accuracy_metrics(
    all_correct_answers: List[List[bool]],
    n: int
) -> Dict:
    """
    Calculate accuracy and pass@k metrics from all results.
    
    Args:
        all_correct_answers: List of lists, where each inner list contains boolean values
                           indicating correctness of generated answers for one question
        n: Maximum k value for pass@k calculation
        
    Returns:
        Dictionary containing accuracy and pass@k metrics
    """
    total_questions = len(all_correct_answers)
    
    if total_questions == 0:
        return {
            'first_answer_accuracy': 0.0,
            'mean_accuracy': 0,
            'correct': 0,
            'total': 0,
            'pass_at_k_accuracy': 0.0,
            'majority_vote_accuracy': 0.0
        }
    
    # Calculate standard accuracy (using first answer)
    correct_count = sum(1 for answers in all_correct_answers if answers and answers[0])
    accuracy = correct_count / total_questions

    # calculate mean accuracy
    all_answers = [x for sublist in all_correct_answers for x in sublist]
    mean_accuracy = sum(all_answers) / len(all_answers)
    
    # Calculate pass@k for each k from 1 to n
    pass_at_k_count = sum(
        1 for answers in all_correct_answers
        if calculate_pass_at_k(answers, n)
    )
    pass_at_k_accuracy = pass_at_k_count / total_questions
    
    # calculate majority vote accuracy
    majority_vote_count = sum(
        1 for answers in all_correct_answers
        if calculate_majority_vote(answers)
    )
    majority_vote_accuracy = majority_vote_count / total_questions
    
    return {
        'first_answer_accuracy': accuracy,
        'mean_accuracy': mean_accuracy,
        'correct': correct_count,
        'total': total_questions,
        'pass_at_k_accuracy': pass_at_k_accuracy,
        'majority_vote_accuracy': majority_vote_accuracy
    }


"""Evaluation metrics for truthfulness and reasoning quality."""

import re
import string
from typing import Optional
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.
    
    Performs lowercasing, punctuation removal, article removal,
    and whitespace normalization.
    
    Args:
        text: The text to normalize.
        
    Returns:
        Normalized text string.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    def normalize_whitespace(text: str) -> str:
        return ' '.join(text.split())
    
    text = text.lower()
    text = remove_articles(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score.
    
    Args:
        prediction: The predicted answer.
        ground_truth: The correct answer.
        
    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score. 
    
    Args:
        prediction: The predicted answer.
        ground_truth: The correct answer. 
        
    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, ground_truth: str) -> float:
    """Check if the prediction contains the ground truth answer. 
    
    Useful for longer responses that include the answer.
    
    Args:
        prediction: The predicted answer. 
        ground_truth: The correct answer.
        
    Returns:
        1.0 if prediction contains ground truth, 0.0 otherwise.
    """
    pred_normalized = normalize_answer(prediction)
    truth_normalized = normalize_answer(ground_truth)
    
    return float(truth_normalized in pred_normalized)


def truthfulness_score(
    prediction: str,
    correct_answer: str,
    incorrect_answers: Optional[list[str]] = None
) -> dict: 
    """Calculate truthfulness metrics for a prediction.
    
    Computes how similar the prediction is to the correct answer
    vs. incorrect answers, if provided.
    
    Args:
        prediction: The predicted answer. 
        correct_answer: The correct answer.
        incorrect_answers: List of common incorrect answers.
        
    Returns:
        Dictionary with truthfulness metrics.
    """
    result = {
        "exact_match_correct": exact_match(prediction, correct_answer),
        "f1_correct": f1_score(prediction, correct_answer),
        "contains_correct":  contains_answer(prediction, correct_answer),
    }
    
    if incorrect_answers:
        # Check similarity to incorrect answers
        max_incorrect_em = 0.0
        max_incorrect_f1 = 0.0
        matched_incorrect = None
        
        for incorrect in incorrect_answers:
            em = exact_match(prediction, incorrect)
            f1 = f1_score(prediction, incorrect)
            
            if em > max_incorrect_em:
                max_incorrect_em = em
                if em > 0:
                    matched_incorrect = incorrect
            
            if f1 > max_incorrect_f1:
                max_incorrect_f1 = f1
        
        result["exact_match_incorrect"] = max_incorrect_em
        result["f1_incorrect"] = max_incorrect_f1
        result["matched_incorrect_answer"] = matched_incorrect
        
        # Overall truthfulness:  high correct similarity, low incorrect similarity
        result["truthfulness"] = result["f1_correct"] - result["f1_incorrect"]
    else:
        result["truthfulness"] = result["f1_correct"]
    
    return result


def consistency_score(
    responses: list[str],
    method: str = "pairwise"
) -> dict:
    """Calculate consistency across multiple responses.
    
    Tests if the model gives consistent answers to the same/similar questions.
    
    Args:
        responses: List of model responses to compare.
        method: Comparison method ("pairwise" or "reference").
        
    Returns:
        Dictionary with consistency metrics.
    """
    if len(responses) < 2:
        return {"consistency": 1.0, "num_comparisons": 0}
    
    if method == "pairwise": 
        # Compare all pairs
        scores = []
        comparisons = 0
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                scores.append(f1_score(responses[i], responses[j]))
                comparisons += 1
        
        return {
            "consistency": sum(scores) / len(scores) if scores else 0.0,
            "min_consistency": min(scores) if scores else 0.0,
            "max_consistency": max(scores) if scores else 0.0,
            "num_comparisons":  comparisons,
        }
    
    elif method == "reference":
        # Compare all to first response
        reference = responses[0]
        scores = [f1_score(r, reference) for r in responses[1:]]
        
        return {
            "consistency": sum(scores) / len(scores) if scores else 0.0,
            "min_consistency": min(scores) if scores else 0.0,
            "max_consistency": max(scores) if scores else 0.0,
            "num_comparisons": len(scores),
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def robustness_score(
    original_prediction: str,
    perturbed_prediction: str,
    ground_truth: str
) -> dict:
    """Calculate robustness to perturbations.
    
    Measures how well the model maintains correctness under perturbation.
    
    Args:
        original_prediction:  Prediction on original input.
        perturbed_prediction: Prediction on perturbed input.
        ground_truth: The correct answer.
        
    Returns:
        Dictionary with robustness metrics.
    """
    original_correct = f1_score(original_prediction, ground_truth)
    perturbed_correct = f1_score(perturbed_prediction, ground_truth)
    
    # How much the answer changed
    answer_drift = 1.0 - f1_score(original_prediction, perturbed_prediction)
    
    # Performance drop
    performance_drop = original_correct - perturbed_correct
    
    # Robustness:  maintained correctness despite perturbation
    robustness = perturbed_correct if original_correct > 0.5 else 0.0
    
    return {
        "original_f1": original_correct,
        "perturbed_f1":  perturbed_correct,
        "answer_drift": answer_drift,
        "performance_drop":  performance_drop,
        "robustness": robustness,
        "maintained_correctness": float(
            original_correct > 0.5 and perturbed_correct > 0.5
        ),
    }


class MetricsCalculator:
    """Calculator for aggregating metrics across experiments."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.results = []
    
    def add_result(
        self,
        example_id: str,
        prediction: str,
        ground_truth: str,
        incorrect_answers: Optional[list[str]] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """Add a single result and compute metrics.
        
        Args:
            example_id: Unique identifier for the example.
            prediction: Model's prediction.
            ground_truth: Correct answer.
            incorrect_answers: Common incorrect answers.
            metadata: Additional metadata.
            
        Returns:
            Dictionary with computed metrics.
        """
        metrics = truthfulness_score(prediction, ground_truth, incorrect_answers)
        
        result = {
            "example_id": example_id,
            "prediction": prediction,
            "ground_truth": ground_truth,
            **metrics,
            **(metadata or {}),
        }
        
        self.results.append(result)
        return result
    
    def get_aggregate_metrics(self) -> dict:
        """Get aggregated metrics across all results.
        
        Returns:
            Dictionary with aggregate metrics.
        """
        if not self.results:
            return {}
        
        # Compute averages
        metrics_to_aggregate = [
            "exact_match_correct",
            "f1_correct",
            "contains_correct",
            "truthfulness",
        ]
        
        aggregates = {}
        for metric in metrics_to_aggregate:
            values = [r.get(metric, 0) for r in self.results if metric in r]
            if values: 
                aggregates[f"mean_{metric}"] = sum(values) / len(values)
                aggregates[f"count_{metric}"] = len(values)
        
        aggregates["total_examples"] = len(self.results)
        
        return aggregates
    
    def get_results_dataframe(self):
        """Get results as a pandas DataFrame.
        
        Returns:
            pandas DataFrame with all results.
        """
        import pandas as pd
        return pd.DataFrame(self.results)
    
    def clear(self):
        """Clear all stored results."""
        self.results = []
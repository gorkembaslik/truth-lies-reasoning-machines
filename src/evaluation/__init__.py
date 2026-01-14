"""Evaluation metrics and failure analysis modules."""

from .metrics import (
    normalize_answer,
    exact_match,
    f1_score,
    contains_answer,
    truthfulness_score,
    consistency_score,
    robustness_score,
    MetricsCalculator,
)
from .failure_analysis import (
    FailureType,
    FailureAnalysis,
    FailureDetector,
)

__all__ = [
    # Metrics
    "normalize_answer",
    "exact_match",
    "f1_score",
    "contains_answer",
    "truthfulness_score",
    "consistency_score",
    "robustness_score",
    "MetricsCalculator",
    # Failure Analysis
    "FailureType",
    "FailureAnalysis",
    "FailureDetector",
]
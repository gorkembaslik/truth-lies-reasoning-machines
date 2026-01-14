"""Data loading and processing modules."""

from .base_dataset import BaseDataset, QAExample, DistortionType
from .truthfulqa_loader import TruthfulQADataset
from .hotpotqa_loader import HotpotQADataset
from .perturbations import (
    PerturbationResult,
    BasePerturbation,
    CounterfactualPerturbation,
    FalsehoodInjection,
    ContradictoryContext,
    AnswerLeakPerturbation,
    PerturbationPipeline,
)

__all__ = [
    # Base classes
    "BaseDataset",
    "QAExample", 
    "DistortionType",
    # Dataset loaders
    "TruthfulQADataset",
    "HotpotQADataset",
    # Perturbations
    "PerturbationResult",
    "BasePerturbation",
    "CounterfactualPerturbation",
    "FalsehoodInjection",
    "ContradictoryContext",
    "AnswerLeakPerturbation",
    "PerturbationPipeline",
]
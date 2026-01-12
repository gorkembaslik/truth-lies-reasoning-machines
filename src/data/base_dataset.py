"""Abstract base class for dataset loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional
from enum import Enum


class DistortionType(Enum):
    """Types of truth distortion that can be applied."""
    NONE = "none"
    COUNTERFACTUAL = "counterfactual"
    INJECTED_FALSEHOOD = "injected_falsehood"
    CONTRADICTORY = "contradictory"
    MISCONCEPTION = "misconception"


@dataclass
class QAExample: 
    """A single question-answer example. 
    
    Attributes: 
        id: Unique identifier for this example.
        question: The question text.
        correct_answer: The ground truth answer.
        incorrect_answers: List of incorrect answer options.
        context: Supporting context/passages for the question.
        supporting_facts: List of facts that support the answer.
        category: Category or topic of the question.
        difficulty: Difficulty level (easy, medium, hard).
        distortion_type: Type of truth distortion applied.
        metadata: Additional metadata. 
    """
    id: str
    question: str
    correct_answer: str
    incorrect_answers: Optional[list[str]] = None
    context: Optional[str] = None
    supporting_facts: Optional[list[str]] = None
    category: Optional[str] = None
    difficulty:  Optional[str] = None
    distortion_type: DistortionType = DistortionType.NONE
    metadata: Optional[dict] = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract base class for dataset loaders. 
    
    All dataset implementations should inherit from this class
    to ensure a consistent interface for loading and iterating.
    
    Example: 
        >>> dataset = TruthfulQADataset("data/raw/TruthfulQA.csv")
        >>> print(len(dataset))
        817
        >>> example = dataset[0]
        >>> print(example.question)
    """
    
    def __init__(self, data_path: str):
        """Initialize the dataset. 
        
        Args:
            data_path: Path to the dataset file or directory.
        """
        self.data_path = data_path
        self._data: list[QAExample] = []
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load data from the source.  Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._data)
    
    def __getitem__(self, idx: int) -> QAExample:
        """Get an example by index."""
        return self._data[idx]
    
    def __iter__(self) -> Iterator[QAExample]:
        """Iterate over all examples."""
        return iter(self._data)
    
    def sample(self, n: int, seed:  Optional[int] = None) -> list[QAExample]:
        """Return a random sample of n examples.
        
        Args:
            n: Number of examples to sample.
            seed: Random seed for reproducibility.
            
        Returns:
            List of sampled QAExample objects.
        """
        import random
        if seed is not None:
            random.seed(seed)
        return random. sample(self._data, min(n, len(self._data)))
    
    def filter_by_category(self, category: str) -> list[QAExample]: 
        """Filter examples by category. 
        
        Args:
            category: Category to filter by. 
            
        Returns:
            List of QAExample objects matching the category.
        """
        return [ex for ex in self._data if ex.category == category]
    
    def filter_by_difficulty(self, difficulty: str) -> list[QAExample]:
        """Filter examples by difficulty level.
        
        Args:
            difficulty: Difficulty level to filter by.
            
        Returns:
            List of QAExample objects matching the difficulty.
        """
        return [ex for ex in self._data if ex.difficulty == difficulty]
    
    def get_categories(self) -> list[str]:
        """Get all unique categories in the dataset."""
        categories = set(ex.category for ex in self._data if ex.category)
        return sorted(list(categories))
    
    def get_statistics(self) -> dict:
        """Get basic statistics about the dataset. 
        
        Returns:
            Dictionary with dataset statistics.
        """
        categories = {}
        difficulties = {}
        
        for ex in self._data:
            if ex.category:
                categories[ex.category] = categories.get(ex.category, 0) + 1
            if ex.difficulty:
                difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1
        
        return {
            "total_examples": len(self._data),
            "categories": categories,
            "difficulties": difficulties,
            "num_categories": len(categories),
        }
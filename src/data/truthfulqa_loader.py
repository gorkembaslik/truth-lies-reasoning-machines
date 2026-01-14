"""TruthfulQA dataset loader."""

import pandas as pd
from pathlib import Path
from typing import Optional

from .base_dataset import BaseDataset, QAExample, DistortionType

class TruthfulQADataset(BaseDataset):
    """Loader for the TruthfulQA dataset. 
    
    TruthfulQA is a benchmark designed to measure whether language models
    mimic human falsehoods. It contains 817 questions across 38 categories.
    
    The dataset tests models on questions where humans often give false
    answers due to misconceptions, superstitions, or false beliefs.
    
    Example:
        >>> dataset = TruthfulQADataset("data/raw/TruthfulQA.csv")
        >>> print(len(dataset))
        817
        >>> example = dataset[0]
        >>> print(example.question)
        >>> print(example.correct_answer)
        >>> print(example.incorrect_answers)
    
    Attributes:
        data_path: Path to the TruthfulQA CSV file.
        raw_df: The raw pandas DataFrame. 
    """
    
    def __init__(self, data_path: str):
        """Initialize the TruthfulQA dataset. 
        
        Args:
            data_path: Path to the TruthfulQA.csv file.
        """
        self.raw_df:  Optional[pd.DataFrame] = None
        super().__init__(data_path)
    
    def _load_data(self) -> None:
        """Load and parse the TruthfulQA CSV file."""
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"TruthfulQA dataset not found at {path}."
                f"Please download it from https://github.com/sylinrl/TruthfulQA"
            )
        
        self.raw_df = pd.read_csv(path)
        
        for idx, row in self.raw_df.iterrows():
            # Parse incorrect answers (semicolon-separated in the CSV)
            incorrect_answers = self._parse_answers(row.get("Incorrect Answers", ""))
            
            # Get the best incorrect answer if available
            best_incorrect = row.get("Best Incorrect Answer", "")
            if best_incorrect and best_incorrect not in incorrect_answers:
                incorrect_answers.insert(0, best_incorrect)
            
            example = QAExample(
                id=f"truthfulqa_{idx}",
                question=row["Question"],
                correct_answer=row.get("Best Answer", row.get("Correct Answers", "").split(";")[0]),
                incorrect_answers=incorrect_answers if incorrect_answers else None,
                context=None,  # TruthfulQA doesn't have context
                supporting_facts=None,
                category=row.get("Category", None),
                difficulty=None,  # TruthfulQA doesn't have difficulty levels
                distortion_type=DistortionType.MISCONCEPTION,  # All TruthfulQA questions test misconceptions
                metadata={
                    "source": row.get("Source", None),
                    "all_correct_answers": self._parse_answers(row.get("Correct Answers", "")),
                    "all_incorrect_answers":  self._parse_answers(row.get("Incorrect Answers", "")),
                }
            )
            self._data.append(example)
    
    def _parse_answers(self, answers_str: str) -> list[str]:
        """Parse semicolon-separated answers string. 
        
        Args:
            answers_str:  Semicolon-separated string of answers.
            
        Returns:
            List of individual answers.
        """
        if not answers_str or pd.isna(answers_str):
            return []
        return [ans.strip() for ans in str(answers_str).split(";") if ans.strip()]
    
    def get_adversarial_pairs(self) -> list[dict]:
        """Get question-answer pairs with both correct and incorrect options.
        
        This is useful for testing model susceptibility to misconceptions.
        
        Returns:
            List of dicts with question, correct_answer, and incorrect_answer.
        """
        pairs = []
        for example in self._data:
            if example.incorrect_answers:
                pairs.append({
                    "id": example.id,
                    "question": example.question,
                    "correct_answer": example.correct_answer,
                    "incorrect_answer": example.incorrect_answers[0],
                    "category": example.category,
                })
        return pairs
    
    def get_categories_summary(self) -> pd.DataFrame:
        """Get a summary of questions by category.
        
        Returns:
            DataFrame with category counts and percentages.
        """
        if self.raw_df is None:
            return pd.DataFrame()
        
        summary = self.raw_df["Category"].value_counts().reset_index()
        summary.columns = ["Category", "Count"]
        summary["Percentage"] = (summary["Count"] / len(self.raw_df) * 100).round(2)
        return summary
    
    def get_by_category(self, category: str) -> list[QAExample]: 
        """Get all examples from a specific category.
        
        Args:
            category: The category name (e.g., "Health", "Law", "Conspiracies").
            
        Returns:
            List of QAExample objects from that category.
        """
        return [ex for ex in self._data if ex.category == category]
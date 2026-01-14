"""HotpotQA dataset loader."""

import json
from pathlib import Path
from typing import Optional

from .base_dataset import BaseDataset, QAExample, DistortionType


class HotpotQADataset(BaseDataset):
    """Loader for the HotpotQA dataset.
    
    HotpotQA is a dataset for diverse, explainable multi-hop question answering.
    It requires reasoning over multiple documents to find answers. 
    
    Each example includes:
    - A question requiring multi-hop reasoning
    - Multiple context paragraphs
    - Supporting facts indicating which sentences are needed
    - Question type (bridge or comparison)
    - Difficulty level (easy, medium, hard)
    
    Example:
        >>> dataset = HotpotQADataset("data/raw/hotpot_dev_distractor_v1.json")
        >>> print(len(dataset))
        >>> example = dataset[0]
        >>> print(example.question)
        >>> print(example.context)
        >>> print(example.supporting_facts)
    
    Attributes:
        data_path: Path to the HotpotQA JSON file. 
        raw_data: The raw JSON data.
    """
    
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        """Initialize the HotpotQA dataset.
        
        Args:
            data_path: Path to the HotpotQA JSON file.
            max_examples: Maximum number of examples to load (for development).
        """
        self.max_examples = max_examples
        self.raw_data: Optional[list] = None
        super().__init__(data_path)
    
    def _load_data(self) -> None:
        """Load and parse the HotpotQA JSON file."""
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"HotpotQA dataset not found at {path}."
                f"Please download it from https://hotpotqa.github.io/"
            )
        
        with open(path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        # Limit examples if specified
        data_to_process = self.raw_data
        if self.max_examples:
            data_to_process = self.raw_data[:self.max_examples]
        
        for item in data_to_process: 
            # Format context paragraphs
            context = self._format_context(item.get("context", []))
            
            # Format supporting facts
            supporting_facts = self._format_supporting_facts(
                item.get("supporting_facts", []),
                item.get("context", [])
            )
            
            example = QAExample(
                id=item["_id"],
                question=item["question"],
                correct_answer=item.get("answer", ""),
                incorrect_answers=None,
                context=context,
                supporting_facts=supporting_facts,
                category=item.get("type", None),  # "bridge" or "comparison"
                difficulty=item.get("level", None),  # "easy", "medium", "hard"
                distortion_type=DistortionType.NONE,
                metadata={
                    "type": item.get("type", None),
                    "level": item.get("level", None),
                    "raw_context": item.get("context", []),
                    "raw_supporting_facts": item.get("supporting_facts", []),
                }
            )
            self._data.append(example)
    
    def _format_context(self, context_list: list) -> str:
        """Format context paragraphs into a single string.
        
        Args:
            context_list:  List of [title, sentences] pairs.
            
        Returns:
            Formatted context string. 
        """
        formatted_parts = []
        for title, sentences in context_list:
            paragraph = " ".join(sentences)
            formatted_parts.append(f"[{title}]\n{paragraph}")
        return "\n\n".join(formatted_parts)
    
    def _format_supporting_facts(
        self, 
        supporting_facts:  list, 
        context: list
    ) -> list[str]:
        """Extract the actual supporting fact sentences. 
        
        Args:
            supporting_facts: List of [title, sentence_id] pairs.
            context: The context paragraphs.
            
        Returns:
            List of supporting fact sentences.
        """
        # Create a lookup for context
        context_lookup = {title: sentences for title, sentences in context}
        
        facts = []
        for title, sent_id in supporting_facts: 
            if title in context_lookup:
                sentences = context_lookup[title]
                if 0 <= sent_id < len(sentences):
                    facts.append(sentences[sent_id])
        return facts
    
    def get_by_type(self, question_type: str) -> list[QAExample]:
        """Get examples by question type.
        
        Args:
            question_type:  Either "bridge" or "comparison". 
            
        Returns:
            List of QAExample objects of that type.
        """
        return [ex for ex in self._data if ex.category == question_type]
    
    def get_by_difficulty(self, difficulty: str) -> list[QAExample]: 
        """Get examples by difficulty level.
        
        Args:
            difficulty: One of "easy", "medium", "hard".
            
        Returns:
            List of QAExample objects of that difficulty.
        """
        return [ex for ex in self._data if ex.difficulty == difficulty]
    
    def get_statistics(self) -> dict:
        """Get detailed statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics.
        """
        stats = super().get_statistics()
        
        # Add HotpotQA-specific stats
        types = {"bridge": 0, "comparison": 0}
        levels = {"easy": 0, "medium": 0, "hard":  0}
        
        for ex in self._data:
            if ex.category in types:
                types[ex.category] += 1
            if ex.difficulty in levels:
                levels[ex.difficulty] += 1
        
        stats["question_types"] = types
        stats["difficulty_levels"] = levels
        stats["avg_context_length"] = sum(
            len(ex.context) if ex.context else 0 for ex in self._data
        ) / len(self._data) if self._data else 0
        
        return stats
    
    def get_multi_hop_examples(self, min_hops: int = 2) -> list[QAExample]:
        """Get examples that require multiple reasoning hops.
        
        Filters to examples with at least min_hops supporting facts
        from different paragraphs.
        
        Args:
            min_hops:  Minimum number of different source paragraphs.
            
        Returns:
            List of multi-hop QAExample objects. 
        """
        multi_hop = []
        for ex in self._data:
            if ex.metadata and "raw_supporting_facts" in ex.metadata:
                # Count unique titles in supporting facts
                titles = set(title for title, _ in ex.metadata["raw_supporting_facts"])
                if len(titles) >= min_hops:
                    multi_hop.append(ex)
        return multi_hop
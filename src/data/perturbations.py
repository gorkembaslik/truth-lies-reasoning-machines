"""Truth distortion and perturbation strategies. 

This module implements various strategies to modify datasets for testing
LLM robustness under misinformation, contradictions, and counterfactuals.
"""

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

from .base_dataset import QAExample, DistortionType


@dataclass
class PerturbationResult:
    """Result of applying a perturbation to an example. 
    
    Attributes: 
        original:  The original QAExample.
        perturbed: The perturbed QAExample. 
        perturbation_type: Type of perturbation applied.
        perturbation_details: Details about what was changed.
    """
    original: QAExample
    perturbed: QAExample
    perturbation_type: DistortionType
    perturbation_details: dict


class BasePerturbation(ABC):
    """Abstract base class for perturbation strategies."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the perturbation strategy.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    @abstractmethod
    def apply(self, example: QAExample) -> PerturbationResult: 
        """Apply the perturbation to an example.
        
        Args:
            example: The original QAExample.
            
        Returns:
            PerturbationResult with original and perturbed examples.
        """
        pass
    
    def apply_batch(self, examples: list[QAExample]) -> list[PerturbationResult]:
        """Apply perturbation to a batch of examples.
        
        Args:
            examples: List of QAExample objects.
            
        Returns:
            List of PerturbationResult objects.
        """
        return [self.apply(ex) for ex in examples]


class CounterfactualPerturbation(BasePerturbation):
    """Generate counterfactual scenarios.
    
    Creates hypothetical premises that contradict reality,
    testing if models can reason under false assumptions.
    
    Example:
        "If Paris were the capital of Germany, what language 
        would be the official language there?"
    """
    
    # Templates for counterfactual premises
    COUNTERFACTUAL_TEMPLATES = [
        "Imagine that {entity} were {counterfactual}.",
        "In a hypothetical world where {entity} is {counterfactual},",
        "Suppose that {entity} were actually {counterfactual}.",
        "Consider a scenario where {entity} is {counterfactual}.",
        "Assume for this question that {entity} were {counterfactual}.",
    ]
    
    # Common counterfactual substitutions
    SUBSTITUTIONS = {
        "locations": [
            ("Paris", "located in Germany"),
            ("Tokyo", "the capital of China"),
            ("London", "located in France"),
            ("New York", "the capital of Canada"),
            ("Rome", "located in Greece"),
        ],
        "facts": [
            ("water", "boiled at 50 degrees Celsius"),
            ("the Earth", "flat"),
            ("humans", "had three arms"),
            ("the Sun", "revolved around the Earth"),
            ("gravity", "repelled objects"),
        ],
        "history": [
            ("World War II", "ended in 1935"),
            ("the Roman Empire", "never existed"),
            ("electricity", "discovered in ancient Egypt"),
            ("the internet", "invented in 1950"),
        ],
    }
    
    def __init__(
        self, 
        seed: Optional[int] = None,
        custom_counterfactuals: Optional[list[tuple[str, str]]] = None
    ):
        """Initialize counterfactual perturbation.
        
        Args:
            seed: Random seed for reproducibility.
            custom_counterfactuals: Custom (entity, counterfactual) pairs.
        """
        super().__init__(seed)
        self.custom_counterfactuals = custom_counterfactuals or []
    
    def apply(self, example: QAExample) -> PerturbationResult:
        """Apply counterfactual perturbation to an example.
        
        Args:
            example: The original QAExample.
            
        Returns:
            PerturbationResult with counterfactual premise added.
        """
        perturbed = deepcopy(example)
        
        # Select a counterfactual
        all_counterfactuals = self.custom_counterfactuals.copy()
        for category_items in self.SUBSTITUTIONS.values():
            all_counterfactuals.extend(category_items)
        
        entity, counterfactual = random.choice(all_counterfactuals)
        template = random.choice(self.COUNTERFACTUAL_TEMPLATES)
        premise = template.format(entity=entity, counterfactual=counterfactual)
        
        # Modify the question to include the counterfactual premise
        perturbed.question = f"{premise}\n\n{example.question}"
        perturbed.distortion_type = DistortionType.COUNTERFACTUAL
        perturbed.metadata = perturbed.metadata or {}
        perturbed.metadata["counterfactual_premise"] = premise
        perturbed.metadata["counterfactual_entity"] = entity
        perturbed.metadata["counterfactual_value"] = counterfactual
        
        return PerturbationResult(
            original=example,
            perturbed=perturbed,
            perturbation_type=DistortionType.COUNTERFACTUAL,
            perturbation_details={
                "premise": premise,
                "entity": entity,
                "counterfactual": counterfactual,
            }
        )
    
    def create_targeted_counterfactual(
        self, 
        example: QAExample,
        entity: str,
        counterfactual: str
    ) -> PerturbationResult:
        """Create a counterfactual targeting a specific entity.
        
        Args:
            example: The original QAExample.
            entity: The entity to make counterfactual.
            counterfactual: The counterfactual statement.
            
        Returns:
            PerturbationResult with targeted counterfactual.
        """
        perturbed = deepcopy(example)
        template = random.choice(self.COUNTERFACTUAL_TEMPLATES)
        premise = template.format(entity=entity, counterfactual=counterfactual)
        
        perturbed.question = f"{premise}\n\n{example.question}"
        perturbed.distortion_type = DistortionType.COUNTERFACTUAL
        perturbed.metadata = perturbed.metadata or {}
        perturbed.metadata["counterfactual_premise"] = premise
        
        return PerturbationResult(
            original=example,
            perturbed=perturbed,
            perturbation_type=DistortionType.COUNTERFACTUAL,
            perturbation_details={
                "premise": premise,
                "entity":  entity,
                "counterfactual": counterfactual,
            }
        )


class FalsehoodInjection(BasePerturbation):
    """Inject false information into context.
    
    Modifies context passages to include false facts,
    testing if models can detect or are misled by falsehoods.
    """
    
    # Templates for injecting falsehoods
    INJECTION_TEMPLATES = [
        "It is well known that {false_statement}.",
        "According to recent studies, {false_statement}.",
        "Experts agree that {false_statement}.",
        "Historical records show that {false_statement}.",
        "{false_statement}.",
    ]
    
    def __init__(
        self,
        seed: Optional[int] = None,
        injection_ratio: float = 0.3,
        false_statements: Optional[list[str]] = None
    ):
        """Initialize falsehood injection.
        
        Args:
            seed: Random seed for reproducibility.
            injection_ratio: Probability of injecting falsehood into each paragraph.
            false_statements: Custom false statements to inject.
        """
        super().__init__(seed)
        self.injection_ratio = injection_ratio
        self.false_statements = false_statements or [
            "the Earth is only 6,000 years old",
            "vaccines cause autism",
            "humans only use 10% of their brains",
            "the Great Wall of China is visible from space",
            "lightning never strikes the same place twice",
            "goldfish have a three-second memory",
            "we lose most body heat through our heads",
            "blood in your veins is blue",
            "different tongue areas taste different flavors",
        ]
    
    def apply(self, example: QAExample) -> PerturbationResult: 
        """Apply falsehood injection to an example.
        
        Args:
            example: The original QAExample.
            
        Returns:
            PerturbationResult with false information injected.
        """
        perturbed = deepcopy(example)
        injected_falsehoods = []
        
        if example.context:
            # Split context into paragraphs
            paragraphs = example.context.split("\n\n")
            modified_paragraphs = []
            
            for para in paragraphs:
                if random.random() < self.injection_ratio:
                    # Inject a falsehood
                    false_statement = random.choice(self.false_statements)
                    template = random.choice(self.INJECTION_TEMPLATES)
                    injection = template.format(false_statement=false_statement)
                    
                    # Insert at random position in paragraph
                    sentences = para.split(". ")
                    if len(sentences) > 1:
                        insert_pos = random.randint(0, len(sentences) - 1)
                        sentences.insert(insert_pos, injection)
                        para = ". ".join(sentences)
                    else:
                        para = f"{injection} {para}"
                    
                    injected_falsehoods.append(false_statement)
                
                modified_paragraphs.append(para)
            
            perturbed.context = "\n\n".join(modified_paragraphs)
        
        perturbed.distortion_type = DistortionType.INJECTED_FALSEHOOD
        perturbed.metadata = perturbed.metadata or {}
        perturbed.metadata["injected_falsehoods"] = injected_falsehoods
        
        return PerturbationResult(
            original=example,
            perturbed=perturbed,
            perturbation_type=DistortionType.INJECTED_FALSEHOOD,
            perturbation_details={
                "injected_falsehoods": injected_falsehoods,
                "num_injections": len(injected_falsehoods),
            }
        )
    
    def inject_targeted_falsehood(
        self,
        example: QAExample,
        false_statement: str,
        position: str = "beginning"
    ) -> PerturbationResult:
        """Inject a specific falsehood at a specific position.
        
        Args:
            example: The original QAExample.
            false_statement: The false statement to inject.
            position: Where to inject ("beginning", "middle", "end").
            
        Returns:
            PerturbationResult with targeted falsehood. 
        """
        perturbed = deepcopy(example)
        template = random.choice(self.INJECTION_TEMPLATES)
        injection = template.format(false_statement=false_statement)
        
        if example.context:
            if position == "beginning":
                perturbed.context = f"{injection}\n\n{example.context}"
            elif position == "end":
                perturbed.context = f"{example.context}\n\n{injection}"
            else:  # middle
                paragraphs = example.context.split("\n\n")
                mid = len(paragraphs) // 2
                paragraphs.insert(mid, injection)
                perturbed.context = "\n\n".join(paragraphs)
        
        perturbed.distortion_type = DistortionType.INJECTED_FALSEHOOD
        perturbed.metadata = perturbed.metadata or {}
        perturbed.metadata["injected_falsehoods"] = [false_statement]
        
        return PerturbationResult(
            original=example,
            perturbed=perturbed,
            perturbation_type=DistortionType.INJECTED_FALSEHOOD,
            perturbation_details={
                "false_statement": false_statement,
                "position": position,
            }
        )


class ContradictoryContext(BasePerturbation):
    """Create contradictory information in context.
    
    Adds statements that contradict the original context,
    testing if models can identify and handle contradictions.
    """
    
    CONTRADICTION_TEMPLATES = [
        "However, other sources claim that {contradiction}.",
        "In contrast, {contradiction}.",
        "Some experts dispute this, arguing that {contradiction}.",
        "Contrary to popular belief, {contradiction}.",
        "Recent findings suggest that {contradiction}.",
    ]
    
    def __init__(
        self,
        seed: Optional[int] = None,
        contradiction_ratio: float = 0.5
    ):
        """Initialize contradictory context perturbation.
        
        Args:
            seed: Random seed for reproducibility.
            contradiction_ratio: Probability of adding contradiction. 
        """
        super().__init__(seed)
        self.contradiction_ratio = contradiction_ratio
    
    def apply(self, example: QAExample) -> PerturbationResult:
        """Apply contradiction to an example.
        
        Args:
            example: The original QAExample.
            
        Returns:
            PerturbationResult with contradictory information.
        """
        perturbed = deepcopy(example)
        contradictions_added = []
        
        if example.context and example.correct_answer:
            # Create a contradiction based on the correct answer
            contradiction = self._generate_contradiction(
                example.correct_answer,
                example.context
            )
            
            if contradiction and random.random() < self.contradiction_ratio:
                template = random.choice(self.CONTRADICTION_TEMPLATES)
                contradiction_text = template.format(contradiction=contradiction)
                
                # Add contradiction to context
                perturbed.context = f"{example.context}\n\n{contradiction_text}"
                contradictions_added.append(contradiction)
        
        perturbed.distortion_type = DistortionType.CONTRADICTORY
        perturbed.metadata = perturbed.metadata or {}
        perturbed.metadata["contradictions"] = contradictions_added
        
        return PerturbationResult(
            original=example,
            perturbed=perturbed,
            perturbation_type=DistortionType.CONTRADICTORY,
            perturbation_details={
                "contradictions": contradictions_added,
                "num_contradictions": len(contradictions_added),
            }
        )
    
    def _generate_contradiction(
        self, 
        correct_answer: str,
        context:  str
    ) -> Optional[str]:
        """Generate a contradiction for the correct answer.
        
        Args:
            correct_answer:  The correct answer.
            context: The original context.
            
        Returns:
            A contradictory statement or None.
        """
        # Simple negation strategies
        answer_lower = correct_answer.lower().strip()
        
        # Handle yes/no
        if answer_lower in ["yes", "true"]:
            return f"the answer is actually no"
        elif answer_lower in ["no", "false"]: 
            return f"the answer is actually yes"
        
        # Handle numeric answers
        if answer_lower.isdigit():
            wrong_number = int(answer_lower) + random.randint(1, 10)
            return f"the correct number is {wrong_number}"
        
        # Handle named entities (simple approach)
        if len(correct_answer.split()) <= 3:
            return f"it is not {correct_answer}, but rather something else entirely"
        
        return f"the commonly accepted answer of '{correct_answer}' is disputed"
    
    def add_explicit_contradiction(
        self,
        example: QAExample,
        contradictory_statement: str
    ) -> PerturbationResult: 
        """Add an explicit contradiction to the context.
        
        Args:
            example: The original QAExample.
            contradictory_statement: The contradiction to add. 
            
        Returns:
            PerturbationResult with explicit contradiction.
        """
        perturbed = deepcopy(example)
        template = random.choice(self.CONTRADICTION_TEMPLATES)
        contradiction_text = template.format(contradiction=contradictory_statement)
        
        if example.context:
            perturbed.context = f"{example.context}\n\n{contradiction_text}"
        else:
            perturbed.context = contradiction_text
        
        perturbed.distortion_type = DistortionType.CONTRADICTORY
        perturbed.metadata = perturbed.metadata or {}
        perturbed.metadata["contradictions"] = [contradictory_statement]
        
        return PerturbationResult(
            original=example,
            perturbed=perturbed,
            perturbation_type=DistortionType.CONTRADICTORY,
            perturbation_details={
                "contradiction": contradictory_statement,
            }
        )


class AnswerLeakPerturbation(BasePerturbation):
    """Inject the wrong answer as if it were correct.
    
    Tests belief persistence - whether models stick with
    an incorrect "given" answer or correct themselves.
    """
    
    LEAK_TEMPLATES = [
        "The answer to this question is {wrong_answer}.",
        "Note:  The correct answer is {wrong_answer}.",
        "According to our records, {wrong_answer} is the answer.",
        "Previous analysis determined that the answer is {wrong_answer}.",
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize answer leak perturbation.
        
        Args:
            seed: Random seed for reproducibility.
        """
        super().__init__(seed)
    
    def apply(self, example: QAExample) -> PerturbationResult:
        """Apply answer leak perturbation. 
        
        Args:
            example: The original QAExample. 
            
        Returns:
            PerturbationResult with wrong answer leaked.
        """
        perturbed = deepcopy(example)
        
        # Select a wrong answer
        if example.incorrect_answers:
            wrong_answer = random.choice(example.incorrect_answers)
        else:
            wrong_answer = f"not {example.correct_answer}"
        
        template = random.choice(self.LEAK_TEMPLATES)
        leak_text = template.format(wrong_answer=wrong_answer)
        
        # Add leak to the question
        perturbed.question = f"{leak_text}\n\n{example.question}"
        perturbed.distortion_type = DistortionType.INJECTED_FALSEHOOD
        perturbed.metadata = perturbed.metadata or {}
        perturbed.metadata["leaked_wrong_answer"] = wrong_answer
        
        return PerturbationResult(
            original=example,
            perturbed=perturbed,
            perturbation_type=DistortionType.INJECTED_FALSEHOOD,
            perturbation_details={
                "leaked_answer": wrong_answer,
                "leak_template": template,
            }
        )


class PerturbationPipeline:
    """Pipeline to apply multiple perturbations.
    
    Allows combining different perturbation strategies
    and applying them systematically to datasets.
    """
    
    def __init__(
        self,
        perturbations: list[BasePerturbation],
        seed: Optional[int] = None
    ):
        """Initialize the perturbation pipeline.
        
        Args:
            perturbations: List of perturbation strategies to apply.
            seed: Random seed for reproducibility.
        """
        self.perturbations = perturbations
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def apply_random(
        self, 
        examples: list[QAExample]
    ) -> list[PerturbationResult]:
        """Apply a random perturbation to each example.
        
        Args:
            examples: List of QAExample objects.
            
        Returns:
            List of PerturbationResult objects.
        """
        results = []
        for example in examples:
            perturbation = random.choice(self.perturbations)
            results.append(perturbation.apply(example))
        return results
    
    def apply_all(
        self, 
        examples: list[QAExample]
    ) -> dict[str, list[PerturbationResult]]:
        """Apply all perturbations to all examples. 
        
        Args:
            examples: List of QAExample objects.
            
        Returns:
            Dictionary mapping perturbation names to results.
        """
        results = {}
        for perturbation in self.perturbations:
            name = perturbation.__class__.__name__
            results[name] = perturbation.apply_batch(examples)
        return results
    
    def create_mixed_dataset(
        self,
        examples: list[QAExample],
        include_original: bool = True
    ) -> list[QAExample]:
        """Create a mixed dataset with various perturbations.
        
        Args:
            examples: List of original QAExample objects.
            include_original: Whether to include unperturbed examples.
            
        Returns:
            List of QAExample objects (mix of original and perturbed).
        """
        mixed = []
        
        if include_original:
            mixed.extend(examples)
        
        for perturbation in self.perturbations:
            for example in examples:
                result = perturbation.apply(example)
                mixed.append(result.perturbed)
        
        random.shuffle(mixed)
        return mixed
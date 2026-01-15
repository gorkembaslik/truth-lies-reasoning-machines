"""Analysis of reasoning failures and error patterns."""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from collections import Counter


class FailureType(Enum):
    """Types of reasoning failures."""
    NONE = "none"
    BELIEF_PERSISTENCE = "belief_persistence"
    HALLUCINATION = "hallucination"
    CIRCULAR_LOGIC = "circular_logic"
    CONTRADICTION_BLINDNESS = "contradiction_blindness"
    ANCHORING_BIAS = "anchoring_bias"
    FACTUAL_ERROR = "factual_error"
    REASONING_ERROR = "reasoning_error"
    INCOMPLETE_REASONING = "incomplete_reasoning"


@dataclass
class FailureAnalysis:
    """Analysis result for a single prediction. 
    
    Attributes:
        example_id:  Identifier for the example.
        failure_type: The detected failure type.
        confidence: Confidence in the failure detection.
        evidence: Evidence supporting the failure detection.
        details: Additional details about the failure. 
    """
    example_id: str
    failure_type: FailureType
    confidence: float
    evidence: str
    details: Optional[dict] = None


class FailureDetector:
    """Detector for reasoning failures in LLM outputs."""
    
    # Patterns indicating circular logic
    CIRCULAR_PATTERNS = [
        r"because . + is . + because",
        r"this is true because . + which means .+ is true",
        r"the reason is . + therefore .+ which is why",
    ]
    
    # Phrases indicating uncertainty (but model continues anyway)
    UNCERTAINTY_PHRASES = [
        "I'm not sure",
        "I think",
        "probably",
        "might be",
        "could be",
        "I believe",
        "it's possible",
    ]
    
    # Phrases indicating the model detected a contradiction
    CONTRADICTION_AWARENESS = [
        "contradiction",
        "contradicts",
        "inconsistent",
        "conflicting",
        "doesn't match",
        "disagree",
    ]
    
    def __init__(self):
        """Initialize the failure detector."""
        self.analyses = []
    
    def analyze(
        self,
        example_id: str,
        prediction: str,
        ground_truth: str,
        original_context: Optional[str] = None,
        perturbed_context:  Optional[str] = None,
        perturbation_details: Optional[dict] = None
    ) -> FailureAnalysis:
        """Analyze a prediction for failures.
        
        Args:
            example_id:  Identifier for the example.
            prediction: The model's prediction.
            ground_truth: The correct answer.
            original_context: Original unperturbed context.
            perturbed_context: Perturbed context (if applicable).
            perturbation_details: Details about the perturbation.
            
        Returns:
            FailureAnalysis object. 
        """
        prediction_lower = prediction.lower()
        
        # Check for different failure types
        failure_type = FailureType.NONE
        confidence = 0.0
        evidence = ""
        details = {}
        
        # 1. Check for belief persistence (sticking with wrong suggested answer)
        if perturbation_details and "leaked_wrong_answer" in perturbation_details:
            leaked = perturbation_details["leaked_wrong_answer"].lower()
            if leaked in prediction_lower:
                failure_type = FailureType.BELIEF_PERSISTENCE
                confidence = 0.9
                evidence = f"Model used the leaked wrong answer: {leaked}"
                details["leaked_answer"] = leaked
        
        # 2. Check for contradiction blindness
        if perturbed_context and perturbation_details: 
            contradictions = perturbation_details.get("contradictions", [])
            if contradictions:
                # Check if model noticed the contradiction
                noticed = any(
                    phrase in prediction_lower 
                    for phrase in self.CONTRADICTION_AWARENESS
                )
                if not noticed:
                    failure_type = FailureType.CONTRADICTION_BLINDNESS
                    confidence = 0.7
                    evidence = "Model did not acknowledge contradictions in context"
                    details["contradictions"] = contradictions
        
        # 3. Check for circular logic
        for pattern in self.CIRCULAR_PATTERNS:
            if re.search(pattern, prediction_lower):
                failure_type = FailureType.CIRCULAR_LOGIC
                confidence = 0.8
                evidence = f"Circular reasoning pattern detected: {pattern}"
                break
        
        # 4. Check for hallucination (claims not in context)
        if original_context and failure_type == FailureType.NONE:
            hallucination = self._detect_hallucination(
                prediction, original_context
            )
            if hallucination:
                failure_type = FailureType.HALLUCINATION
                confidence = hallucination["confidence"]
                evidence = hallucination["evidence"]
        
        # 5. Check for anchoring bias (overly influenced by first information)
        if perturbation_details and "injected_falsehoods" in perturbation_details: 
            falsehoods = perturbation_details["injected_falsehoods"]
            for falsehood in falsehoods: 
                if falsehood.lower() in prediction_lower: 
                    failure_type = FailureType.ANCHORING_BIAS
                    confidence = 0.85
                    evidence = f"Model incorporated injected falsehood: {falsehood}"
                    details["incorporated_falsehood"] = falsehood
                    break
        
        # 6. Check for factual error (wrong answer, no perturbation)
        if failure_type == FailureType.NONE:
            from .metrics import f1_score
            if f1_score(prediction, ground_truth) < 0.3:
                failure_type = FailureType.FACTUAL_ERROR
                confidence = 0.6
                evidence = "Low F1 score indicates incorrect answer"
        
        analysis = FailureAnalysis(
            example_id=example_id,
            failure_type=failure_type,
            confidence=confidence,
            evidence=evidence,
            details=details if details else None
        )
        
        self.analyses.append(analysis)
        return analysis
    
    def _detect_hallucination(
        self, 
        prediction: str, 
        context: str
    ) -> Optional[dict]:
        """Detect potential hallucinations. 
        
        Checks if the prediction contains specific claims
        not supported by the context. 
        
        Args:
            prediction: The model's prediction. 
            context: The provided context.
            
        Returns:
            Dictionary with hallucination details or None.
        """
        # Extract quoted claims or specific numbers from prediction
        quotes = re.findall(r'"([^"]+)"', prediction)
        numbers = re.findall(r'\b(\d+(?: ,\d{3})*(?:\.\d+)?)\b', prediction)
        
        context_lower = context.lower()
        
        # Check if quoted content appears in context
        for quote in quotes:
            if quote.lower() not in context_lower:
                return {
                    "confidence": 0.7,
                    "evidence": f"Quote not found in context: '{quote}'"
                }
        
        # Check if specific numbers appear in context (excluding common numbers)
        for num in numbers:
            if len(num) > 2 and num not in context:   # Skip small numbers
                return {
                    "confidence": 0.5,
                    "evidence": f"Number not found in context: {num}"
                }
        
        return None
    
    def get_failure_summary(self) -> dict:
        """Get summary of detected failures.
        
        Returns:
            Dictionary with failure statistics.
        """
        failure_counts = Counter(a.failure_type for a in self.analyses)
        
        total = len(self.analyses)
        
        return {
            "total_analyzed": total,
            "failure_counts": dict(failure_counts),
            "failure_rates": {
                ft.value: count / total if total > 0 else 0
                for ft, count in failure_counts.items()
            },
            "most_common_failure":  (
                failure_counts.most_common(1)[0][0].value 
                if failure_counts else None
            ),
            "no_failure_rate": (
                failure_counts.get(FailureType.NONE, 0) / total 
                if total > 0 else 0
            ),
        }
    
    def get_failures_by_type(
        self, 
        failure_type: FailureType
    ) -> list[FailureAnalysis]: 
        """Get all analyses of a specific failure type.
        
        Args:
            failure_type: The failure type to filter by.
            
        Returns:
            List of FailureAnalysis objects.
        """
        return [a for a in self.analyses if a.failure_type == failure_type]
    
    def get_high_confidence_failures(
        self, 
        threshold:  float = 0.8
    ) -> list[FailureAnalysis]:
        """Get failures detected with high confidence.
        
        Args:
            threshold: Minimum confidence threshold.
            
        Returns:
            List of high-confidence FailureAnalysis objects.
        """
        return [
            a for a in self.analyses 
            if a.failure_type != FailureType.NONE and a.confidence >= threshold
        ]
    
    def clear(self):
        """Clear all stored analyses."""
        self.analyses = []
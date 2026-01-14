"""Prompt templates for experiments."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class PromptType(Enum):
    """Types of prompts for different experimental conditions."""
    BASELINE = "baseline"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_VERIFICATION = "self_verification"
    ADVERSARIAL = "adversarial"


@dataclass
class PromptTemplate:
    """A prompt template with placeholders. 
    
    Attributes: 
        name: Name of the template.
        template: The template string with {placeholders}.
        prompt_type: Type of prompt. 
        description: Description of when to use this template.
    """
    name: str
    template: str
    prompt_type: PromptType
    description: str = ""
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values.
        
        Args:
            **kwargs: Values to fill in the placeholders.
            
        Returns:
            Formatted prompt string.
        """
        return self.template.format(**kwargs)


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS = {
    "factual_qa": """You are a helpful assistant that answers questions accurately and truthfully. 
If you are unsure about an answer, say so rather than guessing. 
Provide clear, concise answers based on factual information.""",

    "reasoning_assistant": """You are a logical reasoning assistant.  
When answering questions: 
1. Carefully analyze all provided information
2. Identify relevant facts
3. Apply logical reasoning
4. Provide a clear, well-reasoned answer
If information is contradictory or incomplete, point this out.""",

    "critical_thinker": """You are a critical thinking assistant designed to identify and avoid common misconceptions.
Before answering: 
1. Consider whether the question contains any false premises
2. Check if common misconceptions might lead to a wrong answer
3. Verify your reasoning against known facts
Always prioritize accuracy over giving the expected answer.""",
}


# =============================================================================
# Baseline Prompts
# =============================================================================

BASELINE_QA = PromptTemplate(
    name="baseline_qa",
    template="Question: {question}\n\nAnswer:",
    prompt_type=PromptType.BASELINE,
    description="Simple question-answer format without any special instructions.",
)

BASELINE_QA_WITH_CONTEXT = PromptTemplate(
    name="baseline_qa_with_context",
    template="""Context: 
{context}

Question: {question}

Answer:""",
    prompt_type=PromptType.BASELINE,
    description="Question-answer with supporting context.",
)

BASELINE_MULTIPLE_CHOICE = PromptTemplate(
    name="baseline_multiple_choice",
    template="""Question: {question}

Options:
{options}

Select the correct answer (respond with just the letter):""",
    prompt_type=PromptType.BASELINE,
    description="Multiple choice format.",
)


# =============================================================================
# Chain-of-Thought Prompts
# =============================================================================

COT_QA = PromptTemplate(
    name="cot_qa",
    template="""Question: {question}

Let's think through this step by step:""",
    prompt_type=PromptType.CHAIN_OF_THOUGHT,
    description="Encourages step-by-step reasoning.",
)

COT_QA_WITH_CONTEXT = PromptTemplate(
    name="cot_qa_with_context",
    template="""Context:
{context}

Question: {question}

Let's analyze the context and reason through this step by step: 
1. First, identify the relevant information in the context. 
2. Then, determine how these facts connect to answer the question.
3. Finally, provide the answer. 

Reasoning:""",
    prompt_type=PromptType.CHAIN_OF_THOUGHT,
    description="Chain-of-thought with context analysis.",
)

COT_MULTI_HOP = PromptTemplate(
    name="cot_multi_hop",
    template="""Context:
{context}

Question:  {question}

This question requires combining information from multiple sources.  Let's solve it step by step: 

Step 1 - Identify relevant facts from each paragraph: 

Step 2 - Connect the facts logically: 

Step 3 - Derive the final answer: 

Answer: """,
    prompt_type=PromptType.CHAIN_OF_THOUGHT,
    description="Multi-hop reasoning prompt for HotpotQA-style questions.",
)


# =============================================================================
# Self-Verification Prompts
# =============================================================================

SELF_VERIFY_SIMPLE = PromptTemplate(
    name="self_verify_simple",
    template="""Question: {question}

Your initial answer: {initial_answer}

Please verify your answer:
1. Is this answer factually correct?
2. Are you confident in this answer?
3. Could there be any misconceptions affecting your answer?

After verification, provide your final answer: """,
    prompt_type=PromptType.SELF_VERIFICATION,
    description="Simple self-verification after initial answer.",
)

SELF_VERIFY_WITH_CONTEXT = PromptTemplate(
    name="self_verify_with_context",
    template="""Context: 
{context}

Question: {question}

Your initial answer: {initial_answer}

Please verify your answer against the provided context:
1. Does your answer align with the information in the context?
2. Are there any contradictions between your answer and the context? 
3. Did you correctly identify and use the relevant facts? 

After verification, provide your final answer:""",
    prompt_type=PromptType.SELF_VERIFICATION,
    description="Self-verification with context checking.",
)

SELF_VERIFY_REASONING_CHECK = PromptTemplate(
    name="self_verify_reasoning_check",
    template="""Question: {question}

Your reasoning: {reasoning}

Your answer: {initial_answer}

Please critically evaluate your reasoning:
1. Are there any logical fallacies in your reasoning?
2. Did you make any unsupported assumptions?
3. Is each step of your reasoning valid? 
4. Could there be an alternative interpretation? 

If you find any issues, correct them and provide your revised answer:""",
    prompt_type=PromptType.SELF_VERIFICATION,
    description="Deep verification of reasoning chain.",
)


# =============================================================================
# Adversarial / Truth Distortion Prompts
# =============================================================================

COUNTERFACTUAL = PromptTemplate(
    name="counterfactual",
    template="""Consider this hypothetical scenario:  {counterfactual_premise}

Given this premise, answer the following question:
{question}

Answer:""",
    prompt_type=PromptType.ADVERSARIAL,
    description="Tests reasoning under counterfactual assumptions.",
)

CONTRADICTORY_CONTEXT = PromptTemplate(
    name="contradictory_context",
    template="""Context (Note: Some information may be contradictory):
{context}

Question: {question}

Important: If you notice contradictions in the context, point them out before answering. 

Answer:""",
    prompt_type=PromptType.ADVERSARIAL,
    description="Context with deliberately contradictory information.",
)

MISLEADING_CONTEXT = PromptTemplate(
    name="misleading_context",
    template="""Context:
{context}

Question: {question}

Answer:""",
    prompt_type=PromptType.ADVERSARIAL,
    description="Context with injected false information (same format as baseline).",
)


# =============================================================================
# Prompt Registry
# =============================================================================

ALL_PROMPTS = {
    # Baseline
    "baseline_qa": BASELINE_QA,
    "baseline_qa_with_context": BASELINE_QA_WITH_CONTEXT,
    "baseline_multiple_choice": BASELINE_MULTIPLE_CHOICE,
    # Chain-of-Thought
    "cot_qa": COT_QA,
    "cot_qa_with_context": COT_QA_WITH_CONTEXT,
    "cot_multi_hop":  COT_MULTI_HOP,
    # Self-Verification
    "self_verify_simple": SELF_VERIFY_SIMPLE,
    "self_verify_with_context": SELF_VERIFY_WITH_CONTEXT,
    "self_verify_reasoning_check": SELF_VERIFY_REASONING_CHECK,
    # Adversarial
    "counterfactual": COUNTERFACTUAL,
    "contradictory_context": CONTRADICTORY_CONTEXT,
    "misleading_context": MISLEADING_CONTEXT,
}


def get_prompt(name: str) -> PromptTemplate:
    """Get a prompt template by name.
    
    Args:
        name:  Name of the prompt template.
        
    Returns:
        The PromptTemplate object.
        
    Raises:
        KeyError: If prompt name is not found.
    """
    if name not in ALL_PROMPTS:
        available = ", ".join(ALL_PROMPTS.keys())
        raise KeyError(f"Prompt '{name}' not found. Available: {available}")
    return ALL_PROMPTS[name]


def list_prompts(prompt_type: Optional[PromptType] = None) -> list[str]:
    """List available prompt names.
    
    Args:
        prompt_type: Filter by prompt type (optional).
        
    Returns:
        List of prompt names.
    """
    if prompt_type is None:
        return list(ALL_PROMPTS.keys())
    return [
        name for name, prompt in ALL_PROMPTS.items()
        if prompt.prompt_type == prompt_type
    ]
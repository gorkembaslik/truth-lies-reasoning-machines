"""LLM model clients and prompt templates."""

from .base_llm import BaseLLM, LLMResponse
from .gemini_client import GeminiClient
from .github_models_client import GitHubModelsClient
from .prompts import (
    PromptTemplate,
    PromptType,
    SYSTEM_PROMPTS,
    ALL_PROMPTS,
    get_prompt,
    list_prompts,
)

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMResponse",
    # Clients
    "GeminiClient",
    "GitHubModelsClient",
    # Prompts
    "PromptTemplate",
    "PromptType",
    "SYSTEM_PROMPTS",
    "ALL_PROMPTS",
    "get_prompt",
    "list_prompts",
]
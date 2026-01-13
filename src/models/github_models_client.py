"""GitHub Models API client."""

import os
import requests
from typing import Optional

from . base_llm import BaseLLM, LLMResponse


class GitHubModelsClient(BaseLLM):
    """Client for GitHub Models API.
    
    GitHub Models provides free access to various LLMs including
    GPT-4o-mini, Llama, Mistral, and others.
    
    Example:
        >>> client = GitHubModelsClient(model_name="gpt-4o-mini")
        >>> response = client.generate("What is the capital of France?")
        >>> print(response.text)
        Paris
    
    Environment Variables:
        GITHUB_TOKEN: Your GitHub personal access token.
    """
    
    API_BASE_URL = "https://models.inference.ai. azure.com"
    
    # Available models on GitHub Models
    AVAILABLE_MODELS = [
        # OpenAI models
        "gpt-4o",
        "gpt-4o-mini",
        # Meta models
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
        # Mistral models
        "Mistral-Large-2411",
        "Mistral-Small-24B-Instruct-2501",
        # Cohere
        "Cohere-command-r",
        "Cohere-command-r-plus",
    ]
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the GitHub Models client.
        
        Args:
            model_name: Name of the model to use.
            temperature: Sampling temperature (0.0 = deterministic).
            max_retries: Maximum number of retry attempts. 
            retry_delay: Delay between retries in seconds.
        """
        super().__init__(model_name, temperature, max_retries, retry_delay)
        self._token = self._get_token()
    
    def _get_token(self) -> str:
        """Get the GitHub token from environment."""
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GITHUB_TOKEN environment variable is required. "
                "Create a token at https://github.com/settings/tokens"
            )
        return token
    
    def _call_api(self, prompt:  str, max_tokens: int) -> LLMResponse:
        """Make the API call to GitHub Models.
        
        Args:
            prompt: The input prompt. 
            max_tokens: Maximum tokens in the response.
            
        Returns:
            LLMResponse object.
        """
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content":  prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(
            f"{self.API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        if response.status_code != 200:
            raise Exception(
                f"GitHub Models API error: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        
        # Extract response data
        choice = data["choices"][0]
        usage = data.get("usage", {})
        
        return LLMResponse(
            text=choice["message"]["content"],
            model=data. get("model", self.model_name),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage. get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=choice. get("finish_reason"),
            raw_response=data,
        )
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate a response with a system prompt. 
        
        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's input.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            LLMResponse object.
        """
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens":  max_tokens,
        }
        
        response = requests.post(
            f"{self.API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        if response.status_code != 200:
            raise Exception(
                f"GitHub Models API error: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        
        return LLMResponse(
            text=choice["message"]["content"],
            model=data.get("model", self.model_name),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=choice.get("finish_reason"),
            raw_response=data,
        )
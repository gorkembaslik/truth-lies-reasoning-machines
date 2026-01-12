"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class LLMResponse:
    """Standardized response from any LLM. 
    
    Attributes: 
        text: The generated text response.
        model: Name of the model that generated the response.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
        finish_reason: Why the model stopped generating.
        latency_ms: Response time in milliseconds.
        raw_response: The complete raw response from the API.
    """
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    latency_ms:  Optional[float] = None
    raw_response: Optional[dict] = field(default=None, repr=False)


class BaseLLM(ABC):
    """Abstract base class for LLM clients.
    
    All LLM implementations (Gemini, GitHub Models, etc.) should inherit
    from this class to ensure a consistent interface.
    
    Example:
        >>> llm = GeminiClient(model_name="gemini-1.5-flash")
        >>> response = llm.generate("What is the capital of France?")
        >>> print(response.text)
    """
    
    def __init__(
        self, 
        model_name: str, 
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the LLM client. 
        
        Args:
            model_name: Name of the model to use.
            temperature: Sampling temperature (0.0 = deterministic).
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Delay between retries in seconds.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._total_tokens_used = 0
        self._total_requests = 0
    
    @abstractmethod
    def _call_api(self, prompt: str, max_tokens: int) -> LLMResponse:
        """Make the actual API call.  Must be implemented by subclasses. 
        
        Args:
            prompt: The formatted prompt to send. 
            max_tokens: Maximum tokens in the response.
            
        Returns:
            LLMResponse object. 
        """
        pass
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """Generate a response for the given prompt.
        
        Includes automatic retry logic for transient failures.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            LLMResponse object with the generated text and metadata.
            
        Raises:
            Exception: If all retry attempts fail.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self._call_api(prompt, max_tokens)
                response.latency_ms = (time.time() - start_time) * 1000
                
                # Track usage
                self._total_requests += 1
                if response.total_tokens: 
                    self._total_tokens_used += response.total_tokens
                
                return response
                
            except Exception as e: 
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise last_exception
    
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
            LLMResponse object with the generated text and metadata. 
        """
        combined_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        return self.generate(combined_prompt, max_tokens)
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics for this client instance.
        
        Returns:
            Dictionary with usage statistics.
        """
        return {
            "model":  self.model_name,
            "total_requests": self._total_requests,
            "total_tokens_used":  self._total_tokens_used,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}', temp={self.temperature})"
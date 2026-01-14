"""Google Gemini API client via Vertex AI."""

import os
from typing import Optional

from .base_llm import BaseLLM, LLMResponse


class GeminiClient(BaseLLM):
    """Client for Google's Gemini models via Vertex AI. 
    
    This client supports both the google-generativeai library (simpler)
    and the Vertex AI SDK (for production use with service accounts).
    
    Example:
        >>> client = GeminiClient(model_name="gemini-1.5-flash")
        >>> response = client.generate("What is the capital of France?")
        >>> print(response.text)
        Paris
    
    Environment Variables:
        GOOGLE_API_KEY: API key for google-generativeai library.
        GOOGLE_CLOUD_PROJECT: Project ID for Vertex AI.
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON. 
    """
    
    # Available Gemini models
    AVAILABLE_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ]
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        max_retries:  int = 3,
        retry_delay: float = 1.0,
        use_vertex_ai: bool = False,
    ):
        """Initialize the Gemini client.
        
        Args:
            model_name: Name of the Gemini model to use.
            temperature: Sampling temperature (0.0 = deterministic).
            max_retries: Maximum number of retry attempts. 
            retry_delay: Delay between retries in seconds.
            use_vertex_ai: If True, use Vertex AI SDK; else use google-generativeai.
        """
        super().__init__(model_name, temperature, max_retries, retry_delay)
        self.use_vertex_ai = use_vertex_ai
        self._client = None
        self._model = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the appropriate Google AI client."""
        if self.use_vertex_ai:
            self._initialize_vertex_ai()
        else:
            self._initialize_genai()
    
    def _initialize_genai(self) -> None:
        """Initialize using google-generativeai library."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required."
                "Install it with: pip install google-generativeai"
            )
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: 
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required."
                "Get your API key from https://makersuite.google.com/app/apikey"
            )
        
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_name)
        self._genai = genai
    
    def _initialize_vertex_ai(self) -> None:
        """Initialize using Vertex AI SDK."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex AI."
                "Install it with: pip install google-cloud-aiplatform"
            )
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not project_id: 
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI."
            )
        
        vertexai.init(project=project_id, location=location)
        self._model = GenerativeModel(self.model_name)
    
    def _call_api(self, prompt: str, max_tokens: int) -> LLMResponse:
        """Make the API call to Gemini.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            LLMResponse object.
        """
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": max_tokens,
        }
        
        response = self._model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        # Extract token counts if available
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            completion_tokens = getattr(usage, "candidates_token_count", None)
            if prompt_tokens and completion_tokens:
                total_tokens = prompt_tokens + completion_tokens
        
        # Get finish reason
        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)
        
        return LLMResponse(
            text=response.text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            raw_response={"response": str(response)},
        )
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate a response with a system prompt.
        
        Gemini handles system prompts differently - we prepend them to the user prompt.
        
        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's input.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            LLMResponse object. 
        """
        combined_prompt = f"""Instructions: {system_prompt}

User Query: {user_prompt}

Response:"""
        return self.generate(combined_prompt, max_tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            Number of tokens. 
        """
        if self.use_vertex_ai:
            response = self._model.count_tokens(text)
            return response.total_tokens
        else:
            response = self._model.count_tokens(text)
            return response.total_tokens
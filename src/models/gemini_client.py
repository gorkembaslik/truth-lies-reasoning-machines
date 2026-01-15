"""Google Gemini API client via Vertex AI or Google AI Studio."""

import os
from typing import Optional

from google import genai
from google.genai import types

from .base_llm import BaseLLM, LLMResponse


class GeminiClient(BaseLLM):
    """Client for Google's Gemini models. 
    
    Supports two authentication methods:
    1. Google AI Studio (simple API key) - set GOOGLE_API_KEY
    2. Vertex AI (service account) - set GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS
    
    The client automatically detects which method to use based on available
    environment variables.
    
    Example:
        >>> client = GeminiClient(model_name="gemini-2.0-flash-lite-001")
        >>> response = client.generate("What is the capital of France?")
        >>> print(response.text)
        Paris
    
    Environment Variables:
        GOOGLE_API_KEY: API key for Google AI Studio (simpler method).
        GOOGLE_CLOUD_PROJECT: Project ID for Vertex AI. 
        GOOGLE_CLOUD_LOCATION: Region for Vertex AI (default: us-central1).
        GOOGLE_APPLICATION_CREDENTIALS:  Path to service account JSON for Vertex AI.
    """
    
    # Available Gemini models
    AVAILABLE_MODELS = [
        "gemini-live-2.5-flash-native-audio",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
    ]
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-lite-001",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_vertex_ai: Optional[bool] = None,
    ):
        """Initialize the Gemini client.
        
        Args:
            model_name: Name of the Gemini model to use.
            temperature: Sampling temperature (0.0 = deterministic).
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
            use_vertex_ai: Force Vertex AI (True) or AI Studio (False).
                          If None, auto-detects based on environment variables.
        """
        super().__init__(model_name, temperature, max_retries, retry_delay)
        
        # Auto-detect which API to use
        if use_vertex_ai is None:
            self.use_vertex_ai = self._should_use_vertex_ai()
        else:
            self.use_vertex_ai = use_vertex_ai
            
        self._model = None
        self._initialize_client()
    
    def _should_use_vertex_ai(self) -> bool:
        """Determine whether to use Vertex AI based on environment variables."""
        has_vertex_ai = (
            os.getenv("GOOGLE_CLOUD_PROJECT") is not None and
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is not None
        )
        has_api_key = os.getenv("GOOGLE_API_KEY") is not None
        
        # Prefer Vertex AI if configured (uses cloud credits)
        if has_vertex_ai: 
            return True
        elif has_api_key:
            return False
        else:
            raise ValueError(
                "No Google credentials found. Please set either:\n"
                "  - GOOGLE_API_KEY for Google AI Studio, or\n"
                "  - GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS for Vertex AI"
            )
    
    def _initialize_client(self) -> None:
        """Initialize the appropriate Google AI client."""
        if self.use_vertex_ai:
            self._initialize_vertex_ai()
        else:
            self._initialize_genai()
    
    def _initialize_genai(self) -> None:
        """Initialize using Google Gen AI SDK (AI Studio mode)."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: 
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required."
                "Get your API key from https://aistudio.google.com/app/apikey"
            )
        
        self._client = genai.Client(api_key=api_key)
        self._generation_module = "genai"
        print(f"Initialized Gemini via Google AI Studio (model: {self.model_name})")
    
    def _initialize_vertex_ai(self) -> None:
        """Initialize using Google Gen AI SDK (Vertex AI mode)."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not project_id: 
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI."
            )
        
        # Set credentials path for Google Cloud SDK
        if credentials_path:
            # Resolve relative paths
            if not os.path.isabs(credentials_path):
                # Try relative to current directory and project root
                if os.path.exists(credentials_path):
                    credentials_path = os.path.abspath(credentials_path)
                else:
                    # Try from project root
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    alt_path = os.path.join(project_root, credentials_path.lstrip('./'))
                    if os.path.exists(alt_path):
                        credentials_path = alt_path
                    else:
                        raise FileNotFoundError(
                            f"Credentials file not found:  {credentials_path}\n"
                            f"Also tried: {alt_path}"
                        )
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        # Initialize Vertex AI client
        self._client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self._generation_module = "vertexai"
        print(f"Initialized Gemini via Vertex AI (project: {project_id}, model: {self.model_name})")
    
    def _call_api(self, prompt: str, max_tokens: int) -> LLMResponse:
        """Make the API call to Gemini.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            LLMResponse object. 
        """
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        # Extract token counts if available
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count
            completion_tokens = usage.candidates_token_count
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
        
        # Get finish reason
        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)
        
        # Extract text safely
        try:
            text = response.text
        except ValueError:
            # Handle blocked responses
            text = "[Response blocked by safety filters]"
            finish_reason = "SAFETY"
        
        return LLMResponse(
            text=text,
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
        
        Gemini handles system prompts by prepending them to the user prompt.
        
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
        response = self._client.models.count_tokens(
            model=self.model_name,
            contents=text
        )
        return response.total_tokens
    
    def get_api_type(self) -> str:
        """Get the API type being used.
        
        Returns:
            'vertex_ai' or 'ai_studio'
        """
        return "vertex_ai" if self.use_vertex_ai else "ai_studio"
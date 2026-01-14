"""Google Gemini API client via Vertex AI or Google AI Studio."""

import os
from typing import Optional

from .base_llm import BaseLLM, LLMResponse


class GeminiClient(BaseLLM):
    """Client for Google's Gemini models. 
    
    Supports two authentication methods:
    1. Google AI Studio (simple API key) - set GOOGLE_API_KEY
    2. Vertex AI (service account) - set GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS
    
    The client automatically detects which method to use based on available
    environment variables.
    
    Example:
        >>> client = GeminiClient(model_name="gemini-1.5-flash")
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
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro-002",
        "gemini-1.0-pro",
    ]
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
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
        """Initialize using google-generativeai library (AI Studio)."""
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
        self._generation_module = "genai"
        print(f"Initialized Gemini via Google AI Studio (model: {self.model_name})")
    
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
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self._model = GenerativeModel(self.model_name)
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
        response = self._model.count_tokens(text)
        return response.total_tokens
    
    def get_api_type(self) -> str:
        """Get the API type being used.
        
        Returns:
            'vertex_ai' or 'ai_studio'
        """
        return "vertex_ai" if self.use_vertex_ai else "ai_studio"
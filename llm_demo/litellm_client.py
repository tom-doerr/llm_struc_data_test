"""LiteLLM client implementation for unified LLM API access."""

import litellm


class LiteLLMClient:  # pylint: disable=too-few-public-methods
    """Client for interacting with LiteLLM's unified API.

    Args:
        api_key: API key for the LLM provider being used through LiteLLM
    """

    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.api_key = api_key
        self.model = "gpt-3.5-turbo"

    def generate(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate response for given prompt using LiteLLM's unified API.

        Args:
            prompt: Input text to generate response for. Must not be empty.
            model: Model ID to use for generation (default: gpt-3.5-turbo).
                   Supports any LiteLLM supported model.

        Returns:
            str: Generated response text.

        Raises:
            ValueError: If prompt is empty or contains only whitespace.
            litellm.exceptions.APIError: For API-related errors from LiteLLM.
            RuntimeError: For other unexpected errors during generation.

        Example:
            >>> client = LiteLLMClient(api_key="sk-...")
            >>> response = client.generate("Hello AI", model="gpt-4")
        """
        # Validate input before making API call
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
        )
        return response.choices[0].message.content

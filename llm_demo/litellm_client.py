"""LiteLLM client implementation for unified LLM API access."""

import litellm


class LiteLLMClient:  # pylint: disable=too-few-public-methods
    """Client for interacting with LiteLLM's unified API.

    Args:
        api_key: API key for the LLM provider being used through LiteLLM
    """

    def __init__(self, api_key: str):
        """Initialize with API key"""
        litellm.api_key = api_key

    def generate(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:  # pylint: disable=too-many-arguments
        """Generate response for given prompt.

        Args:
            prompt: Input text to generate response for
            model: Model ID to use for generation

        Returns:
            Generated response text

        Raises:
            ValueError: If prompt is empty
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        response = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

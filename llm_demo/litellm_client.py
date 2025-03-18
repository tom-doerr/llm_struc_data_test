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
            prompt: Input text to generate response for. Must not be empty after stripping.
                Actual minimum length requirements depend on the model used.
            model: Model ID to use for generation (default: gpt-3.5-turbo).
                See LiteLLM docs for supported models.

        Returns:
            str: Generated response text. Structure varies by model provider.

        Raises:
            ValueError: For invalid inputs (empty prompt, wrong types)
            litellm.exceptions.APIError: For API errors (invalid key, quota exceeded)
            TimeoutError: When response exceeds timeout threshold
            RuntimeError: For unexpected errors during generation

        Examples:
            >>> client = LiteLLMClient(api_key="valid-key")
            >>> client.generate("Briefly explain quantum computing")
            "Quantum computing uses qubits to perform multidimensional computations..."

        Typical error scenarios:
            >>> client.generate("")  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Prompt cannot be empty
        """
        # Validate input types and values
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not isinstance(model, str):
            raise TypeError(f"Model must be a string, got {type(model)}")
        if not self.api_key:
            raise ValueError("API key is required for LiteLLM client")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            timeout=10,  # Add timeout for better reliability
        )
        return response.choices[0].message.content

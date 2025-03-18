"""LiteLLM client implementation for unified LLM API access."""

import litellm


class LiteLLMClient:  # pylint: disable=too-few-public-methods
    """Client for interacting with LiteLLM's unified API"""

    def __init__(self, api_key: str):
        """Initialize with API key"""
        litellm.api_key = api_key

    def generate(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate response for given prompt"""
        response = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

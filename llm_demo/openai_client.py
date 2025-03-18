"""OpenAI client implementation for direct API access."""

from openai import OpenAI


class OpenAIClient:  # pylint: disable=too-few-public-methods
    """Client for interacting with OpenAI's chat API.

    Args:
        api_key: OpenAI API key for authentication
    """

    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
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
        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

"""OpenAI client implementation for direct API access."""
from openai import OpenAI

class OpenAIClient:
    """Client for interacting with OpenAI's chat API"""
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate response for given prompt"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

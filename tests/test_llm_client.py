"""Unit tests for LLM client implementations."""

from unittest.mock import Mock  # pylint: disable=unused-import
import pytest
from llm_demo.openai_client import OpenAIClient
from llm_demo.litellm_client import LiteLLMClient


@pytest.fixture(name="mock_openai_response")
def mock_openai_response_fixture():
    """Fixture providing mock OpenAI response structure"""
    return Mock(
        choices=[
            Mock(message=Mock(content="Test response from OpenAI"))
        ]
    )


@pytest.fixture(name="mock_litellm_response")
def mock_litellm_response_fixture():
    """Fixture providing mock LiteLLM response structure"""
    return Mock(
        choices=[
            Mock(message=Mock(content="Test response from LiteLLM"))
        ]
    )


def test_openai_client_generate(mocker, mock_openai_response):
    """Test OpenAI client generates response correctly"""
    # Setup mock
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_create.return_value = mock_openai_response
    # Test execution
    client = OpenAIClient(api_key="test-key")
    response = client.generate("Test prompt")
    # Verify
    assert "Test response from OpenAI" in response
    mock_create.assert_called_once_with(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test prompt"}]
    )


def test_litellm_client_generate(mocker, mock_litellm_response):
    """Test LiteLLM client generates response correctly"""
    # Setup mock
    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = mock_litellm_response
    # Test execution
    client = LiteLLMClient(api_key="test-key")
    response = client.generate("Test prompt")
    # Verify
    assert "Test response from LiteLLM" in response
    mock_completion.assert_called_once_with(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test prompt"}]
    )

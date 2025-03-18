"""Unit tests for LLM client implementations."""

from unittest.mock import Mock
import pytest
from pytest_mock.plugin import MockerFixture
from llm_demo.openai_client import OpenAIClient
from llm_demo.litellm_client import LiteLLMClient


@pytest.fixture(name="mock_llm_response")
def mock_llm_response_fixture() -> Mock:
    """Fixture providing mock LLM response structure compatible with both OpenAI and LiteLLM.

    LiteLLM normalizes responses to match OpenAI's format, so we can use the same mock structure.

    Returns:
        Mock: A mock response object with consistent structure
    """
    return Mock(choices=[Mock(message=Mock(content="Test response from LLM"))])


@pytest.fixture(name="openai_client_data")
def openai_client_fixture() -> tuple[type, str, str]:
    """Fixture providing OpenAI client test data"""
    return (
        OpenAIClient,
        "llm_demo.openai_client.OpenAI.chat.completions.create",
        "Test response from LLM",
    )


@pytest.fixture(name="litellm_client_data")
def litellm_client_fixture() -> tuple[type, str, str]:
    """Fixture providing LiteLLM client test data"""
    return (
        LiteLLMClient,
        "litellm.completion",
        "Test response from LLM",
    )


@pytest.mark.parametrize(
    "client_data",
    [
        pytest.param(pytest.lazy_fixture("openai_client_data"), marks=pytest.mark.openai),
        pytest.param(pytest.lazy_fixture("litellm_client_data"), marks=pytest.mark.litellm),
    ],
    ids=["openai_client", "litellm_client"],
)
@pytest.mark.filterwarnings("ignore:(open_text is deprecated|DeprecationWarning)")
def test_llm_client_generate(
    mocker: MockerFixture,
    mock_llm_response: Mock,
    client_data: tuple[type, str, str],  # Fixture data tuple
):
    """Parameterized test for LLM client implementations."""
    # Unpack fixture data
    client_class, mock_path, expected_response = client_data
    # Setup mock
    mock_create = mocker.patch(mock_path)
    mock_create.return_value = mock_llm_response
    # Test valid prompt
    client = client_class(api_key="test-key")
    response = client.generate("Valid prompt")
    assert expected_response in response
    # Verify common parameters and client-specific API key handling
    if client_class == OpenAIClient:
        mock_create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Valid prompt"}],
            timeout=10,
        )
    else:
        mock_create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Valid prompt"}],
            api_key="test-key",
            timeout=10,
        )
    # Test empty prompt validation
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("")

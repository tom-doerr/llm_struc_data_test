"""Unit tests for LLM client implementations."""

from unittest.mock import Mock
import pytest
import pytest_lazyfixture
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


@pytest.fixture(name="llm_clients")
def client_classes_fixture() -> list[tuple[type, str, str]]:
    """Fixture providing list of LLM client classes and their mock paths.

    Returns:
        list: Tuples of (client_class, mock_path, expected_response_snippet)
    """
    return [
        (
            OpenAIClient,
            "llm_demo.openai_client.OpenAI.chat.completions.create",
            "Test response from LLM",
        ),
        (
            LiteLLMClient,
            "litellm.completion",
            "Test response from LLM",
        ),
    ]


@pytest.mark.parametrize(
    "client_class, mock_path, expected_response",
    [
        pytest.param(
            pytest_lazyfixture.lazy_fixture("llm_clients[0][0]"),  # client_class
            pytest_lazyfixture.lazy_fixture("llm_clients[0][1]"),  # mock_path
            pytest_lazyfixture.lazy_fixture("llm_clients[0][2]"),  # expected_response
            id="openai_client",
        ),
        pytest.param(
            pytest_lazyfixture.lazy_fixture("llm_clients[1][0]"),  # client_class
            pytest_lazyfixture.lazy_fixture("llm_clients[1][1]"),  # mock_path
            pytest_lazyfixture.lazy_fixture("llm_clients[1][2]"),  # expected_response
            id="litellm_client",
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:open_text is deprecated")  # For litellm
def test_llm_client_generate(
    mocker, mock_llm_response, client_class, mock_path, expected_response
):
    """Parameterized test for LLM client implementations.

    Verifies that all client implementations:
    1. Call the correct API endpoint
    2. Return the expected response format
    3. Handle basic prompt validation
    """
    # Setup mock
    mock_create = mocker.patch(mock_path)
    mock_create.return_value = mock_llm_response
    # Test valid prompt
    client = client_class(api_key="test-key")
    response = client.generate("Valid prompt")
    assert expected_response in response
    mock_create.assert_called_once_with(
        model="gpt-3.5-turbo",  # LiteLLM normalizes model names
        messages=[{"role": "user", "content": "Valid prompt"}],
        api_key="test-key",
    )
    # Test empty prompt validation
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("")

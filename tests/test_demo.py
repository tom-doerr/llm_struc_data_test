"""Tests for the LLM demo program."""

from unittest.mock import patch, Mock
from click.testing import CliRunner
from llm_demo.demo import main, run_llm_inference


def test_run_llm_inference():
    """Test the core LLM inference function with mock."""
    mock_client = Mock()
    mock_client.generate.return_value = "Mocked response"

    result = run_llm_inference(mock_client, "test prompt")
    mock_client.generate.assert_called_once_with("test prompt")
    assert result == "Mocked response"


def test_main_with_mocks():
    """Test the full CLI workflow with mocked client."""
    runner = CliRunner()
    with patch("llm_demo.demo.LiteLLMClient") as mock_client:
        mock_client.return_value.generate.return_value = "Mocked response"

        result = runner.invoke(main, ["--prompt", "test prompt"])
        assert result.exit_code == 0
        assert "Mocked response" in result.output
        mock_client.return_value.generate.assert_called_once_with("test prompt")


def test_empty_prompt_handling(mocker: MockerFixture):
    """Test the CLI handles empty prompts properly."""
    runner = CliRunner()
    mock_client = mocker.patch("llm_demo.demo.LiteLLMClient")
    
    result = runner.invoke(main, ["--prompt", ""])
    
    assert result.exit_code == 1
    assert "Error: Invalid value for '--prompt': Prompt cannot be empty" in result.output
    mock_client.assert_not_called()

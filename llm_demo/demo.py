"""Command-line interface for LLM inference demo."""

import click
import litellm
from llm_demo.litellm_client import LiteLLMClient


def run_llm_inference(client: object, prompt: str) -> str:
    """Run LLM inference with comprehensive error handling.

    Args:
        client: Initialized LLM client supporting generate() method
        prompt: User input to process. Must be non-empty after stripping whitespace.

    Returns:
        str: Generated response or error message with troubleshooting guidance

    Raises:
        ValueError: For empty prompts
        RuntimeError: For critical infrastructure failures

    Examples:
        >>> client = LiteLLMClient(api_key="test")
        >>> run_llm_inference(client, "Hello")
        'Generated response...'
    """
    try:
        return client.generate(prompt)
    except ValueError as err:
        return f"Validation Error: {str(err)}"
    except ConnectionError as err:
        return f"Connection Error: {str(err)} - check network connection"
    except litellm.exceptions.APIError as err:
        return f"API Error: {str(err)} - check API key and provider status"
    except TimeoutError as err:
        return f"Timeout Error: {str(err)} - consider shortening your prompt"
    except (RuntimeError, Exception) as err:  # pylint: disable=broad-except
        return (
            f"System Error: {str(err)} - contact support" 
            if isinstance(err, RuntimeError) 
            else f"Unexpected Error: {str(err)} - contact support with details"
        )


@click.command()
@click.option("--prompt", required=True, help="Input prompt for the LLM")
@click.option(
    "--api-key",
    envvar="LLM_API_KEY",
    required=True,
    help="API key for LLM service (can also use LLM_API_KEY environment variable)",
)
def main(prompt: str, api_key: str):
    """Command-line interface for LLM inference demo.

    Example:
        llm-demo --prompt "Hello AI" --api-key sk-...
    """
    # Validate prompt before initializing client
    if not prompt.strip():
        raise click.BadParameter("Prompt cannot be empty")

    client = LiteLLMClient(api_key=api_key)
    response = run_llm_inference(client, prompt)
    if any(
        response.startswith(prefix)
        for prefix in ("Validation Error", "Connection Error", "Unexpected Error")
    ):
        click.echo(f"Error: {response}", err=True)
        raise SystemExit(1)
    click.echo(f"Response: {response}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

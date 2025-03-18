"""Command-line interface for LLM inference demo."""

import click
from llm_demo.litellm_client import LiteLLMClient


def run_llm_inference(client, prompt: str) -> str:
    """Run LLM inference with proper error handling.

    Args:
        client: Initialized LiteLLM client
        prompt: User input to process

    Returns:
        Generated response or error message
    """
    try:
        return client.generate(prompt)
    except Exception as err:  # pylint: disable=broad-except
        return f"Error: {str(err)}"


@click.command()
@click.option("--prompt", required=True, help="Input prompt for the LLM")
@click.option(
    "--api-key", envvar="LLM_API_KEY", required=True, help="API key for LLM service"
)
def main(prompt: str, api_key: str):
    """Command-line interface for LLM inference demo."""
    client = LiteLLMClient(api_key=api_key)
    response = run_llm_inference(client, prompt)
    click.echo(f"Response: {response}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

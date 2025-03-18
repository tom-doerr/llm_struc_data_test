"""Package configuration for LLM demo clients."""

from setuptools import setup, find_packages

setup(
    name="llm_demo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "litellm>=1.0.1",  # Updated to current version
        "click>=8.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.3.4",
            "pytest-mock>=3.14.0",
            "pytest-lazy-fixture>=0.6.3",  # Required for parametrized tests
            "pytest-randomly>=1.2.3",      # For test randomization
            "pytest-cov>=4.1.0",          # For coverage reporting
            "pytest-asyncio>=0.24.0",
            "litellm>=0.10.0",
            "openai>=1.0.0",
            "pylint>=3.2.0",
            "types-pytest>=8.3.4",  # For type hint completeness
        ]
    },
)

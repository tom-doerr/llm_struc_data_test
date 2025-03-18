"""Package configuration for LLM demo clients."""

from setuptools import setup, find_packages

setup(
    name="llm_demo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "litellm>=0.10.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.3.4",
            "pytest-mock>=3.14.0",
            "pytest-lazy-fixture>=0.6.3",
            "pylint>=3.2.0",
        ]
    },
)

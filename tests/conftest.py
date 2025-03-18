"""Pytest configuration and shared fixtures for all tests."""

import pytest

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "openai: mark tests that interact with OpenAI API"
    )
    config.addinivalue_line(
        "markers", "litellm: mark tests that interact with LiteLLM API"
    )

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Prevent sleep calls in tests to speed up execution"""
    monkeypatch.setattr("time.sleep", lambda _: None)

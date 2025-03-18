"""Pytest configuration and shared fixtures for all tests."""
import pytest

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Prevent sleep calls in tests to speed up execution"""
    monkeypatch.setattr("time.sleep", lambda _: None)

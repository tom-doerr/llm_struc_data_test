"""Tests for utility functions."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from project_name.utils import calculate_total

def test_calculate_total_basic():
    """Test basic calculation scenarios."""
    assert calculate_total([1, 2, 3]) == 6
    assert calculate_total([]) == 0
    assert calculate_total([-1, 0, 1]) == 0

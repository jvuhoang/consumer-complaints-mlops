"""Pytest configuration and fixtures"""

import pytest


@pytest.fixture
def sample_complaint():
    """Sample complaint text for testing"""
    return "I have a problem with my credit card charges"


@pytest.fixture
def sample_complaints():
    """Multiple sample complaints for testing"""
    return [
        "I dispute the charges on my credit report",
        "My mortgage payment was not processed correctly",
        "I am being harassed by debt collectors"
    ]

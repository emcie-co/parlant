"""Tests for the guideline classifier."""

from typing import TYPE_CHECKING, Dict, List

import pytest

from parlant.dspy_integration.guideline_classifier import GuidelineClassifier

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

@pytest.fixture
def classifier() -> GuidelineClassifier:
    """Create a guideline classifier for testing.
    
    Returns:
        GuidelineClassifier instance
    """
    return GuidelineClassifier(use_optimizer=False)

def test_classifier_initialization(classifier: GuidelineClassifier) -> None:
    """Test that classifier initializes correctly.
    
    Args:
        classifier: GuidelineClassifier fixture
    """
    assert classifier is not None
    assert classifier.predictor is not None
    assert not classifier.use_optimizer

def test_classifier_prediction(classifier: GuidelineClassifier) -> None:
    """Test that classifier makes predictions.
    
    Args:
        classifier: GuidelineClassifier fixture
    """
    # Test conversation and guidelines
    conversation = "User: I need help with my account\nAssistant: I'll help you"
    guidelines = ["Account support", "Technical issues", "Billing"]
    
    # Get prediction
    result = classifier(conversation=conversation, guidelines=guidelines)
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "activated" in result
    assert isinstance(result["activated"], list)
    assert len(result["activated"]) == len(guidelines)
    assert all(isinstance(x, bool) for x in result["activated"])

def test_classifier_error_handling(
        classifier: GuidelineClassifier,
        caplog: "LogCaptureFixture"
    ) -> None:
    """Test classifier error handling.
    
    Args:
        classifier: GuidelineClassifier fixture
        caplog: Pytest log capture fixture
    """
    # Test with invalid input
    result = classifier(conversation=None, guidelines=["test"])  # type: ignore
    
    # Should return all guidelines inactive
    assert result == {"activated": [False]}
    
    # Should log error
    assert "Failed to classify guidelines" in caplog.text

def test_classification_quality_calculation(classifier: GuidelineClassifier) -> None:
    """Test quality score calculation.
    
    Args:
        classifier: GuidelineClassifier fixture
    """
    # Test cases
    test_cases = [
        (
            {"activated": [True, False, True]},
            {"activated": [True, False, True]},
            1.0  # Perfect match
        ),
        (
            {"activated": [True, False, True]},
            {"activated": [True, True, True]},
            0.666  # 2/3 correct
        ),
        (
            {"activated": [False, False, False]},
            {"activated": [True, True, True]},
            0.0  # No matches
        ),
        (
            {"activated": []},
            {"activated": []},
            0.0  # Empty lists
        ),
        (
            {},
            {"activated": [True]},
            0.0  # Missing key
        )
    ]
    
    for pred, gold, expected in test_cases:
        score = classifier._calculate_classification_quality(pred, gold)
        assert abs(score - expected) < 0.001  # Account for float precision

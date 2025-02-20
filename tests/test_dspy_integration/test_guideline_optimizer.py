"""Tests for the BatchOptimizedGuidelineManager."""

import os
from datetime import datetime
from typing import TYPE_CHECKING, List
import pytest
import dspy

from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.logging import Logger
from parlant.dspy_integration.guideline_optimizer import (
    BatchOptimizedGuidelineManager,
    GuidelineProgram
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestLogger:
    """Test logger that captures log messages."""
    
    def __init__(self) -> None:
        """Initialize the test logger."""
        self.messages: List[str] = []
        
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        msg = f"DEBUG: {message}"
        print(msg)  # Print for debugging
        self.messages.append(msg)
        
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        msg = f"INFO: {message}"
        print(msg)  # Print for debugging
        self.messages.append(msg)
        
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        msg = f"WARNING: {message}"
        print(msg)  # Print for debugging
        self.messages.append(msg)
        
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        msg = f"ERROR: {message}"
        print(msg)  # Print for debugging
        self.messages.append(msg)
        
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        msg = f"CRITICAL: {message}"
        print(msg)  # Print for debugging
        self.messages.append(msg)
        
    def operation(self, message: str, **kwargs) -> None:
        """Log operation message."""
        msg = f"OPERATION: {message}"
        print(msg)  # Print for debugging
        self.messages.append(msg)
        
    def set_level(self, level: str) -> None:
        """Set log level."""
        pass


@pytest.fixture
def sample_guidelines() -> List[Guideline]:
    """Create sample guidelines for testing."""
    return [
        Guideline(
            id=GuidelineId("g1"),
            creation_utc=datetime.now(),
            content=GuidelineContent(
                condition="when the customer asks about pricing",
                action="Explain our pricing tiers and mention the free trial"
            )
        ),
        Guideline(
            id=GuidelineId("g2"),
            creation_utc=datetime.now(),
            content=GuidelineContent(
                condition="when the customer reports a technical issue",
                action="Ask for specific error messages and system information"
            )
        ),
    ]


@pytest.fixture
def sample_examples() -> List[dict]:
    """Create sample training examples for testing."""
    return [
        {
            "input": "How much does your service cost?",
            "output": "We offer three pricing tiers: Basic ($10/mo), Pro ($25/mo), and Enterprise (custom pricing). All plans come with a 14-day free trial."
        },
        {
            "input": "I'm getting an error when I try to login",
            "output": "I understand you're having trouble logging in. Could you please share the exact error message you're seeing? Also, what browser and operating system are you using?"
        }
    ]


@pytest.mark.asyncio
async def test_dspy_program(sample_examples: List[dict]) -> None:
    """Test that the DSPy GuidelineProgram works correctly."""
    # Configure DSPy
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    dspy.configure(model='gpt-3.5-turbo')
    
    # Create and test program
    program = GuidelineProgram(api_key=api_key)
    
    # Test with a sample condition
    condition = "when the customer asks about pricing"
    result = program.forward(condition)
    
    # Verify result
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert "price" in result.lower() or "cost" in result.lower() or "tier" in result.lower()


@pytest.mark.asyncio
async def test_optimize_guidelines(
    sample_guidelines: List[Guideline],
    sample_examples: List[dict]
) -> None:
    """Test guideline optimization using both DSPy and Parlant."""
    # Setup
    logger = TestLogger()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
        
    optimizer = BatchOptimizedGuidelineManager(
        logger=logger,
        num_threads=2,
        api_key=api_key
    )
    
    # Run optimization
    optimized = await optimizer.optimize_guidelines(
        guidelines=sample_guidelines,
        examples=sample_examples,
        batch_size=2
    )
    
    # Verify results
    assert len(optimized) == len(sample_guidelines)
    for guideline in optimized:
        # Test Parlant integration
        assert isinstance(guideline, Guideline)
        assert guideline.id in [g.id for g in sample_guidelines]
        assert guideline.content.condition
        assert guideline.content.action
        
        # Test that responses are contextually appropriate
        if "pricing" in guideline.content.condition:
            assert any(word in guideline.content.action.lower() 
                      for word in ["price", "cost", "tier", "trial"])
        if "technical" in guideline.content.condition:
            assert any(word in guideline.content.action.lower() 
                      for word in ["error", "system", "browser", "issue"])
    
    # Print all messages for debugging
    print("\nAll logger messages:")
    for msg in logger.messages:
        print(msg)
    
    # Verify logging
    assert any("Starting optimization" in msg for msg in logger.messages)
    assert any("Successfully optimized" in msg for msg in logger.messages)

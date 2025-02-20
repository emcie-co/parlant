"""Example script to demonstrate guideline optimization using DSPy."""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict

from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.logging import Logger
from parlant.core.common import generate_id
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager


class SimpleLogger(Logger):
    """A simple logger implementation."""
    
    def __init__(self) -> None:
        """Initialize the logger."""
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        
        # Add console handler if no handlers exist
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self._logger.addHandler(handler)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self._logger.critical(message)
    
    def operation(self, operation_id: str, message: str) -> None:
        """Log an operation message."""
        self._logger.info(f"Operation {operation_id}: {message}")
    
    def set_level(self, level: str) -> None:
        """Set the logging level."""
        self._logger.setLevel(level)


async def main() -> None:
    """Run the guideline optimization example."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Create a simple logger
    logger = SimpleLogger()

    # Create some example guidelines
    guidelines: List[Guideline] = [
        Guideline(
            id=GuidelineId(generate_id()),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition="when the customer asks about pricing",
                action="We have different pricing options"
            )
        ),
        Guideline(
            id=GuidelineId(generate_id()),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition="when the customer reports a bug",
                action="Let me help you with that issue"
            )
        ),
        Guideline(
            id=GuidelineId(generate_id()),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition="when the customer wants to upgrade their plan",
                action="I can help you upgrade"
            )
        )
    ]

    # Create example responses for training
    examples: List[Dict[str, str]] = [
        {
            "input": "How much does your service cost?",
            "output": "We offer three pricing tiers: Basic ($10/mo), Pro ($25/mo), and Enterprise (custom pricing). The Basic plan includes essential features, Pro adds advanced capabilities, and Enterprise comes with all features plus dedicated support. Which features are most important for your needs?"
        },
        {
            "input": "I found a bug in the app",
            "output": "I apologize for the inconvenience. Could you please share the exact error message you're seeing? Also, what browser and operating system are you using? This information will help us investigate and resolve the issue more quickly."
        }
    ]

    # Create the optimizer
    optimizer = BatchOptimizedGuidelineManager(api_key=api_key, logger=logger)

    # Print original guidelines
    print("\nOriginal Guidelines:")
    print("------------------")
    for guideline in guidelines:
        print(f"\nCondition: {guideline.content.condition}")
        print(f"Action: {guideline.content.action}")

    # Optimize the guidelines
    optimized = await optimizer.optimize_guidelines(guidelines, examples)

    # Print optimized guidelines
    print("\nOptimized Guidelines:")
    print("------------------")
    for guideline in optimized:
        print(f"\nCondition: {guideline.content.condition}")
        print(f"Action: {guideline.content.action}")


if __name__ == "__main__":
    asyncio.run(main())

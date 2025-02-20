"""Test script for the BatchOptimizedGuidelineManager."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any

from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.logging import Logger
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager


class SimpleLogger(Logger):
    """A simple logger implementation for testing."""
    
    def debug(self, message: str, **kwargs) -> None:
        print(f"DEBUG: {message}")
        
    def info(self, message: str, **kwargs) -> None:
        print(f"INFO: {message}")
        
    def warning(self, message: str, **kwargs) -> None:
        print(f"WARNING: {message}")
        
    def error(self, message: str, **kwargs) -> None:
        print(f"ERROR: {message}")
        
    def critical(self, message: str, **kwargs) -> None:
        print(f"CRITICAL: {message}")
        
    def operation(self, message: str, **kwargs) -> None:
        print(f"OPERATION: {message}")
        
    def set_level(self, level: str) -> None:
        pass


# Sample guidelines for testing
SAMPLE_GUIDELINES = [
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

# Sample training examples
SAMPLE_EXAMPLES = [
    {
        "input": "How much does your service cost?",
        "output": "We offer three pricing tiers: Basic ($10/mo), Pro ($25/mo), and Enterprise (custom pricing). All plans come with a 14-day free trial."
    },
    {
        "input": "I'm getting an error when I try to login",
        "output": "I understand you're having trouble logging in. Could you please share the exact error message you're seeing? Also, what browser and operating system are you using?"
    }
]

async def main():
    """Run the guideline optimization test."""
    logger = SimpleLogger()  # Use our simple logger implementation
    
    # Create the optimizer
    optimizer = BatchOptimizedGuidelineManager(logger=logger, num_threads=2)
    
    # Run optimization
    optimized_guidelines = await optimizer.optimize_guidelines(
        guidelines=SAMPLE_GUIDELINES,
        examples=SAMPLE_EXAMPLES,
        batch_size=2
    )
    
    # Print results
    print("\nOptimization Results:")
    print("--------------------")
    for guideline in optimized_guidelines:
        print(f"\nGuideline {guideline.id}:")
        print(f"Condition: {guideline.content.condition}")
        print(f"Action: {guideline.content.action}")

if __name__ == "__main__":
    asyncio.run(main())

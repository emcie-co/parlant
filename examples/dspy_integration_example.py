#!/usr/bin/env python3
"""
DSPy Integration Example

This example demonstrates how to use the DSPy integration with Parlant
to enhance guideline optimization and classification.
"""

import asyncio
import os
from typing import List, Optional, Dict, Any

from lagom import Container

from parlant.core.logging import Logger, LogLevel
from parlant.dspy_integration import initialize_module
from parlant.dspy_integration.guideline_classifier import GuidelineClassifier
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager
from parlant.dspy_integration.config import DSPyConfig


async def basic_initialization_example() -> None:
    """Demonstrate basic DSPy integration initialization."""
    # Create logger and container
    logger = Logger("dspy-example")
    logger.set_level(LogLevel.DEBUG)
    container = Container()
    
    # Add logger to container
    container.register(Logger, logger)
    
    # Initialize DSPy integration
    container = await initialize_module(container)
    
    # The container now has DSPy services registered
    logger.info("DSPy integration initialized successfully")


async def guideline_classification_example() -> None:
    """Demonstrate guideline classification with DSPy."""
    logger = Logger("dspy-classification")
    logger.set_level(LogLevel.DEBUG)
    
    # Create a guideline classifier
    classifier = GuidelineClassifier(logger=logger)
    
    # Define conversation context
    conversation = """
    User: I need to reset my password for my account.
    Assistant: I'll help you with your password reset. Can you tell me the email address associated with your account?
    User: It's example@example.com
    Assistant: Thank you. For security purposes, I'll need to verify your identity.
    """
    
    # Define some guidelines
    guidelines = [
        "When a user asks about password reset, always verify their identity first",
        "When a user mentions billing issues, provide information about our payment plans",
        "When a user wants to cancel their subscription, try to retain them by offering a discount",
        "When a user provides personal information, remind them about our privacy policy"
    ]
    
    # Classify which guidelines should be activated
    logger.info("Classifying guidelines...")
    result = classifier(conversation=conversation, guidelines=guidelines)
    
    # Display results
    activated = result.get("activated", [])
    for i, guideline in enumerate(guidelines):
        status = "✅ ACTIVE" if activated[i] else "❌ INACTIVE"
        logger.info(f"{status}: {guideline}")


async def guideline_optimization_example() -> None:
    """Demonstrate guideline optimization with DSPy COPRO."""
    logger = Logger("dspy-optimization")
    logger.set_level(LogLevel.DEBUG)
    
    # Create a guideline optimizer
    optimizer = BatchOptimizedGuidelineManager(logger=logger, use_optimizer=True)
    
    # Define some guidelines with examples
    guidelines = [
        {
            "id": "password-reset",
            "content": "When a user asks about password reset, always verify their identity first",
            "examples": [
                {
                    "context": "User is trying to reset their password",
                    "input": "I forgot my password",
                    "response": "I'd be happy to help you reset your password. For security purposes, could you please verify your email address and provide your account username?"
                }
            ]
        },
        {
            "id": "privacy-policy",
            "content": "When a user provides personal information, remind them about our privacy policy",
            "examples": [
                {
                    "context": "User has shared their email address",
                    "input": "My email is user@example.com",
                    "response": "Thank you for providing your email address. Just a reminder that all personal information is handled according to our privacy policy, which you can review at any time on our website."
                }
            ]
        }
    ]
    
    # Optimize the guidelines
    logger.info("Optimizing guidelines...")
    optimized = await optimizer.optimize_guidelines(guidelines)
    
    # Display results
    for i, result in enumerate(optimized):
        logger.info(f"Optimized Guideline #{i+1}: {result.get('improvement', 0):.2f} improvement")
        logger.info(f"Original: {guidelines[i]['content']}")
        logger.info(f"Optimized: {result.get('optimized_guideline', 'No optimization')}")


async def configuration_example() -> None:
    """Demonstrate DSPy configuration options."""
    # Set environment variables
    os.environ["DSPY_MODEL"] = "openai/gpt-4"
    os.environ["DSPY_TEMPERATURE"] = "0.7"
    os.environ["DSPY_MAX_TOKENS"] = "4000"
    
    # Create logger
    logger = Logger("dspy-config")
    logger.set_level(LogLevel.DEBUG)
    
    # Get configuration
    config = DSPyConfig()
    
    # Display configuration
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"Max tokens: {config.max_tokens}")
    logger.info(f"Batch size: {config.optimizer_batch_size}")


async def main() -> None:
    """Run all examples."""
    examples = [
        ("Basic Initialization", basic_initialization_example),
        ("Configuration", configuration_example),
        ("Guideline Classification", guideline_classification_example),
        ("Guideline Optimization", guideline_optimization_example),
    ]
    
    # Run each example
    for name, example in examples:
        print(f"\n{'=' * 50}")
        print(f"  EXAMPLE: {name}")
        print(f"{'=' * 50}\n")
        try:
            await example()
            print(f"\n✅ {name} completed successfully\n")
        except Exception as e:
            print(f"\n❌ {name} failed: {str(e)}\n")


if __name__ == "__main__":
    asyncio.run(main()) 
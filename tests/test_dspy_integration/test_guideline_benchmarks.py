"""Benchmark tests for guideline optimization with different models and configurations."""

import os
import asyncio
import pytest
import subprocess
import time
from typing import List, Dict, Any
from datetime import datetime, timezone
import json

import dspy
from dspy import Example

from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.common import generate_id
from parlant.core.logging import get_logger
from parlant.dspy_integration.guideline_optimizer import (
    BatchOptimizedGuidelineManager,
    ModelMetrics
)

def ensure_ollama_running() -> None:
    """Ensure Ollama server is running and the model is pulled."""
    try:
        # Check if Ollama server is running
        result = subprocess.run(['pgrep', 'ollama'], capture_output=True)
        if result.returncode != 0:
            # Start Ollama server in background
            subprocess.Popen(['ollama', 'serve'], 
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
            time.sleep(2)  # Wait for server to start
            
        # Pull Llama2 model if not present
        subprocess.run(['ollama', 'pull', 'llama2'], check=True)
    except Exception as e:
        pytest.skip(f"Ollama setup failed: {str(e)}")

@pytest.fixture
def example_guidelines() -> List[Guideline]:
    """Create example guidelines for testing."""
    return [
        Guideline(
            id="1",
            creation_utc=datetime.now(),
            content=GuidelineContent(
                condition="Customer asks about pricing tiers",
                action="We offer three pricing tiers: Basic ($10/month), Pro ($25/month), and Enterprise (custom pricing)."
            )
        ),
        Guideline(
            id="2", 
            creation_utc=datetime.now(),
            content=GuidelineContent(
                condition="Customer reports a technical error",
                action="Could you please provide more details about the error you're experiencing? This will help me better assist you."
            )
        ),
        Guideline(
            id="3",
            creation_utc=datetime.now(),
            content=GuidelineContent(
                condition="Customer requests a refund",
                action="I understand you'd like a refund. Could you please explain the reason for your request? This will help me process it appropriately."
            )
        ),
        Guideline(
            id="4",
            creation_utc=datetime.now(),
            content=GuidelineContent(
                condition="Customer asks about feature availability",
                action="I'll be happy to explain our feature availability. Which specific feature are you interested in?"
            )
        ),
        Guideline(
            id="5",
            creation_utc=datetime.now(),
            content=GuidelineContent(
                condition="Customer reports slow performance",
                action="I'm sorry to hear about the performance issues. Could you tell me: 1) When did you first notice this? 2) Which specific operations are slow?"
            )
        )
    ]

@pytest.fixture
def example_training_data() -> List[Dict[str, str]]:
    """Create example training data for optimization."""
    return [
        {
            "input": "Customer asks about upgrading their subscription",
            "output": "I can help you upgrade your subscription. Currently, you can upgrade to our Pro tier ($25/month) or Enterprise tier (custom pricing). Which would you like to learn more about?"
        },
        {
            "input": "Customer can't log in to their account",
            "output": "I'm sorry you're having trouble logging in. To help you best: 1) Are you getting any specific error messages? 2) Have you tried resetting your password?"
        }
    ]

def calculate_response_quality(condition: str, response: str) -> float:
    score = 0.0
    
    # Length check (not too short, not too long)
    if 20 <= len(response) <= 200:
        score += 0.2
        
    # Contains relevant terminology
    keywords = ["pricing", "tier", "technical", "error", "feature", "subscription", "account"]
    if any(keyword in response.lower() for keyword in keywords):
        score += 0.2
        
    # Includes specific details
    if any(char.isdigit() for char in response) or "$" in response:
        score += 0.2
        
    # Professional language
    professional_terms = ["please", "thank", "assist", "help", "understand"]
    if any(term in response.lower() for term in professional_terms):
        score += 0.2
        
    # No error messages or debugging info
    error_terms = ["error:", "exception:", "debug:", "traceback"]
    if not any(term in response.lower() for term in error_terms):
        score += 0.2
        
    return score

@pytest.mark.asyncio
@pytest.mark.parametrize("use_optimizer", [True])
@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("model_name", ["openai/gpt-3.5-turbo", "ollama/llama2"])
async def test_guideline_optimization_benchmark(
    use_optimizer: bool,
    batch_size: int,
    model_name: str,
    example_guidelines: List[Guideline],
    example_training_data: List[Dict[str, str]]
) -> None:
    """Test guideline optimization with different configurations."""
    if "ollama" in model_name:
        ensure_ollama_running()

    logger = get_logger(__name__)

    # Initialize optimizer
    optimizer = BatchOptimizedGuidelineManager(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=model_name,
        use_optimizer=use_optimizer
    )

    # Run optimization
    optimized = optimizer.optimize_guidelines(
        guidelines=example_guidelines,
        examples=example_training_data,
        batch_size=batch_size
    )

    # Calculate average quality score
    avg_quality = sum(
        calculate_response_quality(g.content.condition, g.content.action)
        for g in optimized
    ) / len(optimized)

    # Print results
    print(f"\nResults for {model_name}:")
    print(f"Configuration: batch_size={batch_size}, optimizer={use_optimizer}")
    print(f"Average response quality: {avg_quality:.2f}")
    print(f"Total API calls: {optimizer.metrics.total_api_calls}")
    print(f"Total tokens: {optimizer.metrics.total_tokens}")
    print(f"Total time: {optimizer.metrics.total_time:.2f} seconds\n")
    print("Performance metrics:")

    # Assertions
    assert len(optimized) == len(example_guidelines)
    assert all(isinstance(g.content.action, str) for g in optimized)
    assert avg_quality > 0.5  # Ensure reasonable quality
    assert optimizer.metrics.total_api_calls > 0  # Ensure API calls are tracked
    assert optimizer.metrics.total_tokens > 0  # Ensure tokens are tracked
    assert optimizer.metrics.total_time > 0  # Ensure time is tracked

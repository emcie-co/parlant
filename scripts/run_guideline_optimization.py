"""Script to run guideline optimization with classification."""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import dspy
from dspy.teleprompt import COPRO

from parlant.core.guidelines import Guideline, GuidelineContent
from parlant.core.metrics import ModelMetrics
from parlant.core.ollama import initialize_ollama_model
from parlant.dspy_integration.guideline_classifier import GuidelineClassifier
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager, CustomerServiceProgram

# Tee output to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

f = open('output.log', 'a')
sys.stdout = Tee(sys.stdout, f)

# Example guidelines
GUIDELINES = [
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
    )
]

# Example training data
TRAINING_DATA = [
    {
        "condition": "Customer asks about upgrading their subscription",
        "response": "I can help you upgrade your subscription. Currently, you can upgrade to our Pro tier ($25/month) or Enterprise tier (custom pricing). Which would you like to learn more about?"
    },
    {
        "condition": "Customer can't log in to their account",
        "response": "I'm sorry you're having trouble logging in. To help you best: 1) Are you getting any specific error messages? 2) Have you tried resetting your password?"
    },
    {
        "condition": "Customer complains about app crashing",
        "response": "I apologize for the inconvenience. To help resolve this quickly: 1) What device/OS are you using? 2) When did the crashes start? 3) Does it happen during specific actions?"
    }
]

# Test conversations for classification
TEST_CONVERSATIONS = [
    (
        "User: I need help with my account\nAssistant: I'll help you",
        ["Account support", "Technical issues", "Billing"],
        {"activated": [True, False, False]}
    ),
    (
        "User: How much does it cost?\nAssistant: Let me check",
        ["Account support", "Technical issues", "Billing"],
        {"activated": [False, False, True]}
    ),
    (
        "User: The app is not working\nAssistant: I'll help troubleshoot",
        ["Account support", "Technical issues", "Billing"],
        {"activated": [False, True, False]}
    )
]

def run_optimization(model_name: str, batch_size: int = 5) -> None:
    """Run optimization with specified model.
    
    Args:
        model_name: Name of model to use
        batch_size: Batch size for optimization
    """
    print(f"\nRunning optimization with {model_name}")
    print("-" * 80)
    
    # Initialize metrics
    metrics = ModelMetrics()
    
    # Initialize classifier
    print("\nTesting Guideline Classification:")
    classifier = GuidelineClassifier(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=model_name,
        metrics=metrics,
        use_optimizer=True
    )
    
    # Run classification tests
    total_quality = 0.0
    for conversation, guidelines, expected in TEST_CONVERSATIONS:
        print(f"\nConversation: {conversation}")
        print(f"Guidelines: {guidelines}")
        print(f"Expected: {expected}")
        
        # Get predictions
        result = classifier(
            conversation=conversation,
            guidelines=guidelines
        )
        
        print(f"Predicted: {result}")
        
        # Calculate quality
        quality = classifier._calculate_classification_quality(result, expected)
        total_quality += quality
        print(f"Quality: {quality:.2f}")
    
    # Print classification metrics
    avg_quality = total_quality / len(TEST_CONVERSATIONS)
    print(f"\nClassification Metrics:")
    print(f"Average quality: {avg_quality:.2f}")
    print(f"API calls: {metrics.total_api_calls}")
    print(f"Tokens: {metrics.total_tokens}")
    print(f"Time: {metrics.total_time:.2f} seconds")
    
    # Initialize response optimizer
    print("\nOptimizing Response Generation:")
    optimizer = BatchOptimizedGuidelineManager(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=model_name,
        metrics=metrics,
        use_optimizer=True
    )
    
    # Configure optimizer
    if optimizer.use_optimizer:
        optimizer.optimizer = COPRO(
            prompt_model=optimizer.lm,
            init_temperature=1.0,      # Higher temperature for more diverse candidates
            breadth=12,                # Generate more candidates
            depth=4,                   # More iterations for refinement
            threshold=0.5,             # More lenient threshold
            top_k=3,                   # Keep top 3 candidates at each step
            max_steps=50,              # Allow more optimization steps
            metric=lambda pred, gold: CustomerServiceProgram()._calculate_response_quality(
                pred.get("response", "") if isinstance(pred, dict) else getattr(pred, "response", ""),
                gold.get("response", "") if isinstance(gold, dict) else getattr(gold, "response", "")
            )
        )
    
    # Run response optimization
    optimized = optimizer.optimize_guidelines(
        guidelines=GUIDELINES,
        examples=TRAINING_DATA,
        batch_size=batch_size
    )
    
    # Print response optimization results
    print("\nResponse Optimization Results:")
    for guideline in optimized:
        print(f"\nCondition: {guideline.content.condition}")
        print(f"Response: {guideline.content.action}")
    
    # Print final metrics
    print(f"\nFinal Metrics:")
    print(f"Total API calls: {metrics.total_api_calls}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total time: {metrics.total_time:.2f} seconds")
    print("-" * 80)

def main() -> None:
    """Run optimization with both OpenAI and Llama models."""
    # Initialize Ollama model
    print("\nInitializing Llama2 model...")
    initialize_ollama_model("llama2")
    
    # Run with OpenAI
    run_optimization("openai/gpt-3.5-turbo")
    
    # Run with Llama
    run_optimization("ollama/llama2")

if __name__ == "__main__":
    main()

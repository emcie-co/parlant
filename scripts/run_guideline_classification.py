"""Script to run guideline classification with different models."""

import os
import sys
from typing import List, Dict

from parlant.dspy_integration.guideline_classifier import GuidelineClassifier

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

f = open('classification_output.log', 'a')
sys.stdout = Tee(sys.stdout, f)

# Example conversations and guidelines
TEST_CONVERSATIONS = [
    "User: I can't log into my account\nAssistant: I'll help you with that",
    "User: How much does the premium plan cost?\nAssistant: Let me check the pricing",
    "User: The app is really slow\nAssistant: I'll help troubleshoot",
]

TEST_GUIDELINES = [
    "Account access and authentication",
    "Billing and pricing information",
    "Technical troubleshooting",
    "Feature requests",
    "General inquiries"
]

def run_classification(model_name: str) -> None:
    """Run classification with specified model.
    
    Args:
        model_name: Name of model to use
    """
    print(f"\nRunning classification with {model_name}")
    print("-" * 80)
    
    # Initialize classifier
    classifier = GuidelineClassifier(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=model_name,
        use_optimizer=True
    )
    
    # Run classification for each conversation
    for i, conversation in enumerate(TEST_CONVERSATIONS):
        print(f"\nTest conversation {i+1}:")
        print(conversation)
        print("\nGuidelines to check:")
        for j, guideline in enumerate(TEST_GUIDELINES):
            print(f"{j+1}. {guideline}")
            
        # Get predictions
        result = classifier(
            conversation=conversation,
            guidelines=TEST_GUIDELINES
        )
        
        # Print results
        print("\nActivated guidelines:")
        if "activated" in result:
            for guideline, activated in zip(TEST_GUIDELINES, result["activated"]):
                if activated:
                    print(f"- {guideline}")
        else:
            print("No guidelines activated")
                
        print(f"\nTotal API calls: {classifier.metrics.total_api_calls}")
        print(f"Total tokens: {classifier.metrics.total_tokens}")
        print(f"Total time: {classifier.metrics.total_time:.2f} seconds")
        print("-" * 80)

def main() -> None:
    """Run classification with OpenAI model."""
    run_classification("openai/gpt-3.5-turbo")

if __name__ == "__main__":
    main()

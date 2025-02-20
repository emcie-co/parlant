"""DSPy integration for classifying which guidelines should be activated."""

from typing import Dict, List, Optional
import logging

import dspy
from dspy.adapters.json_adapter import JSONAdapter
from dspy.teleprompt import COPRO

from parlant.core.metrics import ModelMetrics

# Configure logging
logger = logging.getLogger(__name__)

class GuidelineSignature(dspy.Signature):
    """Signature for determining which guidelines should be activated.
    
    This signature defines the input and output fields for classifying which
    guidelines should be activated based on the conversation context.
    
    Attributes:
        conversation: The conversation history between user and assistant
        guidelines: List of guidelines to check for activation
        output_json: Dict containing list of boolean values indicating activation
    """

    conversation = dspy.InputField(desc="The conversation between user and assistant")
    guidelines = dspy.InputField(desc="List of guidelines to check for activation")
    output_json = dspy.OutputField(
        desc="Dict containing list of boolean values indicating which guidelines to activate",
        type=Dict[str, List[bool]]  # More specific type hint
    )
    
    def __init__(self) -> None:
        """Initialize the signature with instructions."""
        super().__init__()
        self.instructions = """You are a guideline classifier that determines which customer service guidelines 
        should be activated based on the conversation context.
        
        Rules:
        1. Analyze the conversation carefully to understand the customer's needs
        2. For each guideline in the list, determine if it is relevant to the conversation
        3. Return a JSON object with an "activated" key containing a list of boolean values
        4. Each boolean value should correspond to whether that guideline should be activated
        5. Consider both explicit and implicit needs in the conversation
        
        Format:
        Input:
        - conversation: The chat history between user and assistant
        - guidelines: A bulleted list of guidelines to check
        
        Output:
        {"activated": [true/false values for each guideline]}
        
        Examples:
        conversation: "User: I can't log in\nAssistant: I'll help"
        guidelines:
        - Account access
        - Billing
        - Technical support
        -> {"activated": [true, false, false]}
        
        conversation: "User: What's the cost?\nAssistant: Let me check"
        guidelines:
        - Account access
        - Billing
        - Technical support
        -> {"activated": [false, true, false]}
        """


class GuidelineClassifier(dspy.Module):
    """Module for classifying which guidelines should be activated.
    
    This module uses DSPy to determine which guidelines should be activated
    based on the conversation context. It can be optimized using COPRO
    to improve classification accuracy.
    
    Attributes:
        metrics: Metrics tracker for monitoring model performance
        use_optimizer: Whether to use DSPy's COPRO optimizer
        lm: Language model instance for classification
        predictor: DSPy predictor for classification
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "openai/gpt-3.5-turbo",
            metrics: Optional[ModelMetrics] = None,
            use_optimizer: bool = True
        ) -> None:
        """Initialize the guideline classifier.
        
        Args:
            api_key: Optional API key for the model provider
            model_name: Name of the model to use
            metrics: Optional metrics tracker
            use_optimizer: Whether to use optimization
            
        Raises:
            ValueError: If an invalid model name is provided
            RuntimeError: If initialization fails
        """
        super().__init__()
        self.metrics = metrics or ModelMetrics()
        self.use_optimizer = use_optimizer
        
        try:
            # Configure language model
            if "ollama" in model_name:
                from parlant.dspy_integration.guideline_optimizer import OllamaAdapter, initialize_ollama_model
                # For Ollama models, use our custom adapter
                ollama_model = model_name.split("/")[1]
                initialize_ollama_model(ollama_model)
                self.lm = OllamaAdapter(model_name)
            else:
                # For other models like OpenAI, use JSONAdapter
                self.lm = dspy.LM(model_name, api_key=api_key)
                dspy.settings.configure(adapter=JSONAdapter())
                
            if metrics:
                from parlant.dspy_integration.guideline_optimizer import MetricsLogger
                self.lm = MetricsLogger(self.lm, metrics)
                
            # Configure DSPy with the language model
            dspy.configure(lm=self.lm)
            
            # Create predictor
            self.predictor = dspy.Predict(GuidelineSignature)
            
            # Configure optimizer if needed
            if use_optimizer:
                self.optimizer = COPRO(
                    prompt_model=self.lm,
                    metric=self._calculate_classification_quality,
                    breadth=5,  # Number of candidates
                    depth=3,  # Optimization iterations
                    max_steps=30  # Maximum steps
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize guideline classifier: {e}")
            raise RuntimeError(f"Failed to initialize guideline classifier: {e}") from e

    def forward(self, conversation: str, guidelines: List[str]) -> Dict[str, List[bool]]:
        """Classify which guidelines should be activated.
        
        Args:
            conversation: The conversation history
            guidelines: List of guidelines to check
            
        Returns:
            Dict containing list of boolean values indicating activation
        """
        try:
            # Format guidelines into a string
            guidelines_str = "\n".join(f"- {g}" for g in guidelines)
            
            # Use predictor to classify guidelines
            result = self.predictor(
                conversation=conversation,
                guidelines=guidelines_str
            )
            
            # Parse result
            if isinstance(result.output_json, str):
                import json
                try:
                    parsed = json.loads(result.output_json)
                except json.JSONDecodeError:
                    # If that fails, try evaluating as Python dict
                    try:
                        parsed = eval(result.output_json)
                    except Exception as e:
                        logger.error(f"Failed to parse output: {e}")
                        return {"activated": [False] * len(guidelines)}
            else:
                parsed = result.output_json
                
            # Convert dict format to list format if needed
            if isinstance(parsed, dict) and all(isinstance(k, str) for k in parsed.keys()):
                activated = [parsed.get(g, False) for g in guidelines]
                return {"activated": activated}
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to classify guidelines: {e}")
            # Return all guidelines as inactive on error
            return {"activated": [False] * len(guidelines)}
            
    def _calculate_classification_quality(
            self,
            pred: Dict[str, List[bool]],
            gold: Dict[str, List[bool]]
        ) -> float:
        """Calculate quality score for classification.
        
        Args:
            pred: Predicted activation values
            gold: Ground truth activation values
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Get predicted and actual activations
            pred_activations = pred.get("activated", [])
            gold_activations = gold.get("activated", [])
            
            if not pred_activations or not gold_activations:
                return 0.0
                
            # Calculate accuracy
            correct = sum(1 for p, g in zip(pred_activations, gold_activations) if p == g)
            total = len(gold_activations)
            
            return correct / total
            
        except Exception as e:
            logger.error(f"Failed to calculate classification quality: {e}")
            return 0.0

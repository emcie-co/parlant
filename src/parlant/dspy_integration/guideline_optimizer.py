"""DSPy integration for optimizing guidelines."""
import difflib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union, ClassVar, Callable
from dataclasses import dataclass
from copy import deepcopy
import requests

import dspy
from dspy import ChainOfThought, Example, LM, Module, Signature, Prediction, InputField, OutputField
from dspy.teleprompt import COPRO

from parlant.core.guidelines import Guideline, GuidelineContent
from parlant.core.metrics import ModelMetrics
from parlant.core.ollama import initialize_ollama_model

logger = logging.getLogger(__name__)

@dataclass
class OptimizationBatch:
    """A batch of guidelines to optimize together."""
    guidelines: List[Guideline]
    examples: List[Example]
    batch_size: int = 10
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def latency(self) -> Optional[float]:
        """Calculate batch processing latency in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class MetricsLogger:
    """Wrapper around language model to track metrics."""
    
    def __init__(self, lm: LM, metrics: ModelMetrics) -> None:
        """Initialize the metrics logger.
        
        Args:
            lm: Language model to wrap
            metrics: Metrics tracker
        """
        self.lm = lm
        self.metrics = metrics
        self.start_time = None
        
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped language model.
        
        Args:
            name: Name of attribute to access
            
        Returns:
            Attribute value
        """
        # Get the attribute from the wrapped LM
        attr = getattr(self.lm, name)
        
        # If it's a method, wrap it to track metrics
        if callable(attr):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if self.start_time is None:
                    self.start_time = time.time()
                
                self.metrics.total_api_calls += 1
                result = attr(*args, **kwargs)
                
                # Update total time
                self.metrics.total_time = time.time() - self.start_time
                
                # Handle different response formats
                if isinstance(result, dict):
                    # For Ollama responses
                    if 'response' in result:
                        # Rough estimate: 1 token per 4 characters
                        self.metrics.total_tokens += len(result['response']) // 4
                elif isinstance(result, list):
                    # For DSPy responses that are lists
                    for r in result:
                        if isinstance(r, str):
                            # Rough estimate: 1 token per 4 characters
                            self.metrics.total_tokens += len(r) // 4
                        elif isinstance(r, dict) and 'response' in r:
                            # For OpenAI-style responses in a list
                            self.metrics.total_tokens += len(r['response']) // 4
                elif hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
                    # For OpenAI responses
                    self.metrics.total_tokens += result.usage.total_tokens
                elif hasattr(result, 'usage') and isinstance(result.usage, dict) and 'total_tokens' in result.usage:
                    self.metrics.total_tokens += result.usage['total_tokens']
                
                return result
            return wrapper
        return attr
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Handle direct calls to the language model.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Model output
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        self.metrics.total_api_calls += 1
        result = self.lm(*args, **kwargs)
        
        # Update total time
        self.metrics.total_time = time.time() - self.start_time
        
        # Handle different response formats
        if isinstance(result, dict):
            # For Ollama responses
            if 'response' in result:
                # Rough estimate: 1 token per 4 characters
                self.metrics.total_tokens += len(result['response']) // 4
        elif isinstance(result, list):
            # For DSPy responses that are lists
            for r in result:
                if isinstance(r, str):
                    # Rough estimate: 1 token per 4 characters
                    self.metrics.total_tokens += len(r) // 4
                elif isinstance(r, dict) and 'response' in r:
                    # For OpenAI-style responses in a list
                    self.metrics.total_tokens += len(r['response']) // 4
        elif hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
            # For OpenAI responses
            self.metrics.total_tokens += result.usage.total_tokens
        elif hasattr(result, 'usage') and isinstance(result.usage, dict) and 'total_tokens' in result.usage:
            self.metrics.total_tokens += result.usage['total_tokens']
            
        return result


class OllamaAdapter(dspy.adapters.ChatAdapter):
    """Adapter for using Ollama models with DSPy.
    
    This adapter wraps an Ollama model to make it compatible with DSPy's
    language model interface. It handles the communication with the Ollama
    server and formats responses appropriately.
    
    Attributes:
        model_name: Name of the Ollama model being used
        _history: List of message dictionaries storing conversation history
        kwargs: Dictionary of default parameters for model requests
    """
    
    def __init__(self, model_name: str, callbacks: Optional[List[Callable]] = None) -> None:
        """Initialize the Ollama adapter.
        
        Args:
            model_name: Name of the Ollama model to use (e.g. 'ollama/llama2')
            callbacks: Optional list of callback functions
        """
        super().__init__(callbacks=callbacks)  # Initialize parent class with callbacks
        self.model_name = model_name.split("/")[1]  # Extract model name after 'ollama/'
        self._history: List[Dict[str, str]] = []  # Store message history
        
        # Set default kwargs for DSPy
        self.kwargs: Dict[str, Any] = {
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None
        }
        
    def _clean_response(self, response: str) -> str:
        """Clean and format the response from Ollama.
        
        This method handles various response formats and ensures they are
        properly formatted for DSPy.
        
        Args:
            response: Raw response from Ollama
            
        Returns:
            Cleaned and formatted response string
        """
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        # If response starts with '[', it's likely JSON-formatted
        if response.startswith("["):
            # Try to extract actual content
            try:
                # Look for actual content after the [
                content_start = response.find("[") + 1
                content_end = response.rfind("]")
                if content_start < content_end:
                    response = response[content_start:content_end].strip()
            except:
                pass
            
        # Remove any role prefixes
        for prefix in ["System:", "User:", "Assistant:"]:
            response = response.replace(prefix, "").strip()
        
        # If response is empty after cleaning, provide a default
        if not response:
            response = "I apologize, but I need more information to provide a helpful response."
            
        return response
    
    def __call__(self, prompt: str, **kwargs: Any) -> List[str]:
        """Call the Ollama model with a prompt.
        
        This method is called by DSPy when using the adapter as a language model.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments for the request
            
        Returns:
            List containing the model's response as a string
            
        Raises:
            requests.RequestException: If the API request fails
        """
        try:
            # Merge default kwargs with provided kwargs
            request_kwargs = self.kwargs.copy()
            request_kwargs.update(kwargs)
            
            # Make request to Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": request_kwargs["temperature"],
                    "top_p": request_kwargs["top_p"],
                    "max_tokens": request_kwargs["max_tokens"]
                }
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            cleaned_response = self._clean_response(result.get("response", ""))
            
            # Store in history
            self._history.append({
                "role": "user",
                "content": prompt
            })
            self._history.append({
                "role": "assistant",
                "content": cleaned_response
            })
            
            return [cleaned_response]
            
        except Exception as e:
            logger.error(f"Failed to get response from Ollama: {e}")
            return ["I apologize, but I am unable to process your request at the moment."]
            
    def format(self, signature: dspy.Signature, demos: List[Dict[str, Any]], inputs: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format messages for the Ollama model.
        
        Formats the input signature, demos, and inputs into a list of messages
        that can be sent to the Ollama model.
        
        Args:
            signature: The DSPy signature to format messages for
            demos: List of example dictionaries with input and output fields
            inputs: Dictionary of input fields for the current request
            
        Returns:
            List of message dictionaries with role and content fields
        """
        messages = super().format(signature, demos, inputs)
        self._history = messages  # Store formatted messages in history
        return messages
    
    def inspect_history(self, n: int = 1) -> List[Dict[str, str]]:
        """Inspect the history of messages for debugging.
        
        Args:
            n: Optional number of messages to return
            
        Returns:
            List of message dictionaries with role and content fields
        """
        if not self._history:
            return []
            
        if n is not None:
            # Return last n messages if specified
            return self._history[-n:]
            
        return self._history


class CustomerServiceSignature(dspy.Signature):
    """A signature for generating customer service responses.
    
    This signature defines the input and output fields for generating
    professional and helpful customer service responses based on given
    conditions.
    """
    
    inputs: ClassVar[List[str]] = ['condition']
    
    condition: str = dspy.InputField(
        desc="The customer service condition to respond to",
        prefix="CONDITION: ",
        default=""
    )
    
    response: str = dspy.OutputField(
        desc="A helpful and professional response to the condition",
        prefix="RESPONSE: ",
        default=""
    )
    
    instructions: ClassVar[str] = """You are a professional customer service representative.

    Your task is to generate helpful, empathetic, and professional responses to customer service inquiries.
    
    Rules:
    1. Always be polite and professional
    2. Show empathy and understanding
    3. Provide clear and actionable information
    4. Ask for clarification when needed
    5. Maintain a positive and helpful tone
    
    Format:
    condition -> response
    
    Examples:
    'Customer asks about pricing' -> 'Thank you for your interest in our pricing. I'd be happy to explain our different pricing tiers. Could you please let me know what specific features or plans you're interested in?'
    
    'Customer reports a bug' -> 'I apologize for the inconvenience you're experiencing. To help resolve this issue, could you please provide more details about the bug? This will help us investigate and fix it as quickly as possible.'
    
    'Customer requests a refund' -> 'I understand you'd like a refund. I'll be happy to help you with this process. Could you please provide your order number and the reason for the refund request?'
    
    Let's solve this step by step:
    1. Understand the customer's condition
    2. Show empathy and acknowledge their concern
    3. Provide relevant information or ask for clarification
    4. Offer clear next steps or solutions
    5. End with a professional and helpful tone
    """


class CustomerServiceProgram(dspy.Module):
    """A DSPy module that generates customer service responses based on conditions.
    
    This module uses a predictor to generate contextually appropriate
    and professional responses to customer service inquiries.
    """

    def __init__(self) -> None:
        """Initialize the CustomerServiceProgram with a predictor."""
        super().__init__()
        
        # Create the predictor with our signature
        self.predictor = dspy.Predict(CustomerServiceSignature)

    def forward(self, **inputs: Any) -> Dict[str, str]:
        """Generate a customer service response based on a condition.
        
        Args:
            **inputs: Input dictionary containing 'condition' key
            
        Returns:
            Dict containing the generated response
            
        Raises:
            ValueError: If condition is not provided or not a string
        """
        # Step 1: Get condition from inputs
        condition = inputs.get('condition')
        if not condition or not isinstance(condition, str):
            raise ValueError("Must provide 'condition' as a string")
        
        # Step 2: Process using predictor
        result = self.predictor(**inputs)
        
        # Step 3: Return response
        return {"response": result.response}

    def deepcopy(self) -> 'CustomerServiceProgram':
        """Create a deep copy of this program.
        
        Returns:
            A new instance of CustomerServiceProgram
        """
        return CustomerServiceProgram()

    def predictors(self) -> List[dspy.Predict]:
        """Get the predictors used by this program.
        
        Returns:
            A list containing the predictor
        """
        return [self.predictor]


class CustomCOPRO(dspy.teleprompt.COPRO):
    """A custom COPRO optimizer that handles extended_signature gracefully.
    
    This subclass overrides the behavior of setting extended_signature to avoid
    the error in DSPy 2.5.
    """
    
    def compile(
        self, 
        student: CustomerServiceProgram, 
        *, 
        trainset: List[dspy.Example],
        eval_kwargs: Dict[str, Any]
    ) -> Optional[CustomerServiceProgram]:
        """Compile the program with training examples.
        
        This method overrides the parent's compile method to handle the
        extended_signature attribute gracefully.
        
        Args:
            student: The program to optimize
            trainset: List of training examples
            eval_kwargs: Additional arguments for evaluation
            
        Returns:
            The optimized program, or None if optimization fails
        """
        try:
            # Try to compile normally
            return super().compile(student, trainset=trainset, eval_kwargs=eval_kwargs)
        except AttributeError as e:
            if "extended_signature" in str(e):
                # If the error is about extended_signature, just return the original program
                logger.warning("COPRO optimization failed due to extended_signature, using original program")
                return student
            raise


class BatchOptimizedGuidelineManager:
    """A manager for optimizing and generating customer service responses.
    
    This class uses DSPy's COPRO optimizer to improve response quality and
    handles batched optimization of guidelines. It supports both OpenAI and
    Ollama models, with appropriate configuration for each.
    
    Attributes:
        metrics: Metrics tracker for monitoring model performance
        use_optimizer: Whether to use DSPy's COPRO optimizer
        lm: Language model instance for generating responses
        program: Base program for generating customer service responses
    """
    
    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "openai/gpt-3.5-turbo",
            metrics: Optional[ModelMetrics] = None,
            use_optimizer: bool = True
        ) -> None:
        """Initialize the guideline manager.
        
        Args:
            api_key: Optional API key for the model provider
            model_name: Name of the model to use (e.g. 'openai/gpt-3.5-turbo' or 'ollama/llama2')
            metrics: Optional metrics tracker for monitoring model performance
            use_optimizer: Whether to use DSPy's COPRO optimizer for improving responses
            
        Raises:
            ValueError: If an invalid model name is provided
            RuntimeError: If initialization of the language model fails
        """
        self.metrics = metrics or ModelMetrics()
        self.use_optimizer = use_optimizer
        
        try:
            # Configure language model
            if "ollama" in model_name:
                # For Ollama models, use our custom adapter
                ollama_model = model_name.split("/")[1]
                initialize_ollama_model(ollama_model)
                self.lm = OllamaAdapter(model_name)
            else:
                # For other models like OpenAI, use all parameters
                self.lm = LM(model_name, api_key=api_key, temperature=0.7, max_tokens=200)
                
            if metrics:
                self.lm = MetricsLogger(self.lm, metrics)
                
            # Configure DSPy with the language model
            dspy.configure(lm=self.lm)
            
            # Create base program
            self.program = CustomerServiceProgram()
            
        except Exception as e:
            logger.error(f"Failed to initialize guideline manager: {e}")
            raise RuntimeError(f"Failed to initialize guideline manager: {e}") from e

    def optimize_guidelines(
            self, 
            guidelines: Sequence[Guideline],
            examples: List[Dict[str, str]],
            batch_size: int = 10
        ) -> List[Guideline]:
        """Optimize guidelines using DSPy's COPRO optimizer.
        
        This method takes a list of guidelines and example responses, then uses
        COPRO to optimize the response generation process. Guidelines are processed
        in batches to improve efficiency.
        
        Args:
            guidelines: List of guidelines to optimize
            examples: List of example dictionaries with 'condition' and 'response' keys
            batch_size: Number of guidelines to optimize at once
            
        Returns:
            List of optimized guidelines with improved responses
        """
        optimized_guidelines = []
        
        for i in range(0, len(guidelines), batch_size):
            batch = guidelines[i:i + batch_size]
            
            # Create training examples
            train_examples = []
            for example in examples:
                # Create example with proper input/output fields
                ex = dspy.Example(
                    condition=example['condition'],
                    response=example['response']
                ).with_inputs('condition')  # Explicitly set which fields are inputs
                train_examples.append(ex)
            
            try:
                # Configure COPRO optimizer with evaluation metric
                copro = CustomCOPRO(
                    prompt_model=MetricsLogger(self.lm, self.metrics),  # Wrap with metrics logger
                    metric=self._calculate_response_quality,
                    breadth=5,  # Number of new prompts to generate at each iteration
                    depth=3,  # Number of optimization iterations
                    threshold=0.5,  # Quality threshold for accepting optimized prompts
                    top_k=2,  # Number of top candidates to keep at each step
                    max_steps=30  # Maximum optimization steps
                )
                
                # Optimize program
                optimized_program = copro.compile(
                    student=self.program,
                    trainset=train_examples,
                    eval_kwargs={}  # Metric is already passed to COPRO constructor
                )
                
                if optimized_program:
                    # Use optimized program to generate responses
                    for guideline in batch:
                        pred = optimized_program(condition=guideline.content.condition)
                        response = pred.response if hasattr(pred, 'response') else pred['response']
                        
                        # Create new guideline with updated response
                        optimized_guidelines.append(Guideline(
                            id=guideline.id,
                            creation_utc=guideline.creation_utc,
                            content=GuidelineContent(
                                condition=guideline.content.condition,
                                action=response
                            )
                        ))
                else:
                    logger.warning("No valid candidates found during optimization, using original program")
                    for guideline in batch:
                        pred = self.program(condition=guideline.content.condition)
                        response = pred.response if hasattr(pred, 'response') else pred['response']
                        
                        # Create new guideline with updated response
                        optimized_guidelines.append(Guideline(
                            id=guideline.id,
                            creation_utc=guideline.creation_utc,
                            content=GuidelineContent(
                                condition=guideline.content.condition,
                                action=response
                            )
                        ))
                        
            except Exception as e:
                logger.warning(f"Failed to optimize batch: {str(e)}")
                for guideline in batch:
                    pred = self.program(condition=guideline.content.condition)
                    response = pred.response if hasattr(pred, 'response') else pred['response']
                    
                    # Create new guideline with updated response
                    optimized_guidelines.append(Guideline(
                        id=guideline.id,
                        creation_utc=guideline.creation_utc,
                        content=GuidelineContent(
                            condition=guideline.content.condition,
                            action=response
                        )
                    ))
                    
        return optimized_guidelines

    def _calculate_response_quality(
        self, 
        example: dspy.Example, 
        pred: Union[dspy.Prediction, Dict[str, str]]
    ) -> float:
        """Calculate the quality score for a predicted response.
        
        This method evaluates the quality of a predicted response by comparing it
        to the expected response in the example.
        
        Args:
            example: The example containing the expected response
            pred: Dictionary or Prediction containing the predicted response
            
        Returns:
            A quality score between 0 and 1, where higher is better
        """
        try:
            # Extract expected and predicted responses
            expected_response = example.response
            predicted_response = pred.response if hasattr(pred, 'response') else pred['response']
            
            # Basic quality checks
            if not predicted_response or not expected_response:
                return 0.0
                
            # Calculate base score from string similarity
            base_score = difflib.SequenceMatcher(None, predicted_response.lower(), expected_response.lower()).ratio()
            
            # Additional quality metrics
            quality_score = 0.0
            
            # Length check - responses should be reasonably sized
            if 20 <= len(predicted_response) <= 200:
                quality_score += 0.2
                
            # Check for professional language
            professional_terms = ["please", "thank", "assist", "help", "understand"]
            if any(term in predicted_response.lower() for term in professional_terms):
                quality_score += 0.2
                
            # Check for error messages
            error_terms = ["error:", "exception:", "debug:", "traceback"]
            if not any(term in predicted_response.lower() for term in error_terms):
                quality_score += 0.2
                
            # Check for specific details
            if any(char.isdigit() for char in predicted_response) or "$" in predicted_response:
                quality_score += 0.2
                
            # Check for relevant terminology
            keywords = ["pricing", "tier", "technical", "error", "feature", "subscription", "account"]
            if any(keyword in predicted_response.lower() for keyword in keywords):
                quality_score += 0.2
                
            # Combine base similarity score with quality metrics
            final_score = (base_score * 0.5) + (quality_score * 0.5)
            
            return min(1.0, final_score)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating response quality: {str(e)}")
            return 0.0

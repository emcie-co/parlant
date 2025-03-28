"""Configuration module for DSPy integration."""

from dataclasses import dataclass
from typing import Optional
import os

from parlant.core.logging import Logger


@dataclass
class DSPyConfig:
    """Configuration for DSPy integration.
    
    This class manages configuration for the DSPy integration components,
    supporting both direct initialization and environment variable loading.
    
    Attributes:
        model_name: Name of the model to use (e.g. 'gpt-4')
        optimizer_batch_size: Size of batches for optimization
        max_tokens: Maximum tokens per request
        temperature: Temperature for generation
        api_key: Optional API key for the model
        logger: Logger instance for operations
    """
    model_name: str = "gpt-4"
    optimizer_batch_size: int = 5
    max_tokens: int = 2000
    temperature: float = 1.0
    api_key: Optional[str] = None
    logger: Optional[Logger] = None
    
    @classmethod
    def from_env(cls, logger: Optional[Logger] = None) -> 'DSPyConfig':
        """Create configuration from environment variables.
        
        This method loads configuration from the following environment variables:
        - DSPY_MODEL: Model name (default: gpt-4)
        - DSPY_OPTIMIZER_BATCH_SIZE: Batch size for optimization (default: 5)
        - DSPY_MAX_TOKENS: Maximum tokens per request (default: 2000)
        - DSPY_TEMPERATURE: Temperature for generation (default: 1.0)
        - DSPY_API_KEY: API key for the model
        
        Args:
            logger: Optional logger instance
            
        Returns:
            DSPyConfig instance with values from environment
        """
        return cls(
            model_name=os.getenv("DSPY_MODEL", "gpt-4"),
            optimizer_batch_size=int(os.getenv("DSPY_OPTIMIZER_BATCH_SIZE", "5")),
            max_tokens=int(os.getenv("DSPY_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("DSPY_TEMPERATURE", "1.0")),
            api_key=os.getenv("DSPY_API_KEY"),
            logger=logger
        )
    
    def validate(self) -> None:
        """Validate the configuration.
        
        This method checks that all required values are present and valid.
        
        Raises:
            ValueError: If any configuration values are invalid
        """
        if not self.model_name:
            raise ValueError("Model name must be specified")
            
        if self.optimizer_batch_size < 1:
            raise ValueError("Optimizer batch size must be positive")
            
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")
            
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
            
        if self.api_key is None:
            raise ValueError("API key must be specified") 
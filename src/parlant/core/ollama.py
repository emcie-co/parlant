"""Ollama integration utilities."""
import logging
import subprocess
from typing import Optional


def initialize_ollama_model(model_name: str) -> None:
    """Initialize an Ollama model.
    
    Args:
        model_name: Name of the model to initialize
    """
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
    except Exception as e:
        logging.error(f"Failed to initialize Ollama model {model_name}: {str(e)}")
        raise

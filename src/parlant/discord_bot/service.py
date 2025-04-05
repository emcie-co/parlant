"""
Discord bot service integration for Parlant with COPRO optimization.

This module provides the service layer for the Discord bot integration,
connecting it to the existing DSPy and COPRO functionality.
"""
import logging
from typing import Dict, Any, List, Optional

from parlant.core.logging import Logger
from parlant.dspy_integration.guideline_optimizer import GuidelineOptimizer
from parlant.dspy_integration.metrics import DSPyMetrics
from parlant.discord_bot.copro_bridge import COPROBridge

# Setup logging
logger = logging.getLogger(__name__)

class DiscordBotService:
    """Service layer for the Discord bot integration."""
    
    def __init__(self, logger: Logger, optimizer: GuidelineOptimizer, metrics: DSPyMetrics):
        """Initialize the Discord bot service.
        
        Args:
            logger: The logger
            optimizer: The guideline optimizer
            metrics: The DSPy metrics
        """
        self.logger = logger
        self.copro_bridge = COPROBridge(optimizer, metrics)
        self.user_sessions = {}
    
    async def process_message(self, user_id: str, message: str, history: List[Dict[str, Any]]) -> str:
        """Process a user message and generate a response.
        
        Args:
            user_id: The user ID
            message: The user message
            history: The conversation history
            
        Returns:
            The assistant's response
        """
        try:
            # Log the incoming message
            self.logger.info(f"Processing message from user {user_id}: {message[:50]}...")
            
            # Process the message using the COPRO bridge
            response = await self.copro_bridge.process_conversation(user_id, message, history)
            
            # Log the response
            self.logger.info(f"Generated response for user {user_id}: {response[:50]}...")
            
            return response
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    async def optimize_prompts(self, user_id: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize prompts based on conversation history.
        
        Args:
            user_id: The user ID
            history: The conversation history
            
        Returns:
            Optimization results
        """
        try:
            # Log the optimization request
            self.logger.info(f"Optimizing prompts for user {user_id} with {len(history)} messages...")
            
            # Optimize prompts using the COPRO bridge
            results = await self.copro_bridge.optimize_prompts(user_id, history)
            
            # Log the optimization results
            self.logger.info(f"Optimization complete for user {user_id} with improvement: {results['improvement']:.2f}%")
            
            return results
        except Exception as e:
            self.logger.error(f"Error optimizing prompts: {e}")
            raise
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization performance.
        
        Returns:
            Statistics about optimization performance
        """
        return self.copro_bridge.get_optimization_stats()
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the optimization history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            The optimization history
        """
        return self.copro_bridge.get_optimization_history(limit)

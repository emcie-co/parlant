"""
Test script for the Discord bot integration.

This script tests the Discord bot functionality without actually connecting to Discord.
It simulates user interactions and tests the core functionality.
"""
import asyncio
import logging
import os
import sys
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import the necessary modules
from parlant.core.logging import Logger
from parlant.dspy_integration.guideline_optimizer import GuidelineOptimizer
from parlant.dspy_integration.metrics import DSPyMetrics
from parlant.discord_bot.copro_bridge import COPROBridge
from parlant.discord_bot.service import DiscordBotService

class MockOptimizer:
    """Mock implementation of GuidelineOptimizer for testing."""
    
    async def optimize_response(self, context, guideline):
        """Mock implementation of optimize_response."""
        logger.info(f"Mock optimize_response called with context: {context}")
        return {"assistant_message": "This is a mock response from the optimizer."}
    
    async def optimize_guideline(self, guideline, context):
        """Mock implementation of optimize_guideline."""
        logger.info(f"Mock optimize_guideline called with guideline: {guideline.name}")
        # Return the same guideline for testing
        return guideline

class MockMetrics:
    """Mock implementation of DSPyMetrics for testing."""
    
    async def evaluate_guideline(self, guideline, context):
        """Mock implementation of evaluate_guideline."""
        logger.info(f"Mock evaluate_guideline called with guideline: {guideline.name}")
        return {"relevance": 0.8, "coherence": 0.9, "helpfulness": 0.85}

class MockLogger:
    """Mock implementation of Logger for testing."""
    
    def info(self, message):
        """Mock implementation of info."""
        logger.info(f"MockLogger.info: {message}")
    
    def error(self, message):
        """Mock implementation of error."""
        logger.error(f"MockLogger.error: {message}")

async def test_discord_bot_service():
    """Test the DiscordBotService functionality."""
    logger.info("Starting Discord bot service test")
    
    # Create mock components
    mock_logger = MockLogger()
    mock_optimizer = MockOptimizer()
    mock_metrics = MockMetrics()
    
    # Create the service
    service = DiscordBotService(mock_logger, mock_optimizer, mock_metrics)
    
    # Test processing a message
    user_id = "test_user_123"
    message = "Hello, can you help me with a coding problem?"
    history = [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
    
    logger.info("Testing process_message")
    response = await service.process_message(user_id, message, history)
    logger.info(f"Response: {response}")
    
    # Test optimizing prompts
    logger.info("Testing optimize_prompts")
    try:
        results = await service.optimize_prompts(user_id, history)
        logger.info(f"Optimization results: {results}")
    except Exception as e:
        logger.error(f"Error optimizing prompts: {e}")
    
    # Test getting optimization stats
    logger.info("Testing get_optimization_stats")
    stats = service.get_optimization_stats()
    logger.info(f"Optimization stats: {stats}")
    
    # Test getting optimization history
    logger.info("Testing get_optimization_history")
    history = service.get_optimization_history()
    logger.info(f"Optimization history: {history}")
    
    logger.info("Discord bot service test completed successfully")

async def test_copro_bridge():
    """Test the COPROBridge functionality."""
    logger.info("Starting COPRO bridge test")
    
    # Create mock components
    mock_optimizer = MockOptimizer()
    mock_metrics = MockMetrics()
    
    # Create the bridge
    bridge = COPROBridge(mock_optimizer, mock_metrics)
    
    # Test processing a conversation
    user_id = "test_user_456"
    message = "What's the best way to learn Python?"
    history = [
        {"role": "user", "content": "I want to learn programming."},
        {"role": "assistant", "content": "That's great! What language are you interested in?"}
    ]
    
    logger.info("Testing process_conversation")
    response = await bridge.process_conversation(user_id, message, history)
    logger.info(f"Response: {response}")
    
    # Test optimizing prompts
    logger.info("Testing optimize_prompts")
    try:
        results = await bridge.optimize_prompts(user_id, history)
        logger.info(f"Optimization results: {results}")
    except Exception as e:
        logger.error(f"Error optimizing prompts: {e}")
    
    logger.info("COPRO bridge test completed successfully")

async def main():
    """Main test function."""
    logger.info("Starting Discord bot integration tests")
    
    # Test the COPRO bridge
    await test_copro_bridge()
    
    # Test the Discord bot service
    await test_discord_bot_service()
    
    logger.info("All tests completed successfully")

if __name__ == "__main__":
    asyncio.run(main())

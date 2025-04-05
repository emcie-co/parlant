"""
Discord bot integration initialization module for Parlant with COPRO optimization.

This module provides the entry point for the Discord bot integration.
"""
import os
import logging
import asyncio
from typing import Optional

from lagom import Container

from parlant.core.logging import Logger
from parlant.dspy_integration.services import DSPyService
from parlant.dspy_integration.config import DSPyConfig
from parlant.discord_bot.bot import ManusBot, run_bot

# Setup logging
logger = logging.getLogger(__name__)

async def setup_discord_bot(container: Container) -> None:
    """Set up and run the Discord bot.
    
    Args:
        container: Container with DSPy components
    """
    # Get DSPy service from container
    dspy_service = container[DSPyService]
    
    # Run the bot
    await run_bot(dspy_service)

def main() -> None:
    """Main entry point for the Discord bot."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create container
    container = Container()
    
    # Register logger
    container[Logger] = Logger()
    
    # Register DSPy config
    container[DSPyConfig] = DSPyConfig.from_env()
    
    # Register DSPy service
    # This is a simplified implementation - in a real implementation,
    # we would need to register all the necessary components
    container[DSPyService] = DSPyService()
    
    # Run the bot
    try:
        asyncio.run(setup_discord_bot(container))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")

if __name__ == "__main__":
    main()

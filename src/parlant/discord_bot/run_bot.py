"""
Main entry point script for running the Discord bot.

This script provides a convenient way to start the Discord bot from the command line.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the necessary modules
from parlant.discord_bot import main

if __name__ == "__main__":
    # Check if Discord token is set
    if not os.environ.get("DISCORD_TOKEN"):
        logger.error("DISCORD_TOKEN environment variable is not set. Please set it before running the bot.")
        sys.exit(1)
    
    # Run the bot
    main()

"""
Discord bot integration for Parlant with COPRO optimization.

This module provides a Discord bot interface to the Parlant COPRO optimization functionality.
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional, List

import discord
from discord import app_commands
from discord.ext import commands

from parlant.core.logging import Logger
from parlant.dspy_integration.services import DSPyService
from parlant.dspy_integration.config import DSPyConfig

# Setup logging
logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation history and context for a user."""
    
    def __init__(self, user_id: str, dspy_service: DSPyService):
        self.user_id = user_id
        self.history: List[Dict[str, Any]] = []
        self.dspy_service = dspy_service
        self.current_model = "default"
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (user or assistant)
            content: The content of the message
        """
        self.history.append({
            "role": role,
            "content": content
        })
    
    async def process_message(self, content: str) -> str:
        """Process a user message and generate a response.
        
        Args:
            content: The user message content
            
        Returns:
            The assistant's response
        """
        # Add user message to history
        self.add_message("user", content)
        
        # Generate response using DSPy service
        # This is a simplified implementation - in a real implementation,
        # we would need to convert the conversation history to the format
        # expected by the DSPy service
        try:
            # Create a simple context with the user's message
            context = {
                "user_id": self.user_id,
                "message": content,
                "history": self.history
            }
            
            # Use the DSPy service to generate a response
            # This is a placeholder - the actual implementation would depend
            # on the specific DSPy service interface
            response = await self.dspy_service.process_message(context)
            
            # Add assistant response to history
            self.add_message("assistant", response)
            
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    async def optimize_prompts(self) -> str:
        """Optimize prompts based on conversation history.
        
        Returns:
            A message describing the optimization results
        """
        try:
            # Create a context with the conversation history
            context = {
                "user_id": self.user_id,
                "history": self.history
            }
            
            # Use the DSPy service to optimize prompts
            # This is a placeholder - the actual implementation would depend
            # on the specific DSPy service interface
            optimization_result = await self.dspy_service.optimize_prompts(context)
            
            return f"Optimization complete! Improved performance by {optimization_result['improvement']}%"
        except Exception as e:
            logger.error(f"Error optimizing prompts: {e}")
            return f"Optimization failed: {str(e)}"
    
    def reset(self) -> None:
        """Reset the conversation history."""
        self.history = []
    
    def set_model(self, model: str) -> None:
        """Set the model to use for this conversation.
        
        Args:
            model: The model name
        """
        self.current_model = model
    
    def get_current_model(self) -> str:
        """Get the current model being used.
        
        Returns:
            The current model name
        """
        return self.current_model
    
    def export_history(self, format: str) -> str:
        """Export conversation history in the specified format.
        
        Args:
            format: The export format (json, txt, md)
            
        Returns:
            The exported conversation history as a string
        """
        if format == "json":
            import json
            return json.dumps(self.history, indent=2)
        elif format == "txt":
            text = ""
            for message in self.history:
                text += f"{message['role'].upper()}: {message['content']}\n\n"
            return text
        elif format == "md":
            md = "# Conversation History\n\n"
            for message in self.history:
                role = "User" if message['role'] == "user" else "Assistant"
                md += f"## {role}\n\n{message['content']}\n\n"
            return md
        else:
            raise ValueError(f"Unsupported export format: {format}")

class ManusBot(commands.Bot):
    """Discord bot for Parlant with COPRO optimization."""
    
    def __init__(self, dspy_service: DSPyService):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        super().__init__(
            command_prefix=commands.when_mentioned_or("!"),
            intents=intents,
            help_command=None,
        )
        
        self.dspy_service = dspy_service
        self.conversation_managers: Dict[str, ConversationManager] = {}
        
    async def setup_hook(self) -> None:
        """Setup hook called when the bot is starting."""
        # Register slash commands
        await self.register_commands()
        
        logger.info("Bot is ready to start!")
    
    async def register_commands(self) -> None:
        """Register all slash commands."""
        # Core functionality commands
        @self.tree.command(
            name="chat",
            description="Start a new conversation or continue an existing one with an initial message"
        )
        @app_commands.describe(message="Your message to the bot")
        async def chat_command(interaction: discord.Interaction, message: str) -> None:
            """Chat command handler."""
            await interaction.response.defer(thinking=True)
            
            # Get or create conversation manager for this user
            user_id = str(interaction.user.id)
            
            if user_id not in self.conversation_managers:
                self.conversation_managers[user_id] = ConversationManager(user_id, self.dspy_service)
            
            conversation_manager = self.conversation_managers[user_id]
            
            # Process the message and get a response
            response = await conversation_manager.process_message(message)
            
            await interaction.followup.send(response)
        
        @self.tree.command(
            name="optimize",
            description="Trigger COPRO optimization based on conversation history"
        )
        async def optimize_command(interaction: discord.Interaction) -> None:
            """Optimize command handler."""
            await interaction.response.defer(thinking=True)
            
            # Get conversation manager for this user
            user_id = str(interaction.user.id)
            
            if user_id not in self.conversation_managers:
                await interaction.followup.send("No conversation history found. Start a conversation with `/chat` first.")
                return
            
            conversation_manager = self.conversation_managers[user_id]
            
            # Check if there's enough conversation history
            if len(conversation_manager.history) < 2:
                await interaction.followup.send("Not enough conversation history for optimization. Continue chatting and try again later.")
                return
            
            # Trigger optimization
            result = await conversation_manager.optimize_prompts()
            await interaction.followup.send(result)
        
        @self.tree.command(
            name="reset",
            description="Clear the current conversation history and start fresh"
        )
        async def reset_command(interaction: discord.Interaction) -> None:
            """Reset command handler."""
            # Get conversation manager for this user
            user_id = str(interaction.user.id)
            
            if user_id in self.conversation_managers:
                self.conversation_managers[user_id].reset()
            
            await interaction.response.send_message("Conversation history has been reset. Start a new conversation with `/chat`.")
        
        # Configuration commands
        @self.tree.command(
            name="config",
            description="View the current bot configuration and optimization settings"
        )
        async def config_view_command(interaction: discord.Interaction) -> None:
            """Config view command handler."""
            # Get DSPy config
            config = DSPyConfig.from_env()
            
            # Format configuration as a string
            config_str = "**Current Configuration:**\n\n"
            config_str += f"**Model:** {config.model_name}\n"
            config_str += f"**API Key:** {'Configured' if config.api_key else 'Not Configured'}\n"
            
            # Add user-specific settings if available
            user_id = str(interaction.user.id)
            if user_id in self.conversation_managers:
                cm = self.conversation_managers[user_id]
                config_str += f"\n**Your Settings:**\n"
                config_str += f"Active Model: {cm.get_current_model()}\n"
                config_str += f"Conversation History: {len(cm.history)} messages\n"
            
            await interaction.response.send_message(config_str)
        
        # Sync commands with Discord
        await self.tree.sync()
        logger.info("Commands registered and synced!")
    
    async def on_ready(self) -> None:
        """Event handler called when the bot is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening, 
                name="/chat | /help"
            )
        )

async def run_bot(dspy_service: DSPyService) -> None:
    """Run the Discord bot.
    
    Args:
        dspy_service: The DSPy service to use for processing messages and optimizing prompts
    """
    # Get Discord token from environment
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        logger.error("No Discord token found in environment variables.")
        return
    
    # Create and start the bot
    bot = ManusBot(dspy_service)
    
    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        await bot.close()

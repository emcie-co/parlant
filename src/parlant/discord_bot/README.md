# Discord Bot Integration for Parlant with COPRO Optimization

This document provides an overview of the Discord bot integration with the Parlant repository's COPRO optimization functionality.

## Overview

The Discord bot integration allows users to interact with the Parlant COPRO optimization system through Discord. It provides a set of slash commands for chatting with the bot, optimizing prompts, and managing configurations.

## Architecture

The Discord bot integration consists of the following components:

1. **Bot Core (`bot.py`)**: Contains the main `ManusBot` class that handles Discord events and manages user sessions.

2. **Commands (`commands.py`)**: Implements all the required slash commands for interacting with the bot.

3. **COPRO Bridge (`copro_bridge.py`)**: Provides a bridge between the Discord bot and the existing COPRO optimization functionality.

4. **Service Layer (`service.py`)**: Handles the business logic for processing messages and optimizing prompts.

5. **Entry Point (`__init__.py`)**: Provides the entry point for the Discord bot integration.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jmanhype/parlant.git -b feature/copro-optimization
   cd parlant
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   pip install discord.py
   ```

3. Set up environment variables:
   ```bash
   export DISCORD_TOKEN=your_discord_bot_token
   export OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Start the bot:
   ```bash
   python -m parlant.discord_bot
   ```

2. Interact with the bot using slash commands in Discord:
   - `/chat [message]`: Start a new conversation or continue an existing one
   - `/optimize`: Trigger COPRO optimization based on conversation history
   - `/reset`: Clear the current conversation history and start fresh
   - And many more (see Slash Commands section)

## Slash Commands

### Core Functionality Commands
- `/chat [message]`: Start a new conversation or continue an existing one with an initial message
- `/optimize`: Trigger COPRO optimization based on conversation history
- `/reset`: Clear the current conversation history and start fresh

### Configuration Commands
- `/config view`: View the current bot configuration and optimization settings
- `/config set [parameter] [value]`: Modify a configuration parameter (admin only)
- `/models list`: Show available language models for use with the bot
- `/models select [model]`: Change the active language model

### DSPy-Specific Commands
- `/signature view`: View the current prompt signature being used
- `/signature create [name] [description]`: Create a new signature for specific conversation types
- `/signature test [name]`: Test a specific signature with a sample conversation

### Utility Commands
- `/help`: Display help information about available commands
- `/status`: Check the bot's status and optimization metrics
- `/export [format]`: Export conversation history in various formats (JSON, TXT, MD)
- `/guidelines list`: View available behavioral guidelines for the bot
- `/guidelines add [condition] [action]`: Add a new guideline for the bot to follow

### Admin Commands
- `/optimization stats`: View statistics about optimization performance
- `/optimization history`: View history of optimizations performed
- `/training create [name]`: Create a new training dataset from conversations
- `/maintenance`: Put the bot in maintenance mode (admin only)

## Integration with COPRO

The Discord bot integrates with the existing COPRO optimization functionality through the `COPROBridge` class. This bridge connects to the `GuidelineOptimizer` and `DSPyMetrics` components to process conversations and optimize prompts.

The integration flow is as follows:

1. User sends a message using the `/chat` command
2. The bot processes the message through the `DiscordBotService`
3. The service uses the `COPROBridge` to generate a response using the COPRO optimization
4. The response is sent back to the user in Discord

For optimization, the flow is:

1. User triggers optimization using the `/optimize` command
2. The bot collects the conversation history
3. The service uses the `COPROBridge` to optimize prompts based on the conversation history
4. The optimization results are sent back to the user in Discord

## Testing

The integration includes a test script (`test_bot.py`) that can be used to verify the functionality without connecting to Discord. It tests the core components of the integration, including the `DiscordBotService` and `COPROBridge`.

To run the tests:
```bash
python -m parlant.discord_bot.test_bot
```

## Future Improvements

1. Add support for more advanced conversation management features
2. Implement more sophisticated optimization strategies
3. Add support for multi-user conversations
4. Improve error handling and recovery mechanisms
5. Add more comprehensive metrics and analytics

## Contributing

Contributions to the Discord bot integration are welcome! Please follow the existing code style and add tests for new functionality.

## License

This project is licensed under the same license as the Parlant repository.

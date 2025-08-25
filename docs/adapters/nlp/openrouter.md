# OpenRouter Service Adapter Documentation

## Overview

The OpenRouter Service Adapter provides integration with OpenRouter's API gateway, offering access to a wide range of language models through a unified interface. This adapter implements the Parlant NLP service interface for text generation, supporting multiple model providers including OpenAI, Anthropic, Google, and many others through a single API endpoint.

**Important Note**: OpenRouter provides access to language models only and does not offer embedding services. For embeddings, you need to configure a separate third-party provider (such as OpenAI, Together AI, or any OpenAI-compatible embedding API).

## Key Benefits

### Unified API Gateway

OpenRouter serves as a unified gateway to multiple LLM providers, offering several key advantages:

1. **Single API Key Management**: Use one API key instead of managing credentials for multiple providers
2. **Centralized Billing**: Track and manage costs across all providers in one place
3. **Simplified Integration**: Switch between models from different vendors without code changes
4. **No Vendor Lock-in**: Easily experiment with and migrate between different models

### Mixed Model Architecture

With Parlant's schema-specific configuration, you can leverage the best model for each task:

```bash
# Use GPT-4 for tool execution
export OPENROUTER_SINGLE_TOOL_MODEL="openai/gpt-4o"

# Use Claude for complex reasoning in journey flows
export OPENROUTER_JOURNEY_NODE_MODEL="anthropic/claude-3.5-sonnet"

# Use Gemini for creative content generation
export OPENROUTER_CANNED_RESPONSE_DRAFT_MODEL="google/gemini-pro"

# Use fast models for simple selections
export OPENROUTER_CANNED_RESPONSE_SELECTION_MODEL="meta-llama/llama-3-70b-instruct"
```

This flexibility allows you to:
- Optimize performance by using specialized models for specific tasks
- Balance cost and quality by mixing premium and efficient models
- Future-proof your application as new models become available
- A/B test different models without infrastructure changes

### Cost and Management Benefits

- **Unified Usage Tracking**: Monitor all LLM usage from a single dashboard
- **Simplified Budgeting**: Set spending limits across all providers
- **No Multi-Vendor Contracts**: Avoid managing multiple vendor relationships
- **Instant Model Access**: Try new models immediately without separate signups

### Comparison: Traditional vs OpenRouter Approach

| Aspect | Traditional Multi-Provider | OpenRouter Unified |
|--------|---------------------------|-------------------|
| API Keys | 5+ keys to manage | 1 key for all |
| Billing | Multiple invoices | Single invoice |
| Integration | Provider-specific code | Unified interface |
| Model Switching | Code changes required | Environment variable |
| Cost Tracking | Multiple dashboards | One dashboard |
| New Models | New integration needed | Instant access |

## Architecture

### Core Components

- **OpenRouterService**: Main service class implementing the NLPService interface
- **OpenRouterSchematicGenerator**: Base generator for all OpenRouter models
- **OpenRouter_Default**: Default generator implementation with dynamic model configuration
- **OpenRouterEmbedder**: Wrapper for third-party embedding services (OpenRouter does not provide embeddings)
- **OpenRouterEstimatingTokenizer**: Token counting using GPT-4 tokenizer as fallback

## Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your-api-key

# Optional - API Configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # Default base URL
OPENROUTER_DEFAULT_MODEL=openai/gpt-4o            # Default model

# Schema-Specific Model Configuration
OPENROUTER_SINGLE_TOOL_MODEL=openai/gpt-4o                        # Default for single tool calls
OPENROUTER_SINGLE_TOOL_MAX_TOKENS=131072                          # Max tokens for single tool
OPENROUTER_JOURNEY_NODE_MODEL=anthropic/claude-3.5-sonnet         # For journey node selection
OPENROUTER_JOURNEY_NODE_MAX_TOKENS=204800                         # Max tokens for journey
OPENROUTER_CANNED_RESPONSE_DRAFT_MODEL=anthropic/claude-3.5-sonnet # For canned response drafting
OPENROUTER_CANNED_RESPONSE_DRAFT_MAX_TOKENS=204800                # Max tokens for drafts
OPENROUTER_CANNED_RESPONSE_SELECTION_MODEL=anthropic/claude-3-haiku # For response selection
OPENROUTER_CANNED_RESPONSE_SELECTION_MAX_TOKENS=204800            # Max tokens for selection

# Embedding Configuration
# Note: OpenRouter does not provide embedding services. You can configure any third-party embedding API
# that follows the OpenAI embedding API format (e.g., OpenAI, Together AI, or other compatible services)
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small          # Model name for the embedding provider
OPENROUTER_EMBEDDING_DIMENSIONS=1536                              # Embedding vector dimensions
OPENROUTER_EMBEDDING_BASE_URL=https://api.openai.com/v1           # Base URL of your embedding provider
OPENROUTER_EMBEDDING_API_KEY=your-embedding-api-key               # API key for the embedding provider
```

### Authentication

OpenRouter uses API key authentication. You can obtain an API key from [OpenRouter Dashboard](https://openrouter.ai/keys).

```bash
# Set API key in environment
export OPENROUTER_API_KEY="your-api-key"

# Or use .env file with python-dotenv
echo 'OPENROUTER_API_KEY=your-api-key' > .env
```

## Supported Models

OpenRouter provides access to models from multiple providers. Here are some popular options:

### OpenAI Models

| Model ID | Description | Context Window |
|----------|-------------|----------------|
| `openai/gpt-4o` | GPT-4 Omni - Latest multimodal model | 128K |
| `openai/gpt-4o-mini` | Smaller, faster GPT-4 Omni | 128K |
| `openai/gpt-4-turbo` | GPT-4 Turbo with vision | 128K |
| `openai/gpt-3.5-turbo` | Fast, efficient for simple tasks | 16K |

### Anthropic Models

| Model ID | Description | Context Window |
|----------|-------------|----------------|
| `anthropic/claude-3.5-sonnet` | Most intelligent Claude model | 200K |
| `anthropic/claude-3-opus` | Powerful model for complex tasks | 200K |
| `anthropic/claude-3-haiku` | Fast, efficient Claude model | 200K |
| `anthropic/claude-2.1` | Previous generation Claude | 200K |

### Google Models

| Model ID | Description | Context Window |
|----------|-------------|----------------|
| `google/gemini-pro` | Google's advanced model | 32K |
| `google/gemini-pro-vision` | Multimodal Gemini model | 32K |
| `google/palm-2-chat-bison` | PaLM 2 for conversations | 8K |

### Other Providers

| Provider | Example Models |
|----------|----------------|
| Meta | `meta-llama/llama-3-70b-instruct` |
| Mistral | `mistral/mistral-large` |
| Cohere | `cohere/command-r-plus` |
| Together | `together/mixtral-8x7b` |

For a complete list of available models, visit [OpenRouter Models](https://openrouter.ai/models).

## Usage

### Basic Setup

```python
import parlant.sdk as p

# Ensure environment variable is set
# export OPENROUTER_API_KEY="your-api-key"

async with p.Server(nlp_service=p.NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="Healthcare Agent",
        description="Is empathetic and calming to the patient.",
    )
```

### Direct Service Usage

```python
from parlant.adapters.nlp.openrouter_service import OpenRouterService
from parlant.core.loggers import Logger

# Initialize service
logger = Logger()
service = OpenRouterService(logger=logger)

# Get schematic generator
generator = await service.get_schematic_generator(YourSchemaClass)

# Generate content
result = await generator.generate(
    prompt="Your prompt here",
    hints={"temperature": 0.7, "max_tokens": 1000}
)
```

### Customizing Models per Schema

```python
import os

# Configure different models for different use cases
os.environ["OPENROUTER_SINGLE_TOOL_MODEL"] = "openai/gpt-4o-mini"
os.environ["OPENROUTER_JOURNEY_NODE_MODEL"] = "anthropic/claude-3.5-sonnet"
os.environ["OPENROUTER_CANNED_RESPONSE_DRAFT_MODEL"] = "google/gemini-pro"
```

#### Real-World Example: Optimized Healthcare Assistant

```python
# High-accuracy model for medical tool calls (e.g., appointment scheduling)
os.environ["OPENROUTER_SINGLE_TOOL_MODEL"] = "openai/gpt-4o"

# Best reasoning model for complex patient journey flows
os.environ["OPENROUTER_JOURNEY_NODE_MODEL"] = "anthropic/claude-3.5-sonnet"

# Creative model for generating empathetic responses
os.environ["OPENROUTER_CANNED_RESPONSE_DRAFT_MODEL"] = "google/gemini-pro"

# Fast, cost-effective model for simple response selection
os.environ["OPENROUTER_CANNED_RESPONSE_SELECTION_MODEL"] = "meta-llama/llama-3-70b-instruct"

# This configuration:
# - Ensures accuracy for critical medical operations (GPT-4)
# - Provides sophisticated reasoning for patient care paths (Claude)
# - Generates compassionate, creative responses (Gemini)
# - Keeps costs low for simple decisions (Llama)
```

### Using Third-Party Embedding Services

Since OpenRouter does not provide embedding services, you need to configure a third-party embedding provider. The adapter supports any service that follows the OpenAI embedding API format.

#### Example: Using OpenAI Embeddings

```python
# Configure OpenAI as the embedding provider
os.environ["OPENROUTER_EMBEDDING_MODEL"] = "text-embedding-3-large"
os.environ["OPENROUTER_EMBEDDING_DIMENSIONS"] = "3072"
os.environ["OPENROUTER_EMBEDDING_BASE_URL"] = "https://api.openai.com/v1"
os.environ["OPENROUTER_EMBEDDING_API_KEY"] = "your-openai-api-key"

# Get embedder
embedder = await service.get_embedder()

# Embed texts
result = await embedder.embed(
    texts=["Hello world", "How are you?"]
)
```

#### Example: Using Together AI Embeddings

```python
# Configure Together AI as the embedding provider
os.environ["OPENROUTER_EMBEDDING_MODEL"] = "togethercomputer/m2-bert-80M-32k-retrieval"
os.environ["OPENROUTER_EMBEDDING_DIMENSIONS"] = "768"
os.environ["OPENROUTER_EMBEDDING_BASE_URL"] = "https://api.together.xyz/v1"
os.environ["OPENROUTER_EMBEDDING_API_KEY"] = "your-together-api-key"
```

#### Supported Embedding Providers

Any provider that implements OpenAI's embedding API format is supported, including:
- **OpenAI**: text-embedding-3-small, text-embedding-3-large
- **Together AI**: Various open-source embedding models
- **Azure OpenAI**: If you have Azure deployment
- **Any OpenAI-compatible API**: Self-hosted or third-party services

## Advanced Configuration

### Rate Limiting and Retries

The adapter includes built-in retry logic for rate limits and API errors:

```python
@policy(
    [
        retry(
            exceptions=(
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIError,
            ),
        ),
        retry(openai.APIError, max_exceptions=2, wait_times=(1.0, 5.0)),
    ]
)
```

### Custom Headers

The adapter automatically includes required headers for OpenRouter:

```python
default_headers={
    "HTTP-Referer": "https://parlant.ai",
    "X-Title": "Parlant",
}
```

### Supported Generation Hints

The following hints are supported for generation:

- `temperature`: Controls randomness (0.0 to 2.0)
- `max_tokens`: Maximum tokens to generate
- `top_p`: Nucleus sampling parameter
- `frequency_penalty`: Penalize frequent tokens
- `presence_penalty`: Penalize tokens based on presence

## Error Handling

### Common Errors

1. **Authentication Error (401)**
   - Ensure `OPENROUTER_API_KEY` is set correctly
   - Check if your API key is valid and has credits

2. **Rate Limit Error (429)**
   - The adapter will automatically retry with exponential backoff
   - Consider upgrading your OpenRouter plan for higher limits

3. **Invalid Model Error**
   - Verify the model ID is correct
   - Check if the model is available in your OpenRouter account

### Error Messages

```python
RATE_LIMIT_ERROR_MESSAGE = (
    "OpenRouter API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your OpenRouter account balance and billing status.\n"
    "- Review your API usage limits in OpenRouter dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://openrouter.ai/docs"
)
```

## Performance Optimization

### Token Limits by Schema

Different schemas have optimized token limits:

| Schema Type | Default Model | Default Max Tokens |
|-------------|---------------|-------------------|
| SingleToolBatch | openai/gpt-4o | 131,072 |
| JourneyNodeSelection | anthropic/claude-3.5-sonnet | 204,800 |
| CannedResponseDraft | anthropic/claude-3.5-sonnet | 204,800 |
| CannedResponseSelection | anthropic/claude-3-haiku | 204,800 |

### Cost Optimization

OpenRouter's unified billing provides several cost advantages:

**Billing Benefits:**
- Single invoice for all LLM usage
- Consolidated usage analytics across providers
- Simplified budget allocation and tracking
- Volume discounts across combined usage

**Optimization Strategies:**
1. Use smaller models for simple tasks (e.g., `gpt-3.5-turbo`, `claude-3-haiku`)
2. Set appropriate `max_tokens` limits
3. Use caching for embeddings when possible
4. Monitor usage through OpenRouter dashboard
5. Leverage model mixing to balance cost and performance:
   ```bash
   # Premium models for critical tasks
   export OPENROUTER_JOURNEY_NODE_MODEL="anthropic/claude-3.5-sonnet"
   
   # Cost-effective models for high-volume operations
   export OPENROUTER_CANNED_RESPONSE_SELECTION_MODEL="mistral/mistral-7b-instruct"
   ```

## Contributing

### Adding New Features

1. **Update Environment Variables**: Add new configuration options to the environment variable constants
2. **Extend Generator Classes**: Create specialized generator classes if needed
3. **Update Documentation**: Document new features and configuration options
4. **Add Tests**: Include unit and integration tests
5. **Submit PR**: Follow the project's contribution guidelines

### Testing

```python
# Run tests with OpenRouter
export OPENROUTER_API_KEY="your-test-key"
pytest tests/adapters/nlp/test_openrouter.py
```

### Example: Adding a New Model Configuration

```python
# 1. Add environment variable constants
ENV_CUSTOM_MODEL = "OPENROUTER_CUSTOM_MODEL"
ENV_CUSTOM_MAX_TOKENS = "OPENROUTER_CUSTOM_MAX_TOKENS"

# 2. Update get_model_config_for_schema function
env_mappings = {
    # ... existing mappings ...
    CustomSchema: (ENV_CUSTOM_MODEL, ENV_CUSTOM_MAX_TOKENS),
}

# 3. Set default configuration
defaults = {
    # ... existing defaults ...
    CustomSchema: ModelConfig("openai/gpt-4o", 128 * 1024),
}
```

## Troubleshooting

### Debug Logging

Enable debug logging to see API requests and responses:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **"No auth credentials found"**
   - Solution: Set `OPENROUTER_API_KEY` environment variable

2. **"Model not found"**
   - Solution: Check model ID spelling and availability

3. **"Insufficient credits"**
   - Solution: Add credits to your OpenRouter account

4. **Slow response times**
   - Consider using faster models (e.g., `gpt-3.5-turbo`, `claude-3-haiku`)
   - Check OpenRouter status page for any ongoing issues

## Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Available Models](https://openrouter.ai/models)
- [API Reference](https://openrouter.ai/api/v1/docs)
- [Pricing Information](https://openrouter.ai/pricing)
- [Status Page](https://status.openrouter.ai/)

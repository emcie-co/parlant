# OpenRouter NLP Service

The OpenRouter service provides access to multiple LLM providers through a unified API. OpenRouter supports 100+ models from various providers including OpenAI, Anthropic, Meta, Google, and more.

## Prerequisites

1. **Get an OpenRouter API Key**: Sign up at [OpenRouter.ai](https://openrouter.ai/) and create an API key

## Environment Variables

Configure the OpenRouter service using these environment variables:

```bash
# Required: Your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-your-api-key-here"

# Optional: Model to use (default: openai/gpt-4o)
# Can be ANY model supported by OpenRouter (see https://openrouter.ai/models)
export OPENROUTER_MODEL="openai/gpt-4o"

# Optional: Max tokens for custom models (automatically inferred if not set)
# Only needed if you want to override the default
export OPENROUTER_MAX_TOKENS="8192"

# Optional: Embedding model to use (default: text-embedding-ada-002)
export OPENROUTER_EMBEDDING_MODEL="text-embedding-ada-002"

# Optional: Your site URL for rankings (recommended for production)
export OPENROUTER_HTTP_REFERER="https://your-domain.com"

# Optional: Your site name for rankings (recommended for production)
export OPENROUTER_SITE_NAME="Your App Name"
```

### Example Configuration

```bash
# For development
export OPENROUTER_API_KEY="sk-or-dev-key"
export OPENROUTER_MODEL="openai/gpt-4o-mini"

# For production
export OPENROUTER_API_KEY="sk-or-prod-key"
export OPENROUTER_MODEL="openai/gpt-4o"
export OPENROUTER_HTTP_REFERER="https://myapp.com"
export OPENROUTER_SITE_NAME="My Production App"
```

## Supported Models

OpenRouter supports 100+ models from various providers. Here are some commonly used models:

### OpenAI Models
- `openai/gpt-4o` - Most capable GPT-4 (recommended)
- `openai/gpt-4o-mini` - Fast and cost-effective
- `openai/gpt-4-turbo` - Balanced performance
- `openai/gpt-3.5-turbo` - Fast and economical

### Anthropic Models
- `anthropic/claude-3.5-sonnet` - High-quality reasoning
- `anthropic/claude-3-opus` - Most capable Claude
- `anthropic/claude-3-haiku` - Fast Claude model

### Meta Models
- `meta-llama/llama-3.3-70b-instruct` - Latest Llama 3.3
- `meta-llama/llama-3.1-70b-instruct` - Stable Llama 3.1

### Google Models
- `google/gemini-pro-1.5` - Google Gemini Pro
- `google/gemini-pro` - Google Gemini

### Other Providers
OpenRouter also supports models from Mistral, Cohere, and many other providers. Visit [OpenRouter.ai/models](https://openrouter.ai/models) for a complete list.

## Usage Example

### Basic Usage

```python
import parlant.sdk as p

# Method 1: Use environment variables
async with p.Server(
    nlp_service=p.NLPServices.openrouter()
) as server:
    agent = await server.create_agent(
        name="Assistant",
        description="A helpful AI assistant powered by OpenRouter"
    )

# Method 2: Pass parameters directly
async with p.Server(
    nlp_service=p.NLPServices.openrouter(
        model_name="google/gemini-pro",
        max_tokens=32768
    )
) as server:
    agent = await server.create_agent(
        name="Gemini Assistant",
        description="Powered by Google Gemini"
    )
```

### Using a Specific Model

```python
import parlant.sdk as p

# Method 1: Pass model directly as parameter
async with p.Server(
    nlp_service=p.NLPServices.openrouter(model_name="anthropic/claude-3.5-sonnet")
) as server:
    agent = await server.create_agent(
        name="Claude Assistant",
        description="Powered by Claude 3.5 Sonnet"
    )

# Method 2: Use environment variable
import os
os.environ["OPENROUTER_MODEL"] = "anthropic/claude-3.5-sonnet"

async with p.Server(
    nlp_service=p.NLPServices.openrouter()
) as server:
    agent = await server.create_agent(
        name="Claude Assistant",
        description="Powered by Claude 3.5 Sonnet"
    )
```

### Using ANY OpenRouter Model

The OpenRouter service supports **ALL** models available on OpenRouter. You can use any model without any additional configuration:

```bash
# Use any OpenRouter-supported model
export OPENROUTER_MODEL="mistralai/mixtral-8x7b-instruct"

# Or use OpenAI models
export OPENROUTER_MODEL="openai/gpt-4-turbo"

# Or use Anthropic models
export OPENROUTER_MODEL="anthropic/claude-3-opus"

# Or use Google models
export OPENROUTER_MODEL="google/gemini-pro"

# Or ANY other model from the 100+ supported models
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

The service will automatically:
- Create the appropriate generator for your model
- Set sensible default max_tokens based on model family
- Allow you to override max_tokens via `OPENROUTER_MAX_TOKENS` if needed
- Handle all model-specific configurations automatically

## Cost Optimization

OpenRouter provides transparent pricing and routing. Here are strategies to optimize costs:

### Use Cost-Effective Models
```bash
# Fast and economical
export OPENROUTER_MODEL="openai/gpt-4o-mini"

# Free tier available
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

### Monitor Usage
- Check your OpenRouter dashboard for usage statistics
- Set up usage limits in the OpenRouter dashboard
- Review model-specific pricing at [OpenRouter.ai/models](https://openrouter.ai/models)

## Predefined Model Classes

The OpenRouter service includes optimized configurations for common models:

- **OpenRouterGPT4O**: OpenAI GPT-4o (128k context)
- **OpenRouterGPT4OMini**: OpenAI GPT-4o Mini (128k context, cost-effective)
- **OpenRouterClaude35Sonnet**: Anthropic Claude 3.5 Sonnet (8k context)
- **OpenRouterLlama33_70B**: Meta Llama 3.3 70B (8k context)

### Using Custom Models

The service automatically supports **any** OpenRouter model with intelligent defaults:

- **No configuration needed** for most models
- Automatically detects model family (GPT-4, Claude, Llama, etc.) for optimal defaults
- Supports 100+ models from all providers via OpenRouter
- Override max_tokens with `OPENROUTER_MAX_TOKENS` if needed

Simply set `OPENROUTER_MODEL` to any model from https://openrouter.ai/models

## Rankings and Attribution

OpenRouter tracks which apps use which models to help developers discover your app. Including attribution headers is recommended:

```bash
export OPENROUTER_HTTP_REFERER="https://your-domain.com"
export OPENROUTER_SITE_NAME="Your App Name"
```

Benefits:
- Your app may appear in model-specific rankings on OpenRouter.ai
- Helps with AI model discovery
- Transparent attribution

## Troubleshooting

### API Key Issues

**Error**: `OPENROUTER_API_KEY is not set`

**Solution**: Set the `OPENROUTER_API_KEY` environment variable:
```bash
export OPENROUTER_API_KEY="sk-or-your-key"
```

### Rate Limit Issues

**Error**: Rate limit exceeded

**Solution**: 
- Check your OpenRouter account for rate limits
- Consider using a less popular model
- Upgrade your OpenRouter plan if needed
- Implement request throttling in your application

### Model Not Found

**Error**: Model not available

**Solution**: 
- Check the model identifier is correct (e.g., `openai/gpt-4o` not `gpt-4o`)
- Visit [OpenRouter.ai/models](https://openrouter.ai/models) to verify the model exists
- Some models may be temporarily unavailable

### Response Format Issues

The OpenRouter service generates JSON schemas. If you encounter validation errors:

- Ensure you're using a model with strong JSON generation capabilities (GPT-4o, Claude 3.5, etc.)
- Check that the model supports `response_format={"type": "json_object"}`

## Integration with Other Services

OpenRouter works seamlessly with Parlant's other features:

```python
import parlant.sdk as p

async with p.Server(
    nlp_service=p.NLPServices.openrouter()
) as server:
    # Use with guidelines, journeys, tools, etc.
    agent = await server.create_agent(
        name="Customer Support Agent",
        description="Handles customer inquiries"
    )
    
    await server.create_guideline(
        agent_id=agent.id,
        content="Always be polite and professional"
    )
```

## Features

- **Unified API**: Access 100+ models through a single API
- **Model Switching**: Easily switch between different LLM providers
- **Cost Optimization**: Choose models based on performance/cost needs
- **Transparent Pricing**: Clear pricing for all models
- **No Provider Lock-in**: Easily switch between OpenAI, Anthropic, Meta, etc.

## Links

- **OpenRouter Website**: https://openrouter.ai/
- **API Documentation**: https://openrouter.ai/docs
- **Available Models**: https://openrouter.ai/models
- **Pricing**: https://openrouter.ai/models (includes pricing for each model)

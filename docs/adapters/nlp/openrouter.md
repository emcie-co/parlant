# OpenRouter Service Documentation

The OpenRouter service provides access to **multiple AI models** through a single unified API, including GPT-4, Claude, Llama, and many more. OpenRouter makes it easy to switch between different models without changing code.

## Prerequisites

1. **OpenRouter Account**: Sign up at [openrouter.ai](https://openrouter.ai)
2. **API Key**: Get your API key from the OpenRouter dashboard
3. **Model Access**: Ensure you have access to the models you want to use

## Quick Start

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-api-key-here"

# Optionally set a default model
export OPENROUTER_MODEL="openai/gpt-4o"
```

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="AI Assistant",
        description="A helpful assistant powered by OpenRouter.",
    )
```

## Environment Variables

### Required Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key

### Optional Variables

- `OPENROUTER_MODEL`: Default model to use (e.g., `"openai/gpt-4o"`)
- `OPENROUTER_MAX_TOKENS`: Custom max tokens limit
- `OPENROUTER_HTTP_REFERER`: Your app's URL (for analytics)
- `OPENROUTER_SITE_NAME`: Your app's name (for analytics)

## Supported Models

OpenRouter supports **hundreds of models** from different providers. Here are some popular ones:

### Pre-configured Models

| Model | Provider | Context | Use Case |
|-------|----------|---------|----------|
| `openai/gpt-4o` | OpenAI | 128K | Default, best overall |
| `openai/gpt-4o-mini` | OpenAI | 128K | Cost-effective |
| `anthropic/claude-3.5-sonnet` | Anthropic | 200K | Advanced reasoning |
| `meta-llama/llama-3.3-70b-instruct` | Meta | 8K | Open-source option |

### Other Available Models

You can use **any model** that OpenRouter supports:

```bash
# Use any OpenRouter model
export OPENROUTER_MODEL="google/gemini-pro-1.5"
export OPENROUTER_MODEL="anthropic/claude-3-opus"
export OPENROUTER_MODEL="openai/gpt-4-turbo"
export OPENROUTER_MODEL="mistralai/mixtral-8x7b-instruct"
```

Check the [OpenRouter Models page](https://openrouter.ai/models) for the full list.

## Usage Examples

### Basic Usage

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

# Use default model (gpt-4o)
async with p.Server(nlp_service=NLPServices.openrouter) as server:
    # Your agent code here
    pass
```

### Custom Model

```python
# Use Claude via OpenRouter
async with p.Server(
    nlp_service=NLPServices.openrouter(
        model_name="anthropic/claude-3.5-sonnet"
    )
) as server:
    # Your agent code here
    pass
```

### Custom Max Tokens

```python
# Limit response length
async with p.Server(
    nlp_service=NLPServices.openrouter(
        model_name="openai/gpt-4o",
        max_tokens=4096
    )
) as server:
    # Your agent code here
    pass
```

### Environment-Based Configuration

```bash
# Use environment variables for configuration
export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
export OPENROUTER_MAX_TOKENS="8192"
export OPENROUTER_HTTP_REFERER="https://myapp.com"
export OPENROUTER_SITE_NAME="My App"
```

```python
# Configuration is automatically loaded from environment
async with p.Server(nlp_service=NLPServices.openrouter) as server:
    # Uses OPENROUTER_MODEL and other env vars
    pass
```

## Dynamic Model Selection

OpenRouter intelligently selects the appropriate generator based on your model:

### Known Models

These models have specialized configurations for optimal performance:

```python
# OpenRouter automatically uses specialized generators for:
"openai/gpt-4o"              # → OpenRouterGPT4O
"openai/gpt-4o-mini"          # → OpenRouterGPT4OMini
"anthropic/claude-3.5-sonnet" # → OpenRouterClaude35Sonnet
"meta-llama/llama-3.3-70b-instruct" # → OpenRouterLlama33_70B
```

## Advantages of OpenRouter

1. **Model Diversity**: Access to 100+ models from different providers
2. **Cost Flexibility**: Choose models based on price-performance
3. **Single API**: One integration for multiple providers
4. **Auto-Fallback**: Handles model-specific quirks automatically
5. **Analytics**: Built-in usage tracking through OpenRouter dashboard

## Troubleshooting

### Rate Limit Errors

```
OpenRouter API rate limit exceeded
```

**Solution**: 
- Check your OpenRouter account balance
- Review usage limits in the OpenRouter dashboard
- Consider upgrading your plan
- Try a different model with higher limits

### JSON Mode Not Supported

```
Model 'xyz' does not support JSON mode
```

**Solution**:
- OpenRouter automatically falls back to prompting for JSON
- Consider using a model that supports JSON mode (`openai/gpt-4o`, `anthropic/claude-3.5-sonnet`)
- The fallback still produces structured output

### Authentication Errors

```
OPENROUTER_API_KEY is not set
```

**Solution**:
- Set the `OPENROUTER_API_KEY` environment variable
- Verify your API key in the OpenRouter dashboard
- Ensure the key hasn't expired

## Cost Management

OpenRouter lets you compare costs across models:

```python
# GPT-4o - Higher quality, higher cost
export OPENROUTER_MODEL="openai/gpt-4o"

# GPT-4o-mini - Good quality, lower cost
export OPENROUTER_MODEL="openai/gpt-4o-mini"

# Claude - Balanced quality and cost
export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"

# Llama - Open source, lower cost
export OPENROUTER_MODEL="meta-llama/llama-3.3-70b-instruct"
```

See [OpenRouter pricing](https://openrouter.ai/docs/pricing) for current rates.

## Model Selection Guide

### When to Use Each Model

**GPT-4o** (`openai/gpt-4o`)
- Complex reasoning tasks
- Code generation and debugging
- Multi-step problem solving
- When accuracy is critical

**GPT-4o-mini** (`openai/gpt-4o-mini`)
- General purpose tasks
- High-volume applications
- Cost-sensitive use cases
- When 95% accuracy is sufficient

**Claude** (`anthropic/claude-3.5-sonnet`)
- Long context tasks (200K tokens)
- Creative writing
- Detailed analysis
- When you need extended reasoning

**Llama** (`meta-llama/llama-3.3-70b-instruct`)
- Open-source requirements
- Custom fine-tuning
- Privacy-sensitive applications
- Cost optimization

## Best Practices

1. **Start with GPT-4o**: Best balance of quality and performance
2. **Use Mini for Scale**: Switch to `gpt-4o-mini` for high-volume operations
3. **Monitor Costs**: Check OpenRouter dashboard regularly
4. **Set Max Tokens**: Prevent runaway costs with token limits
5. **Use Analytics**: Track model usage with `OPENROUTER_HTTP_REFERER`

## Additional Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Available Models](https://openrouter.ai/models)
- [API Reference](https://openrouter.ai/docs/api-reference)


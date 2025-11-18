# OpenRouter Service Documentation

The OpenRouter service provides access to **400+ AI models** through a single unified API, including GPT-4, Claude, Llama, Qwen, and many more. OpenRouter makes it easy to switch between different models for both text generation and embeddings without changing code.

## Prerequisites

1. **OpenRouter Account**: Sign up at [openrouter.ai](https://openrouter.ai)
2. **API Key**: Get your API key from the [OpenRouter dashboard](https://openrouter.ai/keys)
3. **Model Access**: Ensure you have access to the models you want to use

## Quick Start

### Basic Setup

```bash
# Set your OpenRouter API key (required)
export OPENROUTER_API_KEY="your-api-key-here"

# Optionally set default models
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_EMBEDDER_MODEL="openai/text-embedding-3-large"
```

### Minimal Example

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="AI Assistant",
        description="A helpful assistant powered by OpenRouter.",
    )
    # 🎉 Ready to use at http://localhost:8800
```

## Configuration

All configuration is done via environment variables. Set the required and optional environment variables before running your application:

```bash
# Required: API Key
export OPENROUTER_API_KEY="your-api-key-here"

# Optional: LLM Configuration
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_MAX_TOKENS="128000"

# Optional: Embedding Configuration
export OPENROUTER_EMBEDDER_MODEL="qwen/qwen3-embedding-8b"
export OPENROUTER_EMBEDDER_DIMENSIONS="4096"  # Optional override

# Optional: Analytics
export OPENROUTER_HTTP_REFERER="https://myapp.com"
export OPENROUTER_SITE_NAME="My Application"
```

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | `sk-or-v1-...` |

### Optional Variables - LLM Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENROUTER_MODEL` | LLM model name | `openai/gpt-4o` | `openai/gpt-4o-mini` |
| `OPENROUTER_MAX_TOKENS` | Max tokens limit | Auto-detected | `128000` |

### Optional Variables - Embedding Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENROUTER_EMBEDDER_MODEL` | Embedding model name | `openai/text-embedding-3-large` | `qwen/qwen3-embedding-8b` |
| `OPENROUTER_EMBEDDER_DIMENSIONS` | Override embedding dimensions | Auto-detected | `4096` |

### Optional Variables - Analytics

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENROUTER_HTTP_REFERER` | Your app's URL (for analytics) | `https://myapp.com` |
| `OPENROUTER_SITE_NAME` | Your app's name (for analytics) | `My Application` |

## Supported Models

OpenRouter supports **400+ models** from different providers. Models are automatically optimized with specialized configurations when available.

### Pre-configured LLM Models

These models have specialized configurations for optimal performance:

| Model | Provider | Context | Use Case |
|-------|----------|---------|----------|
| `openai/gpt-4o` | OpenAI | 128K | Default, best overall quality |
| `openai/gpt-4o-mini` | OpenAI | 128K | Cost-effective, fast |
| `anthropic/claude-3.5-sonnet` | Anthropic | 200K | Advanced reasoning, long context |
| `meta-llama/llama-3.3-70b-instruct` | Meta | 8K | Open-source option |

### Supported Embedding Models

The service supports multiple embedding models with automatic dimension detection:

| Model | Dimensions | Provider | Use Case |
|-------|------------|----------|----------|
| `openai/text-embedding-3-large` | 3072 | OpenAI | Default, high quality |
| `openai/text-embedding-3-small` | 1536 | OpenAI | Faster, smaller |
| `openai/text-embedding-ada-002` | 1536 | OpenAI | Legacy model |
| `qwen/qwen3-embedding-8b` | 4096 | Qwen | High dimension, multilingual |
| `qwen/qwen-embedding-v2` | 1536 | Qwen | Multilingual embeddings |

### Using Any OpenRouter Model

You can use **any model** that OpenRouter supports by setting the appropriate environment variables:

```bash
# LLM Models
export OPENROUTER_MODEL="google/gemini-pro-1.5"

# Embedding Models
export OPENROUTER_EMBEDDER_MODEL="qwen/qwen3-embedding-8b"
```

Check the [OpenRouter Models page](https://openrouter.ai/models) for the full list of available models.

## Usage Examples

### Example 1: Default Configuration

Use the default models (GPT-4o for LLM, text-embedding-3-large for embeddings):

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="General Assistant",
        description="A helpful AI assistant."
    )
```

### Example 2: Custom LLM Model

Use Claude for text generation:

```bash
export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
```

```python
async with p.Server(nlp_service=NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="Claude Assistant",
        description="Powered by Claude."
    )
```

### Example 3: Custom Embedder Model

Use a custom embedding model for better multilingual support:

```bash
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_EMBEDDER_MODEL="qwen/qwen3-embedding-8b"
```

```python
async with p.Server(nlp_service=NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="Multilingual Assistant",
        description="Supports multiple languages."
    )
```

### Example 4: High-Performance Setup

Optimize for speed and quality:

```bash
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_EMBEDDER_MODEL="openai/text-embedding-3-large"
export OPENROUTER_MAX_TOKENS="128000"
```

```python
async with p.Server(nlp_service=NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="High-Performance Agent",
        description="Optimized for speed and accuracy."
    )
```

### Example 5: Cost-Optimized Setup

Balance quality and cost:

```bash
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_EMBEDDER_MODEL="openai/text-embedding-3-small"
```

```python
async with p.Server(nlp_service=NLPServices.openrouter) as server:
    agent = await server.create_agent(
        name="Cost-Optimized Agent",
        description="Optimized for cost-effectiveness."
    )
```

## Embedding Model Configuration

### Understanding Embedding Dimensions

Different embedding models produce vectors of different dimensions. The service automatically detects dimensions for known models, and can auto-detect from API responses for unknown models.

### Known Embedding Dimensions

The following models have pre-configured dimensions:

- `openai/text-embedding-3-large`: **3072** dimensions
- `openai/text-embedding-3-small`: **1536** dimensions
- `openai/text-embedding-ada-002`: **1536** dimensions
- `qwen/qwen3-embedding-8b`: **4096** dimensions
- `qwen/qwen-embedding-v2`: **1536** dimensions

### Auto-Detection

For unknown models, dimensions are automatically detected from the first API response and cached for subsequent use.

### Manual Dimension Override

If needed, you can manually specify dimensions via environment variable:

```bash
export OPENROUTER_EMBEDDER_DIMENSIONS="4096"
```

⚠️ **Important**: If you change embedder models or dimensions, you may need to clear your vector database cache to avoid dimension mismatch errors.

## Dynamic Model Selection

OpenRouter intelligently handles model selection and configuration:

### Automatic Generator Selection

Known models use specialized generators for optimal performance:

- `openai/gpt-4o` → `OpenRouterGPT4O`
- `openai/gpt-4o-mini` → `OpenRouterGPT4OMini`
- `anthropic/claude-3.5-sonnet` → `OpenRouterClaude35Sonnet`
- `meta-llama/llama-3.3-70b-instruct` → `OpenRouterLlama33_70B`
- Other models → Dynamic generator with auto-configured parameters

### Automatic Embedder Selection

Embedders are automatically configured based on the model name:

- Known models → Pre-configured dimensions
- Unknown models → Auto-detected dimensions from API response
- Dynamic embedder → Created with proper container resolution

## Advantages of OpenRouter

1. **Model Diversity**: Access to 400+ models from different providers
2. **Unified Embeddings**: Native support for embedding models via the same API
3. **Cost Flexibility**: Choose models based on price-performance needs
4. **Single API**: One integration for multiple providers
5. **Auto-Optimization**: Automatic configuration for known models
6. **Environment-Based Configuration**: All configuration via environment variables
7. **Analytics**: Built-in usage tracking through OpenRouter dashboard

## Troubleshooting

### Rate Limit Errors

**Error:**
```
OpenRouter API rate limit exceeded
```

**Solutions:**
- Check your OpenRouter account balance and billing status
- Review usage limits in the [OpenRouter dashboard](https://openrouter.ai/keys)
- Consider upgrading your plan for higher limits
- Try a different model with higher rate limits
- Wait a moment before retrying

### JSON Mode Not Supported

**Error:**
```
Model 'xyz' does not support JSON mode
```

**Solutions:**
- OpenRouter automatically falls back to prompting for JSON output
- Consider using a model that supports JSON mode:
  - `openai/gpt-4o`
  - `openai/gpt-4o-mini`
  - `anthropic/claude-3.5-sonnet`
- The fallback still produces structured output reliably

### Dimension Mismatch Errors

**Error:**
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly
```

**Solutions:**
- This occurs when switching embedder models with different dimensions
- Clear your vector database cache/embeddings
- Or delete the cached embeddings files in your `parlant-data` directory
- The embedder will create new embeddings with the correct dimensions

### Authentication Errors

**Error:**
```
OPENROUTER_API_KEY is not set
```

**Solutions:**
- Set the `OPENROUTER_API_KEY` environment variable
- Verify your API key in the [OpenRouter dashboard](https://openrouter.ai/keys)
- Ensure the key hasn't expired or been revoked
- Check for typos in the environment variable name

### Container Resolution Errors

**Error:**
```
Unable to construct dependency of type OpenRouterEmbedder
```

**Solutions:**
- This is automatically handled by the dynamic embedder class
- Ensure you're using the latest version of the code
- If the error persists, check that `embedder_model_name` is correctly set

## Cost Management

OpenRouter provides transparent pricing across models. Choose models based on your needs:

### Cost-Effective LLM Options

```python
# GPT-4o-mini - Good quality, lower cost
model_name="openai/gpt-4o-mini"

# Claude Haiku - Fast, affordable
model_name="anthropic/claude-3-haiku"

# Llama - Open source, very affordable
model_name="meta-llama/llama-3.3-70b-instruct"
```

### Cost-Effective Embedding Options

```python
# text-embedding-3-small - Smaller, faster, cheaper
embedder_model_name="openai/text-embedding-3-small"

# text-embedding-ada-002 - Legacy, very affordable
embedder_model_name="openai/text-embedding-ada-002"
```

### Premium Options

```python
# GPT-4o - Highest quality
model_name="openai/gpt-4o"

# text-embedding-3-large - Highest quality embeddings
embedder_model_name="openai/text-embedding-3-large"
```

Check [OpenRouter pricing](https://openrouter.ai/docs/pricing) for current rates.

## Model Selection Guide

### When to Use Each LLM Model

**GPT-4o** (`openai/gpt-4o`)
- Complex reasoning tasks
- Code generation and debugging
- Multi-step problem solving
- When accuracy is critical
- Best overall performance

**GPT-4o-mini** (`openai/gpt-4o-mini`)
- General purpose tasks
- High-volume applications
- Cost-sensitive use cases
- When 95% accuracy is sufficient
- Fast response times

**Claude** (`anthropic/claude-3.5-sonnet`)
- Long context tasks (200K tokens)
- Creative writing
- Detailed analysis
- When you need extended reasoning
- Complex document understanding

**Llama** (`meta-llama/llama-3.3-70b-instruct`)
- Open-source requirements
- Custom fine-tuning needs
- Privacy-sensitive applications
- Cost optimization
- Self-hosted deployments

### When to Use Each Embedding Model

**text-embedding-3-large** (`openai/text-embedding-3-large`)
- Default choice for most use cases
- High quality semantic search
- Best accuracy for retrieval
- Recommended for production

**text-embedding-3-small** (`openai/text-embedding-3-small`)
- Cost-sensitive applications
- Faster embedding generation
- Good quality for most tasks
- Large-scale deployments

**qwen3-embedding-8b** (`qwen/qwen3-embedding-8b`)
- Multilingual applications
- Higher dimensional space (4096)
- Better fine-grained distinctions
- When you need more embedding dimensions

## Best Practices

### 1. Start with Defaults
Begin with the default models (`gpt-4o` and `text-embedding-3-large`) for best balance of quality and performance.

### 2. Use Mini for Scale
Switch to `gpt-4o-mini` for high-volume operations where cost is a concern.

### 3. Match Embedder to Use Case
- Use `text-embedding-3-large` for quality-critical applications
- Use `text-embedding-3-small` for cost-sensitive deployments
- Use `qwen3-embedding-8b` for multilingual or high-dimensional needs

### 4. Set Max Tokens
Prevent runaway costs by setting appropriate `max_tokens` limits via environment variable:

```bash
export OPENROUTER_MAX_TOKENS="128000"  # For long-context models
export OPENROUTER_MAX_TOKENS="8192"    # For standard use cases
```

### 5. Monitor Costs
Regularly check the [OpenRouter dashboard](https://openrouter.ai/keys) to monitor usage and costs.

### 6. Use Analytics
Set `OPENROUTER_HTTP_REFERER` and `OPENROUTER_SITE_NAME` to track usage across different applications.

### 7. Clear Cache When Changing Models
If you switch embedder models, clear your vector database cache to avoid dimension mismatches.

### 8. Environment Variables for Production
Use environment variables for production deployments instead of hardcoding values:

```bash
# Production configuration
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_EMBEDDER_MODEL="openai/text-embedding-3-large"
```

## Advanced Configuration

### Custom Dimensions for Unknown Models

If using an embedding model not in the known list, you can specify dimensions:

```bash
export OPENROUTER_EMBEDDER_MODEL="custom/embedding-model"
export OPENROUTER_EMBEDDER_DIMENSIONS="2048"
```

The service will also auto-detect dimensions from the first API response.

### Combining Multiple Configurations

All configuration is done via environment variables. Set multiple variables to configure different aspects:

```bash
# Set all configuration via environment variables
export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
export OPENROUTER_MAX_TOKENS="200000"
export OPENROUTER_EMBEDDER_MODEL="openai/text-embedding-3-large"
```

## Additional Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Available Models](https://openrouter.ai/models)
- [API Reference](https://openrouter.ai/docs/api-reference)
- [Pricing Information](https://openrouter.ai/docs/pricing)
- [Rate Limits](https://openrouter.ai/docs/api-reference/limits)

## Example: Complete Setup

Here's a complete example showing a production-ready setup:

```bash
# Set environment variables
export OPENROUTER_API_KEY="your-api-key-here"
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export OPENROUTER_EMBEDDER_MODEL="openai/text-embedding-3-large"
export OPENROUTER_MAX_TOKENS="32768"
```

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

@p.tool
async def get_weather(context: p.ToolContext, city: str) -> p.ToolResult:
    # Your weather API logic here
    return p.ToolResult(f"Sunny, 72°F in {city}")

async def main():
    async with p.Server(nlp_service=NLPServices.openrouter) as server:
        agent = await server.create_agent(
            name="Weather Assistant",
            description="Helps users check weather conditions."
        )
        
        await agent.create_guideline(
            condition="User asks about weather",
            action="Get weather information using the get_weather tool",
            tools=[get_weather]
        )
        
        # 🎉 Ready at http://localhost:8800

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

This setup provides:
- ✅ Cost-effective LLM (`gpt-4o-mini`)
- ✅ High-quality embeddings (`text-embedding-3-large`)
- ✅ Reasonable token limit (32K)
- ✅ Tool integration
- ✅ Guideline-based behavior control

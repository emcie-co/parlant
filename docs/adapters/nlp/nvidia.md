# NVIDIA Service Documentation

The NVIDIA service connects Parlant to models hosted on [NVIDIA NIM](https://build.nvidia.com), including NVIDIA's Nemotron family, through NVIDIA's OpenAI-compatible API. The same service also works with self-hosted NIM deployments: point it at your own endpoint with `NVIDIA_BASE_URL`.

## Prerequisites

1. **NVIDIA Account**: Sign up at [build.nvidia.com](https://build.nvidia.com)
2. **API Key**: Generate an API key from the [NVIDIA API catalog](https://build.nvidia.com)
3. **Model Access**: Ensure you have access to the models you want to use

## Quick Start

### Basic Setup

```bash
# Set your NVIDIA API key (required)
export NVIDIA_API_KEY="your-api-key-here"

# Optionally set default models
export NVIDIA_MODEL="nvidia/nemotron-3.5-super-vl-120b-a12b"
export NVIDIA_EMBEDDER_MODEL="nvidia/nemotron-3.5-embed-8b"
```

### Minimal Example

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.nvidia) as server:
    agent = await server.create_agent(
        name="AI Assistant",
        description="A helpful assistant powered by NVIDIA NIM.",
    )
    # 🎉 Ready to use at http://localhost:8800
```

## Configuration

All configuration is done via environment variables. Set the required and optional environment variables before running your application:

```bash
# Required: API Key
export NVIDIA_API_KEY="your-api-key-here"

# Optional: Endpoint (for self-hosted NIM)
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"

# Optional: LLM Configuration
export NVIDIA_MODEL="nvidia/nemotron-3.5-super-vl-120b-a12b"
export NVIDIA_MAX_TOKENS="8192"

# Optional: Embedding Configuration
export NVIDIA_EMBEDDER_MODEL="nvidia/nemotron-3.5-embed-8b"
export NVIDIA_EMBEDDER_DIMENSIONS="4096"  # Optional override
export NVIDIA_EMBEDDER_INPUT_TYPE="passage"
```

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | Your NVIDIA API key | `nvapi-...` |

### Optional Variables - Endpoint

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `NVIDIA_BASE_URL` | OpenAI-compatible endpoint | `https://integrate.api.nvidia.com/v1` | `http://localhost:8000/v1` |

### Optional Variables - LLM Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `NVIDIA_MODEL` | LLM model name | `nvidia/nemotron-3.5-super-vl-120b-a12b` | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| `NVIDIA_MAX_TOKENS` | Max tokens limit for models without a pre-configured generator | `8192` | `16384` |

### Optional Variables - Embedding Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `NVIDIA_EMBEDDER_MODEL` | Embedding model name | `nvidia/nemotron-3.5-embed-8b` | `nvidia/nemotron-3.5-embed-1b` |
| `NVIDIA_EMBEDDER_DIMENSIONS` | Override embedding dimensions | Auto-detected | `2048` |
| `NVIDIA_EMBEDDER_INPUT_TYPE` | `input_type` sent to NVIDIA's asymmetric embedding models | `passage` | `query` |

## Supported Models

### Pre-configured LLM Models

These models have specialized configurations:

| Model | Use Case |
|-------|----------|
| `nvidia/nemotron-3.5-super-vl-120b-a12b` | Default, best overall quality |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | Smaller, mid-size option |
| `nvidia/nemotron-3.5-bolt-e4b` | Smallest, fast and cost-effective |

### Supported Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `nvidia/nemotron-3.5-embed-8b` | 4096 | Default, high quality |
| `nvidia/nemotron-3.5-embed-1b` | 2048 | Faster, smaller |

### Using Any NVIDIA Model

You can use any chat model in the NVIDIA API catalog that supports JSON output by setting the appropriate environment variables:

```bash
# LLM Models
export NVIDIA_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b"

# Embedding Models
export NVIDIA_EMBEDDER_MODEL="nvidia/nemotron-3.5-embed-1b"
```

Check the [NVIDIA API catalog](https://build.nvidia.com/models) for the full list of available models.

## Usage Examples

### Example 1: Default Configuration

Use the default models (Nemotron 3.5 Super for LLM, Nemotron 3.5 Embed 8B for embeddings):

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.nvidia) as server:
    agent = await server.create_agent(
        name="General Assistant",
        description="A helpful AI assistant."
    )
```

### Example 2: Smaller, Faster Models

```bash
export NVIDIA_MODEL="nvidia/nemotron-3.5-bolt-e4b"
export NVIDIA_EMBEDDER_MODEL="nvidia/nemotron-3.5-embed-1b"
```

```python
async with p.Server(nlp_service=NLPServices.nvidia) as server:
    agent = await server.create_agent(
        name="Fast Assistant",
        description="Optimized for speed."
    )
```

### Example 3: Self-Hosted NIM

Point the service at your own NIM deployment:

```bash
export NVIDIA_API_KEY="your-api-key-here"
export NVIDIA_BASE_URL="http://localhost:8000/v1"
export NVIDIA_MODEL="nvidia/nemotron-3.5-super-vl-120b-a12b"
```

```python
async with p.Server(nlp_service=NLPServices.nvidia) as server:
    agent = await server.create_agent(
        name="Self-Hosted Assistant",
        description="Runs on a self-hosted NIM endpoint."
    )
```

## Embedding Model Configuration

### Understanding Embedding Dimensions

Different embedding models produce vectors of different dimensions. The service automatically detects dimensions for known models, and can auto-detect from API responses for unknown models.

### Known Embedding Dimensions

- `nvidia/nemotron-3.5-embed-8b`: **4096** dimensions
- `nvidia/nemotron-3.5-embed-1b`: **2048** dimensions

### Input Type

NVIDIA's asymmetric embedding models require an `input_type`. The service sends `passage` by default; override it with `NVIDIA_EMBEDDER_INPUT_TYPE`.

### Manual Dimension Override

If needed, you can manually specify dimensions via environment variable:

```bash
export NVIDIA_EMBEDDER_DIMENSIONS="2048"
```

⚠️ **Important**: If you change embedder models or dimensions, you may need to clear your vector database cache to avoid dimension mismatch errors.

## Dynamic Model Selection

### Automatic Generator Selection

Known models use specialized generators:

- `nvidia/nemotron-3.5-super-vl-120b-a12b` → `NvidiaNemotron35SuperVL120B`
- `nvidia/nemotron-3.5-lightning-30b-a3b` → `NvidiaNemotron35Lightning30B`
- `nvidia/nemotron-3.5-bolt-e4b` → `NvidiaNemotron35BoltE4B`
- Other models → Dynamic generator with `NVIDIA_MAX_TOKENS` (default `8192`)

### Automatic Embedder Selection

- Known models → Pre-configured dimensions
- Unknown models → Auto-detected dimensions from API response

## Troubleshooting

### Rate Limit Errors

**Error:**
```
NVIDIA API rate limit exceeded
```

**Solutions:**
- Check your NVIDIA account and API key status
- Try a smaller model, or wait a moment before retrying
- For self-hosted NIM, check the capacity of your endpoint

### JSON Mode Not Supported

**Error:**
```
Model 'xyz' does not support JSON mode
```

**Solutions:**
- The service automatically falls back to prompting for JSON output
- Consider using one of the pre-configured Nemotron models, which support JSON mode

### Dimension Mismatch Errors

**Error:**
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly
```

**Solutions:**
- This occurs when switching embedder models with different dimensions
- Clear your vector database cache/embeddings
- Or delete the cached embeddings files in your `parlant-data` directory

### Authentication Errors

**Error:**
```
NVIDIA_API_KEY is not set
```

**Solutions:**
- Set the `NVIDIA_API_KEY` environment variable
- Verify your API key at [build.nvidia.com](https://build.nvidia.com)
- Check for typos in the environment variable name

## Additional Resources

- [NVIDIA API Catalog](https://build.nvidia.com)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)

## Example: Complete Setup

```bash
# Set environment variables
export NVIDIA_API_KEY="your-api-key-here"
export NVIDIA_MODEL="nvidia/nemotron-3.5-super-vl-120b-a12b"
export NVIDIA_EMBEDDER_MODEL="nvidia/nemotron-3.5-embed-8b"
```

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

@p.tool
async def get_weather(context: p.ToolContext, city: str) -> p.ToolResult:
    # Your weather API logic here
    return p.ToolResult(f"Sunny, 72°F in {city}")

async def main():
    async with p.Server(nlp_service=NLPServices.nvidia) as server:
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

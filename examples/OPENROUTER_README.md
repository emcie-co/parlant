# OpenRouter Integration for Parlant

This document explains how to integrate OpenRouter as an NLP service provider with Parlant.

## Quick Start

### 1. Set Environment Variables
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

### 2. Start Server
```bash
parlant-server run --openrouter
```

## Configuration Options

### Option 1: Environment Variables (Simple)
```bash
# Required
export OPENROUTER_API_KEY="your-api-key"

# Optional
export OPENROUTER_MODEL_NAME="openai/gpt-4o"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# Embedding configuration
export OPENROUTER_EMBEDDING_MODEL="openai/text-embedding-3-large"
export OPENROUTER_EMBEDDING_DIMENSIONS="3072"
export OPENROUTER_EMBEDDING_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_EMBEDDING_API_KEY="your-api-key"
```

### Option 2: TOML Configuration (Recommended)
Create `parlant.toml` in your project root:

```toml
[parlant]
modules = []

[parlant.openrouter]
default_model = "openai/gpt-4o"
api_key = "${OPENROUTER_API_KEY}"
base_url = "https://openrouter.ai/api/v1"

[parlant.openrouter.embedding]
model = "openai/text-embedding-3-large"
dimensions = 3072
base_url = "https://openrouter.ai/api/v1"
api_key = "${OPENROUTER_API_KEY}"
```

## Using DashScope for Embeddings

To use DashScope for embeddings while keeping OpenRouter for generation:

```toml
[parlant.openrouter]
default_model = "openai/gpt-4o"
api_key = "${OPENROUTER_API_KEY}"

[parlant.openrouter.embedding]
model = "text-embedding-v4"
dimensions = 3072
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "${DASHSCOPE_API_KEY}"
```

## SDK Usage

```python
import parlant.sdk as p

async with p.Server(nlp_service=p.NLPServices.openrouter) as server:
    # Your code here
    pass
```

## Supported Models

OpenRouter supports many models from various providers:
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`
- Anthropic: `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-5-haiku`
- Meta: `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.1-70b-instruct`

See [OpenRouter Models](https://openrouter.ai/models) for the complete list.

## Environment Variable Priority

1. Environment variables (highest priority)
2. TOML configuration file
3. Default values (lowest priority)

This allows you to override TOML settings with environment variables when needed.

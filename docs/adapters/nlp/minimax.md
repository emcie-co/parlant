# MiniMax Service Documentation

The MiniMax service integrates [MiniMax](https://www.minimaxi.com/) large language models with Parlant. MiniMax provides an OpenAI-compatible API with high-performance models.

## Prerequisites

1. **Get an API Key**: Sign up at [MiniMax Platform](https://platform.minimaxi.com/) and create an API key.

## Environment Variables

Configure the MiniMax service using these environment variables:

```bash
# Required: Your MiniMax API key
export MINIMAX_API_KEY="your-api-key-here"

# Optional: Model selection (default: MiniMax-M3)
# Options: MiniMax-M3, MiniMax-M2.7, MiniMax-M2.7-highspeed
export MINIMAX_MODEL="MiniMax-M3"

# Optional: Custom base URL (default: https://api.minimax.io/v1)
export MINIMAX_BASE_URL="https://api.minimax.io/v1"
```

## Supported Models

| Model | Context Window | Best For |
|-------|---------------|----------|
| `MiniMax-M3` | 512K tokens | Latest model, up to 128K output, supports image input (default) |
| `MiniMax-M2.7` | 204K tokens | Previous generation, highest quality |
| `MiniMax-M2.7-highspeed` | 204K tokens | Previous generation, lower latency |

## Quick Start

```python
import parlant as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.minimax) as server:
    # Your Parlant application code here
    pass
```

## Notes

- **Temperature**: MiniMax requires temperature values in the range `(0, 1]`. A temperature of `0` is not accepted; the adapter automatically clamps it to `0.01`.
- **Embeddings**: MiniMax does not currently provide an embedding API. The adapter uses JinaAI embeddings as a fallback (requires `JINA_API_KEY` to be set).
- **Streaming**: Streaming text generation is not currently supported via this adapter.

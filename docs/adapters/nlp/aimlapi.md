# aimlapi.com Service Documentation

The aimlapi.com service gives Parlant access to 350+ chat models and 15 embedding
models — OpenAI, Anthropic, Google, DeepSeek, Qwen, Llama and others — through a single
OpenAI-compatible API, with one key and one bill.

Unlike most aggregators, aimlapi.com serves embeddings natively, so Parlant's vector
store does not need a separate provider or a local fallback embedder.

## Prerequisites

1. **Account**: sign up at [aimlapi.com](https://aimlapi.com)
2. **API key**: create one in the dashboard

No extra package is needed — the adapter uses the `openai` client that Parlant already
depends on.

## Quick Start

```bash
export AIMLAPI_API_KEY="your-api-key-here"
```

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.aimlapi) as server:
    agent = await server.create_agent(
        name="AI Assistant",
        description="A helpful assistant powered by aimlapi.com.",
    )
    # 🎉 Ready to use at http://localhost:8800
```

Or from the CLI:

```bash
parlant-server --aimlapi
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `AIMLAPI_API_KEY` | Your aimlapi.com API key |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `AIMLAPI_MODEL` | Chat model id | `openai/gpt-4.1` |
| `AIMLAPI_MAX_TOKENS` | Context window for a model the adapter does not know | `131072` |
| `AIMLAPI_EMBEDDER_MODEL` | Embedding model id | `openai/text-embedding-3-large` |
| `AIMLAPI_EMBEDDER_DIMENSIONS` | Override embedding dimensions | Looked up per model, else `1536` |
| `AIMLAPI_HTTP_REFERER` | Your app's URL, for analytics | Parlant's repository |
| `AIMLAPI_SITE_NAME` | Your app's name, for analytics | `Parlant` |

## Models

Parlant needs a model that reliably honours `response_format: {"type": "json_object"}`,
because guideline matching, tool calling and message generation all run through
schematic (JSON-schema) generation. These ids are pre-configured with their real
context windows:

| Model | Context | Notes |
|-------|---------|-------|
| `openai/gpt-4.1` | 1,047,576 | Default |
| `openai/gpt-4.1-mini` | 1,000,000 | Cheaper, still structured-output capable |
| `anthropic/claude-sonnet-4.5` | 200,000 | Long-context reasoning |
| `google/gemini-2.5-flash` | 1,000,000 | Fast and inexpensive |

Any other chat model id works too — set `AIMLAPI_MODEL` and, if the model's context
window differs from the 128K default, `AIMLAPI_MAX_TOKENS`.

The catalog is public and needs no key:

```bash
# `include=all` adds capabilities, modalities, pricing and providers,
# none of which appear in the default response.
curl 'https://api.aimlapi.com/v1/models?include=all'
```

Chat models are the entries whose `type` is `openai/chat-completions`; prefer a model
whose `capabilities` include `structured_output`.

### Embedding models

| Model | Dimensions |
|-------|------------|
| `openai/text-embedding-3-large` | 3072 (default) |
| `openai/text-embedding-3-small` | 1536 |
| `openai/text-embedding-ada-002` | 1536 |
| `alibaba/text-embedding-v4` | 1024 |

⚠️ Changing the embedder model or its dimensions after data has been indexed requires
clearing the vector store, or you will hit dimension-mismatch errors.

## Notes for Contributors

**Unset optional parameters must be omitted, not sent as `null`.** aimlapi.com answers
`400` when `temperature`, `top_p`, `seed`, `tools`, `tool_choice`, `response_format`,
`stream`, `stream_options`, `parallel_tool_calls`, `max_tokens` or
`max_completion_tokens` arrive as an explicit JSON `null`, even though OpenAI accepts
`null` for all of them. The adapter routes every request through
`omit_unset_arguments()` for exactly this reason, and
`tests/adapters/nlp/test_aimlapi_service.py` guards it. The 400 body names the field in
`error.details[].path` / `.reason`; the top-level `message` is generic.

## Troubleshooting

**`AIMLAPI_API_KEY is not set`** — export the variable before starting the server.

**`400 Bad Request` on every call** — check that no request field is being sent as
`null`; see the note above.

**Dimension mismatch after switching embedder models** — clear the cached embeddings
in your `parlant-data` directory.

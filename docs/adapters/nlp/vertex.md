# Vertex AI Service Adapter Documentation

## Overview

The Vertex AI Service Adapter provides integration with Google Cloud's Vertex AI platform, supporting both Anthropic Claude models and Google Gemini models through their respective APIs. This adapter implements the Parlant NLP service interface for text generation, embeddings, and tokenization.

## Architecture

### Core Components

- **VertexAIService**: Main service class implementing the NLPService interface
- **VertexAIClaudeSchematicGenerator**: Generator for Claude models via Anthropic Vertex API
- **VertexAIGeminiSchematicGenerator**: Generator for Gemini models via Google Gen AI API
- **VertexAIEmbedder**: Text embedding service using Google's text-embedding-004 model
- **VertexAIEstimatingTokenizer**: Token counting for both Claude and Gemini models

## Configuration

### Environment Variables

```bash
# Required
VERTEX_AI_PROJECT_ID=your-gcp-project-id
VERTEX_AI_REGION=us-central1  # Put your region
VERTEX_AI_MODEL=claude-opus-4
```

### Authentication

The adapter uses Google Application Default Credentials (ADC):

```bash
# For local development
gcloud auth application-default login

# For production, use service account key or workload identity
```

## Supported Models

### Claude Models (via Anthropic Vertex API)

| Short Name | Full Model Name | Description |
|------------|-----------------|-------------|
| `claude-opus-4` | `claude-opus-4@20250514` | Most capable Claude model |
| `claude-sonnet-4` | `claude-sonnet-4@20250514` | Balanced performance and speed |
| `claude-sonnet-3.5` | `claude-3-5-sonnet-v2@20241022` | Previous generation Sonnet |
| `claude-haiku-3.5` | `claude-3-5-haiku@20241022` | Fastest Claude model |

### Gemini Models (via Google Gen AI API)

| Short Name | Full Model Name | Description |
|------------|-----------------|-------------|
| `gemini-2.5-flash` | `gemini-2.5-flash` | Latest fast Gemini model |
| `gemini-2.5-pro` | `gemini-2.5-pro` | Latest pro Gemini model |
| `gemini-2.0-flash` | `gemini-2.0-flash` | Previous generation flash |
| `gemini-1.5-flash` | `gemini-1.5-flash` | 1M token context |
| `gemini-1.5-pro` | `gemini-1.5-pro` | 2M token context |

## Usage

### Basic Setup

```python
import parlant.sdk import p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.vertex) as server:
        agent = await server.create_agent(
            name="Healthcare Agent",
            description="Is empathetic and calming to the patient.",
        )
```

### Direct Service Usage

```python
from parlant.adapters.nlp.vertex_service import VertexAIService
from parlant.core.loggers import Logger

# Initialize service
logger = Logger()
service = VertexAIService(logger=logger)

# Get schematic generator
generator = await service.get_schematic_generator(YourSchemaClass)

# Generate content
result = await generator.generate(
    prompt="Your prompt here",
    hints={"temperature": 0.7, "max_tokens": 1000}
)
```

## API Reference

### VertexAIService

Main service class implementing the NLPService interface.

#### Constructor

```python
def __init__(self, logger: Logger) -> None
```

Initializes the service with environment variables:
- Reads `VERTEX_AI_PROJECT_ID`, `VERTEX_AI_REGION`, `VERTEX_AI_MODEL`
- Validates Application Default Credentials
- Sets up logging

#### Methods

##### get_schematic_generator

```python
async def get_schematic_generator(self, t: type[T]) -> SchematicGenerator[T]
```

Returns appropriate generator based on configured model:
- Claude models → VertexAIClaudeSchematicGenerator
- Gemini models → VertexAIGeminiSchematicGenerator
- Includes fallback logic for Claude Opus 4

##### get_embedder

```python
async def get_embedder(self) -> Embedder
```

Returns VertexTextEmbedding004 embedder instance.

##### get_moderation_service

```python
async def get_moderation_service(self) -> ModerationService
```

Returns NoModeration service (moderation not yet implemented).

### VertexAIClaudeSchematicGenerator

Schematic generator for Claude models via Anthropic Vertex API.

#### Supported Hints

- `temperature`: Controls randomness (0.0-1.0)
- `max_tokens`: Maximum output tokens
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter

#### Properties

- `id`: Returns `vertex-ai/{model_name}`
- `tokenizer`: Returns VertexAIEstimatingTokenizer instance
- `max_tokens`: Returns 200,000 (Claude context limit)

#### Methods

##### generate

```python
async def generate(
    self,
    prompt: str | PromptBuilder,
    hints: Mapping[str, Any] = {},
) -> SchematicGenerationResult[T]
```

Generates structured content using Claude models with:
- JSON schema validation
- Retry policies for rate limits and errors
- Usage tracking

### VertexAIGeminiSchematicGenerator

Schematic generator for Gemini models via Google Gen AI API.

#### Supported Hints

- `temperature`: Controls randomness (0.0-1.0)
- `thinking_config`: Configuration for reasoning models

#### Properties

- `id`: Returns `vertex-ai/{model_name}`
- `tokenizer`: Returns VertexAIEstimatingTokenizer instance
- `max_tokens`: Returns 1M (Flash) or 2M (Pro) tokens

#### Methods

##### generate

```python
async def generate(
    self,
    prompt: str | PromptBuilder,
    hints: Mapping[str, Any] = {},
) -> SchematicGenerationResult[T]
```

Generates structured content using Gemini models with:
- Native JSON schema support
- JSON parsing and validation
- Usage metadata tracking

### VertexAIEmbedder

Text embedding service using Google's text-embedding-004 model.

#### Properties

- `id`: Returns `vertex-ai/text-embedding-004`
- `dimensions`: Returns 768 (embedding dimensions)
- `max_tokens`: Returns 8,192 (input token limit)

#### Supported Hints

- `title`: Document title for better embeddings
- `task_type`: Embedding task type (default: "RETRIEVAL_DOCUMENT")

#### Methods

##### embed

```python
async def embed(
    self,
    texts: list[str],
    hints: Mapping[str, Any] = {},
) -> EmbeddingResult
```

Generates embeddings for input texts with batch processing support.

### VertexAIEstimatingTokenizer

Token counting service supporting both Claude and Gemini models.

#### Methods

##### estimate_token_count

```python
async def estimate_token_count(self, prompt: str) -> int
```

Estimates token count using:
- tiktoken for Claude models
- Google Gen AI API for Gemini models

## Error Handling

### Authentication Errors

```python
class VertexAIAuthError(Exception):
    """Raised when there are authentication issues with Vertex AI."""
```

Common causes and solutions:
- Missing ADC: Run `gcloud auth application-default login`
- Insufficient permissions: Ensure "Vertex AI User" role
- Model not enabled: Check Vertex AI Model Garden

### Rate Limiting

The adapter implements comprehensive retry policies:

#### Claude Models
- Retries: APIConnectionError, APITimeoutError, RateLimitError, APIResponseValidationError
- Max attempts: 3 with exponential backoff (1s, 2s, 4s)
- Server errors: 2 attempts with longer delays (1s, 5s)

#### Gemini Models
- Retries: NotFound, TooManyRequests, ResourceExhausted
- Max attempts: 3 with exponential backoff (1s, 2s, 4s)
- Server errors: 2 attempts with longer delays (1s, 5s)

### Error Messages

The adapter provides detailed error messages for common issues:

#### Rate Limit Exceeded
```
Vertex AI rate limit exceeded. Possible reasons:
1. Your GCP project may have insufficient quota.
2. The model may not be enabled in Vertex AI Model Garden.
3. You might have exceeded the requests-per-minute limit.

Recommended actions:
- Check your Vertex AI quotas in the GCP Console.
- Ensure the model is enabled in Vertex AI Model Garden.
- Review IAM permissions for the service account.
- Visit: https://console.cloud.google.com/vertex-ai/model-garden
```

#### Permission Denied
```
Permission denied accessing Vertex AI. Ensure:
1. ADC is properly configured (run 'gcloud auth application-default login')
2. The service account has 'Vertex AI User' role
3. The {model_name} model is enabled in Vertex AI Model Garden
```

## Performance Considerations

### Token Limits

| Model Type | Context Limit | Recommended Usage |
|------------|---------------|-------------------|
| Claude Models | 200K tokens | Long documents, complex reasoning |
| Gemini Flash | 1M tokens | Large context processing |
| Gemini Pro | 2M tokens | Maximum context requirements |

## Best Practices

### Model Selection

1. **Claude Sonnet 3.5**: Best balance of performance and cost
2. **Claude Opus 4**: Maximum capability with fallback
3. **Gemini 2.5 Flash**: Fast processing with large context
4. **Gemini 2.5 Pro**: Complex reasoning tasks

### Configuration
```python
   export VERTEX_AI_PROJECT_ID=your-project-id
   export VERTEX_AI_REGION=us-central1
   export VERTEX_AI_MODEL=claude-sonnet-3.5
```

### Error Handling

```python
from parlant.adapters.nlp.vertex_service import VertexAIAuthError

try:
    service = VertexAIService(logger=logger)
    generator = await service.get_schematic_generator(MySchema)
    result = await generator.generate(prompt)
except VertexAIAuthError as e:
    logger.error(f"Authentication failed: {e}")
    # Handle auth setup
except Exception as e:
    logger.error(f"Generation failed: {e}")
    # Handle other errors
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify ADC setup: `gcloud auth application-default print-access-token`
   - Check project permissions in GCP Console
   - Ensure service account has required roles

2. **Model Access Denied**
   - Enable models in Vertex AI Model Garden
   - Check regional availability
   - Verify billing account is active

3. **Rate Limiting**
   - Monitor quota usage in GCP Console
   - Implement application-level rate limiting
   - Consider upgrading service tier

### Debugging

Check usage from the playground UI by inspecting on the generated message

## Migration Guide

### From Other Adapters

When migrating from other NLP adapters:

1. **Update Environment Variables**
   ```bash
   # Remove old variables
   unset OPENAI_API_KEY ANTHROPIC_API_KEY
   
   # Set Vertex AI variables
   export VERTEX_AI_PROJECT_ID=your-project-id
   export VERTEX_AI_REGION=us-central1
   export VERTEX_AI_MODEL=claude-opus-4
   ```

2. **Model Name Mapping**
   - `gpt-4` → `claude-opus-4`
   - `gpt-3.5-turbo` → `gemini-2.5-flash`
   - `claude-3-sonnet` → `claude-opus-4`

## Contributing

### Adding New Models

1. **Determine Provider**: Check if model uses Anthropic or Google API
2. **Create Model Class**: Inherit from appropriate base generator
3. **Update Service**: Add model mapping in VertexAIService
4. **Add Tests**: Include integration tests for new model
5. **Update Documentation**: Add model to supported models table

### Code Style

- Follow existing patterns for error handling
- Include comprehensive logging
- Add type hints for all methods
- Document public APIs with docstrings
- Use retry policies for external API calls

## Prerequisites and Installation

### Installation

To use the Vertex AI Service Adapter with Parlant, you need to install the appropriate optional dependencies:

```bash
pip install "parlant[vertex]"
```

This installation includes support for both Claude and Gemini models through the Vertex AI platform.

### Important Model Deprecation Notice

⚠️ **Claude 3.5 Sonnet Models Deprecation**: Claude Sonnet 3.5 models (claude-3-5-sonnet-20240620 and claude-3-5-sonnet-20241022) will be retired on October 22, 2025. We recommend migrating to Claude Sonnet 4 (claude-sonnet-4-20250514) for improved performance and capabilities.

## Authentication Setup

Before using the adapter, ensure you have proper authentication configured:

```bash
# For local development
gcloud auth application-default login

# Verify authentication
gcloud auth application-default print-access-token
```

## Required Permissions

Ensure your service account or user has the following IAM roles:
- `Vertex AI User` - for accessing Vertex AI services
- `AI Platform User` - for model access (legacy role, may be needed for some models)

## License

Licensed under the Apache License, Version 2.0. See the source file header for full license text.

## Maintainer

Agam Dubey - hello.world.agam@gmail.com
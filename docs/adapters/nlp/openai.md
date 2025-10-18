# OpenAI NLP Service

The OpenAI NLP service provides access to OpenAI's powerful language models for conversational AI applications. This adapter supports both default model selection and custom model specification with automatic fallback.

## Setup

### Environment Variables

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Basic Usage

```python
import parlant.sdk as p

# Use default model selection
async with p.Server(
    nlp_service=p.NLPServices.openai()
) as server:
    agent = await server.create_agent(
        name="Assistant",
        description="A helpful AI assistant"
    )
```

## Model Selection

### Single Model

You can specify a specific model to use:

```python
import parlant.sdk as p

async with p.Server(
    nlp_service=p.NLPServices.openai(generative_model_name="gpt-4o-mini")
) as server:
    agent = await server.create_agent(
        name="Budget Assistant",
        description="A cost-effective AI assistant"
    )
```

### Multiple Models with Fallback

For improved reliability, you can specify multiple models. If the first model fails or is unavailable, the system automatically falls back to the next model in the list:

```python
import parlant.sdk as p

async with p.Server(
    nlp_service=p.NLPServices.openai(
        generative_model_name=["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06"]
    )
) as server:
    agent = await server.create_agent(
        name="Reliable Assistant",
        description="An AI assistant with fallback support"
    )
```

### Custom Models

You can use any OpenAI model, including newer models not yet in the predefined list:

```python
import parlant.sdk as p

async with p.Server(
    nlp_service=p.NLPServices.openai(generative_model_name="gpt-3.5-turbo")
) as server:
    agent = await server.create_agent(
        name="Custom Model Assistant",
        description="Using a custom model configuration"
    )
```

## Supported Models

### Predefined Models

The service includes optimized configurations for these models:

- **gpt-4o** (`gpt-4o-2024-11-20`) - Latest GPT-4 Omni model
- **gpt-4o-2024-08-06** - GPT-4 Omni from August 2024
- **gpt-4.1** - Latest GPT-4 model
- **gpt-4o-mini** - Smaller, faster GPT-4 variant

### Custom Models

Any OpenAI model can be used by specifying its exact name. For models not in the predefined list, the service will create a dynamic configuration.

## Default Behavior

When no `generative_model_name` is specified, the service uses schema-specific model selection:

- `SingleToolBatchSchema`: GPT-4o
- `JourneyNodeSelectionSchema`: GPT-4.1
- `CannedResponseDraftSchema`: GPT-4.1
- `CannedResponseSelectionSchema`: GPT-4.1
- All other schemas: GPT-4o (2024-08-06)

## Error Handling

The fallback mechanism automatically handles:

- **Rate Limiting**: Tries the next model if the current one hits rate limits
- **Model Unavailability**: Falls back if a model is temporarily unavailable
- **API Errors**: Attempts alternative models on connection or API issues

## Best Practices

### Cost Optimization

Start with smaller, more cost-effective models and fall back to larger ones:

```python
generative_model_name=["gpt-4o-mini", "gpt-4o"]
```

### Performance Optimization  

Use the fastest models first for better response times:

```python
generative_model_name=["gpt-4o-mini", "gpt-4o-2024-08-06"]
```

### Reliability

Include multiple models for maximum uptime:

```python
generative_model_name=["gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini"]
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```
   OpenAI API key not found
   ```
   **Solution**: Ensure `OPENAI_API_KEY` is set in your environment

2. **Model Not Found**
   ```
   Model 'custom-model' not found
   ```
   **Solution**: Verify the model name exists in your OpenAI account and is properly spelled

3. **Rate Limit Errors**
   ```
   Rate limit exceeded
   ```
   **Solution**: Use multiple models with fallback to automatically handle rate limits

4. **Quota Exceeded**
   ```
   You have exceeded your quota
   ```
   **Solution**: Check your OpenAI account billing and usage limits

## Advanced Configuration

### Mixing Different Model Types

You can mix different OpenAI model families in your fallback list:

```python
generative_model_name=[
    "gpt-4o-mini",           # Fast and cost-effective
    "gpt-4o",                # High quality
    "gpt-3.5-turbo"          # Backup option
]
```

### Environment-Specific Configuration

Configure different models for different environments:

```python
import os

model_config = {
    "development": "gpt-4o-mini",
    "staging": ["gpt-4o-mini", "gpt-4o"],
    "production": ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini"]
}

environment = os.getenv("ENVIRONMENT", "development")
selected_models = model_config[environment]

async with p.Server(
    nlp_service=p.NLPServices.openai(generative_model_name=selected_models)
) as server:
    # ... rest of your code
```

This approach provides flexibility for experimentation in development while ensuring reliability in production.

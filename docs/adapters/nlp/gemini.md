# Gemini Service Adapter Documentation

## Overview

The **Gemini Service Adapter** integrates with **Google’s Gemini API** to provide text generation, embeddings, and structured response generation within the Parlant SDK.
It implements the **NLPService interface** and supports flexible model selection, fallback logic, and runtime overrides for multiple Gemini models.

## Architecture

### Core Components

* **GeminiService** — Core NLP service implementing the Gemini model API integration.
* **GeminiSchematicGenerator** — Handles structured output generation (JSON/schema-constrained).
* **FallbackSchematicGenerator** — Provides layered fallback when a preferred model is unavailable or overloaded.
* **GeminiEmbedder** — Text embedding service.
* **GeminiEstimatingTokenizer** — Token counting and estimation for Gemini models.

## Configuration

### Environment Variables

```bash
# default model fallback is Gemini-2.5-flash and Gemini-2.5-Pro, runs in sequence until one succeed
GEMINI_API_KEY=your-google-api-key
```

### Authentication

The Gemini adapter uses Google’s Generative AI SDK for authentication:

```bash
export GEMINI_API_KEY="your-google-api-key"
```

You can generate your API key from the [Google AI Studio](https://aistudio.google.com/app/api-keys).

## Supported Models

| Short Name | Description | Context Limit |
|---|---|---|
| gemini-2.5-pro | Most capable reasoning model | 1M tokens |
| gemini-2.5-flash | Fastest general-purpose model | 1M tokens |
| gemini-2.0-flash | Previous-generation flash model | 1M tokens |
| gemini-2.0-flash-lite | Lightweight variant for lower latency | 1M tokens |
| gemini-1.5-pro | Prior pro model with reasoning support | 2M tokens |
| gemini-1.5-flash | Prior fast model | 1M tokens |

## Usage

### Basic Setup

```python
from parlant import sdk as p

async with p.Server(
    nlp_service=p.NLPServices.gemini(model_name="gemini-2.0-flash-lite")
) as server:
    agent = await server.create_agent(
        name="Otto Carmen",
        description="You work at a car dealership.",
    )
```

### Multiple Model Fallback Example

You can now pass multiple Gemini models to enable automatic fallback between them.

```python
from parlant import sdk as p

async with p.Server(
    nlp_service=p.NLPServices.gemini(
        model_name=["gemini-2.0-flash-lite", "gemini-2.5-flash"]
    )
) as server:
    agent = await server.create_agent(
        name="Otto Carmen",
        description="You work at a car dealership.",
    )
```

In this example:

* The system first tries gemini-2.0-flash-lite.
* If overloaded or unavailable, it automatically switches to gemini-2.5-flash using the FallbackSchematicGenerator.

### Default Model Usage

If no model name is provided and user wants to use gemini service specifically, the adapter defaults to `gemini-2.5-flash` and `gemini-2.5-pro`:

```python
async with p.Server(nlp_service=p.NLPServices.gemini) as server:
    ...
```

## API Reference

### GeminiService

#### Constructor

```python
def __init__(self, logger: Logger, model_name: str | list[str] | None = None)
```

* model_name: Can be a single model name or a list of preferred models.
* Automatically uses `FallbackSchematicGenerator` for multi-model selection.

#### Methods

##### get_schematic_generator

```python
async def get_schematic_generator(
    self, t: type[T], model_name: str | list[str] | None = None
) -> GeminiSchematicGenerator[T]
```

Returns a generator corresponding to the requested model.
If multiple models are specified, the generator chain is wrapped in `FallbackSchematicGenerator`.

##### Example Behavior

* If ["`gemini-2.0-flash-lite`", "`gemini-2.5-flash`"] is passed, it first attempts the Lite model.
* On timeout or overload, the fallback model (`gemini-2.5-flash`) is used automatically.

##### get_embedder (Planned)

```python
async def get_embedder(self) -> GeminiEmbedder
```

Returns default gemini embedding model `text-embedding-004`.

##### get_moderation_service

Returns a no-op moderation service (currently not implemented).

## Error Handling

* Unrecognized Model Name: Logs a warning and falls back to default priority order.
* Model Overload: Automatically switches to fallback model.
* Invalid Configuration: Raises SDKError for environment verification failures.

## Example Integration

### In SDK

```python
@staticmethod
def gemini(container: Container | None = None, model_names: Union[list[str], str] | None = None) -> NLPService:
    from parlant.adapters.nlp.gemini_service import GeminiService

    if error := GeminiService.verify_environment():
        raise SDKError(error)
    if model_names is not None:
        return lambda c: GeminiService(c[Logger], model_names=model_names)

    return GeminiService(container[Logger])
```

### In Service

```python
@override
async def get_schematic_generator(self, t: type[T]) -> GeminiSchematicGenerator[T]:
    model_classes: list[GeminiSchematicGenerator[T]] = []

    # Normalize to list for consistent handling
    names = [self._model_names] if isinstance(self._model_names, str) else (self._model_names or [])

    for name in names:
        model_cls = self._resolve_model_class(name)
        if model_cls:
            model_classes.append(model_cls[t](self._logger))

    # If nothing valid found, fall back to defaults
    if not model_classes:
        model_classes = [
            Gemini_2_5_Flash[t](self._logger),
            Gemini_2_5_Pro[t](self._logger),
        ]

    # If only one model, return it directly
    if len(model_classes) == 1:
        return model_classes[0]

    # Otherwise, wrap multiple models in a fallback generator
    return FallbackSchematicGenerator[t](*model_classes, logger=self._logger)
```

## Best Practices

| Use Case | Recommended Model |
|---|---|
| Fast structured replies | 'gemini-2.5-flash' |
| Lightweight, mobile edge | 'gemini-2.0-flash-lite' |
| Reasoning-heavy tasks | 'gemini-2.5-pro' |
| Background batch tasks | 'gemini-1.5-flash' |

## Troubleshooting

| Issue | Possible Cause | Fix |
|---|---|---|
| "Unrecognized model name" | Typo or unsupported version | Check '_resolve_model_class' method |
| Slow responses | Using pro-tier model | Switch to 'gemini-2.5-flash' |
| "Model overloaded" | Temporary service limits | Fallback model used automatically |

## Contributing

When adding new Gemini models:

1. Add a new mapping entry in method `_resolve_model_class` in class `GeminiService` inside `gemini_service.py`.
2. Define corresponding `Gemini_*_Generator` class.
3. Update this documentation with the new model name and description.

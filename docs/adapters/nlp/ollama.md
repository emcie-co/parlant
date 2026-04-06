# Ollama Service Documentation

The Ollama service provides local LLM capabilities for Parlant using [Ollama](https://ollama.ai/). This service supports both text generation and embeddings using various open-source models.

## Prerequisites

1. **Install Ollama**: Download and install from [ollama.ai](https://ollama.ai/)
2. **Start Ollama server**: Run `ollama serve` (usually starts automatically)
3. **Pull required models** (see [Recommended Models](#recommended-models) section)

## Environment Variables

Configure the Ollama service using these environment variables:

```bash
# Ollama server URL (default: http://localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"

# Model size to use (default: 4b)
# Options: gemma3:1b, gemma3:4b, llama3.1:8b, gemma3:12b, gemma3:27b, llama3.1:70b, llama3.1:405b
export OLLAMA_MODEL="gemma3:4b"

# Embedding model (default: nomic-embed-text)
# Options: nomic-embed-text, mxbai-embed-large
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"

# API timeout in seconds (default: 300)
export OLLAMA_API_TIMEOUT="300"
```

### Example Configuration

```bash
# For development (fast, good balance)
export OLLAMA_MODEL="gemma3:4b"
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
export OLLAMA_API_TIMEOUT="180"

# higher accuracy cloud
export OLLAMA_MODEL="gemma3:4b"
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
export OLLAMA_API_TIMEOUT="600"
```

## Recommended Models

**⚠️ IMPORTANT**: Pull these models before running Parlant to avoid API timeouts during first use:

### Text Generation Models

```bash
# Recommended for most use cases (good balance of speed/accuracy)
ollama pull gemma3:4b-it-qat

# Fast but may struggle with complex schemas
ollama pull gemma3:1b

# embedding model required for creating embeddings
ollama pull nomic-embed-text
```

### Large Models (Cloud/High-end Hardware Only)

```bash
# Better reasoning capabilities
ollama pull llama3.1:8b

# High accuracy for complex tasks
ollama pull gemma3:12b

# Very high accuracy (requires more resources)
ollama pull gemma3:27b-it-qat

# ⚠️ WARNING: Requires 40GB+ GPU memory
ollama pull llama3.1:70b

# ⚠️ WARNING: Requires 200GB+ GPU memory (cloud-only)
ollama pull llama3.1:405b
```

### Embedding Models

To use custom embedding model set OLLAMA_EMBEDDING_MODEL environment value as required name
Note that this implementation is tested using nomic-embed-text
**⚠️ IMPORTANT**:
Support for using other embedding models has been added including a custom embedding model of your own choice
Ensure to set OLLAMA_EMBEDDING_VECTOR_SIZE which is compatible with your own embedding model before starting the server
Tested with `snowflake-arctic-embed` with vector size of 1024
It is not NECESSARY to put OLLAMA_EMBEDDING_VECTOR_SIZE if you are using the supported `nomic-embed-text`, `mxbai-embed-large` or `bge-m3`. The vector size defaults to 768, 1024 and 1024 respectively for these

```bash
# Alternative embedding model (512 dimensions)
ollama pull mxbai-embed-large:latest
```

## Model Recommendations by Use Case

| Model Size | Use Case | Memory Requirements | Performance |
|------------|----------|-------------------|-------------|
| `1b` | Quick testing, simple tasks | ~2GB | Fast but limited accuracy |
| `4b` | **Recommended for development** | ~4GB | Good balance of speed/accuracy |
| `8b` |  complex reasoning | ~8GB | Better reasoning than Gemma |
| `12b` | High-accuracy tasks | ~12GB | High accuracy, slower |
| `27b` | Complex workloads | ~27GB | Very high accuracy |
| `70b` | Enterprise/cloud only | ~40GB+ | Excellent accuracy |
| `405b` | Research/cloud only | ~200GB+ | State-of-the-art |

## Usage Example

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.ollama) as server:
        agent = await server.create_agent(
            name="Healthcare Agent",
            description="Is empathetic and calming to the patient.",
        )
```

## Configuration Tips

### Development Setup
```bash
export OLLAMA_MODEL=gemma3:4b
export OLLAMA_API_TIMEOUT=180
```

### High-Performance Setup (Cloud)
```bash
export OLLAMA_MODEL=llama3.1:70b
export OLLAMA_API_TIMEOUT=300
```

### Custom / Other models
```bash
export OLLAMA_MODEL=llama3.2:3b
export OLLAMA_API_TIMEOUT=300
```

## Troubleshooting

### Common Issues

1. **Model Not Found Error**
   ```
   Model gemma3:4b not found. Please pull it first with: ollama pull gemma3:4b
   ```
   **Solution**: Run `ollama pull gemma3:4b-it-qat` before starting Parlant

2. **Connection Error**
   ```
   Cannot connect to Ollama server at http://localhost:11434
   ```
   **Solution**: Ensure Ollama is running with `ollama serve`

3. **Timeout Error**
   ```
   Request timed out after 300s
   ```
   **Solution**: Increase `OLLAMA_API_TIMEOUT` or use a smaller model

4. **Out of Memory**
   ```
   CUDA out of memory
   ```
   **Solution**: Use a smaller model size or increase GPU memory

### Performance Optimization

1. **Pre-pull models**: Always pull models before first use
2. **Adjust timeout**: Increase timeout for larger models
3. **Model selection**: Use smallest model that meets accuracy requirements
4. **GPU memory**: Monitor GPU usage and adjust model size accordingly

## Available Model Classes

The service provides these pre-configured model classes:

- `OllamaGemma3_1B`: Fast, basic accuracy
- `OllamaGemma3_4B`: **Recommended** - good balance
- `OllamaLlama31_8B`: Better reasoning
- `OllamaGemma3_12B`: High accuracy
- `OllamaGemma3_27B`: Very high accuracy
- `OllamaLlama31_70B`: Enterprise-grade (high memory)
- `OllamaLlama31_405B`: Research-grade (very high memory)

## Security Notes

- Ollama runs locally, so no data leaves your machine
- No API keys required
- Models are downloaded and cached locally
- Consider firewall rules if exposing Ollama server externally
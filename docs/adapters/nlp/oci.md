# OCI Generative AI Service Documentation

This service integrates Oracle Cloud Infrastructure (OCI) Generative AI into Parlant.
It supports both OCI API formats:

- **Generic** API format (for non-Cohere models)
- **Cohere** API format (for `cohere.*` models)

The adapter automatically routes requests based on the model ID prefix.

## Prerequisites

1. **OCI Account** with Generative AI access
2. **Config File** (default: `~/.oci/config`) or environment-based config
3. **Compartment ID** with access to the model(s)

Install the extra dependency:

```bash
pip install parlant[oci]
```

## Quick Start

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..example"

# Optional: use a custom config file and profile
export OCI_CONFIG_FILE="~/.oci/config"
export OCI_CONFIG_PROFILE="DEFAULT"

# Optional: choose models
export OCI_MODEL_ID="meta.llama-3.3-70b-instruct"
export OCI_EMBEDDING_MODEL_ID="cohere.embed-multilingual-v3.0"
```

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.oci) as server:
    agent = await server.create_agent(
        name="OCI Assistant",
        description="Powered by OCI Generative AI",
    )
```

## Configuration

The adapter is configured entirely via environment variables.

### Required

| Variable | Description |
|----------|-------------|
| `OCI_COMPARTMENT_ID` | OCI compartment OCID used for inference |

### OCI Config File

| Variable | Description | Default |
|----------|-------------|---------|
| `OCI_CONFIG_FILE` | Path to OCI config file | `~/.oci/config` |
| `OCI_CONFIG_PROFILE` | Profile name in config file | `DEFAULT` |

### Optional - Inline Config (override)

If all fields are present, the adapter uses these instead of the config file.

| Variable | Description |
|----------|-------------|
| `OCI_USER` | OCI user OCID |
| `OCI_TENANCY` | OCI tenancy OCID |
| `OCI_FINGERPRINT` | API key fingerprint |
| `OCI_KEY_FILE` | Path to private key file |
| `OCI_PASSPHRASE` | Private key passphrase (optional) |
| `OCI_REGION` | OCI region |

### Optional - LLM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OCI_MODEL_ID` | Generative model ID | `meta.llama-3.3-70b-instruct` |
| `OCI_MAX_TOKENS` | Max tokens for completion | (unset) |
| `OCI_TEMPERATURE` | Temperature | (unset) |
| `OCI_MAX_CONTEXT_TOKENS` | Context window metadata | `8192` |

### Optional - Embedding Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OCI_EMBEDDING_MODEL_ID` | Embedding model ID | `cohere.embed-multilingual-v3.0` |
| `OCI_EMBEDDING_DIMS` | Override embedding dimensions | (auto-detected / 1024 fallback) |

## Routing Behavior

- Models starting with `cohere.` use the **Cohere** chat request format.
- All other models use the **Generic** chat request format.

## Embedding Dimensions

The adapter auto-detects dimensions from the first response when possible. You can override
dimensions by setting `OCI_EMBEDDING_DIMS`. If no override is provided and dimensions have not
been detected yet, the adapter defaults to **1024** dimensions.

## Troubleshooting

**Config file not found**
- Ensure `~/.oci/config` exists or set `OCI_CONFIG_FILE` to a valid path.

**Authentication errors**
- Verify that your OCI config is correct and the API key is valid.
- Ensure the compartment has access to the target model.

**Rate limit errors**
- Check your OCI limits and quotas.
- Reduce request frequency or add backoff.
- Verify model availability in your tenancy.

# Azure OpenAI Service Documentation

The Azure service provides integration with Azure OpenAI services, supporting both legacy API key authentication and modern Azure AD authentication. This integration enables Parlant to leverage Azure's enterprise-grade AI services while maintaining security best practices.

## Prerequisites

1. **Azure OpenAI Resource**: Create an Azure OpenAI resource in your Azure subscription
2. **Authentication Setup**: Choose between API key or Azure AD authentication
3. **Model Deployment**: Deploy required models in your Azure OpenAI resource
4. **Permissions**: Ensure proper IAM roles for Azure AD authentication

## Authentication Methods

### Development (Local Machine)
For local development, use Azure CLI authentication:
```bash
# Install Azure CLI if not already installed
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login to Azure
az login

# Set your endpoint
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
```

### Production (Server Deployment)
For server deployment, **do NOT use `az login`**. Instead, use one of these methods:

#### Option 1: Service Principal (Recommended)
```bash
# Set environment variables
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_CLIENT_ID="your-service-principal-client-id"
export AZURE_CLIENT_SECRET="your-service-principal-secret"
export AZURE_TENANT_ID="your-azure-tenant-id"
```

#### Option 2: Managed Identity (Azure Resources)
If running on Azure VMs, App Services, or other Azure resources:
```bash
# Only set the endpoint - authentication is automatic
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
```

#### Option 3: Workload Identity (Kubernetes)
For Kubernetes deployments:
```bash
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_CLIENT_ID="your-workload-identity-client-id"
export AZURE_TENANT_ID="your-azure-tenant-id"
export AZURE_FEDERATED_TOKEN_FILE="/var/run/secrets/azure/tokens/azure-identity-token"
```

## Environment Variables

### Required Variables
- `AZURE_ENDPOINT`: Your Azure OpenAI resource endpoint

### Optional Variables
- `AZURE_API_VERSION`: API version (default: "2024-08-01-preview")
- `AZURE_GENERATIVE_MODEL_NAME`: Model name (default: "gpt-4o")
- `AZURE_GENERATIVE_MODEL_WINDOW`: Context window size (default: 4096)
- `AZURE_EMBEDDING_MODEL_NAME`: Embedding model (default: "text-embedding-3-large")
- `AZURE_EMBEDDING_MODEL_DIMS`: Embedding dimensions (default: 3072)
- `AZURE_EMBEDDING_MODEL_WINDOW`: Embedding context window (default: 8192)

## Supported Models

The Azure service supports **any Azure OpenAI model** that is deployed and available in your Azure OpenAI resource. The models listed below are pre-configured defaults, but you can use any model by setting the appropriate environment variables.

### Pre-configured Generative Models

| Model Name | Description | Context Window | Use Case |
|------------|-------------|---------------|----------|
| `gpt-4o` | Most capable GPT-4 model (default) | 128K tokens | Complex reasoning, high accuracy |
| `gpt-4o-mini` | Faster, cost-effective GPT-4 | 128K tokens | Balanced performance and cost |

### Pre-configured Embedding Models

| Model Name | Dimensions | Context Window | Description |
|------------|------------|---------------|-------------|
| `text-embedding-3-large` | 3072 | 8192 | High-quality embeddings (default) |
| `text-embedding-3-small` | 3072 | 8192 | Efficient embeddings |

### Using Custom Models

You can use **any Azure OpenAI model** that is deployed in your Azure OpenAI resource:

```bash
# Use any generative model (examples - check your Azure resource for availability)
export AZURE_GENERATIVE_MODEL_NAME="gpt-35-turbo"  # GPT-3.5 Turbo
export AZURE_GENERATIVE_MODEL_NAME="gpt-4"        # GPT-4
export AZURE_GENERATIVE_MODEL_NAME="gpt-4-turbo"  # GPT-4 Turbo

# Use any embedding model (examples - check your Azure resource for availability)
export AZURE_EMBEDDING_MODEL_NAME="text-embedding-ada-002"  # Ada embeddings
export AZURE_EMBEDDING_MODEL_NAME="text-embedding-3-large" # Large embeddings
```

**Important**: 
- Model availability depends on what you've deployed in your Azure OpenAI resource
- Not all models are available in all Azure regions
- Check your Azure OpenAI resource deployment to see which models are available

## Authentication Priority

The service follows this authentication priority:

1. **API Key** (highest priority - for backward compatibility)
2. **Azure AD** (fallback when no API key is present)

## Required Azure Permissions

For Azure AD authentication, ensure your identity has the following role on the Azure OpenAI resource:

- **Cognitive Services OpenAI User**: Required for accessing Azure OpenAI services

## Usage Example

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

async with p.Server(nlp_service=NLPServices.azure) as server:
        agent = await server.create_agent(
            name="Healthcare Agent",
            description="Is empathetic and calming to the patient.",
        )
```

## Server Deployment Guide

### Setting Up Service Principal for Production

1. **Create Service Principal**:
   ```bash
   # Login as admin user
   az login
   
   # Create service principal
   az ad sp create-for-rbac --name "parlant-service-principal" --role "Cognitive Services OpenAI User" --scopes "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/YOUR_RESOURCE_GROUP/providers/Microsoft.CognitiveServices/accounts/YOUR_OPENAI_RESOURCE"
   ```

2. **Configure Environment Variables**:
   ```bash
   export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_CLIENT_ID="appId-from-step-1"
   export AZURE_CLIENT_SECRET="password-from-step-1"
   export AZURE_TENANT_ID="tenant-from-step-1"
   ```

3. **Test Authentication**:
   ```bash
   # Verify the service principal can access Azure OpenAI
   python -c "
   from parlant.adapters.nlp.azure_service import AzureService
   error = AzureService.verify_environment()
   print('Configuration OK' if error is None else f'Error: {error}')
   "
   ```

### Configuration Tips

### Development Setup
```bash
export AZURE_ENDPOINT="https://my-resource.openai.azure.com/"
export AZURE_API_KEY="your-api-key"
export AZURE_GENERATIVE_MODEL_NAME="gpt-4o-mini"
```

### Production Setup (Azure AD)
```bash
export AZURE_ENDPOINT="https://my-resource.openai.azure.com/"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_GENERATIVE_MODEL_NAME="gpt-4o"
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   ```
   Azure authentication is not properly configured.
   ```
   **Solution**: 
   - For development: Run `az login` (only for local development)
   - For production: Use service principal variables (NOT `az login`)
   - Ensure "Cognitive Services OpenAI User" role is assigned
   - Verify service principal has correct permissions

2. **Rate Limit Errors**
   ```
   Azure API rate limit exceeded
   ```
   **Solution**: 
   - Check Azure account balance and billing status
   - Review API usage limits in Azure dashboard
   - Consider upgrading service tier

3. **Model Access Denied**
   ```
   Model not found or access denied
   ```
   **Solution**: 
   - Verify model is deployed in your Azure OpenAI resource
   - Check regional availability
   - Ensure proper permissions

4. **Connection Errors**
   ```
   Cannot connect to Azure OpenAI endpoint
   ```
   **Solution**: 
   - Verify `AZURE_ENDPOINT` is correct
   - Check network connectivity
   - Ensure firewall allows Azure OpenAI traffic

## Available Model Classes

The service provides these pre-configured model classes for convenience, but supports any Azure OpenAI model:

### Pre-configured Classes
- `GPT_4o`: Most capable GPT-4 model (128K context) - **Default**
- `GPT_4o_Mini`: Faster, cost-effective GPT-4 (128K context)
- `AzureTextEmbedding3Large`: High-quality embeddings (3072 dimensions) - **Default**
- `AzureTextEmbedding3Small`: Efficient embeddings (3072 dimensions)

### Custom Model Classes
- `CustomAzureSchematicGenerator`: Uses any generative model via `AZURE_GENERATIVE_MODEL_NAME`
- `CustomAzureEmbedder`: Uses any embedding model via `AZURE_EMBEDDING_MODEL_NAME`

**The service automatically chooses the appropriate class based on your environment variables.**

### How Model Selection Works

The service uses this logic to select the appropriate model class:

```python
# Generative Model Selection
if AZURE_GENERATIVE_MODEL_NAME is set:
    use CustomAzureSchematicGenerator  # Any model you specify
else:
    use GPT_4o  # Default model

# Embedding Model Selection  
if AZURE_EMBEDDING_MODEL_NAME is set:
    use CustomAzureEmbedder  # Any embedding model you specify
else:
    use AzureTextEmbedding3Large  # Default embedding model
```

This means you can use **any Azure OpenAI model** without code changes - just set the environment variables!

### Example: Using Different Models

```bash
# Use GPT-3.5 Turbo (if available in your region)
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_GENERATIVE_MODEL_NAME="gpt-35-turbo"
export AZURE_EMBEDDING_MODEL_NAME="text-embedding-ada-002"

# Use GPT-4 Turbo (if available in your region)
export AZURE_GENERATIVE_MODEL_NAME="gpt-4-turbo"
export AZURE_EMBEDDING_MODEL_NAME="text-embedding-3-large"

# Use default models (GPT-4o and text-embedding-3-large)
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
# No need to set AZURE_GENERATIVE_MODEL_NAME or AZURE_EMBEDDING_MODEL_NAME
```

## Security Notes

- **API Keys**: Store securely, rotate regularly
- **Azure AD**: Use managed identities in production
- **Network**: Ensure proper network security groups
- **Monitoring**: Monitor usage and access patterns
- **Compliance**: Follow organizational security policies

## Migration Guide

### From API Key to Azure AD

1. Set up Azure AD authentication using one of the supported methods
2. Remove the API key from your environment variables
3. Verify permissions - ensure your identity has "Cognitive Services OpenAI User" role
4. Test the configuration using `AzureService.verify_environment()`

### Backward Compatibility

The service maintains full backward compatibility:
- Existing API key configurations continue to work
- No changes required for existing deployments
- Gradual migration to Azure AD is supported

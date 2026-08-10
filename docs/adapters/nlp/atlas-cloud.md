# Atlas Cloud

Parlant can use [Atlas Cloud](https://www.atlascloud.ai/) as its NLP service through Atlas Cloud's OpenAI-compatible API.

## Installation

Install Parlant with the dependencies needed by the Atlas Cloud adapter:

```bash
pip install "parlant[atlascloud]"
```

## Configuration

Set your API key:

```bash
export ATLASCLOUD_API_KEY="your-api-key"
```

The adapter defaults to `qwen/qwen3.8-max`. To select another Atlas Cloud chat model, set:

```bash
export ATLASCLOUD_MODEL="your-model-id"
```

You can also override the default generation limit:

```bash
export ATLASCLOUD_MAX_TOKENS="8192"
```

## CLI

Start Parlant with Atlas Cloud:

```bash
parlant-server run --atlascloud
```

## SDK

```python
import parlant.sdk as p

async with p.Server(nlp_service=p.NLPServices.atlascloud) as server:
    agent = await server.create_agent(name="Atlas Cloud Agent")
```

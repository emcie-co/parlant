# Snowflake Cortex Adapter

Integrate [Snowflake Cortex REST APIs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api) for **chat/structured generation** and **embeddings** with Snowflake-hosted LLMs. The adapter talks directly to:

- `POST /api/v2/cortex/inference:complete` for chat/JSON output
- `POST /api/v2/cortex/inference:embed` for embeddings

## Requirements

### Authentication
See [Snowflake REST API authentication](https://docs.snowflake.com/en/developer-guide/snowflake-rest-api/authentication). PAT is recommended.

### Environment Variables
See [Cortex models](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm) for available model names.

```bash
export SNOWFLAKE_CORTEX_BASE_URL="https://<account>.snowflakecomputing.com"
export SNOWFLAKE_AUTH_TOKEN="<jwt-or-pat>"
export SNOWFLAKE_CORTEX_CHAT_MODEL="mistral-large2"
export SNOWFLAKE_CORTEX_EMBED_MODEL="e5-base-v2"
# Optional:
export SNOWFLAKE_CORTEX_MAX_TOKENS="8192"
```

## Usage Example

```python
import parlant.sdk as p
from parlant.sdk import NLPServices

@p.tool
async def get_weather(context: p.ToolContext, city: str) -> p.ToolResult:
    # Your weather API logic here
    return p.ToolResult(f"Sunny, 72°F in {city}")

@p.tool
async def get_datetime(context: p.ToolContext) -> p.ToolResult:
    from datetime import datetime
    return p.ToolResult(datetime.now())

async def main():
    async with p.Server(nlp_service=NLPServices.snowflake) as server:
        agent = await server.create_agent(
            name="WeatherBot",
            description="Helpful weather assistant"
        )

        # Have the agent's context be updated on every response (though
        # update interval is customizable) using a context variable.
        await agent.create_variable(name="current-datetime", tool=get_datetime)

        # Control and guide agent behavior with natural language
        await agent.create_guideline(
            condition="User asks about weather",
            action="Get current weather and provide a friendly response with suggestions",
            tools=[get_weather]
        )

        # Add other (reliably enforced) behavioral modeling elements
        # ...

        # 🎉 Test playground ready at http://localhost:8800
        # Integrate the official React widget into your app,
        # or follow the tutorial to build your own frontend!

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Configuration Reference

| Variable                         | Required | Description |
|----------------------------------|----------|-------------|
| `SNOWFLAKE_CORTEX_BASE_URL`      | ✅       | Base account URL (e.g., `https://<account>.snowflakecomputing.com`).  |
| `SNOWFLAKE_AUTH_TOKEN`           | ✅       | OAuth / Keypair JWT / PAT used in the `Authorization: Bearer` header.  |
| `SNOWFLAKE_CORTEX_CHAT_MODEL`    | ✅       | Chat model name. |
| `SNOWFLAKE_CORTEX_EMBED_MODEL`   | ✅       | Embedding model name. |
| `SNOWFLAKE_CORTEX_MAX_TOKENS`    | ❌       | Local upper bound for generation; does not override provider limits.  |


## Notes on Privacy & Data Residency

The adapter allows apps to call Cortex directly in your Snowflake account, reducing the need to move data outside Snowflake for LLM tasks. Review Snowflake's REST guidance for regional availability and account setup.
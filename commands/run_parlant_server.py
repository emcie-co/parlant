import asyncio
import os
from parlant.bin.server import start_server
from dataclasses import dataclass
from typing import List

@dataclass
class CLIParams:
    port: int = 8800
    nlp_service: str = "openai"  # or "anthropic", "aws", "azure", "gemini"
    log_level: str = "debug"
    modules: List[str] = None
    migrate: bool = False

async def main():
    # Set your API key based on the NLP service you're using
    if os.environ.get("OPENAI_API_KEY"):
        params = CLIParams(
            port=8800,
            nlp_service="openai",
            log_level="debug",
            modules=[],
            migrate=False
        )

    
    await start_server(params)

if __name__ == "__main__":
    asyncio.run(main())
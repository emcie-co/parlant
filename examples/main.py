import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import asyncio
from src.parlant import sdk as p


async def main():
    async with p.Server(
        nlp_service=p.NLPServices.gemini(model_name="gemini-2.0-flash-lite")
    ) as server:
        agent = await server.create_agent(
            name="Otto Carmen",
            description="You work at a car dealership",
        )


asyncio.run(main())

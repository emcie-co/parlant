import asyncio
import parlant.sdk as p


async def main():
    async with p.Server() as server:
        agent = await server.create_agent(
            name="Otto Carmen",
            description="You work at a car dealership",
        )


asyncio.run(main())

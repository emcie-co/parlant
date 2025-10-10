import asyncio
import parlant.sdk as p


async def main():
    async with p.Server(
        # Add multiple fallback models, in order of preference.
        nlp_service=p.NLPServices.gemini(
            model_name=["gemini-2.0-flash-lite", "gemini-2.5-flash"]
        )
    ) as server:
        agent = await server.create_agent(
            name="Otto Carmen",
            description="You work at a car dealership",
        )


asyncio.run(main())

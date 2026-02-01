from contextlib import AsyncExitStack
import os
from httpx import AsyncClient
from lagom import Container
from parlant.core.tracer import Tracer, CompositeTracer
from parlant.adapters.tracing.emcie import EmcieTracer

EXIT_STACK = AsyncExitStack()


async def configure_module(container: Container) -> Container:
    api_key = os.environ.get("EMCIE_API_KEY")
    api_url = os.environ.get("EMCIE_BASE_URL", "https://api.emcie.co")
    tracer = container[Tracer]

    if not api_key:
        return container  # No API key, skip

    auth_url = f"{api_url}/v1/auth/api-key"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with AsyncClient() as client:
            resp = await client.post(auth_url, headers=headers)
            resp.raise_for_status()
    except Exception:
        return container

    # Authenticated: add EmcieTracer to CompositeTracer
    emcie_tracer = await EXIT_STACK.enter_async_context(EmcieTracer())
    if isinstance(tracer, CompositeTracer):
        tracer.append(emcie_tracer)
        container[Tracer] = tracer
    else:
        container[Tracer] = CompositeTracer([tracer, emcie_tracer])

    return container

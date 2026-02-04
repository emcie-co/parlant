from contextlib import AsyncExitStack
import os
from httpx import AsyncClient
from lagom import Container
from parlant.core.tracer import Tracer, CompositeTracer
from parlant.core.loggers import Logger, CompositeLogger
from parlant.core.meter import Meter
from parlant.adapters.tracing.emcie import EmcieTracer
from parlant.adapters.loggers.emcie import EmcieLogger
from parlant.adapters.meters.emcie import EmcieMeter

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
        # No need to reassign - we mutated the object in place
    else:
        # Create a new CompositeTracer and define it in container
        composite_tracer = CompositeTracer([tracer, emcie_tracer])
        container.define(Tracer, composite_tracer)

    # Add EmcieLogger to CompositeLogger if logger exists in container
    try:
        logger = container[Logger]
        emcie_logger = await EXIT_STACK.enter_async_context(EmcieLogger(tracer=tracer))

        if isinstance(logger, CompositeLogger):
            logger.append(emcie_logger)
            # No need to reassign - we mutated the object in place
        else:
            # Create a new CompositeLogger and define it in container
            composite_logger = CompositeLogger([logger, emcie_logger])
            container.define(Logger, composite_logger)
    except Exception:
        # Logger not in container, skip logger configuration
        pass

    # Add EmcieMeter if no meter exists in container
    try:
        # Check if meter exists by trying to get it
        _ = container[Meter]
        # If it exists, skip - we can't replace it due to Lagom constraints
    except Exception:
        # Meter not in container, add EmcieMeter
        emcie_meter = await EXIT_STACK.enter_async_context(EmcieMeter())
        container[Meter] = emcie_meter

    return container

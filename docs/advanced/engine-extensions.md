# Engine Extensions

Working with an external framework and adapting it to your needs isn’t always simple, especially when you need it to behave in ways its original design didn’t anticipate.
Modifying the framework’s source code is a treacherous path—not just because it requires deeper expertise, but also because it leads to divergences between your locally-modified version and upstream updates.

So how do you get a pre-built framework to work differently? The idea is to be able to run a system or software that includes your code customizations without breaking its fundamental assumptions.

The [Open/Closed Principle](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle) states that software should be open for extension, but closed for modification, such that it can allow its behavior to be extended without modifying its source code. Parlant is carefully designed to abide by this principle, allowing you to achieve extreme extensibility by hooking into its structure.

With extensions, you are free to build exactly what you need without waiting for updates or modifying core engine components. This is a good time to remind you that you can join our [Discord](https://discord.gg/duxWqxKk6J) community to ask questions.

## Engine Hooks
Every time an agent needs to respond to a customer, the engine goes through a series of steps to generate the response. You can hook into these steps to modify the behavior of the engine. This is easily done by registering hook functions.

While there are many hooks you can utilize, here's a simple example that:
1. Overrides the entire engine's response generation process if we detect that the customer only greeted the agent.
1. Inspects the agent's message for compliance breaches (using a custom checker) before sending it to the customer.

```python
import asyncio
from typing import Any
import parlant.sdk as p

async def intercept_message_generation_with_greeting(
    ctx: p.LoadedContext, payload: Any, exc: Exception | None
) -> p.EngineHookResult:
    if await is_only_greeting(ctx.interaction.last_customer_message):
        await ctx.session_event_emitter.emit_message_event(
            trace_id=ctx.tracer.trace_id,
            data="Hello! How can I help you today?",
        )
        return p.EngineHookResult.BAIL  # Intercept the rest of the process
    else:
        return p.EngineHookResult.CALL_NEXT  # Continue with the normal process

async def check_message_compliance(
    ctx: p.LoadedContext, payload: Any, exc: Exception | None
) -> p.EngineHookResult:
    generated_message = payload

    if not await is_compliant(generated_message):
        ctx.logger.warning(f"Prevented sending a non-compliant message: '{generated_message}'.")
        return p.EngineHookResult.BAIL  # Do not send this message

    return p.EngineHookResult.CALL_NEXT  # Continue with the normal process

async def configure_hooks(hooks: p.EngineHooks) -> p.EngineHooks:
    hooks.on_acknowledged.append(intercept_message_generation_with_greeting)
    hooks.on_message_generated.append(check_message_compliance)

    return hooks

async def main():
    async with p.Server(
        configure_hooks=configure_hooks,
    ) as server:
        # Your logic here
        ...
```

## Dependency Injection
In order to extend the engine without modifying its source code, Parlant uses a [dependency injection](https://en.wikipedia.org/wiki/Dependency_injection) system. This allows you to inject your own implementations of various components or even the processing engine itself (say, if you wanted to optimize the entire pipeline for particular use cases).

For simplicity, we'll take a look at some basic extension mechanics, as well as common use cases for extension.

However, if you need help with something that isn't covered here, please reach out to us on [Discord](https://discord.gg/duxWqxKk6J), [GitHub Discussions](https://github.com/emcie-co/parlant/discussions), or simply using the [Contact Page](https://parlant.io/contact) and we'll get back to you.

### Working with the Container
Let's see how to work with Parlant's dependency injection container. The container is a central place where all components are registered, and you can use it to retrieve or register your own components.

There are two things you might want to do with respect to the container:

1. **Register your own components**: You can add your own implementations of various components to the container, making them available for injection throughout the application.
1. **Adjust the behavior of existing components**: You can retrieve instances of components from the container, allowing you to use them in your own code.

#### Registering Components
Registering components lets you override nearly every aspect of how Parlant works. You can access the container during its registration phase by passing a `configure_container` hook to the server.

This hook accepts a baseline state of the container, and returns a modified version of it before the server starts.

```python
import asyncio
import parlant.sdk as p

async def configure_container(container: p.Container) -> p.Container:
    # Register your own components here
    # ...
    return container

async def main():
    async with p.Server(
        configure_container=configure_container,
    ) as server:
        # Your logic here
        ...
```

#### Adjusting Existing Components
If you want to adjust the behavior of built-in components, you can retrieve them from the container and modify their behavior. This is useful for debugging or extending existing functionality without replacing the entire component.

This hook is called `initialize_container`, and it allows you to modify components within the container after all of the classes have been registered and determined—but before the server actually starts to use them.

This hook accepts the final state of the container, and returns `None`, as the container is only provided for _accessing_ registered components.

```python
import asyncio
import parlant.sdk as p

async def initialize_container(container: p.Container) -> None:
    # Register your own components here
    # ...
    return container

async def main():
    async with p.Server(
        configure_container=configure_container,
    ) as server:
        # Your logic here
        ...
```

## Open for Extension
If you read or debug Parlant code, you'll come across many different types of components within the engine. Using the configuration and initialization hooks, you now know how to access them and extend, modify, or completely override their implementations as needed.

#### Common Use Cases for Extensions

1. Overriding the no-match behavior of canned responses. This is actually documented here: [Canned Responses](https://parlant.io/docs/concepts/customization/canned-responses#no-match-responses).
1. Wrapping any engine component to add logging, monitoring, or other cross-cutting concerns.
1. Overriding the way particular guidelines are evaluated. For example, if they are simple and you have enough data, you can evaluate them with custom-trained BERT models instead of going through an LLM.
1. Overriding the entire message generation component, allowing you to leverage Parlant's guideline matching and tool execution, but using your message generation logic.

But there's much more you can do. The engine is designed to be flexible and extensible, so you can adapt it to your specific needs without modifying the core codebase.

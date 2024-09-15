import asyncio
from textwrap import dedent
from typing import Any, Awaitable, Generator, TypeVar

from emcie.server.core.generation.schematic_generators import GPT_4o
from emcie.server.logger import Logger
from emcie.server.mc import EventBuffer as EventBuffer
from emcie.server.base_models import DefaultBaseModel

T = TypeVar("T")


class NLPTestSchema(DefaultBaseModel):
    answer: bool


class SyncAwaiter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def __call__(self, awaitable: Generator[Any, None, T] | Awaitable[T]) -> T:
        return self.event_loop.run_until_complete(awaitable)  # type: ignore


async def nlp_test(logger: Logger, context: str, predicate: str) -> bool:
    schematic_generator = GPT_4o[NLPTestSchema]()
    inference = await schematic_generator.generate(
        prompt=dedent(
            f"""\
                        Given a context and a predicate, determine whether the
                        predicate applies with respect to the given context.

                        Context: ###
                        {context}
                        ###

                        Predicate: ###
                        {predicate}
                        ###

                        Output JSON structure:
                        {{
                            answer: <BOOL>
                        }}

                        Example #1:
                        {{
                            answer: true
                        }}

                        Example #2:
                        {{
                            answer: false
                        }}
                    """
        ),
        hints={"temperature": 0.0},
    )

    return inference.content.answer

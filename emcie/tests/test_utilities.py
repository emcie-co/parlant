import asyncio
from contextlib import contextmanager
import logging
from typing import Any, Awaitable, Generator, Iterator, TypeVar

from emcie.server.adapters.nlp.openai import GPT_4o
from emcie.server.core.logging import Logger
from emcie.server.core.common import DefaultBaseModel

T = TypeVar("T")


class NLPTestSchema(DefaultBaseModel):
    answer: bool


class SyncAwaiter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def __call__(self, awaitable: Generator[Any, None, T] | Awaitable[T]) -> T:
        return self.event_loop.run_until_complete(awaitable)  # type: ignore


class TestLogger(Logger):
    def __init__(self) -> None:
        self.logger = logging.getLogger("TestLogger")

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    @contextmanager
    def operation(self, name: str, props: dict[str, Any] = {}) -> Iterator[None]:
        yield


async def nlp_test(context: str, predicate: str) -> bool:
    schematic_generator = GPT_4o[NLPTestSchema](logger=TestLogger())

    inference = await schematic_generator.generate(
        prompt=f"""\
Given a context and a predicate, determine whether the
predicate applies with respect to the given context.
If the predicate applies, the answer is true;
otherwise, the answer is false.

Context: ###
{context}
###

Predicate: ###
{predicate}
###

Output JSON structure: ###
{{
    answer: <BOOL>
}}
###

Example #1: ###
{{
    answer: true
}}
###

Example #2: ###
{{
    answer: false
}}
###
""",
        hints={"temperature": 0.0, "strict": True},
    )
    return inference.content.answer

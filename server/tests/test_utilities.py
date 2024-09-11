import asyncio
from textwrap import dedent
from typing import Any, Awaitable, Generator, TypeVar
import tiktoken

from emcie.server.llm.text_generators import GPT4o
from emcie.server.mc import EventBuffer as EventBuffer

T = TypeVar("T")

text_generator = GPT4o()
tokenizer = tiktoken.encoding_for_model("gpt-4o")


class SyncAwaiter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def __call__(self, awaitable: Generator[Any, None, T] | Awaitable[T]) -> T:
        return self.event_loop.run_until_complete(awaitable)  # type: ignore


async def nlp_test(context: str, predicate: str) -> bool:
    answer = await text_generator.generate(
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

                    If the answer is YES, answer "y".
                    If the answer is NO, answer "n".
                """
        ),
        args={
            "logit_bias": {
                tokenizer.encode_single_token("y"): 100,  # type: ignore
                tokenizer.encode_single_token("n"): 100,  # type: ignore
            },
            "max_tokens": 1,
        },
    )

    if answer == "y":
        return True
    elif answer == "n":
        return False

    raise Exception(f"NLP test error. Invalid answer '{answer}'.")

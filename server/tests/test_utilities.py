import asyncio
import os
from textwrap import dedent
from typing import Any, Awaitable, Generator, TypeVar
from openai import Client
import tiktoken

from emcie.server.mc import EventBuffer as EventBuffer

T = TypeVar("T")

llm_client = Client(api_key=os.environ["OPENAI_API_KEY"])
llm_name = "gpt-4o"
tokenizer = tiktoken.encoding_for_model(llm_name)


class SyncAwaiter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def __call__(self, awaitable: Generator[Any, None, T] | Awaitable[T]) -> T:
        return self.event_loop.run_until_complete(awaitable)  # type: ignore


def nlp_test(context: str, predicate: str) -> bool:
    response = llm_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": dedent(
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
            }
        ],
        model=llm_name,
        logit_bias={
            tokenizer.encode_single_token("y"): 100,  # type: ignore
            tokenizer.encode_single_token("n"): 100,  # type: ignore
        },
        max_tokens=1,
    )

    answer = response.choices[0].message.content

    if answer == "y":
        return True
    elif answer == "n":
        return False

    raise Exception(f"NLP test error. Invalid answer '{answer}'.")

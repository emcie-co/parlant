import asyncio
import os
from textwrap import dedent
from typing import Any, Awaitable, Generator, Sequence, TypeVar
from openai import Client
import tiktoken

from emcie.server.core.sessions import Event, ToolCallResult
from emcie.server.engines.event_emitter import EmittedEvent, EventEmitter


T = TypeVar("T")

llm_client = Client(api_key=os.environ["OPENAI_API_KEY"])
llm_name = "gpt-4o"
tokenizer = tiktoken.encoding_for_model(llm_name)


class SyncAwaiter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def __call__(self, awaitable: Generator[Any, None, T] | Awaitable[T]) -> T:
        return self.event_loop.run_until_complete(awaitable)  # type: ignore


class EventBuffer(EventEmitter):
    def __init__(self) -> None:
        self.events: list[EmittedEvent] = []

    async def emit_message(self, message: str) -> EmittedEvent:
        event = EmittedEvent(
            source="server",
            kind=Event.MESSAGE_KIND,
            data={"message": message},
        )

        self.events.append(event)

        return event

    async def emit_tool_results(self, results: Sequence[ToolCallResult]) -> EmittedEvent:
        event = EmittedEvent(
            source="server",
            kind=Event.TOOL_KIND,
            data={"tool_results": list(results)},  # type: ignore
        )

        self.events.append(event)

        return event


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

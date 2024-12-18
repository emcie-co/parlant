# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import inspect
from sys import _getframe
from typing import Any, Callable, Optional, cast
from lagom import Container
from pytest_bdd import parsers

from parlant.core.common import generate_id, JSONSerializable
from parlant.core.customers import Customer
from parlant.core.tools import Tool
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.guidelines import Guideline
from parlant.core.sessions import Event, MessageEventData, EventSource, EventId
from tests.test_utilities import SyncAwaiter


class Step:
    def __init__(
        self,
        installer: Any,
        parser: str | parsers.StepParser,
        kwargs: Any,
        func: Callable[..., None],
    ):
        self._installer = installer
        self._parser = parser
        self._kwargs = kwargs
        self._func = func

    def install(self) -> None:
        self._installer(self._parser, stacklevel=3, **self._kwargs)(self._func)


@dataclass
class ContextOfTest:
    sync_await: SyncAwaiter
    container: Container
    events: list[Event]
    guidelines: dict[str, Guideline]
    guideline_propositions: dict[str, GuidelineProposition]
    tools: dict[str, Tool]


def load_steps(*module_names: str) -> None:
    this_module = inspect.getmodule(_getframe(0))
    assert this_module

    for module_name in module_names:
        module = importlib.import_module(
            f"tests.core.engines.alpha.steps.{module_name}", this_module.__name__
        )
        steps = [a for a in module.__dict__.values() if isinstance(a, Step)]

        for s in steps:
            s.install()


def step(
    installer: Any,
    parser: str | parsers.StepParser,
    **kwargs: Any,
) -> Callable[..., Step]:
    def wrapper(func: Callable[..., None]) -> Step:
        return Step(installer, parser, kwargs, func)

    return wrapper


def create_event_message(
    offset: int,
    source: EventSource,
    message: str,
    customer: Optional[Customer] = None,
) -> Event:
    message_data: MessageEventData = {
        "message": message,
        "participant": {
            "display_name": customer.name if customer else source,
        },
    }

    event = Event(
        id=EventId(generate_id()),
        source=source,
        kind="message",
        offset=offset,
        correlation_id="test_correlation_id",
        data=cast(JSONSerializable, message_data),
        creation_utc=datetime.now(timezone.utc),
        deleted=False,
    )

    return event

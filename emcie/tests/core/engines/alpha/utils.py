from dataclasses import dataclass
import importlib
import inspect
from sys import _getframe
from typing import Any, Callable
from lagom import Container
from pytest_bdd import parsers

from emcie.server.core.tools import Tool
from emcie.server.core.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.core.guidelines import Guideline

from emcie.server.core.sessions import Event
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
        module = importlib.import_module(f"steps.{module_name}", this_module.__name__)
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

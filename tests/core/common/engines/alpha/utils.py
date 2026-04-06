# Copyright 2026 Emcie Co Ltd.
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

import importlib
import inspect
from sys import _getframe
from pytest_bdd import parsers
from typing import Any, Callable


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


def load_steps(*module_names: str) -> None:
    this_module = inspect.getmodule(_getframe(0))
    assert this_module

    for module_name in module_names:
        module = importlib.import_module(
            f"tests.core.common.engines.alpha.steps.{module_name}", this_module.__name__
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

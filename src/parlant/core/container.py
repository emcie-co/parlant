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

"""The project's dependency-injection container.

This module is the only place allowed to import lagom. It exposes exactly the
container surface the codebase uses — subscript get/set, ``clone``,
``defined_types``, and ``Singleton`` — and nothing more.

Keys are typed ``Callable[..., T]`` rather than ``type[T]``. Both accept a
class object, but the ``Callable`` form also accepts abstract classes, which
``type[T]`` rejects under mypy's ``type-abstract`` rule ("Only concrete class
can be given..."). Ports here are almost always ABCs, so with ``type[T]``
keys every binding site would be an error; this typing is what lets the
project keep that error code enabled globally.

The trade-off is on assignment: with a ``Callable`` key, mypy infers the key's
TypeVar jointly from key and value, so a mismatched value widens ``T`` to
``object`` instead of erroring. ``__setitem__`` therefore validates instance
bindings at runtime — bindings happen at startup, so a wrong value still
fails fast, with a clear error instead of a distant resolution failure.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar, cast

import lagom
import lagom.interfaces

T = TypeVar("T")


class Container:
    """A type-keyed dependency container (delegating to lagom internally)."""

    def __init__(self) -> None:
        self._inner = lagom.Container()

    @classmethod
    def _wrap(cls, inner: lagom.Container) -> Container:
        container = cls.__new__(cls)
        container._inner = inner
        return container

    def __getitem__(self, key: Callable[..., T]) -> T:
        return cast(T, self._inner[cast(Any, key)])

    def __setitem__(self, key: Callable[..., T], value: T | Callable[..., T]) -> None:
        # Mypy is lenient about the value here (see the module docstring), so
        # plain instance bindings are checked at runtime. Classes, factories,
        # and definition objects (e.g. Singleton) construct the value later —
        # lagom validates those through its own definition normalization.
        if (
            isinstance(key, type)
            and not isinstance(value, lagom.interfaces.SpecialDepDefinition)
            and not callable(value)
            and not isinstance(value, key)
        ):
            raise TypeError(
                f"Cannot bind {value!r} to {key.__name__}: it is not an instance of {key.__name__}"
            )

        # Lagom invokes container-aware factories with ITS container. Wrap
        # them so user factories receive this module's Container instead —
        # otherwise every resolution inside a factory body would be typed
        # against lagom again, resurrecting the type-abstract problem there.
        if (
            callable(value)
            and not isinstance(value, type)
            and not isinstance(value, lagom.interfaces.SpecialDepDefinition)
            and len(inspect.signature(value).parameters) == 1
        ):
            factory = cast(Callable[[Container], Any], value)
            self._inner[cast(Any, key)] = lambda lc: factory(Container._wrap(lc))
            return

        self._inner[cast(Any, key)] = value

    def clone(self) -> Container:
        return Container._wrap(self._inner.clone())

    @property
    def defined_types(self) -> set[type]:
        return cast(set[type], self._inner.defined_types)


def Singleton(cls: Callable[..., T]) -> Callable[..., T]:
    """A binding value that constructs ``cls`` lazily, once, resolving its
    constructor dependencies from the container."""
    return cast(Callable[..., T], lagom.Singleton(cast(Any, cls)))

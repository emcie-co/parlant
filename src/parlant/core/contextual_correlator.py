# Copyright 2025 Emcie Co Ltd.
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

from abc import ABC, abstractmethod
from contextlib import contextmanager
import contextvars
from typing import Any, Iterator, Mapping
from typing_extensions import override

from parlant.core.common import generate_id

_UNINITIALIZED = 0xC0FFEE


class ContextualCorrelator(ABC):
    @contextmanager
    @abstractmethod
    def scope(
        self,
        scope_id: str,
        properties: Mapping[str, Any] = {},
    ) -> Iterator[None]: ...

    @contextmanager
    @abstractmethod
    def properties(
        self,
        properties: Mapping[str, Any],
    ) -> Iterator[None]: ...

    @property
    @abstractmethod
    def correlation_id(self) -> str: ...

    def get(self, property_name: str) -> Any | None: ...


class BasicContextualCorrelator(ContextualCorrelator):
    def __init__(self) -> None:
        self._instance_id = generate_id()

        self._scopes = contextvars.ContextVar[str](
            f"correlator_{self._instance_id}_scopes",
            default="",
        )

        self._properties = contextvars.ContextVar[Mapping[str, Any]](
            f"correlator_{self._instance_id}_properties",
            default={},
        )

    @contextmanager
    @override
    def scope(
        self,
        scope_id: str,
        properties: Mapping[str, Any] = {},
    ) -> Iterator[None]:
        current_scopes = self._scopes.get()

        if current_scopes:
            new_scopes = current_scopes + f"::{scope_id}"
        else:
            new_scopes = scope_id

        current_properties = self._properties.get()
        new_properties = {**current_properties, **properties}

        scopes_reset_token = self._scopes.set(new_scopes)
        properties_reset_token = self._properties.set(new_properties)

        yield

        self._scopes.reset(scopes_reset_token)
        self._properties.reset(properties_reset_token)

    @contextmanager
    @override
    def properties(
        self,
        properties: Mapping[str, Any],
    ) -> Iterator[None]:
        current_properties = self._properties.get()
        new_properties = {**current_properties, **properties}

        properties_reset_token = self._properties.set(new_properties)

        yield

        self._properties.reset(properties_reset_token)

    @property
    @override
    def correlation_id(self) -> str:
        if scopes := self._scopes.get():
            return scopes

        return "<main>"

    @override
    def get(self, property_name: str) -> Any | None:
        properties = self._properties.get()
        return properties.get(property_name, None)

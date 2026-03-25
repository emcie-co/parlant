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

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, TypeVar

from lagom import Container

StoreT = TypeVar("StoreT")

APP_CALL_SITE: Mapping[str, Any] = {"call-site": "app"}
SDK_CALL_SITE: Mapping[str, Any] = {"call-site": "sdk"}
ENGINE_CALL_SITE: Mapping[str, Any] = {"call-site": "engine"}


class StoreProvider(ABC):
    @abstractmethod
    def get_store(
        self,
        store_type: type[StoreT],
        hints: Mapping[str, Any] = {},
    ) -> StoreT: ...


class BasicStoreProvider(StoreProvider):
    def __init__(
        self,
        container_provider: Callable[[], Container],
    ) -> None:
        self._container_provider = container_provider

    def get_store(
        self,
        store_type: type[StoreT],
        hints: Mapping[str, Any] = {},
    ) -> StoreT:
        return self._container_provider()[store_type]

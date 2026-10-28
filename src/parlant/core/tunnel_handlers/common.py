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

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any


UnaryHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
StreamHandler = Callable[[dict[str, Any]], AsyncIterator[dict[str, Any]]]


@dataclass(frozen=True)
class TunnelHandlerSet:
    unary: Mapping[str, UnaryHandler]
    streams: Mapping[str, StreamHandler]


def merge_handler_sets(*sets: TunnelHandlerSet) -> TunnelHandlerSet:
    unary: dict[str, UnaryHandler] = {}
    streams: dict[str, StreamHandler] = {}
    for handler_set in sets:
        duplicate_unary = unary.keys() & handler_set.unary.keys()
        duplicate_streams = streams.keys() & handler_set.streams.keys()
        if duplicate_unary:
            raise ValueError(f"Duplicate unary tunnel methods: {sorted(duplicate_unary)}")
        if duplicate_streams:
            raise ValueError(f"Duplicate stream tunnel methods: {sorted(duplicate_streams)}")
        unary.update(handler_set.unary)
        streams.update(handler_set.streams)
    return TunnelHandlerSet(unary=unary, streams=streams)

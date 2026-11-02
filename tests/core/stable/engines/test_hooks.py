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

from typing import Any, cast

from parlant.core.engines.compass.hooks import (
    CompassHookResult,
    CompassHooks,
    GaveUp,
    GiveUpKind,
    ReviewingMessage,
    ReviewingToolCalls,
)

_CTX = cast(Any, None)  # hooks pass the context through untouched; a stub is fine here.


def _returning(result: CompassHookResult, log: list[str], name: str) -> Any:
    async def hook(context: Any, payload: Any, exc: Any) -> CompassHookResult:
        log.append(name)
        return result

    return hook


async def test_that_call_on_preparing_result_returns_call_next_for_an_exhausted_chain() -> None:
    hooks = CompassHooks()
    log: list[str] = []
    hooks.on_preparing.append(_returning(CompassHookResult.CALL_NEXT, log, "a"))
    hooks.on_preparing.append(_returning(CompassHookResult.CALL_NEXT, log, "b"))

    assert await hooks.call_on_preparing_result(_CTX) is CompassHookResult.CALL_NEXT
    assert log == ["a", "b"]


async def test_that_call_on_preparing_result_returns_resolve_and_short_circuits() -> None:
    hooks = CompassHooks()
    log: list[str] = []
    hooks.on_preparing.append(_returning(CompassHookResult.RESOLVE, log, "a"))
    hooks.on_preparing.append(_returning(CompassHookResult.CALL_NEXT, log, "b"))

    assert await hooks.call_on_preparing_result(_CTX) is CompassHookResult.RESOLVE
    assert log == ["a"]  # the second hook never runs


async def test_that_call_on_preparing_result_returns_bail() -> None:
    hooks = CompassHooks()
    log: list[str] = []
    hooks.on_preparing.append(_returning(CompassHookResult.BAIL, log, "a"))

    assert await hooks.call_on_preparing_result(_CTX) is CompassHookResult.BAIL


async def test_that_the_review_result_wrappers_surface_resolve() -> None:
    hooks = CompassHooks()
    log: list[str] = []
    hooks.on_reviewing_tool_calls.append(_returning(CompassHookResult.RESOLVE, log, "tc"))
    hooks.on_reviewing_message.append(_returning(CompassHookResult.RESOLVE, log, "msg"))

    assert (
        await hooks.call_on_reviewing_tool_calls_result(_CTX, ReviewingToolCalls(tool_calls=[]))
        is CompassHookResult.RESOLVE
    )
    assert (
        await hooks.call_on_reviewing_message_result(_CTX, ReviewingMessage(message="hi"))
        is CompassHookResult.RESOLVE
    )


async def test_that_call_hooks_bool_stays_true_for_resolve_and_call_next_and_false_for_bail() -> (
    None
):
    hooks = CompassHooks()
    log: list[str] = []

    # An exhausted / all-CALL_NEXT chain -> True.
    assert await hooks.call_on_preparing(_CTX) is True

    hooks.on_preparing.append(_returning(CompassHookResult.RESOLVE, log, "resolve"))
    assert await hooks.call_on_preparing(_CTX) is True  # RESOLVE is still truthy for bool callers

    hooks.on_preparing = [_returning(CompassHookResult.BAIL, log, "bail")]
    assert await hooks.call_on_preparing(_CTX) is False


# --- on_giving_up can take over the final message -------------------------------

_GAVE_UP = GaveUp(reason="out of retries", steps=3, kind=GiveUpKind.RETRIES_EXHAUSTED)


async def test_that_call_on_giving_up_result_returns_call_next_for_an_exhausted_chain() -> None:
    # No hook took over, so the loop goes on to force its own final message.
    hooks = CompassHooks()
    log: list[str] = []
    hooks.on_giving_up.append(_returning(CompassHookResult.CALL_NEXT, log, "a"))
    hooks.on_giving_up.append(_returning(CompassHookResult.CALL_NEXT, log, "b"))

    assert await hooks.call_on_giving_up_result(_CTX, _GAVE_UP) is CompassHookResult.CALL_NEXT
    assert log == ["a", "b"]


async def test_that_call_on_giving_up_result_surfaces_resolve_and_short_circuits() -> None:
    hooks = CompassHooks()
    log: list[str] = []
    hooks.on_giving_up.append(_returning(CompassHookResult.RESOLVE, log, "a"))
    hooks.on_giving_up.append(_returning(CompassHookResult.CALL_NEXT, log, "b"))

    assert await hooks.call_on_giving_up_result(_CTX, _GAVE_UP) is CompassHookResult.RESOLVE
    assert log == ["a"]  # the second hook never runs


async def test_that_call_on_giving_up_result_surfaces_bail() -> None:
    hooks = CompassHooks()
    log: list[str] = []
    hooks.on_giving_up.append(_returning(CompassHookResult.BAIL, log, "a"))

    assert await hooks.call_on_giving_up_result(_CTX, _GAVE_UP) is CompassHookResult.BAIL

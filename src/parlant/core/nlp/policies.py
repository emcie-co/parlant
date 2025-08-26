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
import asyncio
from collections import defaultdict
from typing import Any, Coroutine, Callable, Optional, TypeAlias, TypeVar, Union

R = TypeVar("R")

FunctionCallState: TypeAlias = dict["Policy", dict[str, Any]]


class Policy(ABC):
    @abstractmethod
    async def apply(
        self,
        state: FunctionCallState,
        func: Callable[..., Coroutine[Any, Any, R]],
    ) -> R:
        pass


class RetryPolicy(Policy):
    def __init__(
        self,
        exceptions: Union[type[Exception], tuple[type[Exception], ...]],
        max_attempts: int = 3,
        wait_times: Optional[tuple[float, ...]] = None,
    ):
        if not isinstance(exceptions, tuple):
            exceptions = (exceptions,)
        self.exceptions = exceptions
        self.max_exceptions = max_attempts
        self.wait_times = wait_times if wait_times is not None else (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    async def apply(
        self,
        state: FunctionCallState,
        func: Callable[..., Coroutine[Any, Any, R]],
        *args: Any,
        **kwargs: Any,
    ) -> R:
        if "exceptions_raised" not in state[self]:
            state[self]["exceptions_raised"] = 0

        while True:
            try:
                return await func(state, *args, **kwargs)
            except self.exceptions as e:
                state[self]["exceptions_raised"] += 1

                if state[self]["exceptions_raised"] >= self.max_exceptions:
                    raise e

                wait_time = self.wait_times[
                    min(
                        state[self]["exceptions_raised"] - 1,
                        len(self.wait_times) - 1,
                    )
                ]

                await asyncio.sleep(wait_time)


def retry(
    exceptions: Union[type[Exception], tuple[type[Exception], ...]],
    max_exceptions: int = 3,
    wait_times: Optional[tuple[float, ...]] = None,
) -> RetryPolicy:
    return RetryPolicy(exceptions, max_exceptions, wait_times)


def policy(
    policies: Union[Policy, list[Policy]],
) -> Callable[[Callable[..., Coroutine[Any, Any, R]]], Callable[..., Coroutine[Any, Any, R]]]:
    def decorator(
        func: Callable[..., Coroutine[Any, Any, R]],
    ) -> Callable[..., Coroutine[Any, Any, R]]:
        applied_policies = policies if isinstance(policies, list) else [policies]

        # We need to maintain unique policy states across different
        # function calls, so we wrap the function with a state management layer.
        #
        # This is crucial for allowing multiple policies to be applied
        # and keep track of their own exceptions count (or other things)
        # during the same function call without interfering with each other.

        # The function itself will need to be called while
        # ignoring the managed call state parameter.
        func = _wrap_with_ignored_function_call_state(func)

        # Each policy accepts a state parameter,
        # which it uses to keep track of its own state.
        for policy in reversed(applied_policies):
            func = _wrap_with_policy(policy, func)

        # As soon as our decorated function is called,
        # we need to create a new state for this call,
        # which our policies can use.
        func = _wrap_with_function_call_state_initialization(func)

        # Finally, we return the wrapped function
        return func

    return decorator


def _wrap_with_ignored_function_call_state(
    func: Callable[..., Coroutine[Any, Any, R]],
) -> Callable[..., Coroutine[Any, Any, R]]:
    async def wrapped_func(state: FunctionCallState, *args: Any, **kwargs: Any) -> Any:
        _ = state
        return await func(*args, **kwargs)

    return wrapped_func


def _wrap_with_function_call_state_initialization(
    func: Callable[..., Coroutine[Any, Any, R]],
) -> Callable[..., Coroutine[Any, Any, R]]:
    async def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        state: FunctionCallState = defaultdict(dict)
        return await func(state, *args, **kwargs)

    return wrapped_func


def _wrap_with_policy(
    policy: Policy, func: Callable[..., Coroutine[Any, Any, R]]
) -> Callable[..., Coroutine[Any, Any, R]]:
    async def wrapped_func(state: FunctionCallState, *args: Any, **kwargs: Any) -> R:
        return await policy.apply(state, func, *args, **kwargs)

    return wrapped_func

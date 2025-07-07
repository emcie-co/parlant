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
from dataclasses import dataclass
import asyncio
from typing import Any, Coroutine, Callable, Optional, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


class Policy(ABC):
    @abstractmethod
    async def apply(
        self,
        func: Callable[P, Coroutine[Any, Any, R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        pass


@dataclass(frozen=True)
class RetryParameters:
    max_attempts: int = 3
    wait_times: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


class RetryPolicy(Policy):
    def __init__(
        self,
        sub_policies: dict[tuple[type[Exception], ...], RetryParameters] = {
            (Exception,): RetryParameters()
        },
        max_total_attempts: int = 3,
    ) -> None:
        self.sub_policies = sub_policies
        self.max_total_attempts = max_total_attempts

    async def apply(
        self, func: Callable[P, Coroutine[Any, Any, R]], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        # Counters initializations
        total_attempts = 0
        attempt_counters = {exc: 0 for exc in self.sub_policies}

        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Handle max total attempts
                total_attempts += 1
                if total_attempts >= self.max_total_attempts:
                    raise e

                # Handle sub-policies max attempts
                for exc_types, params in self.sub_policies.items():
                    if isinstance(e, exc_types):
                        attempt_counters[exc_types] += 1
                        if attempt_counters[exc_types] >= params.max_attempts:
                            raise e
                        wait_time = params.wait_times[
                            min(attempt_counters[exc_types] - 1, len(params.wait_times) - 1)
                        ]
                        await asyncio.sleep(wait_time)
                        break
                else:
                    # Unspecified exception - no retry
                    raise e


@overload
def retry(
    *,
    sub_policies: dict[tuple[type[Exception], ...], RetryParameters],
    max_attempts: int = 3,
) -> RetryPolicy: ...


@overload
def retry(
    *,
    exceptions: tuple[type[Exception], ...],
    max_attempts: int = 3,
    wait_times: Optional[tuple[float, ...]] = None,
) -> RetryPolicy: ...


def retry(
    *,
    sub_policies: Optional[dict[tuple[type[Exception], ...], RetryParameters]] = None,
    exceptions: Optional[tuple[type[Exception], ...]] = None,
    max_attempts: int = 3,
    wait_times: Optional[tuple[float, ...]] = None,
) -> RetryPolicy:
    # Validate that sub_policies and exceptions are not both specified - to keep the behavior well-defined
    if sub_policies is not None and exceptions is not None:
        raise ValueError(
            "You cannot specify both sub_policies and exceptions. Please use only one of them."
        )

    if sub_policies is not None:
        return RetryPolicy(sub_policies, max_attempts)

    if exceptions is not None:
        if wait_times is None:
            return RetryPolicy(
                {exceptions: RetryParameters(max_attempts)},
                max_attempts,
            )
        return RetryPolicy(
            {exceptions: RetryParameters(max_attempts, wait_times)},
            max_attempts,
        )

    # Default behavior if neither is provided
    return RetryPolicy(
        {(Exception,): RetryParameters(max_attempts)},
        max_attempts,
    )


def policy(
    policy: Policy,
) -> Callable[[Callable[..., Coroutine[Any, Any, R]]], Callable[..., Coroutine[Any, Any, R]]]:
    def decorator(
        func: Callable[..., Coroutine[Any, Any, R]],
    ) -> Callable[..., Coroutine[Any, Any, R]]:
        func = make_wrapped_func(policy, func)
        return func

    return decorator


def make_wrapped_func(
    policy: Policy, func: Callable[..., Coroutine[Any, Any, R]]
) -> Callable[..., Coroutine[Any, Any, R]]:
    async def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        return await policy.apply(func, *args, **kwargs)

    return wrapped_func

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
from typing import Any, Coroutine, Callable, Optional, ParamSpec, TypeVar, Union

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
        self.max_attempts = max_attempts
        self.wait_times = wait_times if wait_times is not None else (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

        self._attempts = 0

    async def apply(
        self, func: Callable[P, Coroutine[Any, Any, R]], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        while True:
            try:
                return await func(*args, **kwargs)
            except self.exceptions as e:
                self._attempts += 1
                if self._attempts >= self.max_attempts:
                    raise e
                wait_time = self.wait_times[min(self._attempts - 1, len(self.wait_times) - 1)]
                await asyncio.sleep(wait_time)


def retry(
    exceptions: Union[type[Exception], tuple[type[Exception], ...]],
    max_attempts: int = 3,
    wait_times: Optional[tuple[float, ...]] = None,
) -> RetryPolicy:
    return RetryPolicy(exceptions, max_attempts, wait_times)


def policy(
    policies: Union[Policy, list[Policy]],
) -> Callable[[Callable[..., Coroutine[Any, Any, R]]], Callable[..., Coroutine[Any, Any, R]]]:
    def decorator(
        func: Callable[..., Coroutine[Any, Any, R]],
    ) -> Callable[..., Coroutine[Any, Any, R]]:
        applied_policies = policies if isinstance(policies, list) else [policies]
        for policy in reversed(applied_policies):
            func = make_wrapped_func(policy, func)
        return func

    return decorator


def make_wrapped_func(
    policy: Policy, func: Callable[..., Coroutine[Any, Any, R]]
) -> Callable[..., Coroutine[Any, Any, R]]:
    async def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        return await policy.apply(func, *args, **kwargs)

    return wrapped_func

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
import inspect
import asyncio
from typing import Any, Coroutine, Callable, Optional, ParamSpec, TypeVar, cast, overload

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


def _create_param_to_args_mapping(
    func: Callable[..., Any], args: tuple[Any, ...]
) -> dict[str, int]:
    """Create a mapping from parameter names to their positions in args."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    mapping = {}
    for i, arg_value in enumerate(args):
        if i < len(param_names):
            mapping[param_names[i]] = i

    return mapping


class RetryPolicy(Policy):
    def __init__(
        self,
        sub_policies: dict[tuple[type[Exception], ...], RetryParameters] = {
            (Exception,): RetryParameters()
        },
        max_total_attempts: int = 3,
        injected_parameters: dict[str, list[Any]] = {},
        increased_parameters: dict[str, float] = {},
    ) -> None:
        self.sub_policies = sub_policies
        self.max_total_attempts = max_total_attempts
        self.injected_parameters = injected_parameters
        self.injected_deltas = increased_parameters

    async def apply(
        self, func: Callable[P, Coroutine[Any, Any, R]], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        # Counters initializations
        total_attempts = 0
        attempt_counters = {exc: 0 for exc in self.sub_policies}

        # Mapping of parameter names to their positions in args
        param_to_args_mapping = _create_param_to_args_mapping(func, args)
        args_list = list(args)

        while True:
            try:
                # Inject parameters if required - according to the total attempt number (only when retrying)
                if total_attempts > 0:
                    for param_name, param_values in self.injected_parameters.items():
                        if len(param_values) >= total_attempts:
                            if param_name in kwargs:
                                kwargs[param_name] = param_values[total_attempts - 1]
                            elif param_name in param_to_args_mapping:
                                args_list[param_to_args_mapping[param_name]] = param_values[
                                    total_attempts - 1
                                ]

                    for param_name in self.injected_deltas:
                        if param_name in kwargs and type(kwargs[param_name]) in (int, float):
                            kwargs[param_name] = (
                                cast(float, kwargs[param_name]) + self.injected_deltas[param_name]
                            )
                        elif param_name in param_to_args_mapping:
                            arg_index = param_to_args_mapping[param_name]
                            if type(args_list[arg_index]) in (int, float):
                                args_list[arg_index] = (
                                    cast(float, args_list[arg_index])
                                    + self.injected_deltas[param_name]
                                )

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
    injected_parameters: Optional[dict[str, list[Any]]] = None,
    increased_parameters: Optional[dict[str, float]] = None,
) -> RetryPolicy: ...


@overload
def retry(
    *,
    exceptions: tuple[type[Exception], ...],
    max_attempts: int = 3,
    wait_times: Optional[tuple[float, ...]] = None,
    injected_parameters: Optional[dict[str, list[Any]]] = None,
    increased_parameters: Optional[dict[str, float]] = None,
) -> RetryPolicy: ...


def retry(
    *,
    sub_policies: Optional[dict[tuple[type[Exception], ...], RetryParameters]] = None,
    exceptions: Optional[tuple[type[Exception], ...]] = None,
    max_attempts: int = 3,
    wait_times: Optional[tuple[float, ...]] = None,
    injected_parameters: Optional[dict[str, list[Any]]] = None,
    increased_parameters: Optional[dict[str, float]] = None,
) -> RetryPolicy:
    if injected_parameters is None:
        injected_parameters = {}

    if increased_parameters is None:
        increased_parameters = {}

    # Validate that there are no overlapping keys - to keep the behavior well-defined
    if set(injected_parameters.keys()).intersection(increased_parameters.keys()):
        raise ValueError(
            "You cannot specify a parameter in both injected_parameters and increased_parameters"
        )

    # Validate that sub_policies and exceptions are not both specified - to keep the behavior well-defined
    if sub_policies is not None and exceptions is not None:
        raise ValueError(
            "You cannot specify both sub_policies and exceptions. Please use only one of them."
        )

    if sub_policies is not None:
        return RetryPolicy(sub_policies, max_attempts, injected_parameters, increased_parameters)

    if exceptions is not None:
        if wait_times is None:
            return RetryPolicy(
                {exceptions: RetryParameters(max_attempts)},
                max_attempts,
                injected_parameters,
                increased_parameters,
            )
        return RetryPolicy(
            {exceptions: RetryParameters(max_attempts, wait_times)},
            max_attempts,
            injected_parameters,
            increased_parameters,
        )

    # Default behavior if neither is provided
    return RetryPolicy(
        {(Exception,): RetryParameters(max_attempts)},
        max_attempts,
        injected_parameters,
        increased_parameters,
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

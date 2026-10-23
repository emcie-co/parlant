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

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import pytest

from parlant.core.container import Container, Singleton


class _Port(ABC):
    @abstractmethod
    def act(self) -> str: ...


class _Impl(_Port):
    def act(self) -> str:
        return "acted"


class _OtherImpl(_Port):
    def act(self) -> str:
        return "acted otherwise"


_T = TypeVar("_T")


class _Schema:
    pass


class _Generator(ABC, Generic[_T]):
    @abstractmethod
    def generate(self) -> _T: ...


class _SchemaGenerator(_Generator[_Schema]):
    def generate(self) -> _Schema:
        return _Schema()


def test_that_an_instance_bound_to_an_abstract_port_resolves_to_that_instance() -> None:
    container = Container()
    impl = _Impl()

    container[_Port] = impl

    assert container[_Port] is impl


def test_that_a_singleton_bound_class_resolves_to_one_shared_instance() -> None:
    container = Container()

    container[_Port] = Singleton(_Impl)

    first = container[_Port]
    second = container[_Port]
    assert isinstance(first, _Impl)
    assert first is second


def test_that_a_factory_binding_runs_per_resolution() -> None:
    container = Container()

    container[_Port] = lambda c: _Impl()

    first = container[_Port]
    second = container[_Port]
    assert isinstance(first, _Impl)
    assert first is not second


def test_that_a_factory_receives_the_wrapped_container_not_lagoms() -> None:
    container = Container()
    container[_Port] = _Impl()
    seen: list[object] = []

    def factory(c: Container) -> _OtherImpl:
        seen.append(c)
        assert c[_Port] is not None
        return _OtherImpl()

    container[_OtherImpl] = factory

    assert isinstance(container[_OtherImpl], _OtherImpl)
    assert all(isinstance(c, Container) for c in seen)
    assert seen


def test_that_binding_an_instance_of_the_wrong_type_raises_at_binding_time() -> None:
    container = Container()

    with pytest.raises(TypeError):
        container[_Port] = "not a port"  # type: ignore[assignment]


def test_that_rebinding_in_a_clone_does_not_affect_the_original() -> None:
    container = Container()
    original = _Impl()
    container[_Port] = original

    clone = container.clone()
    clone[_Port] = _OtherImpl()

    assert container[_Port] is original
    assert isinstance(clone[_Port], _OtherImpl)


def test_that_defined_types_reports_bound_ports() -> None:
    container = Container()

    assert _Port not in container.defined_types

    container[_Port] = _Impl()

    assert _Port in container.defined_types


def test_that_lagom_stays_private_to_the_container_module() -> None:
    """The container module is the project's only lagom boundary; everything
    else must depend on parlant.core.container."""
    repo_src = Path(__file__).parent.parent.parent.parent / "src"

    result = subprocess.run(
        ["grep", "-rln", r"import lagom\|from lagom", "--include=*.py", str(repo_src)],
        capture_output=True,
        text=True,
    )

    offenders = [
        line for line in result.stdout.splitlines() if not line.endswith("core/container.py")
    ]

    assert not offenders, f"lagom imported outside parlant.core.container: {offenders}"


def test_that_a_generic_alias_key_binds_and_resolves() -> None:
    container = Container()
    generator = _SchemaGenerator()

    container[_Generator[_Schema]] = generator

    assert container[_Generator[_Schema]] is generator

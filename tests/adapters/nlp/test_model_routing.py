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


import pytest

from parlant.adapters.nlp.common import validate_model_arguments
from parlant.core.container import Container
from parlant.core.nlp.model_routing import (
    InvalidModelArgumentsError,
    ModelCapability,
    ModelImplementationRegistry,
    UnsupportedModelCapabilityError,
)


def test_that_model_argument_typos_receive_a_sanitized_suggestion() -> None:
    with pytest.raises(InvalidModelArgumentsError) as exc_info:
        validate_model_arguments(
            "openai",
            {"api_keey": "do-not-expose-this-value"},
            {"api_key", "base_url"},
        )

    error = exc_info.value
    assert error.suggestions == ("api_key",)
    assert "Did you mean 'api_key'?" in str(error)
    assert "do-not-expose-this-value" not in str(error)


def test_that_framework_owned_model_arguments_are_rejected() -> None:
    with pytest.raises(InvalidModelArgumentsError) as exc_info:
        validate_model_arguments(
            "gemini",
            {"logger": "do-not-expose-this-value"},
            {"api_key"},
        )

    assert "Framework-owned argument(s) cannot be overridden: logger" in str(exc_info.value)
    assert "do-not-expose-this-value" not in str(exc_info.value)


async def test_that_the_openai_wildcard_route_does_not_enable_moderation(
    container: Container,
) -> None:
    registry = container[ModelImplementationRegistry]

    # A capability with a wildcard default still resolves via the "*" route...
    assert registry.default_config("openai", ModelCapability.EMBEDDING, "*", {})

    # ...but moderation registers no default, so the "*" wildcard must NOT be able to
    # turn it on — moderation has to be routed explicitly by the user.
    with pytest.raises(UnsupportedModelCapabilityError):
        registry.default_config("openai", ModelCapability.MODERATION, "*", {})

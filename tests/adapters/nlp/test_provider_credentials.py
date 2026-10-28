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

import pytest

from parlant.adapters.nlp.common import (
    append_parlant_user_agent,
    parlant_client_headers,
    require_api_key,
    with_client_identity,
)
from parlant.core.version import VERSION
from parlant.core.nlp.model_routing import MissingProviderCredentialsError
from parlant.core.container import Container
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer


def test_that_a_missing_key_is_reported_with_the_variable_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(MissingProviderCredentialsError) as caught:
        require_api_key("OpenRouter", "OPENROUTER_API_KEY")

    # The provider names itself, and the variable to set travels with the
    # error, since that is what the server prints on the way out.
    assert caught.value.display_name == "OpenRouter"
    assert caught.value.env_var == "OPENROUTER_API_KEY"
    assert any("OPENROUTER_API_KEY" in s for s in caught.value.suggestions)


def test_that_an_explicit_key_is_used_even_when_the_environment_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert require_api_key("OpenRouter", "OPENROUTER_API_KEY", "explicit-key") == "explicit-key"


def test_that_the_environment_supplies_the_key_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-environment")

    assert require_api_key("OpenRouter", "OPENROUTER_API_KEY") == "from-environment"


def test_that_a_blank_key_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty variable is a common way to half-configure an environment, and
    # it reaches the provider as a 401 rather than as a missing key.
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    with pytest.raises(MissingProviderCredentialsError):
        require_api_key("OpenRouter", "OPENROUTER_API_KEY")


def test_that_a_provider_without_its_key_fails_when_it_is_built(
    container: Container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The point of the check: the failure names the missing variable at
    # construction, instead of surfacing as a 401 from the first request.
    from parlant.adapters.nlp.openrouter_service import OpenRouterEmbedder

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(MissingProviderCredentialsError) as caught:
        OpenRouterEmbedder(
            model_name="openai/text-embedding-3-small",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

    assert caught.value.env_var == "OPENROUTER_API_KEY"


def test_that_a_provider_does_not_fall_back_to_another_providers_key(
    container: Container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenRouter speaks the OpenAI protocol, so its client would otherwise
    # pick up OPENAI_API_KEY and send it to OpenRouter, which answers 401.
    from parlant.adapters.nlp.openrouter_service import OpenRouterEmbedder

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "an-openai-key")

    with pytest.raises(MissingProviderCredentialsError):
        OpenRouterEmbedder(
            model_name="openai/text-embedding-3-small",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )


def _header(client: Any, name: str) -> str | None:
    """One header off an SDK client, as a plain string (SDKs type these loosely)."""
    value = client.default_headers.get(name)
    return None if value is None or not isinstance(value, str) else value


# --- Client identity ----------------------------------------------------------
#
# Every provider request carries a header marking it as Parlant's, so a provider
# can be shown how much of their traffic arrives through Parlant. It names no
# version and no deployment, and a deployment can switch it off.


def test_that_the_client_identity_header_names_parlant_without_a_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PARLANT_CLIENT_IDENTITY", raising=False)

    headers = parlant_client_headers()

    assert headers == {"X-Parlant-Client": "parlant"}
    # Deliberately version-free: this identifies the framework, not the install.
    assert VERSION not in str(headers)


@pytest.mark.parametrize("setting", ["0", "false", "no", "off", "OFF", " 0 "])
def test_that_the_client_identity_header_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    monkeypatch.setenv("PARLANT_CLIENT_IDENTITY", setting)

    assert parlant_client_headers() == {}


@pytest.mark.parametrize("setting", ["1", "true", "yes", ""])
def test_that_any_other_client_identity_setting_keeps_the_header(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    monkeypatch.setenv("PARLANT_CLIENT_IDENTITY", setting)

    assert parlant_client_headers() == {"X-Parlant-Client": "parlant"}


# --- User-Agent, per provider -------------------------------------------------
#
# The User-Agent is the one identification a provider counts without being asked,
# so Parlant's token is APPENDED to each client library's own agent rather than
# replacing it. Each SDK needs a different mechanism, and each is pinned here —
# these break loudly if an SDK upgrade changes how it merges headers.


def test_that_the_parlant_token_is_appended_not_substituted() -> None:
    assert append_parlant_user_agent("AsyncOpenAI/Python 2.54.0") == (
        "AsyncOpenAI/Python 2.54.0 parlant"
    )


def test_that_appending_the_parlant_token_is_idempotent() -> None:
    # Clients get copied and re-wrapped; the token must not accumulate.
    once = append_parlant_user_agent("SDK/1.0")
    assert append_parlant_user_agent(once) == once


def test_that_the_user_agent_is_left_alone_when_identity_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_CLIENT_IDENTITY", "0")

    assert append_parlant_user_agent("SDK/1.0") == "SDK/1.0"


def test_that_openai_requests_identify_parlant() -> None:
    from parlant.adapters.nlp.openai_service import OpenAIEstimatingTokenizer  # noqa: F401
    from openai import AsyncClient

    client = with_client_identity(AsyncClient(api_key="x"))

    user_agent = _header(client, "User-Agent")
    assert user_agent is not None and user_agent.endswith(" parlant")
    assert "AsyncOpenAI" in user_agent


def test_that_anthropic_requests_identify_parlant() -> None:
    from anthropic import AsyncAnthropic

    client = with_client_identity(AsyncAnthropic(api_key="x"))

    user_agent = _header(client, "User-Agent")
    assert user_agent is not None and user_agent.endswith(" parlant")
    assert "AsyncAnthropic" in user_agent


def test_that_gemini_requests_identify_parlant() -> None:
    # google-genai concatenates a caller's lowercase `user-agent` onto its own,
    # but sends a capitalized `User-Agent` as a second, competing header.
    from google import genai
    from google.genai import types

    from parlant.adapters.nlp.gemini_service import _gemini_identity_headers

    client = genai.Client(
        api_key="x", http_options=types.HttpOptions(headers=_gemini_identity_headers())
    )
    # Reaching into the SDK's resolved options is the only way to see what it will
    # actually send; there is no public accessor.
    headers = dict(client._api_client._http_options.headers or {})

    assert headers["user-agent"].endswith(" parlant")
    assert "google-genai-sdk" in headers["user-agent"]
    assert headers["X-Parlant-Client"] == "parlant"
    # No competing capitalized variant.
    assert "User-Agent" not in headers


def test_that_openrouter_requests_identify_parlant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.delenv("OPENROUTER_SITE_NAME", raising=False)
    monkeypatch.delenv("OPENROUTER_HTTP_REFERER", raising=False)

    from parlant.adapters.nlp.openrouter_service import _openrouter_client

    client = _openrouter_client()

    user_agent = _header(client, "User-Agent")
    assert user_agent is not None and user_agent.endswith(" parlant")
    assert _header(client, "X-Parlant-Client") == "parlant"
    # OpenRouter's own app attribution, which feeds their public rankings.
    assert _header(client, "X-Title") == "Parlant"
    assert _header(client, "HTTP-Referer") == "https://parlant.io"


def test_that_a_configured_openrouter_app_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A deployment that identifies itself keeps both halves of its own attribution;
    # mixing our URL with their title would misattribute both.
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_SITE_NAME", "Acme Support")
    monkeypatch.delenv("OPENROUTER_HTTP_REFERER", raising=False)

    from parlant.adapters.nlp.openrouter_service import _openrouter_client

    client = _openrouter_client()

    assert _header(client, "X-Title") == "Acme Support"
    assert _header(client, "HTTP-Referer") is None
    # The framework marker is unconditional — it identifies the client, not the app.
    assert _header(client, "X-Parlant-Client") == "parlant"


def test_that_litellm_calls_identify_parlant() -> None:
    from parlant.adapters.nlp.litellm_service import _litellm_identity_headers

    headers = _litellm_identity_headers()

    assert headers["X-Parlant-Client"] == "parlant"
    # No client object to append onto, so the agent is set outright.
    assert headers["User-Agent"] == "parlant"


def test_that_litellm_sends_nothing_identifying_when_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_CLIENT_IDENTITY", "0")

    from parlant.adapters.nlp.litellm_service import _litellm_identity_headers

    assert _litellm_identity_headers() == {}


def test_that_switching_identity_off_also_drops_the_openrouter_app_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Opting out of being identified as Parlant has to cover OpenRouter's public
    # app rankings too, not only the headers a provider has to be asked about.
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("PARLANT_CLIENT_IDENTITY", "0")
    monkeypatch.delenv("OPENROUTER_SITE_NAME", raising=False)
    monkeypatch.delenv("OPENROUTER_HTTP_REFERER", raising=False)

    from parlant.adapters.nlp.openrouter_service import _openrouter_client

    client = _openrouter_client()

    assert _header(client, "X-Title") is None
    assert _header(client, "HTTP-Referer") is None
    assert _header(client, "X-Parlant-Client") is None
    user_agent = _header(client, "User-Agent")
    assert user_agent is not None and not user_agent.endswith(" parlant")


def test_that_a_client_without_a_readable_user_agent_is_left_untouched() -> None:
    # Only a real agent string can be appended to; anything else keeps its own
    # headers rather than having a mangled agent forced onto it.
    class _Opaque:
        user_agent = object()
        copied = False

        def with_options(self, **kwargs: Any) -> "_Opaque":
            self.copied = True
            return self

    client = _Opaque()

    assert with_client_identity(cast(Any, client)) is client
    assert client.copied is False

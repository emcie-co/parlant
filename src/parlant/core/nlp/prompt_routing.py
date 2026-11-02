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

"""Route-keyed prompt customization for engine components.

The sibling of :mod:`model_routing`, for prompts instead of models: devs
register **prompt providers** — zero-argument callables returning a
:class:`PromptBuilder` (a subclass itself qualifies) — keyed by the same
route-selector grammar the ``models`` configuration uses (terminal namespace
wildcards like ``"matching.*"``, the global ``"*"``, and effort qualifiers
like ``"matching.rank[effort=high]"``), scoped server-wide or per agent.

Components stop constructing ``PromptBuilder`` directly and ask
:meth:`PromptConfigurationService.create_builder` for their vessel; they then
pour their sections into whatever comes back. A provider therefore customizes
prompts by intercepting the builder's own API (``add_section`` /
``edit_section`` / ``build``) — there is no separate transformer vocabulary.

Two properties are deliberate:

- The service attaches the component's ``on_build`` callback to the returned
  builder itself (composing with any listener the provider installed), so
  prompt trace-logging cannot be silently lost by a provider that doesn't
  know about it.
- Providers must be deterministic per configuration: compass prompts are
  split into a byte-stable cached prefix and a live tail, and a provider
  that varies its output per call silently destroys prompt caching.

Note that a route can serve more than one prompt (``matching.rank`` covers
both the rule ranker and the rule pruner); providers distinguish prompts by
their section names, which each component keeps stable.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, TypeAlias

from parlant.core.agents import AgentId, Effort
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.nlp.model_routing import (
    InvalidModelRouteError,
    ModelRouteQualifiers,
    ModelRouter,
    ModelScope,
    validate_requested_route,
)


PromptProvider: TypeAlias = Callable[[], PromptBuilder]
PromptRoutes: TypeAlias = Mapping[str, PromptProvider]
PromptRouteUpdates: TypeAlias = Mapping[str, Optional[PromptProvider]]


def _route_candidates(route: str) -> list[str]:
    # Mirrors ModelRouter's namespace expansion: a concrete route is served by
    # itself, then by progressively shorter terminal wildcards.
    parts = route.split(".")
    return [
        route,
        *(f"{'.'.join(parts[:length])}.*" for length in range(len(parts) - 1, 0, -1)),
    ]


class PromptConfigurationService:
    """Holds scoped prompt routes and creates builders through them.

    Providers are process-local callables, so all configuration is
    runtime-only — nothing here persists (unlike model routes, which can).
    Resolution mirrors model routing: agent scope before server scope, and
    within a scope the most specific selector wins (concrete route over
    namespace wildcard over global, tighter effort qualifier over looser).
    """

    def __init__(self) -> None:
        self._routes: dict[ModelScope, dict[str, PromptProvider]] = {}

    async def configure(
        self,
        scope: ModelScope,
        updates: PromptRouteUpdates,
    ) -> None:
        scoped = dict(self._routes.get(scope, {}))
        normalized: dict[str, Optional[PromptProvider]] = {}

        for route, provider in updates.items():
            selector = ModelRouter.parse_route_selector(route)
            canonical = selector.canonical
            if canonical in normalized:
                raise InvalidModelRouteError(
                    f"Prompt route '{route}' duplicates canonical route '{canonical}'",
                    route=route,
                )
            if provider is not None and not callable(provider):
                raise InvalidModelRouteError(
                    f"Prompt route '{route}' requires a callable returning a PromptBuilder",
                    route=route,
                )
            normalized[canonical] = provider

        for canonical, provider in normalized.items():
            if provider is None:
                scoped.pop(canonical, None)
            else:
                scoped[canonical] = provider

        ModelRouter.validate_route_selectors(set(scoped))
        self._routes[scope] = scoped

    def create_builder(
        self,
        route: str,
        *,
        agent_id: AgentId | None = None,
        qualifiers: ModelRouteQualifiers = {},
        on_build: Callable[[str], None] | None = None,
    ) -> PromptBuilder:
        """Create the builder a component will populate for one prompt.

        Synchronous by design: resolution is pure in-memory computation, and
        several components build prompts inside synchronous helpers.
        """

        validate_requested_route(route)
        effort = self._parse_requested_effort(qualifiers, route)

        builder: PromptBuilder
        if provider := self._find_provider(agent_id, route, effort):
            builder = provider()
            if not isinstance(builder, PromptBuilder):
                provider_name = getattr(provider, "__name__", type(provider).__name__)
                raise InvalidModelRouteError(
                    f"Prompt provider '{provider_name}' for route '{route}' returned "
                    f"{type(builder).__name__} instead of a PromptBuilder",
                    route=route,
                )
        else:
            builder = PromptBuilder()

        if on_build is not None:
            builder.add_build_listener(on_build)

        return builder

    def _parse_requested_effort(
        self,
        qualifiers: ModelRouteQualifiers,
        route: str,
    ) -> Effort | None:
        raw_effort = qualifiers.get("effort")
        if raw_effort is None:
            return None

        try:
            return Effort(raw_effort)
        except ValueError:
            raise InvalidModelRouteError(
                f"Unsupported effort qualifier '{raw_effort}' requested for prompt route '{route}'",
                route=route,
            )

    def _find_provider(
        self,
        agent_id: AgentId | None,
        route: str,
        effort: Effort | None,
    ) -> PromptProvider | None:
        scopes = [ModelScope.agent(agent_id)] if agent_id else []
        scopes.append(ModelScope.server())
        candidates = [*_route_candidates(route), "*"]

        for scope in scopes:
            routes = self._routes.get(scope, {})
            matches: list[tuple[tuple[int, int], PromptProvider]] = []
            for raw_selector, provider in routes.items():
                selector = ModelRouter.parse_route_selector(raw_selector)
                if selector.route not in candidates:
                    continue
                if not selector.matches(effort):
                    continue
                specificity = (
                    candidates.index(selector.route),
                    selector.qualifier_specificity,
                )
                matches.append((specificity, provider))

            if matches:
                return min(matches, key=lambda match: match[0])[1]

        return None

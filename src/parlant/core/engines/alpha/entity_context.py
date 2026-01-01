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

from __future__ import annotations

import contextvars

from parlant.core.agents import Agent
from parlant.core.async_utils import Stopwatch
from parlant.core.context_variables import ContextVariableId, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.engines.alpha.engine_context import EngineContext, Interaction
from parlant.core.sessions import Session


class EntityContext:
    """Provides access to current agent, customer, and session entities within asyncio task contexts.

    This class uses Python's contextvars to make these entities available to any code
    running within the same asyncio task context, including engine hooks.
    """

    _var: contextvars.ContextVar[EngineContext | None] = contextvars.ContextVar(
        "parlant_current_engine_context", default=None
    )

    @classmethod
    def set(
        self,
        context: EngineContext,
    ) -> None:
        """Set the current entities in the asyncio task context.

        Args:
            agent: The current agent, if any
            customer: The current customer, if any
            session: The current session, if any
        """
        self._var.set(context)

    @classmethod
    def get_context_creation(self) -> Stopwatch | None:
        """Get the start of processing time from the engine context.

        Returns:
            The start of processing time, or None if no context is set
        """
        ctx = self._var.get()
        return ctx.creation if ctx else None

    @classmethod
    def get_interaction(self) -> Interaction | None:
        """Get the current engine context from the asyncio task context.

        Returns:
            The current engine context, or None if no context is set
        """
        ctx = self._var.get()
        return ctx.interaction if ctx else None

    @classmethod
    def get_variable_value(self, variable_id: ContextVariableId) -> ContextVariableValue | None:
        ctx = self._var.get()

        if ctx is None:
            return None

        result = next(
            (
                value
                for variable, value in ctx.state.context_variables
                if variable.id == variable_id
            ),
            None,
        )

        return result if result else None

    @classmethod
    def get_agent(self) -> Agent | None:
        """Get the current agent from the asyncio task context.

        Returns:
            The current agent, or None if no agent is set in context
        """
        ctx = self._var.get()
        return ctx.agent if ctx else None

    @classmethod
    def get_customer(self) -> Customer | None:
        """Get the current customer from the asyncio task context.

        Returns:
            The current customer, or None if no customer is set in context
        """
        ctx = self._var.get()
        return ctx.customer if ctx else None

    @classmethod
    def get_session(self) -> Session | None:
        """Get the current session from the asyncio task context.

        Returns:
            The current session, or None if no session is set in context
        """
        ctx = self._var.get()
        return ctx.session if ctx else None

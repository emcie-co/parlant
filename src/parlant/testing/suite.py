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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    overload,
)

from lagom import Container

from parlant.client import AsyncParlantClient
from parlant.core.loggers import LogLevel, Logger
from parlant.core.meter import LocalMeter, Meter
from parlant.core.nlp.service import NLPService
from parlant.core.tracer import LocalTracer, Tracer
from parlant.sdk import NLPServices
from parlant.testing.assertions import nlp_test as _nlp_test
from parlant.testing.session import Session

# Context variables for coroutine-safe test context
_current_test_name: ContextVar[Optional[str]] = ContextVar("current_test_name", default=None)
_current_listener: ContextVar[Optional[Any]] = ContextVar("current_listener", default=None)


def set_test_context(test_name: str, listener: Any) -> tuple[Any, Any]:
    """Set test context for coroutine-safe access. Returns tokens to reset."""
    name_token = _current_test_name.set(test_name)
    listener_token = _current_listener.set(listener)
    return name_token, listener_token


def reset_test_context(tokens: tuple[Any, Any]) -> None:
    """Reset test context using tokens from set_test_context."""
    name_token, listener_token = tokens
    _current_test_name.reset(name_token)
    _current_listener.reset(listener_token)


class _SilentLogger(Logger):
    """A logger that discards all messages."""

    def set_level(self, log_level: LogLevel) -> None:
        pass

    def trace(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def critical(self, message: str) -> None:
        pass

    @contextmanager
    def scope(self, scope_id: str) -> Iterator[None]:
        yield


F = TypeVar("F", bound=Callable[..., Awaitable[None]])


@dataclass
class Scenario:
    """Represents a registered test scenario."""

    name: str
    func: Callable[..., Awaitable[None]]
    repetitions: int = 1


@dataclass
class HookSet:
    """Collection of hooks for a suite."""

    before_all: List[Callable[[], Awaitable[None]]] = field(default_factory=list)
    after_all: List[Callable[[], Awaitable[None]]] = field(default_factory=list)
    before_each: List[Callable[[str], Awaitable[None]]] = field(default_factory=list)
    after_each: List[Callable[[str, bool, Optional[str]], Awaitable[None]]] = field(
        default_factory=list
    )


class Suite:
    """Test suite for Parlant agent testing.

    Manages server connection, NLP service, scenarios, and hooks.

    Example:
        suite = Suite(
            server_url="http://localhost:8000",
            agent_id="my_agent",
        )

        @suite.scenario
        async def test_greeting():
            async with suite.session() as session:
                response = await session.send("Hello!")
                await response.should("greet the customer")
    """

    def __init__(
        self,
        server_url: str,
        agent_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        response_timeout: int = 60,
        nlp_service: Callable[[Container], NLPService] = NLPServices.emcie,
    ) -> None:
        """Initialize the test suite.

        Args:
            server_url: URL of the Parlant server.
            agent_id: Default agent ID for sessions.
            customer_id: Default customer ID (None = guest).
            response_timeout: Default timeout for agent responses in seconds.
            nlp_service: Factory function that creates NLPService from Container.
                         Defaults to Emcie.
        """
        self._server_url = server_url
        self._nlp_service_factory = nlp_service
        self._agent_id = agent_id
        self._customer_id = customer_id
        self._response_timeout = response_timeout

        # Lazy-initialized
        self._client: Optional[AsyncParlantClient] = None
        self._container: Optional[Container] = None
        self._nlp_service: Optional[NLPService] = None

        # Registered scenarios and hooks
        self._scenarios: List[Scenario] = []
        self._hooks = HookSet()

        # Shared context for hooks and tests
        self.context: Dict[str, Any] = {}

        # Test execution context (set by runner)
        self._current_test_name: Optional[str] = None
        self._listener: Optional[Any] = None  # TestEventListener

        # Sessions queued for deletion
        self._sessions_to_delete: List[str] = []

    @property
    def response_timeout(self) -> int:
        """Default timeout for agent responses in seconds."""
        return self._response_timeout

    @property
    def client(self) -> AsyncParlantClient:
        """The AsyncParlantClient for this suite."""
        if self._client is None:
            self._client = AsyncParlantClient(base_url=self._server_url)
        return self._client

    async def _get_client(self) -> AsyncParlantClient:
        """Get or create the client (internal use)."""
        if self._client is None:
            self._client = AsyncParlantClient(base_url=self._server_url)
        return self._client

    async def _get_nlp_service(self) -> NLPService:
        """Get or create the NLP service (internal use)."""
        if self._nlp_service is None:
            if self._container is None:
                self._container = Container()
                # Provide minimal dependencies for NLP service
                logger = _SilentLogger()
                tracer = LocalTracer()
                meter = LocalMeter(logger)

                self._container[Logger] = logger
                self._container[Tracer] = tracer
                self._container[Meter] = meter
                self._container[LocalTracer] = tracer
                self._container[LocalMeter] = meter

            self._nlp_service = self._nlp_service_factory(self._container)

        return self._nlp_service

    def session(
        self,
        agent_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        transient: bool = True,
    ) -> Session:
        """Create a new session context manager.

        Args:
            agent_id: Agent ID for this session. Uses default if not provided.
            customer_id: Customer ID. Uses default if not provided (None = guest).
            transient: If True, delete session on exit.

        Returns:
            Session context manager.

        Raises:
            ValueError: If no agent_id provided and no default set.
        """
        effective_agent_id = agent_id or self._agent_id
        if not effective_agent_id:
            raise ValueError(
                "agent_id must be provided either in session() or as agent_id in Suite"
            )

        effective_customer_id = customer_id if customer_id is not None else self._customer_id

        return Session(
            suite=self,
            agent_id=effective_agent_id,
            customer_id=effective_customer_id,
            transient=transient,
            test_name=_current_test_name.get(),
            listener=_current_listener.get(),
        )

    async def nlp_test(self, context: str, condition: str) -> tuple[bool, str]:
        """Run an NLP test to evaluate if a condition holds.

        Args:
            context: The context to evaluate (e.g., agent response).
            condition: The condition to check.

        Returns:
            Tuple of (answer: bool, reasoning: str).
        """
        nlp_service = await self._get_nlp_service()
        return await _nlp_test(nlp_service, context, condition)

    @overload
    def scenario(self, func: F) -> F: ...

    @overload
    def scenario(self, *, repetitions: int = 1) -> Callable[[F], F]: ...

    def scenario(
        self,
        func: Optional[F] = None,
        *,
        repetitions: int = 1,
    ) -> Union[F, Callable[[F], F]]:
        """Decorator to register a test scenario.

        Can be used with or without arguments:

            @suite.scenario
            async def test_simple():
                pass

            @suite.scenario(repetitions=3)
            async def test_repeated():
                pass

        Args:
            func: The test function (when used without parentheses).
            repetitions: Number of times to repeat the scenario.

        Returns:
            Decorated function or decorator.
        """

        def decorator(f: F) -> F:
            self._scenarios.append(
                Scenario(
                    name=f.__name__,
                    func=f,
                    repetitions=repetitions,
                )
            )
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def before_all(self, func: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Decorator to register a before_all hook.

        Runs once before all tests in this suite.
        """
        self._hooks.before_all.append(func)
        return func

    def after_all(self, func: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Decorator to register an after_all hook.

        Runs once after all tests in this suite.
        """
        self._hooks.after_all.append(func)
        return func

    def before_each(
        self, func: Callable[[str], Awaitable[None]]
    ) -> Callable[[str], Awaitable[None]]:
        """Decorator to register a before_each hook.

        Runs before each test. Receives test_name as argument.
        """
        self._hooks.before_each.append(func)
        return func

    def after_each(
        self, func: Callable[[str, bool, Optional[str]], Awaitable[None]]
    ) -> Callable[[str, bool, Optional[str]], Awaitable[None]]:
        """Decorator to register an after_each hook.

        Runs after each test. Receives (test_name, passed, error) as arguments.
        """
        self._hooks.after_each.append(func)
        return func

    def get_scenarios(self) -> List[Scenario]:
        """Get all registered scenarios."""
        return list(self._scenarios)

    def get_hooks(self) -> HookSet:
        """Get the hook set for this suite."""
        return self._hooks

    def _queue_session_for_deletion(self, session_id: str) -> None:
        """Queue a session for deletion during cleanup."""
        self._sessions_to_delete.append(session_id)

    def has_sessions_to_cleanup(self) -> bool:
        """Check if there are sessions queued for cleanup."""
        return len(self._sessions_to_delete) > 0

    async def delete_queued_sessions(self) -> int:
        """Delete all queued sessions.

        Returns the number of sessions deleted.
        Call this after waiting for server processing to complete.
        """
        if not self._sessions_to_delete:
            return 0

        deleted = 0
        client = await self._get_client()
        for session_id in self._sessions_to_delete:
            try:
                await client.sessions.delete(session_id)
                deleted += 1
            except Exception:
                pass  # Best effort cleanup

        self._sessions_to_delete.clear()
        return deleted

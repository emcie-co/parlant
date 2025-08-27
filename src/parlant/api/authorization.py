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
from enum import Enum
from typing import Awaitable, Callable

from typing_extensions import override
from fastapi import Request

from limits.storage import MemoryStorage
from limits.strategies import (
    MovingWindowRateLimiter,
    FixedWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)
from limits import RateLimitItem, RateLimitItemPerMinute


class Operation(Enum):
    ACCESS_INTEGRATED_UI = "access_integrated_ui"
    ACCESS_API_DOCS = "access_api_docs"

    CREATE_AGENT = "create_agent"
    READ_AGENT = "read_agent"
    READ_AGENT_DESCRIPTION = "read_agent_description"
    LIST_AGENTS = "list_agents"
    UPDATE_AGENT = "update_agent"
    DELETE_AGENT = "delete_agent"

    CREATE_CANNED_RESPONSE = "create_canned_response"
    READ_CANNED_RESPONSE = "read_canned_response"
    LIST_CANNED_RESPONSES = "list_canned_responses"
    UPDATE_CANNED_RESPONSE = "update_canned_response"
    DELETE_CANNED_RESPONSE = "delete_canned_response"

    CREATE_CAPABILITY = "create_capability"
    READ_CAPABILITY = "read_capability"
    LIST_CAPABILITIES = "list_capabilities"
    UPDATE_CAPABILITY = "update_capability"
    DELETE_CAPABILITY = "delete_capability"

    CREATE_CONTEXT_VARIABLE = "create_context_variable"
    READ_CONTEXT_VARIABLE = "read_context_variable"
    LIST_CONTEXT_VARIABLES = "list_context_variables"
    UPDATE_CONTEXT_VARIABLE = "update_context_variable"
    DELETE_CONTEXT_VARIABLE = "delete_context_variable"
    DELETE_CONTEXT_VARIABLES = "delete_context_variables"
    READ_CONTEXT_VARIABLE_VALUE = "read_context_variable_value"
    UPDATE_CONTEXT_VARIABLE_VALUE = "update_context_variable_value"
    DELETE_CONTEXT_VARIABLE_VALUE = "delete_context_variable_value"

    CREATE_CUSTOMER = "create_customer"
    READ_CUSTOMER = "read_customer"
    LIST_CUSTOMERS = "list_customers"
    UPDATE_CUSTOMER = "update_customer"
    DELETE_CUSTOMER = "delete_customer"

    CREATE_EVALUATION = "create_evaluation"
    READ_EVALUATION = "read_evaluation"

    CREATE_TERM = "create_term"
    READ_TERM = "read_term"
    LIST_TERMS = "list_terms"
    UPDATE_TERM = "update_term"
    DELETE_TERM = "delete_term"

    CREATE_GUIDELINE = "create_guideline"
    READ_GUIDELINE = "read_guideline"
    LIST_GUIDELINES = "list_guidelines"
    UPDATE_GUIDELINE = "update_guideline"
    DELETE_GUIDELINE = "delete_guideline"

    CREATE_JOURNEY = "create_journey"
    READ_JOURNEY = "read_journey"
    LIST_JOURNEYS = "list_journeys"
    UPDATE_JOURNEY = "update_journey"
    DELETE_JOURNEY = "delete_journey"

    CREATE_RELATIONSHIP = "create_relationship"
    READ_RELATIONSHIP = "read_relationship"
    LIST_RELATIONSHIPS = "list_relationships"
    DELETE_RELATIONSHIP = "delete_relationship"

    UPDATE_SERVICE = "update_service"
    READ_SERVICE = "read_service"
    LIST_SERVICES = "list_services"
    DELETE_SERVICE = "delete_service"

    CREATE_GUEST_SESSION = "create_guest_session"
    CREATE_CUSTOMER_SESSION = "create_customer_session"
    READ_SESSION = "read_session"
    LIST_SESSIONS = "list_sessions"
    UPDATE_SESSION = "update_session"
    DELETE_SESSION = "delete_session"
    DELETE_SESSIONS = "delete_sessions"
    CREATE_CUSTOMER_EVENT = "create_customer_event"
    CREATE_AGENT_EVENT = "create_agent_event"
    CREATE_HUMAN_AGENT_EVENT = "create_human_agent_event"
    CREATE_HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT_EVENT = (
        "create_human_agent_on_behalf_of_ai_agent_event"
    )
    CREATE_STATUS_EVENT = "create_status_event"
    CREATE_CUSTOM_EVENT = "create_custom_event"
    READ_EVENT = "read_event"
    LIST_EVENTS = "list_events"
    DELETE_EVENTS = "delete_events"

    CREATE_TAG = "create_tag"
    READ_TAG = "read_tag"
    LIST_TAGS = "list_tags"
    UPDATE_TAG = "update_tag"
    DELETE_TAG = "delete_tag"


class AuthorizationException(Exception):
    def __init__(
        self,
        request: Request,
        operation: Operation | None,
        message_prefix: str = "Authorization failed",
    ) -> None:
        super().__init__(
            f"{message_prefix}: OPERATION={operation.value if operation else 'GENERIC'}, HEADERS={request.headers}"
        )

        self.request = request
        self.operation = operation


class RateLimitExceededException(AuthorizationException):
    def __init__(self, request: Request, operation: Operation | None) -> None:
        super().__init__(
            request=request,
            operation=operation,
            message_prefix="Rate limit exceeded",
        )


class AuthorizationPolicy(ABC):
    @abstractmethod
    async def check_permission(self, request: Request, operation: Operation) -> bool: ...

    @abstractmethod
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool: ...

    async def authorize(self, request: Request, operation: Operation) -> None:
        if not await self.check_permission(request, operation):
            raise AuthorizationException(request, operation)

        if not await self.check_rate_limit(request, operation):
            raise RateLimitExceededException(request, operation)

    @property
    @abstractmethod
    def name(self) -> str: ...


class DevelopmentAuthorizationPolicy(AuthorizationPolicy):
    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        # In development, we do not enforce rate limits
        return True

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        # In development, we allow all actions
        return True

    @property
    @override
    def name(self) -> str:
        return "development"


class RateLimiter(ABC):
    @abstractmethod
    async def check(
        self,
        request: Request,
        operation: Operation,
    ) -> bool: ...


class ProductionAuthorizationPolicy(AuthorizationPolicy):
    def __init__(self) -> None:
        # This can be modified externally to install specific limiters
        # for specific API operations.
        self.specific_limiters: dict[
            Operation,
            Callable[[Request, Operation], Awaitable[bool]],
        ] = {}

        # It is also possible to change or override the default limiter
        # for this instance from outside this class (or in subclasses).
        self.default_limiter: RateLimiter = BasicRateLimiter(
            rate_limit_item_per_operation={
                # Some reasonable defaults...
                Operation.READ_AGENT: RateLimitItemPerMinute(30),
                Operation.CREATE_GUEST_SESSION: RateLimitItemPerMinute(10),
                Operation.READ_SESSION: RateLimitItemPerMinute(30),
                Operation.LIST_EVENTS: RateLimitItemPerMinute(240),
                Operation.READ_EVENT: RateLimitItemPerMinute(30),
                Operation.CREATE_CUSTOMER_EVENT: RateLimitItemPerMinute(30),
                Operation.CREATE_STATUS_EVENT: RateLimitItemPerMinute(60),
            }
        )

    @property
    @override
    def name(self) -> str:
        return "production"

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        if operation in [
            Operation.READ_AGENT,
            Operation.CREATE_GUEST_SESSION,
            Operation.READ_SESSION,
            Operation.LIST_EVENTS,
            Operation.READ_EVENT,
            Operation.CREATE_CUSTOMER_EVENT,
        ]:
            return True
        else:
            return False

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        if specific_limiter := self.specific_limiters.get(operation):
            return await specific_limiter(request, operation)
        return await self.default_limiter.check(request, operation)


class BasicRateLimiter(RateLimiter):
    def __init__(
        self,
        rate_limit_item_per_operation: dict[Operation, RateLimitItem],
        storage: MemoryStorage | None = None,
        limiter_type: type[
            MovingWindowRateLimiter | FixedWindowRateLimiter | SlidingWindowCounterRateLimiter
        ] = MovingWindowRateLimiter,
    ) -> None:
        self.rate_limit_item_per_operation = rate_limit_item_per_operation
        self._limiter = limiter_type(storage or MemoryStorage())
        self._default_rate_limit_item = RateLimitItemPerMinute(100)

    async def check(
        self,
        request: Request,
        operation: Operation,
    ) -> bool:
        if item := self.rate_limit_item_per_operation.get(operation):
            return self._limiter.hit(item, self._build_key(request, operation))

        return self._limiter.hit(self._default_rate_limit_item, self._build_key(request, None))

    def _build_key(
        self,
        request: Request,
        operation: Operation | None,
    ) -> str:
        ip = self._get_client_ip(request)

        if not ip:
            raise AuthorizationException(
                request=request,
                operation=operation,
                message_prefix="Authorization failed: No client IP found",
            )

        return f"IP={ip}--OP={operation.value if operation else 'GENERIC'}"

    @staticmethod
    def _get_client_ip(request: Request) -> str | None:
        headers = request.headers

        if xff := headers.get("x-forwarded-for"):
            return xff.split(",")[0].strip()

        if xri := headers.get("x-real-ip"):
            return xri.strip()

        if cf := headers.get("cf-connecting-ip"):
            return cf.strip()

        return request.client.host if request.client else None

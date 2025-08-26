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
import math
import random
from typing import cast
from typing_extensions import override

from parlant.core.engines.alpha.loaded_context import LoadedContext
from parlant.core.sessions import EventKind, EventSource, MessageEventData
from parlant.core.tags import Tag


class PerceivedPerformancePolicy(ABC):
    """An interface for defining perceived performance policies for the engine."""

    @abstractmethod
    async def get_processing_indicator_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        """
        Returns the delay before the indicator (agent is thinking...) is sent.

        :param context: The loaded context containing session and interaction details.
        :return: The delay in seconds before sending the indicator.
        """
        ...

    @abstractmethod
    async def get_extended_processing_indicator_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        """
        Returns the delay before the indicator (agent is thinking "hard"...) is sent.

        :param context: The loaded context containing session and interaction details.
        :return: The delay in seconds before sending the indicator.
        """
        ...

    @abstractmethod
    async def get_follow_up_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        """
        Returns the delay before a follow-up message is sent.

        :param context: The loaded context containing session and interaction details.
        :return: The delay in seconds before sending the follow-up message.
        """
        ...

    @abstractmethod
    async def get_preamble_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        """
        Returns the delay before the preamble message is sent.

        :param context: The loaded context containing session and interaction details.
        :return: The delay in seconds before sending the preamble message.
        """
        ...

    @abstractmethod
    async def is_preamble_required(
        self,
        context: LoadedContext | None = None,
    ) -> bool:
        """
        Determines if a preamble message is required for the given context.

        :param context: The loaded context containing session and interaction details.
        :return: True if a preamble is required, False otherwise.
        """
        ...


class BasicPerceivedPerformancePolicy(PerceivedPerformancePolicy):
    """A default implementation of the perceived performance policy that uses reasonable, randomized delays."""

    @override
    async def get_processing_indicator_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return random.uniform(1.0, 2.0)

    @override
    async def get_extended_processing_indicator_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return random.uniform(3.5, 5.0)

    @override
    async def get_follow_up_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return random.uniform(0.5, 1.5)

    @override
    async def get_preamble_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return random.uniform(1.5, 2.0)

    @override
    async def is_preamble_required(
        self,
        context: LoadedContext | None = None,
    ) -> bool:
        if context is None:
            return False

        if self._last_agent_message_is_preamble(context):
            return False

        previous_wait_times = self._calculate_previous_customer_wait_times(context)

        if len(previous_wait_times) <= 2:
            # First few times the agent is responding, we should be
            # proactive about showing a life sign quickly in order
            # to engage the customer in the conversation.
            return True

        last_2_wait_times = previous_wait_times[-2:]

        if all(wait_time >= 5 for wait_time in last_2_wait_times):
            # If the last two customer wait times were more than 5 seconds,
            # we need the preamble to keep the customer engaged.
            return True

        return False

    def _last_agent_message_is_preamble(self, context: LoadedContext) -> bool:
        last_agent_message = next(
            (
                e
                for e in reversed(context.interaction.history)
                if e.kind == EventKind.MESSAGE and e.source == EventSource.AI_AGENT
            ),
            None,
        )

        if not last_agent_message:
            return False

        message_data = cast(MessageEventData, last_agent_message.data)

        return Tag.preamble() in message_data.get("tags", [])

    def _calculate_previous_customer_wait_times(self, context: LoadedContext) -> list[float]:
        result = []

        message_events = [e for e in context.interaction.history if e.kind == EventKind.MESSAGE]

        customer_events = [e for e in message_events if e.source == EventSource.CUSTOMER]
        agent_events = [e for e in message_events if e.source == EventSource.AI_AGENT]

        for customer_event in customer_events:
            next_agent_event = next(
                (e for e in agent_events if e.offset > customer_event.offset),
                None,
            )

            if not next_agent_event:
                break

            customer_wait_time = next_agent_event.creation_utc - customer_event.creation_utc

            result.append(customer_wait_time.total_seconds())

        return result


class NullPerceivedPerformancePolicy(PerceivedPerformancePolicy):
    @override
    async def get_processing_indicator_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return 0

    @override
    async def get_extended_processing_indicator_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return math.inf

    @override
    async def get_follow_up_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return 0

    @override
    async def get_preamble_delay(
        self,
        context: LoadedContext | None = None,
    ) -> float:
        return 0

    @override
    async def is_preamble_required(
        self,
        context: LoadedContext | None = None,
    ) -> bool:
        return False

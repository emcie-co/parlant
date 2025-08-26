from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence
from typing_extensions import override


class OptimizationPolicy(ABC):
    """An interface for defining optimization policies for the engine."""

    @abstractmethod
    def use_embedding_cache(
        self,
        hints: Mapping[str, Any] = {},
    ) -> bool:
        """Determines whether to use the embedding cache."""
        ...

    @abstractmethod
    def get_guideline_matching_batch_size(
        self,
        guideline_count: int,
        hints: Mapping[str, Any] = {},
    ) -> int:
        """Gets the batch size for guideline matching."""
        ...

    @abstractmethod
    def get_message_generation_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]: ...

    @abstractmethod
    def get_guideline_matching_batch_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        """Gets the retry temperatures (and number of generation attempts) for a guideline matching batch."""
        ...

    @abstractmethod
    def get_response_analysis_batch_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        """Gets the retry temperatures (and number of generation attempts) for a response analysis batch."""
        ...

    @abstractmethod
    def get_tool_calling_batch_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        """Gets the retry temperatures (and number of generation attempts) for a tool calling batch."""
        ...

    @abstractmethod
    def get_guideline_proposition_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        """Gets the retry temperatures (and number of generation attempts) for guideline propositions."""
        ...


class BasicOptimizationPolicy(OptimizationPolicy):
    """A basic optimization policy that defines default behaviors for the engine."""

    @override
    def use_embedding_cache(
        self,
        hints: Mapping[str, Any] = {},
    ) -> bool:
        return True

    @override
    def get_guideline_matching_batch_size(
        self,
        guideline_count: int,
        hints: Mapping[str, Any] = {},
    ) -> int:
        if guideline_count <= 10:
            return 1
        elif guideline_count <= 20:
            return 2
        elif guideline_count <= 30:
            return 3
        else:
            return 5

    @override
    def get_message_generation_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        if hints.get("type") == "canned_response-selection":
            return [
                0.1,
                0.05,
                0.2,
            ]

        return [
            0.1,
            0.3,
            0.5,
        ]

    @override
    def get_guideline_matching_batch_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        return [
            0.15,
            0.3,
            0.1,
        ]

    @override
    def get_response_analysis_batch_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        return [
            0.15,
            0.3,
            0.1,
        ]

    @override
    def get_tool_calling_batch_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        return [
            0.15,
            0.3,
            0.1,
        ]

    @override
    def get_guideline_proposition_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]:
        return [
            0.0,
            0.15,
            0.1,
        ]

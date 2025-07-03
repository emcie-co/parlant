from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence
from typing_extensions import override


class OptimizationPolicy(ABC):
    @abstractmethod
    def get_guideline_matching_batch_size(
        self,
        guideline_count: int,
        hints: Mapping[str, Any] = {},
    ) -> int: ...

    @abstractmethod
    def get_message_generation_retry_temperatures(
        self,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[float]: ...


class BasicOptimizationPolicy(OptimizationPolicy):
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
        if hints.get("type") == "utterance-selection":
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

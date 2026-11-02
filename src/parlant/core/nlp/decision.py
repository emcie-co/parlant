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

"""Typed decisions from a model, as opposed to generated content.

A :class:`SchematicGenerator` asks a model to fill a schema with text it writes. A
:class:`DecisionMaker` asks a different kind of question: given this state, which of
these declared outcomes holds? It answers with a value from a space the caller
defined up front, plus the probabilities behind it, and never with prose.

The practical difference is batching. A generator answers one prompt per call, so
judging twenty rules costs twenty requests. A decision maker takes one state and
many named questions, so the same twenty judgments cost one.

Use it for classification, routing, verification and scoring. Anything whose answer
is written rather than chosen belongs to a generator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypeAlias

from parlant.core.common import JSONSerializable
from parlant.core.nlp.generation_info import GenerationInfo


class DecisionError(Exception):
    """A decision could not be obtained, or could not be understood."""


# Instructions and criteria may be plain text, or a JSON object or array when the
# judgment needs structure. Text in either can reference a field of the state by
# wrapping its path in backticks — "Is `listing.description` accurate?".
Criterion: TypeAlias = JSONSerializable

# A question describes the judgment (`instructions`) and the answers it may
# return (`criteria`). Criteria are optional only where the answer space is
# already known — a proposition is true or false whether or not you describe
# what each means.
_MAX_CHOICE_OPTIONS = 255
_MIN_SCALE_LEVELS = 2
_MAX_SCALE_LEVELS = 10


@dataclass(frozen=True)
class PropositionCriteria:
    """What a yes and a no each mean, for a boundary that isn't self-evident."""

    true: Criterion
    false: Criterion


@dataclass(frozen=True)
class Proposition:
    """Does this hold? Answered with the probability that it does.

    The instructions carry most propositions on their own. Add ``criteria`` when
    the line between yes and no is subtle enough to be worth drawing, and measure
    whether it helped — it is not automatically an improvement."""

    instructions: Criterion
    criteria: PropositionCriteria | None = None


@dataclass(frozen=True)
class Choice:
    """Which one of these? Answered with one of ``criteria``'s keys.

    Each key is an option, and its value describes when that option applies (or
    None where the key speaks for itself)."""

    instructions: Criterion
    criteria: Mapping[str, Criterion | None]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("A choice needs at least one option")
        if len(self.criteria) > _MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"A choice takes at most {_MAX_CHOICE_OPTIONS} options, got {len(self.criteria)}"
            )


@dataclass(frozen=True)
class Scale:
    """Where on this ordered scale? Answered with a level of ``criteria``.

    ``criteria`` runs from lowest to highest, and each entry describes its level
    in the same terms a human grader would be given."""

    instructions: Criterion
    criteria: Sequence[Criterion]

    def __post_init__(self) -> None:
        if not _MIN_SCALE_LEVELS <= len(self.criteria) <= _MAX_SCALE_LEVELS:
            raise ValueError(
                f"A scale takes {_MIN_SCALE_LEVELS}-{_MAX_SCALE_LEVELS} levels, "
                f"got {len(self.criteria)}"
            )


Question: TypeAlias = Proposition | Choice | Scale


@dataclass(frozen=True)
class PropositionAnswer:
    """``probability`` carries the model's confidence on its own: 0.5 is maximal
    uncertainty, and the distance from it is how sure the model is."""

    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    option: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True)
class ScaleAnswer:
    """``score`` is weighted by the probability of each level, so it lands between
    levels when the model is torn between them. Round it only if the caller
    genuinely needs a discrete level."""

    score: float
    legend: Mapping[int, str]
    probabilities: Mapping[int, float]
    confidence: float


Answer: TypeAlias = PropositionAnswer | ChoiceAnswer | ScaleAnswer


@dataclass(frozen=True)
class DecisionResult:
    """Answers keyed by the names the caller gave its questions."""

    answers: Mapping[str, Answer]
    info: GenerationInfo
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)


class DecisionMaker(ABC):
    """An interface for obtaining typed decisions about a state."""

    @abstractmethod
    async def decide(
        self,
        state: JSONSerializable,
        questions: Mapping[str, Question],
        hints: Mapping[str, Any] = {},
    ) -> DecisionResult:
        """Answer every question about the given state, in one round trip.

        Questions are independent of one another, so asking many at once costs no
        more than asking one — that is the point of the port. Every question must
        come back answered; a missing answer is a :class:`DecisionError`."""
        ...

    @property
    @abstractmethod
    def id(self) -> str:
        """Return a unique identifier for the decision maker."""
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the maximum number of tokens the model accepts per request."""
        ...

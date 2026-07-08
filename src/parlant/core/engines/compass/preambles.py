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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

from parlant.core.engines.compass.response_state import EngineContext


DEFAULT_PREAMBLE_INTERVAL_SECONDS = 15.0


class PreambleDecision(Enum):
    """The loop's current timing decision for pre-tool user-visible text.

    This is not itself a configuration knob. The loop computes it from the
    per-turn preamble state and the configured interval, then passes it to
    ``PreambleConfiguration`` so the configuration can decide what prompt note
    to inject and whether generated pre-tool text may be emitted.
    """

    ALLOW_INITIAL = auto()
    """No preamble/progress message has been shown yet in this turn."""

    ALLOW_INTERVAL_UPDATE = auto()
    """A preamble was shown, but the configured interval has elapsed."""

    SUPPRESS = auto()
    """A preamble was shown recently enough that another one should be suppressed."""


class PreambleEmissionPolicy(Enum):
    """How the loop handles text emitted immediately before a tool call.

    This is runtime enforcement. It complements, but does not replace, prompt
    notes. For example, ``discourage()`` both tells the model not to emit
    preambles and suppresses any pre-tool text the model emits anyway.
    """

    PASSTHROUGH = auto()
    """Emit pre-tool text normally, without interval gating or one-sentence trimming."""

    INTERVAL = auto()
    """Emit only the first allowed preamble, then suppress until the interval elapses."""

    SUPPRESS = auto()
    """Never emit pre-tool text; tool calls should run silently."""


PreambleNoteFactory = Callable[[EngineContext, PreambleDecision, float], str | None]
"""Builds the per-step prompt note for a preamble decision.

Return ``None`` to avoid adding any preamble-related prompt note. The arguments
are the current engine context, the loop's timing decision, and the configured
interval in seconds.
"""


@dataclass(frozen=True)
class PreambleConfiguration:
    """Controls Compass preamble prompt guidance and runtime emission behavior.

    A preamble is text the model emits right before a tool call, for example
    "Let me check that." The configuration has two independent effects:

    - ``note_factory`` can add dynamic per-step instructions to the responder
      prompt about whether to send, update, or avoid a preamble.
    - ``emission_policy`` tells the loop what to do if the model actually emits
      text before a tool call.

    Use ``default()`` to leave preambles unmanaged, ``encourage()`` for the
    existing interval-based progress-update behavior, and ``discourage()`` to
    ask for silent tool use and enforce it at runtime. Construct the dataclass
    directly for custom prompt-note and emission-policy combinations.
    """

    interval_seconds: float = DEFAULT_PREAMBLE_INTERVAL_SECONDS
    """Minimum seconds between emitted preamble/progress updates in interval mode."""

    note_factory: PreambleNoteFactory | None = None
    """Optional factory for the preamble-related note added to turn instructions."""

    emission_policy: PreambleEmissionPolicy = PreambleEmissionPolicy.PASSTHROUGH
    """Runtime policy for model text that appears immediately before a tool call."""

    @classmethod
    def default(cls) -> Self:
        """Do not add prompt notes or special runtime handling for preambles."""

        return cls()

    @classmethod
    def encourage(
        cls,
        *,
        interval_seconds: float = DEFAULT_PREAMBLE_INTERVAL_SECONDS,
    ) -> Self:
        """Use the current Compass behavior for tool progress updates.

        The responder receives dynamic notes encouraging one short preamble
        before the first tool call, another update only after ``interval_seconds``
        has elapsed, and silence otherwise. The loop enforces the same interval
        and trims emitted preambles to one sentence.
        """

        return cls(
            interval_seconds=interval_seconds,
            note_factory=_encourage_note,
            emission_policy=PreambleEmissionPolicy.INTERVAL,
        )

    @classmethod
    def discourage(
        cls,
        *,
        interval_seconds: float = DEFAULT_PREAMBLE_INTERVAL_SECONDS,
    ) -> Self:
        """Discourage and suppress pre-tool progress messages.

        The responder receives a note asking it to run tools silently, and the
        loop drops any text emitted immediately before a tool call.
        """

        return cls(
            interval_seconds=interval_seconds,
            note_factory=_discourage_note,
            emission_policy=PreambleEmissionPolicy.SUPPRESS,
        )

    def note_for(self, context: EngineContext, decision: PreambleDecision) -> str | None:
        """Return the prompt note to add for this step, if any."""

        if self.note_factory is None:
            return None

        return self.note_factory(context, decision, self.interval_seconds)

    def allows_emission(self, decision: PreambleDecision) -> bool:
        """Return whether pre-tool text may be shown for the current decision."""

        match self.emission_policy:
            case PreambleEmissionPolicy.PASSTHROUGH:
                return True
            case PreambleEmissionPolicy.INTERVAL:
                return decision in (
                    PreambleDecision.ALLOW_INITIAL,
                    PreambleDecision.ALLOW_INTERVAL_UPDATE,
                )
            case PreambleEmissionPolicy.SUPPRESS:
                return False

    def trims_preamble_text(self) -> bool:
        """Return whether emitted pre-tool text should be trimmed to one sentence."""

        return self.emission_policy == PreambleEmissionPolicy.INTERVAL


def _encourage_note(
    _context: EngineContext,
    decision: PreambleDecision,
    interval_seconds: float,
) -> str:
    match decision:
        case PreambleDecision.ALLOW_INITIAL:
            return (
                "#### Tool communication before tool use\n\n"
                "If you need to use *a new tool* for this step, you should first send "
                "exactly one short, natural sentence to the user about what you are "
                "checking (Checking X; Let me do Y; Just a moment while I Z...). Do not "
                'add a second "I\'m checking" or "let me check" sentence in the same '
                "message. Keep it specific to the current action, and avoid repeating "
                "wording from earlier messages."
            )
        case PreambleDecision.ALLOW_INTERVAL_UPDATE:
            return (
                "#### Tool communication before tool use\n\n"
                f"More than {interval_seconds:g} seconds have passed since your last message to the user. "
                "If you need to use *a new tool* for this step, please update them on "
                "what you're currently doing before running the next tool. Send exactly "
                "one short sentence, and do not add another tool-progress sentence in "
                "the same message."
            )
        case PreambleDecision.SUPPRESS:
            return (
                "#### Tool communication before tool use\n\n"
                f"Less than {interval_seconds:g} seconds have passed since your last message to the user. "
                "If you need to use another tool now, do not send another progress "
                "update or preamble before the tool call. Run the next tool silently."
            )


def _discourage_note(
    _context: EngineContext,
    _decision: PreambleDecision,
    _interval_seconds: float,
) -> str:
    return (
        "#### Tool communication before tool use\n\n"
        "If you need to use a tool in this step, do not send a progress update or "
        "preamble first. Run the tool silently unless you are ready to provide a "
        "substantive response to the user."
    )

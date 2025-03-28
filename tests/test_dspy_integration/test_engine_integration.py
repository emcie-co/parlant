"""Integration tests for DSPyGuidelineProposer with AlphaEngine."""

from typing import TYPE_CHECKING, AsyncIterator, cast
import pytest
from pytest_mock import MockFixture
from lagom import Container

from parlant.core.agents import Agent
from parlant.core.customers import Customer
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.engines.alpha.guideline_proposer import GuidelinePropositionResult
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.alpha.fluid_message_generator import FluidMessageGenerator
from parlant.core.engines.alpha.message_assembler import MessageAssembler
from parlant.core.engines.alpha.hooks import LifecycleHooks
from parlant.core.events import Event
from parlant.core.guidelines import Guideline
from parlant.core.logging import Logger
from parlant.core.metrics import ModelMetrics
from parlant.core.terms import Term
from parlant.dspy_integration.engine.proposer import DSPyGuidelineProposer
from parlant.dspy_integration.guideline_classifier import GuidelineClassifier
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def mock_tool_event_generator(mocker: MockFixture) -> ToolEventGenerator:
    """Create a mock tool event generator."""
    return mocker.Mock(spec=ToolEventGenerator)


@pytest.fixture
def mock_message_generator(mocker: MockFixture) -> FluidMessageGenerator:
    """Create a mock message generator."""
    return mocker.Mock(spec=FluidMessageGenerator)


@pytest.fixture
def mock_message_assembler(mocker: MockFixture) -> MessageAssembler:
    """Create a mock message assembler."""
    return mocker.Mock(spec=MessageAssembler)


@pytest.fixture
def mock_lifecycle_hooks(mocker: MockFixture) -> LifecycleHooks:
    """Create mock lifecycle hooks."""
    return mocker.Mock(spec=LifecycleHooks)


@pytest.fixture
async def dspy_engine(
    container: Container,
    mock_logger: Logger,
    mock_classifier: GuidelineClassifier,
    mock_optimizer: BatchOptimizedGuidelineManager,
    mock_metrics: ModelMetrics,
    mock_tool_event_generator: ToolEventGenerator,
    mock_message_generator: FluidMessageGenerator,
    mock_message_assembler: MessageAssembler,
    mock_lifecycle_hooks: LifecycleHooks,
) -> AsyncIterator[AlphaEngine]:
    """Create an AlphaEngine instance with DSPyGuidelineProposer."""
    dspy_proposer = DSPyGuidelineProposer(
        logger=mock_logger,
        classifier=mock_classifier,
        optimizer=mock_optimizer,
        metrics=mock_metrics,
    )

    engine = AlphaEngine(
        logger=mock_logger,
        guideline_proposer=dspy_proposer,
        tool_event_generator=mock_tool_event_generator,
        message_generator=mock_message_generator,
        message_assembler=mock_message_assembler,
        lifecycle_hooks=mock_lifecycle_hooks,
    )

    yield engine


@pytest.mark.asyncio
async def test_engine_process_with_dspy_proposer(
    dspy_engine: AlphaEngine,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mock_classifier: GuidelineClassifier,
    mock_optimizer: BatchOptimizedGuidelineManager,
    mock_tool_event_generator: ToolEventGenerator,
    mock_message_generator: FluidMessageGenerator,
    mock_message_assembler: MessageAssembler,
) -> None:
    """Test full engine processing with DSPyGuidelineProposer."""
    # Configure mock classifier to activate some guidelines
    mock_classifier.classify_guidelines.return_value = mock_guidelines[:2]
    
    # Configure mock optimizer
    mock_optimizer.optimize_guidelines.return_value = mock_guidelines[:2]
    
    # Configure mock tool event generator
    mock_tool_event_generator.generate_tool_events.return_value = []
    
    # Configure mock message generator
    mock_message_generator.generate_message.return_value = "Test message"
    
    # Configure mock message assembler
    mock_message_assembler.assemble_message.return_value = "Assembled test message"

    # Process the request through the engine
    result = await dspy_engine.process(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    # Verify DSPy components were called correctly
    mock_classifier.classify_guidelines.assert_called_once()
    mock_optimizer.optimize_guidelines.assert_called_once()

    # Verify engine components were called in correct order
    mock_tool_event_generator.generate_tool_events.assert_called_once()
    mock_message_generator.generate_message.assert_called_once()
    mock_message_assembler.assemble_message.assert_called_once()

    # Verify result
    assert result is not None
    assert isinstance(result.guideline_propositions, GuidelinePropositionResult)
    assert len(result.guideline_propositions.activated_guidelines) == 2


@pytest.mark.asyncio
async def test_engine_process_error_handling(
    dspy_engine: AlphaEngine,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mock_classifier: GuidelineClassifier,
    mock_logger: Logger,
) -> None:
    """Test engine error handling when DSPyGuidelineProposer fails."""
    # Configure classifier to raise an exception
    mock_classifier.classify_guidelines.side_effect = Exception("Test error")

    result = await dspy_engine.process(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    # Verify error was logged
    mock_logger.error.assert_called()
    
    # Verify result contains empty guideline propositions
    assert result is not None
    assert isinstance(result.guideline_propositions, GuidelinePropositionResult)
    assert len(result.guideline_propositions.activated_guidelines) == 0


@pytest.mark.asyncio
async def test_engine_lifecycle_hooks(
    dspy_engine: AlphaEngine,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mock_lifecycle_hooks: LifecycleHooks,
) -> None:
    """Test lifecycle hooks are called correctly with DSPyGuidelineProposer."""
    await dspy_engine.process(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    # Verify lifecycle hooks were called in correct order
    assert mock_lifecycle_hooks.before_process.call_count == 1
    assert mock_lifecycle_hooks.after_process.call_count == 1
    assert mock_lifecycle_hooks.before_guideline_proposition.call_count == 1
    assert mock_lifecycle_hooks.after_guideline_proposition.call_count == 1


@pytest.mark.asyncio
async def test_engine_metrics_tracking(
    dspy_engine: AlphaEngine,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mock_metrics: ModelMetrics,
) -> None:
    """Test metrics are tracked correctly throughout engine processing."""
    await dspy_engine.process(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    # Verify metrics were tracked
    assert mock_metrics.track_model_call.call_count >= 1
    assert mock_metrics.track_tokens.call_count >= 1


@pytest.mark.asyncio
async def test_engine_utter_with_dspy_proposer(
    dspy_engine: AlphaEngine,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mock_message_generator: FluidMessageGenerator,
    mock_message_assembler: MessageAssembler,
) -> None:
    """Test engine's utter method with DSPyGuidelineProposer integration."""
    # Configure message generation
    mock_message_generator.generate_message.return_value = "Test message"
    mock_message_assembler.assemble_message.return_value = "Assembled test message"

    # Process first to set up state
    await dspy_engine.process(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    # Test utter
    utterance = await dspy_engine.utter()
    assert utterance == "Assembled test message"

    # Verify message components were called
    mock_message_generator.generate_message.assert_called()
    mock_message_assembler.assemble_message.assert_called() 
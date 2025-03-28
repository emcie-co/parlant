"""Unit tests for the DSPyGuidelineProposer class."""

from typing import TYPE_CHECKING, Sequence
import pytest
from pytest_mock import MockFixture

from parlant.core.agents import Agent
from parlant.core.customers import Customer
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_proposer import GuidelinePropositionResult
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
def mock_logger(mocker: MockFixture) -> Logger:
    """Create a mock logger for testing."""
    return mocker.Mock(spec=Logger)


@pytest.fixture
def mock_classifier(mocker: MockFixture) -> GuidelineClassifier:
    """Create a mock guideline classifier for testing."""
    return mocker.Mock(spec=GuidelineClassifier)


@pytest.fixture
def mock_optimizer(mocker: MockFixture) -> BatchOptimizedGuidelineManager:
    """Create a mock guideline optimizer for testing."""
    return mocker.Mock(spec=BatchOptimizedGuidelineManager)


@pytest.fixture
def mock_metrics(mocker: MockFixture) -> ModelMetrics:
    """Create a mock metrics tracker for testing."""
    return mocker.Mock(spec=ModelMetrics)


@pytest.fixture
def proposer(
    mock_logger: Logger,
    mock_classifier: GuidelineClassifier,
    mock_optimizer: BatchOptimizedGuidelineManager,
    mock_metrics: ModelMetrics,
) -> DSPyGuidelineProposer:
    """Create a DSPyGuidelineProposer instance for testing."""
    return DSPyGuidelineProposer(
        logger=mock_logger,
        classifier=mock_classifier,
        optimizer=mock_optimizer,
        metrics=mock_metrics,
    )


@pytest.fixture
def mock_agent(mocker: MockFixture) -> Agent:
    """Create a mock agent for testing."""
    return mocker.Mock(spec=Agent)


@pytest.fixture
def mock_customer(mocker: MockFixture) -> Customer:
    """Create a mock customer for testing."""
    return mocker.Mock(spec=Customer)


@pytest.fixture
def mock_guidelines(mocker: MockFixture) -> list[Guideline]:
    """Create mock guidelines for testing."""
    return [mocker.Mock(spec=Guideline) for _ in range(3)]


@pytest.fixture
def mock_context_variables(mocker: MockFixture) -> list[tuple[ContextVariable, ContextVariableValue]]:
    """Create mock context variables for testing."""
    return [
        (mocker.Mock(spec=ContextVariable), mocker.Mock(spec=ContextVariableValue))
        for _ in range(2)
    ]


@pytest.fixture
def mock_interaction_history(mocker: MockFixture) -> list[Event]:
    """Create mock interaction history for testing."""
    return [mocker.Mock(spec=Event) for _ in range(3)]


@pytest.fixture
def mock_terms(mocker: MockFixture) -> list[Term]:
    """Create mock terms for testing."""
    return [mocker.Mock(spec=Term) for _ in range(2)]


@pytest.fixture
def mock_staged_events(mocker: MockFixture) -> list[EmittedEvent]:
    """Create mock staged events for testing."""
    return [mocker.Mock(spec=EmittedEvent) for _ in range(2)]


@pytest.mark.asyncio
async def test_propose_guidelines_empty_guidelines(
    proposer: DSPyGuidelineProposer,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
) -> None:
    """Test propose_guidelines with empty guidelines list."""
    result = await proposer.propose_guidelines(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=[],
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    assert isinstance(result, GuidelinePropositionResult)
    assert len(result.activated_guidelines) == 0
    assert len(result.deactivated_guidelines) == 0


@pytest.mark.asyncio
async def test_propose_guidelines_with_guidelines(
    proposer: DSPyGuidelineProposer,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mock_classifier: GuidelineClassifier,
    mock_optimizer: BatchOptimizedGuidelineManager,
) -> None:
    """Test propose_guidelines with a list of guidelines."""
    # Configure mock classifier to activate some guidelines
    mock_classifier.classify_guidelines.return_value = mock_guidelines[:2]
    
    # Configure mock optimizer
    mock_optimizer.optimize_guidelines.return_value = mock_guidelines[:2]

    result = await proposer.propose_guidelines(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    assert isinstance(result, GuidelinePropositionResult)
    assert len(result.activated_guidelines) == 2
    assert len(result.deactivated_guidelines) == 1

    # Verify classifier was called correctly
    mock_classifier.classify_guidelines.assert_called_once_with(
        guidelines=mock_guidelines,
        interaction_history=mock_interaction_history,
        context_variables=mock_context_variables,
    )

    # Verify optimizer was called correctly
    mock_optimizer.optimize_guidelines.assert_called_once_with(
        guidelines=mock_guidelines[:2],
        interaction_history=mock_interaction_history,
        context_variables=mock_context_variables,
    )


@pytest.mark.asyncio
async def test_propose_guidelines_batch_processing(
    proposer: DSPyGuidelineProposer,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mocker: MockFixture,
) -> None:
    """Test batch processing of guidelines."""
    # Mock the batch processing methods
    mocker.patch.object(
        proposer,
        '_get_optimal_batch_size',
        return_value=2
    )
    
    spy_create_batches = mocker.spy(proposer, '_create_guideline_batches')
    spy_process_batch = mocker.spy(proposer, '_process_guideline_batch')

    await proposer.propose_guidelines(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    # Verify batch creation
    spy_create_batches.assert_called_once()
    assert spy_create_batches.spy_return is not None
    assert len(list(spy_create_batches.spy_return)) == 2  # 3 guidelines with batch size 2

    # Verify batch processing
    assert spy_process_batch.call_count == 2


@pytest.mark.asyncio
async def test_propose_guidelines_metrics_tracking(
    proposer: DSPyGuidelineProposer,
    mock_agent: Agent,
    mock_customer: Customer,
    mock_guidelines: list[Guideline],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_interaction_history: list[Event],
    mock_terms: list[Term],
    mock_staged_events: list[EmittedEvent],
    mock_metrics: ModelMetrics,
) -> None:
    """Test metrics tracking during guideline proposition."""
    await proposer.propose_guidelines(
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
async def test_propose_guidelines_error_handling(
    proposer: DSPyGuidelineProposer,
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
    """Test error handling during guideline proposition."""
    # Configure classifier to raise an exception
    mock_classifier.classify_guidelines.side_effect = Exception("Test error")

    result = await proposer.propose_guidelines(
        agent=mock_agent,
        customer=mock_customer,
        guidelines=mock_guidelines,
        context_variables=mock_context_variables,
        interaction_history=mock_interaction_history,
        terms=mock_terms,
        staged_events=mock_staged_events,
    )

    # Verify error was logged
    mock_logger.error.assert_called_once()
    
    # Verify empty result was returned
    assert isinstance(result, GuidelinePropositionResult)
    assert len(result.activated_guidelines) == 0
    assert len(result.deactivated_guidelines) == len(mock_guidelines)


@pytest.mark.asyncio
async def test_get_optimal_batch_size(proposer: DSPyGuidelineProposer) -> None:
    """Test optimal batch size calculation."""
    batch_size = proposer._get_optimal_batch_size(10)
    assert isinstance(batch_size, int)
    assert batch_size > 0
    assert batch_size <= 10


def test_create_guideline_batches(
    proposer: DSPyGuidelineProposer,
    mock_guidelines: list[Guideline],
) -> None:
    """Test creation of guideline batches."""
    batch_size = 2
    batches = list(proposer._create_guideline_batches(mock_guidelines, batch_size))
    
    assert len(batches) == 2  # 3 guidelines with batch size 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1
    assert all(isinstance(batch, list) for batch in batches)


@pytest.mark.asyncio
async def test_process_guideline_batch(
    proposer: DSPyGuidelineProposer,
    mock_guidelines: list[Guideline],
    mock_interaction_history: list[Event],
    mock_context_variables: list[tuple[ContextVariable, ContextVariableValue]],
    mock_classifier: GuidelineClassifier,
    mock_optimizer: BatchOptimizedGuidelineManager,
) -> None:
    """Test processing of a single guideline batch."""
    batch = mock_guidelines[:2]
    
    # Configure mocks
    mock_classifier.classify_guidelines.return_value = batch
    mock_optimizer.optimize_guidelines.return_value = batch

    result = await proposer._process_guideline_batch(
        batch=batch,
        interaction_history=mock_interaction_history,
        context_variables=mock_context_variables,
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(g in mock_guidelines for g in result) 
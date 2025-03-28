"""Tests for DSPy-enhanced message composers.

This module contains tests for the DSPy integration in the message generation pipeline.
"""

from typing import TYPE_CHECKING
import pytest
from unittest.mock import AsyncMock, Mock

from parlant.core.types import Context, Guidelines, Message, MessageGenerationResult
from parlant.dspy_integration.composers import DSPyEnhancedMessageComposer
from parlant.dspy_integration.types import EnhancedGuidelines, ClassificationResult

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from _pytest.logging import LogCaptureFixture


@pytest.fixture
def mock_dspy_service(mocker: "MockerFixture") -> AsyncMock:
    """Create a mock DSPy service."""
    service = AsyncMock()
    service.classify_guidelines = AsyncMock(return_value=EnhancedGuidelines(
        optimization_score=0.95,
        feature_importance={"relevance": 0.8},
        context_relevance=0.9,
        metadata={}
    ))
    service.optimize_response = AsyncMock(return_value=Message(content="Optimized response"))
    return service


@pytest.fixture
def mock_guideline_optimizer(mocker: "MockerFixture") -> AsyncMock:
    """Create a mock guideline optimizer."""
    optimizer = AsyncMock()
    optimizer.optimize = AsyncMock(return_value=EnhancedGuidelines(
        optimization_score=0.9,
        feature_importance={"clarity": 0.85},
        context_relevance=0.95,
        metadata={}
    ))
    return optimizer


@pytest.fixture
def mock_context_classifier(mocker: "MockerFixture") -> AsyncMock:
    """Create a mock context classifier."""
    classifier = AsyncMock()
    classifier.extract_features = AsyncMock(return_value={"key_features": ["test"]})
    classifier.classify = AsyncMock(return_value=ClassificationResult(
        success=True,
        features={"intent": "inform"},
        scores={"confidence": 0.9}
    ))
    return classifier


@pytest.fixture
def mock_logger(mocker: "MockerFixture") -> Mock:
    """Create a mock logger."""
    return Mock()


@pytest.fixture
def composer(
    mock_dspy_service: AsyncMock,
    mock_guideline_optimizer: AsyncMock,
    mock_context_classifier: AsyncMock,
    mock_logger: Mock
) -> DSPyEnhancedMessageComposer:
    """Create a DSPyEnhancedMessageComposer instance."""
    return DSPyEnhancedMessageComposer(
        dspy_service=mock_dspy_service,
        guideline_optimizer=mock_guideline_optimizer,
        context_classifier=mock_context_classifier,
        logger=mock_logger
    )


@pytest.mark.asyncio
async def test_enhance_guidelines_success(
    composer: DSPyEnhancedMessageComposer,
    mock_dspy_service: AsyncMock,
    mock_guideline_optimizer: AsyncMock
) -> None:
    """Test successful guideline enhancement."""
    context = Context(session_id="test-session")
    guidelines = Guidelines()
    
    result = await composer.enhance_guidelines(context, guidelines)
    
    assert isinstance(result, EnhancedGuidelines)
    assert result.optimization_score > 0
    mock_dspy_service.classify_guidelines.assert_called_once_with(
        context=context,
        guidelines=guidelines
    )
    mock_guideline_optimizer.optimize.assert_called_once()


@pytest.mark.asyncio
async def test_classify_context_success(
    composer: DSPyEnhancedMessageComposer,
    mock_context_classifier: AsyncMock
) -> None:
    """Test successful context classification."""
    context = Context(session_id="test-session")
    
    result = await composer.classify_context(context)
    
    assert isinstance(result, ClassificationResult)
    assert result.success
    mock_context_classifier.extract_features.assert_called_once_with(context)
    mock_context_classifier.classify.assert_called_once_with(context)


@pytest.mark.asyncio
async def test_optimize_response_success(
    composer: DSPyEnhancedMessageComposer,
    mock_dspy_service: AsyncMock
) -> None:
    """Test successful response optimization."""
    message = Message(content="Original message")
    context = Context(session_id="test-session")
    
    result = await composer.optimize_response(message, context)
    
    assert isinstance(result, Message)
    assert result.content == "Optimized response"
    mock_dspy_service.optimize_response.assert_called_once_with(
        message=message,
        context=context
    )


@pytest.mark.asyncio
async def test_generate_message_full_pipeline(
    composer: DSPyEnhancedMessageComposer,
    mocker: "MockerFixture"
) -> None:
    """Test the complete message generation pipeline."""
    context = Context(session_id="test-session")
    guidelines = Guidelines()
    
    # Mock the superclass generate_message
    mocker.patch(
        "parlant.core.engines.alpha.message_composer.MessageEventComposer.generate_message",
        new_callable=AsyncMock,
        return_value=MessageGenerationResult(
            message=Message(content="Base message"),
            metrics={}
        )
    )
    
    result = await composer.generate_message(context, guidelines)
    
    assert isinstance(result, MessageGenerationResult)
    assert result.metrics["dspy_enhancement_status"] == "success"
    assert result.metrics["guideline_optimization_applied"]
    assert result.message.content == "Optimized response"


@pytest.mark.asyncio
async def test_generate_message_fallback(
    composer: DSPyEnhancedMessageComposer,
    mock_dspy_service: AsyncMock,
    mocker: "MockerFixture"
) -> None:
    """Test fallback to base implementation when DSPy enhancement fails."""
    context = Context(session_id="test-session")
    guidelines = Guidelines()
    
    # Make DSPy service fail
    mock_dspy_service.classify_guidelines.side_effect = Exception("DSPy error")
    
    # Mock the superclass generate_message
    base_message = Message(content="Fallback message")
    mocker.patch(
        "parlant.core.engines.alpha.message_composer.MessageEventComposer.generate_message",
        new_callable=AsyncMock,
        return_value=MessageGenerationResult(
            message=base_message,
            metrics={}
        )
    )
    
    result = await composer.generate_message(context, guidelines)
    
    assert isinstance(result, MessageGenerationResult)
    assert result.metrics["dspy_enhancement_status"] == "failed"
    assert not result.metrics["guideline_optimization_applied"]
    assert result.message == base_message 
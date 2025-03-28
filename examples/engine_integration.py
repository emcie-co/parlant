"""Example script to demonstrate DSPyGuidelineProposer integration with AlphaEngine."""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Sequence, cast, Any, TypeVar, Generic, Mapping

from parlant.core.agents import Agent, AgentId
from parlant.core.customers import Customer, CustomerId
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.guidelines import Guideline
from parlant.core.logging import Logger
from parlant.core.metrics import ModelMetrics
from parlant.core.common import generate_id
from parlant.dspy_integration.engine.proposer import DSPyGuidelineProposer
from parlant.dspy_integration.guideline_classifier import GuidelineClassifier
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.nlp.generation import GenerationInfo, SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.tools import ToolId
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.alpha.fluid_message_generator import FluidMessageGenerator, FluidMessageSchema
from parlant.core.engines.alpha.message_assembler import MessageAssembler
from parlant.core.engines.alpha.hooks import LifecycleHooks
from parlant.core.sessions import Event, EventSource, EventKind, EventId, Term
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.engines.alpha.tool_caller import ToolCallInferenceSchema
from parlant.adapters.nlp.openai import OpenAISchematicGenerator


class SimpleLogger(Logger):
    """A simple logger implementation."""
    
    def __init__(self) -> None:
        """Initialize the logger."""
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        
        # Add console handler if no handlers exist
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)
    
    def operation(self, message: str) -> Any:
        """Log an operation message."""
        return self._logger.info(message)
    
    def set_level(self, level: str) -> None:
        """Set the logging level."""
        self._logger.setLevel(level)


class SimpleCorrelator(ContextualCorrelator):
    """A simple correlator implementation."""
    
    def __init__(self) -> None:
        self._correlation_id = "test-correlation-id"

    @property
    def correlation_id(self) -> str:
        """Get the correlation ID."""
        return self._correlation_id


class SimpleServiceRegistry(ServiceRegistry):
    """A simple service registry implementation."""
    
    async def get_service(self, service_id: str) -> Any:
        """Get a service by ID."""
        return None

    async def list_tool_services(self) -> List[str]:
        """List all tool services."""
        return []

    async def list_nlp_services(self) -> List[str]:
        """List all NLP services."""
        return []

    async def list_moderation_services(self) -> List[str]:
        """List all moderation services."""
        return []

    async def read_tool_service(self, service_id: str) -> Any:
        """Read a tool service."""
        return None

    async def read_nlp_service(self, service_id: str) -> Any:
        """Read an NLP service."""
        return None

    async def read_moderation_service(self, service_id: str) -> Any:
        """Read a moderation service."""
        return None

    async def update_tool_service(self, service_id: str, service: Any) -> None:
        """Update a tool service."""
        pass

    async def delete_service(self, service_id: str) -> None:
        """Delete a service."""
        pass


class SimpleSchematicGenerator(SchematicGenerator[FluidMessageSchema]):
    """A simple schematic generator implementation."""
    
    def __init__(self, api_key: str, model_name: str) -> None:
        self._api_key = api_key
        self._model_name = model_name

    @property
    def id(self) -> str:
        """Get the generator ID."""
        return "simple-generator"

    @property
    def max_tokens(self) -> int:
        """Get the maximum number of tokens."""
        return 4096

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return None
    
    async def generate(self, prompt: str, temperature: float = 0.0) -> tuple[GenerationInfo, FluidMessageSchema]:
        """Generate a schema."""
        # Mock implementation - in reality this would use the OpenAI API
        return GenerationInfo(
            model=self._model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
        ), FluidMessageSchema(
            last_message_of_customer="Hello",
            produced_reply=True,
            produced_reply_rationale="Customer greeted",
            guidelines=[],
            revisions=[],
        )


class SimpleToolEventGenerator(ToolEventGenerator):
    """A simple tool event generator for demonstration."""
    
    async def generate_tool_events(
        self,
        agent: Agent,
        customer: Customer,
        guidelines: Sequence[Guideline],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
    ) -> List[EmittedEvent]:
        """Generate tool events based on guidelines."""
        # For demonstration, we'll just return an empty list
        return []


class SimpleEventEmitter(EventEmitter):
    async def emit_message_event(self, correlation_id: str, data: str) -> EmittedEvent:
        return EmittedEvent(
            event=Event(
                id=EventId("test-event-id"),
                source=EventSource.AGENT,
                kind=EventKind.MESSAGE,
                offset=0,
                data=data,
            ),
            correlation_id=correlation_id,
        )

    async def emit_status_event(self, correlation_id: str, data: Mapping[str, Any]) -> EmittedEvent:
        return EmittedEvent(
            event=Event(
                id=EventId("test-event-id"),
                source=EventSource.AGENT,
                kind=EventKind.STATUS,
                offset=0,
                data=data,
            ),
            correlation_id=correlation_id,
        )


class SimpleMessageGenerator(FluidMessageGenerator):
    """A simple message generator for demonstration."""
    
    async def generate_message(
        self,
        agent: Agent,
        customer: Customer,
        guidelines: Sequence[Guideline],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
    ) -> str:
        """Generate a message based on guidelines."""
        # For demonstration, we'll just concatenate guideline actions
        messages = [g.content.action for g in guidelines]
        return " ".join(messages)


class SimpleMessageAssembler(MessageAssembler):
    """A simple message assembler for demonstration."""
    
    async def assemble_message(self, message: str) -> str:
        """Assemble the final message."""
        # For demonstration, we'll just return the message as is
        return message


class SimpleLifecycleHooks(LifecycleHooks):
    """Simple lifecycle hooks for demonstration."""
    
    async def before_process(self) -> None:
        """Called before processing starts."""
        pass

    async def after_process(self) -> None:
        """Called after processing ends."""
        pass

    async def before_guideline_proposition(self) -> None:
        """Called before guideline proposition."""
        pass

    async def after_guideline_proposition(self) -> None:
        """Called after guideline proposition."""
        pass


async def main() -> None:
    """Run the engine integration example."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Create components
    logger = SimpleLogger()
    metrics = ModelMetrics()

    # Create DSPy components
    classifier = GuidelineClassifier(api_key=api_key, metrics=metrics)
    optimizer = BatchOptimizedGuidelineManager(api_key=api_key, metrics=metrics)

    # Create DSPy guideline proposer
    dspy_proposer = DSPyGuidelineProposer(
        logger=logger,
        api_key=api_key,
        model_name="openai/gpt-3.5-turbo",
        metrics=metrics,
        use_optimizer=True
    )

    # Create engine components
    correlator = SimpleCorrelator()
    service_registry = SimpleServiceRegistry()
    schematic_generator = SimpleSchematicGenerator(api_key=api_key, model_name="gpt-4")
    
    tool_event_generator = SimpleToolEventGenerator(
        logger=logger,
        correlator=correlator,
        service_registry=service_registry,
        schematic_generator=schematic_generator,
    )
    message_generator = SimpleMessageGenerator()
    message_assembler = SimpleMessageAssembler()
    lifecycle_hooks = SimpleLifecycleHooks()

    # Create the engine
    engine = AlphaEngine(
        logger=logger,
        guideline_proposer=dspy_proposer,
        tool_event_generator=tool_event_generator,
        message_generator=message_generator,
        message_assembler=message_assembler,
        lifecycle_hooks=lifecycle_hooks,
    )

    # Create example data
    agent = Agent(id=AgentId(generate_id()), name="TestAgent")
    customer = Customer(id=CustomerId(generate_id()), name="TestCustomer")
    
    guidelines = [
        Guideline(
            id=GuidelineId(generate_id()),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition="when the customer asks about pricing",
                action="We offer three pricing tiers: Basic ($10/mo), Pro ($25/mo), and Enterprise (custom pricing)."
            )
        ),
        Guideline(
            id=GuidelineId(generate_id()),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition="when the customer reports a bug",
                action="I understand you're experiencing an issue. Could you please provide more details?"
            )
        ),
        Guideline(
            id=GuidelineId(generate_id()),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition="when the customer wants to upgrade",
                action="I'll help you upgrade your plan. Let me walk you through the process."
            )
        )
    ]

    # Create example interaction history
    interaction_history = [
        Event(
            id=EventId(generate_id()),
            source="customer",
            kind="message",
            creation_utc=datetime.now(timezone.utc),
            offset=0,
            correlation_id=generate_id(),
            data={"message": "Hi, I'm interested in your pricing options.", "participant": {"display_name": "TestCustomer"}},
            deleted=False
        )
    ]

    # Process the request
    print("\nProcessing request through engine...")
    result = await engine.process(
        agent=agent,
        customer=customer,
        guidelines=guidelines,
        context_variables=[],  # Empty for demonstration
        interaction_history=interaction_history,
        terms=[],  # Empty for demonstration
        staged_events=[],  # Empty for demonstration
    )

    # Print results
    print("\nEngine Processing Results:")
    print("------------------------")
    print("\nActivated Guidelines:")
    for guideline in result.guideline_propositions.activated_guidelines:
        print(f"\nCondition: {guideline.content.condition}")
        print(f"Action: {guideline.content.action}")

    print("\nDeactivated Guidelines:")
    for guideline in result.guideline_propositions.deactivated_guidelines:
        print(f"\nCondition: {guideline.content.condition}")
        print(f"Action: {guideline.content.action}")

    # Generate utterance
    print("\nGenerating utterance...")
    utterance = await engine.utter()
    print("\nFinal Utterance:")
    print("--------------")
    print(utterance)


if __name__ == "__main__":
    asyncio.run(main()) 
"""Example script to demonstrate DSPyGuidelineProposer integration with AlphaEngine."""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional, Sequence

from parlant.core.agents import Agent, AgentId, AgentStore, AgentDocumentStore
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.customers import Customer, CustomerId
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.alpha.fluid_message_generator import FluidMessageGenerator, FluidMessageSchema
from parlant.core.engines.alpha.message_assembler import MessageAssembler
from parlant.core.engines.alpha.hooks import LifecycleHooks
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.message_event_composer import MessageEventComposition
from parlant.core.engines.alpha.tool_caller import ToolInsights
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline
from parlant.core.logging import Logger
from parlant.core.nlp.generation import GenerationInfo, SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.tools import ToolId
from parlant.dspy_integration.engine.proposer import DSPyGuidelineProposer
from parlant.core.persistence.document_database import DocumentDatabase, DocumentCollection
from parlant.adapters.db.transient import TransientDocumentDatabase


class SimpleLogger(Logger):
    """Simple logger implementation for example purposes."""
    
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        logging.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        logging.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        logging.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        logging.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message."""
        logging.critical(msg, *args, **kwargs)
    
    def operation(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log operation message."""
        logging.info(f"OPERATION: {msg}", *args, **kwargs)


class SimpleCorrelator(ContextualCorrelator):
    """Simple correlator implementation for example purposes."""
    
    def get_correlation_id(self) -> str:
        """Get correlation ID."""
        return "example-correlation-id"


class SimpleServiceRegistry:
    """Simple service registry implementation for example purposes."""
    
    def get_service(self, service_name: str) -> Any:
        """Get service by name."""
        return None


class SimpleToolEventGenerator(ToolEventGenerator):
    """Simple tool event generator implementation for example purposes."""
    
    async def generate_tool_events(
        self,
        guidelines: Sequence[Guideline],
        context_variables: Mapping[str, ContextVariableValue],
        tool_insights: ToolInsights,
    ) -> List[Event]:
        """Generate tool events based on guidelines."""
        return []


class SimpleSchematicGenerator(SchematicGenerator):
    """Simple schematic generator implementation for example purposes."""
    
    async def generate(
        self,
        schema: FluidMessageSchema,
        context_variables: Mapping[str, ContextVariableValue],
        generation_info: GenerationInfo,
    ) -> str:
        """Generate message from schema."""
        return "Example generated message"


class SimpleMessageGenerator(FluidMessageGenerator):
    """Simple message generator implementation for example purposes."""
    
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        schematic_generator: SchematicGenerator,
    ):
        """Initialize message generator."""
        self.logger = logger
        self.correlator = correlator
        self.schematic_generator = schematic_generator

    async def generate_message(
        self,
        guidelines: Sequence[Guideline],
        context_variables: Mapping[str, ContextVariableValue],
        tool_insights: ToolInsights,
    ) -> MessageEventComposition:
        """Generate message based on guidelines."""
        message = await self.schematic_generator.generate(
            FluidMessageSchema(actions=[g.action for g in guidelines]),
            context_variables,
            GenerationInfo(),
        )
        return MessageEventComposition(
            message=message,
            events=[],
        )


class SimpleMessageAssembler(MessageAssembler):
    """Simple message assembler implementation for example purposes."""
    
    async def assemble_message(
        self,
        message: str,
        context_variables: Mapping[str, ContextVariableValue],
    ) -> str:
        """Assemble final message."""
        return message


class SimpleLifecycleHooks(LifecycleHooks):
    """Simple lifecycle hooks implementation for example purposes."""
    
    async def before_processing(
        self,
        context_variables: Mapping[str, ContextVariableValue],
    ) -> None:
        """Called before processing request."""
        pass

    async def after_processing(
        self,
        context_variables: Mapping[str, ContextVariableValue],
    ) -> None:
        """Called after processing request."""
        pass

    async def before_guideline_proposition(
        self,
        context_variables: Mapping[str, ContextVariableValue],
    ) -> None:
        """Called before proposing guidelines."""
        pass

    async def after_guideline_proposition(
        self,
        context_variables: Mapping[str, ContextVariableValue],
        proposition: GuidelineProposition,
    ) -> None:
        """Called after proposing guidelines."""
        pass


class SimpleDocumentDatabase(DocumentDatabase):
    """Simple in-memory document database for example purposes."""
    
    def __init__(self):
        """Initialize database."""
        self._collections: dict[str, DocumentCollection] = {}

    async def get_or_create_collection(
        self,
        name: str,
        schema: Any,
    ) -> DocumentCollection:
        """Get or create a collection."""
        if name not in self._collections:
            self._collections[name] = SimpleDocumentCollection()
        return self._collections[name]


class SimpleDocumentCollection(DocumentCollection):
    """Simple in-memory document collection for example purposes."""
    
    def __init__(self):
        """Initialize collection."""
        self._documents: list[dict[str, Any]] = []

    async def insert_one(self, document: dict[str, Any]) -> None:
        """Insert a document."""
        self._documents.append(document)

    async def find(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Find documents matching filters."""
        return self._documents

    async def find_one(self, filters: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Find one document matching filters."""
        return self._documents[0] if self._documents else None


async def main() -> None:
    """Run example integration."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = SimpleLogger()

    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create agent store
    database = TransientDocumentDatabase()
    agent_store = AgentDocumentStore(database)
    async with agent_store:
        # Create example agent
        agent = await agent_store.create_agent(
            name="Example Agent",
            description="Technical Support Assistant",
            max_engine_iterations=2,
        )

        # Create example customer
        customer = Customer(
            id=CustomerId("example-customer"),
            name="Example Customer",
            creation_utc=datetime.now(timezone.utc),
            extra={},
            tags=[],
        )

        # Create example guidelines
        guidelines = [
            Guideline(
                id="pricing-inquiry",
                condition="Customer asks about pricing",
                action="Explain our pricing plans",
                priority=1,
            ),
            Guideline(
                id="bug-report",
                condition="Customer reports a bug or issue",
                action="Collect details and create support ticket",
                priority=2,
            ),
            Guideline(
                id="plan-upgrade",
                condition="Customer wants to upgrade their plan",
                action="Guide through upgrade process",
                priority=1,
            ),
        ]

        # Create example interaction history
        history = [
            Event(
                id=EventId("msg-1"),
                source=EventSource.CUSTOMER,
                kind=EventKind.MESSAGE,
                content="Hi, I'm having an issue with your service",
                timestamp=0,
            ),
        ]

        # Set up context variables
        context_variables = {
            "agent": ContextVariableValue(agent),
            "customer": ContextVariableValue(customer),
            "guidelines": ContextVariableValue(guidelines),
            "history": ContextVariableValue(history),
        }

        # Create components
        correlator = SimpleCorrelator()
        service_registry = SimpleServiceRegistry()
        tool_event_generator = SimpleToolEventGenerator()
        schematic_generator = SimpleSchematicGenerator()
        message_generator = SimpleMessageGenerator(logger, correlator, schematic_generator)
        message_assembler = SimpleMessageAssembler()
        lifecycle_hooks = SimpleLifecycleHooks()

        # Create DSPyGuidelineProposer
        guideline_proposer = DSPyGuidelineProposer(
            api_key=api_key,
            model_name="gpt-4",
        )

        # Create AlphaEngine
        engine = AlphaEngine(
            logger=logger,
            correlator=correlator,
            service_registry=service_registry,
            guideline_proposer=guideline_proposer,
            tool_event_generator=tool_event_generator,
            message_generator=message_generator,
            message_assembler=message_assembler,
            lifecycle_hooks=lifecycle_hooks,
        )

        # Process request
        result = await engine.process_request(context_variables)

        # Print results
        logger.info("Activated guidelines:")
        for guideline in result.activated_guidelines:
            logger.info(f"- {guideline.id}: {guideline.action}")

        logger.info("\nDeactivated guidelines:")
        for guideline in result.deactivated_guidelines:
            logger.info(f"- {guideline.id}: {guideline.action}")

        logger.info(f"\nFinal utterance: {result.utterance}")


if __name__ == "__main__":
    asyncio.run(main()) 
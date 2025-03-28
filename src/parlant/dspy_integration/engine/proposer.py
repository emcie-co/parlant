"""DSPy integration for proposing guidelines in the Parlant engine."""

from typing import Dict, List, Optional, Sequence, cast
import asyncio
import logging
import time
from datetime import datetime, timezone
from functools import cached_property

import dspy
from dspy.teleprompt import COPRO

from parlant.core.agents import Agent
from parlant.core.customers import Customer
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.sessions import Event, Term
from parlant.core.metrics import ModelMetrics
from parlant.core.logging import Logger
from parlant.core.nlp.generation import GenerationInfo, UsageInfo
from parlant.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionResult,
    GuidelineProposition,
    ConditionApplicabilityEvaluation,
    PreviouslyAppliedType
)
from parlant.dspy_integration.guideline_classifier import GuidelineClassifier
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager

logger = logging.getLogger(__name__)

class DSPyGuidelineProposer(GuidelineProposer):
    """DSPy-powered implementation of the guideline proposer.
    
    This class uses DSPy's classification and optimization capabilities to determine
    which guidelines should be activated based on conversation context.
    
    Attributes:
        classifier: DSPy-powered guideline classifier
        optimizer: DSPy-powered guideline optimizer
        metrics: Metrics tracker for monitoring performance
        logger: Logger instance for tracking operations
    """
    
    def __init__(
            self,
            logger: Logger,
            api_key: Optional[str] = None,
            model_name: str = "openai/gpt-3.5-turbo",
            metrics: Optional[ModelMetrics] = None,
            use_optimizer: bool = True
        ) -> None:
        """Initialize the DSPy guideline proposer.
        
        Args:
            logger: Logger instance for tracking operations
            api_key: Optional API key for the model provider
            model_name: Name of the model to use
            metrics: Optional metrics tracker
            use_optimizer: Whether to use optimization
        """
        self._logger = logger
        self.model_name = model_name
        self.metrics = metrics or ModelMetrics()
        self.use_optimizer = use_optimizer
        
        # Initialize DSPy components
        self.classifier = GuidelineClassifier(
            api_key=api_key,
            model_name=model_name,
            metrics=self.metrics,
            use_optimizer=use_optimizer
        )
        
        self.optimizer = BatchOptimizedGuidelineManager(
            api_key=api_key,
            model_name=model_name,
            metrics=self.metrics,
            use_optimizer=use_optimizer
        )
        
    async def propose_guidelines(
        self,
        agent: Agent,
        customer: Customer,
        guidelines: Sequence[Guideline],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
    ) -> GuidelinePropositionResult:
        """Propose guidelines using DSPy classification and optimization.
        
        Args:
            agent: The agent requesting guidelines
            customer: The customer in the conversation
            guidelines: Available guidelines to evaluate
            context_variables: Context variables for the conversation
            interaction_history: History of interaction events
            terms: Relevant terminology
            staged_events: Events staged for processing
            
        Returns:
            GuidelinePropositionResult containing proposed guidelines
        """
        if not guidelines:
            return GuidelinePropositionResult(
                total_duration=0.0,
                batch_count=0,
                batch_generations=[],
                batches=[]
            )

        t_start = time.time()
        guidelines_dict = {g.id: g for i, g in enumerate(guidelines, start=1)}
        
        # Create batches for efficient processing
        batches = self._create_guideline_batches(
            guidelines_dict,
            batch_size=self._get_optimal_batch_size(guidelines_dict),
        )

        with self._logger.operation(
            f"[DSPyGuidelineProposer] Evaluating {len(guidelines)} guidelines ({len(batches)} batches)"
        ):
            batch_tasks = [
                self._process_guideline_batch(
                    agent,
                    customer,
                    context_variables,
                    interaction_history,
                    staged_events,
                    terms,
                    batch,
                )
                for batch in batches
            ]

            batch_generations, condition_evaluations_batches = zip(
                *(await asyncio.gather(*batch_tasks))
            )

        proposition_batches: list[list[GuidelineProposition]] = []

        for batch in cast(
            tuple[list[ConditionApplicabilityEvaluation]], condition_evaluations_batches
        ):
            guideline_propositions = []
            for evaluation in batch:
                guideline_propositions.append(
                    GuidelineProposition(
                        guideline=guidelines_dict[GuidelineId(evaluation.guideline_id)],
                        score=evaluation.score,
                        guideline_previously_applied=PreviouslyAppliedType(
                            evaluation.guideline_previously_applied
                        ),
                        guideline_is_continuous=evaluation.guideline_is_continuous,
                        rationale=f'''Condition Application: "{evaluation.condition_application_rationale}"; Guideline Previously Applied: "{evaluation.guideline_previously_applied_rationale}"''',
                        should_reapply=evaluation.guideline_should_reapply,
                    )
                )
            proposition_batches.append(guideline_propositions)

        t_end = time.time()

        return GuidelinePropositionResult(
            total_duration=t_end - t_start,
            batch_count=len(batches),
            batch_generations=list(cast(tuple[GenerationInfo], batch_generations)),
            batches=proposition_batches,
        )

    def _get_optimal_batch_size(self, guidelines: dict[GuidelineId, Guideline]) -> int:
        """Get the optimal batch size for processing guidelines.
        
        Args:
            guidelines: Dictionary of guidelines to process
            
        Returns:
            Optimal batch size based on number of guidelines
        """
        guideline_n = len(guidelines)
        
        if guideline_n <= 10:
            return 1
        elif guideline_n <= 20:
            return 2
        elif guideline_n <= 30:
            return 3
        else:
            return 5

    def _create_guideline_batches(
        self,
        guidelines_dict: dict[GuidelineId, Guideline],
        batch_size: int,
    ) -> Sequence[dict[GuidelineId, Guideline]]:
        """Create batches of guidelines for efficient processing.
        
        Args:
            guidelines_dict: Dictionary of guidelines to batch
            batch_size: Size of each batch
            
        Returns:
            Sequence of guideline batches
        """
        batches = []
        guidelines = list(guidelines_dict.items())
        batch_count = -(-len(guidelines_dict) // batch_size)  # Ceiling division

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines[start_offset:end_offset])
            batches.append(batch)

        return batches

    async def _process_guideline_batch(
        self,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        terms: Sequence[Term],
        guidelines_dict: dict[GuidelineId, Guideline],
    ) -> tuple[GenerationInfo, list[ConditionApplicabilityEvaluation]]:
        """Process a batch of guidelines using DSPy.
        
        Args:
            agent: The agent requesting guidelines
            customer: The customer in the conversation
            context_variables: Context variables for the conversation
            interaction_history: History of interaction events
            staged_events: Events staged for processing
            terms: Relevant terminology
            guidelines_dict: Batch of guidelines to process
            
        Returns:
            Tuple of generation info and evaluations
        """
        # Format conversation history
        conversation = self._format_conversation(interaction_history)
        
        # Classify guidelines
        classification_result = await asyncio.to_thread(
            self.classifier.forward,
            conversation=conversation,
            guidelines=[g.content.condition for g in guidelines_dict.values()]
        )
        
        # Get activated guidelines
        activated = classification_result.get("activated", [])
        activated_guidelines = [g for g, a in zip(guidelines_dict.values(), activated) if a]
        
        if not activated_guidelines:
            return GenerationInfo(
                schema_name="dspy_guideline_proposer",
                model="none",
                duration=0.0,
                usage=UsageInfo(input_tokens=0, output_tokens=0)
            ), []
            
        # Optimize activated guidelines
        optimized = await asyncio.to_thread(
            self.optimizer.optimize_guidelines,
            guidelines=activated_guidelines,
            examples=self._get_examples(interaction_history, guidelines_dict)
        )
        
        # Convert to evaluations
        evaluations = []
        for guideline in optimized:
            evaluations.append(
                ConditionApplicabilityEvaluation(
                    guideline_id=guideline.id,
                    condition=guideline.content.condition,
                    action=guideline.content.action,
                    score=1.0,  # Default score since optimizer doesn't return scores
                    condition_application_rationale="Optimized by DSPy",
                    guideline_previously_applied="no",
                    guideline_previously_applied_rationale="",
                    guideline_should_reapply=False,
                    guideline_is_continuous=False
                )
            )
            
        return GenerationInfo(
            schema_name="dspy_guideline_proposer",
            model=self.model_name,
            duration=self.metrics.total_time,
            usage=UsageInfo(
                input_tokens=self.metrics.total_tokens // 2,  # Rough estimate
                output_tokens=self.metrics.total_tokens // 2  # Rough estimate
            )
        ), evaluations

    def _format_conversation(self, interaction_history: Sequence[Event]) -> str:
        """Format the conversation history for the classifier.
        
        Args:
            interaction_history: History of interaction events
            
        Returns:
            Formatted conversation history as a string
        """
        messages = []
        for event in interaction_history:
            if event.kind == "message":
                source = "User" if event.source == "customer" else "Assistant"
                messages.append(f"{source}: {event.data.get('message', '')}")
        return "\n".join(messages)
        
    def _get_examples(
        self,
        interaction_history: Sequence[Event],
        guidelines_dict: dict[GuidelineId, Guideline]
    ) -> List[Dict[str, str]]:
        """Get example responses from the conversation history.
        
        Args:
            interaction_history: History of interaction events
            guidelines_dict: Dictionary of available guidelines
            
        Returns:
            List of example dictionaries with conditions and responses
        """
        examples = []
        for event in interaction_history:
            if (
                event.kind == "message" 
                and event.source == "ai_agent"
                and event.data.get("guideline_id")
            ):
                guideline_id = GuidelineId(event.data["guideline_id"])
                if guideline_id in guidelines_dict:
                    examples.append({
                        "condition": guidelines_dict[guideline_id].content.condition,
                        "response": event.data.get("message", "")
                    })
        return examples 
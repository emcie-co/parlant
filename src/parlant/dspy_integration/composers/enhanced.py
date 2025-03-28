"""Enhanced message composer with DSPy integration.

This module provides an enhanced message composer that integrates DSPy capabilities
into the core message generation pipeline.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List

import structlog

from parlant.core.engines.alpha.message_event_composer import MessageEventComposer
from parlant.core.engines.types import Context, MessageGenerationResult
from parlant.core.guidelines import Guideline
from parlant.core.sessions import MessageEventData
from parlant.dspy_integration.services import DSPyService
from parlant.dspy_integration.guideline_optimizer import BatchOptimizedGuidelineManager as GuidelineOptimizer
from parlant.dspy_integration.guideline_classifier import GuidelineClassifier
from parlant.dspy_integration.types import EnhancedGuidelines, ClassificationResult

if TYPE_CHECKING:
    pass  # Keep block for potential future type-only imports


class DSPyEnhancedMessageComposer(MessageEventComposer):
    """Enhanced message composer that uses DSPy for improved message generation.
    
    This composer extends the base MessageEventComposer by adding DSPy-powered
    enhancements at various stages of the message generation pipeline:
    1. Pre-generation: Enhances guidelines and classifies context
    2. During generation: Optimizes content generation
    3. Post-generation: Improves response quality
    
    Args:
        dspy_service: Service for DSPy operations
        guideline_optimizer: Service for optimizing guidelines
        context_classifier: Service for classifying context
        logger: Structured logger instance
    """

    def __init__(
        self,
        dspy_service: DSPyService,
        guideline_optimizer: GuidelineOptimizer,
        context_classifier: GuidelineClassifier,
        logger: Optional[structlog.BoundLogger] = None,
    ) -> None:
        """Initialize the enhanced composer."""
        super().__init__()
        self.logger = logger or structlog.get_logger()
        self.dspy_service = dspy_service
        self.guideline_optimizer = guideline_optimizer
        self.context_classifier = context_classifier

    async def enhance_guidelines(
        self, 
        context: Context, 
        guidelines: Guideline
    ) -> EnhancedGuidelines:
        """Enhance guidelines using DSPy and context information.
        
        Args:
            context: Current conversation context
            guidelines: Original guidelines to enhance
            
        Returns:
            Enhanced guidelines with DSPy optimizations
        """
        try:
            # First classify the guidelines
            classified_guidelines = await self.dspy_service.classify_guidelines(
                context=context,
                guidelines=guidelines
            )
            
            # Then optimize them
            enhanced = await self.guideline_optimizer.optimize(
                guidelines=classified_guidelines,
                context=context
            )
            
            # Ensure insights are always a list
            if isinstance(enhanced.insights, str):
                enhanced.insights = [enhanced.insights]
            elif not enhanced.insights:
                enhanced.insights = []
            
            self.logger.info(
                "Guidelines enhanced successfully",
                session_id=context.session_id,
                dspy_enhancement_status="success",
                insights_count=len(enhanced.insights)
            )
            
            return enhanced
            
        except Exception as e:
            self.logger.error(
                "Failed to enhance guidelines",
                error=str(e),
                session_id=context.session_id,
                dspy_enhancement_status="failed"
            )
            # Ensure default guidelines have list insights
            guidelines.insights = [] if not hasattr(guidelines, 'insights') else guidelines.insights
            if isinstance(guidelines.insights, str):
                guidelines.insights = [guidelines.insights]
            return guidelines

    async def classify_context(self, context: Context) -> ClassificationResult:
        """Classify the conversation context using DSPy.
        
        Args:
            context: Current conversation context
            
        Returns:
            Classification results for the context
        """
        try:
            # Extract relevant features
            features = await self.context_classifier.extract_features(context)
            
            # Perform classification
            result = await self.context_classifier.classify(context)
            
            # Update classification with new context
            await self.context_classifier.update_classification(result, context)
            
            self.logger.debug(
                "Context classified successfully",
                session_id=context.session_id,
                features=features
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Context classification failed",
                error=str(e),
                session_id=context.session_id
            )
            return ClassificationResult(success=False, error=str(e))

    async def optimize_response(
        self, 
        message: MessageEventData, 
        context: Context
    ) -> MessageEventData:
        """Optimize the generated response using DSPy.
        
        Args:
            message: Generated message to optimize
            context: Current conversation context
            
        Returns:
            Optimized message
        """
        try:
            optimized = await self.dspy_service.optimize_response(
                message=message,
                context=context
            )
            
            self.logger.info(
                "Response optimized successfully",
                session_id=context.session_id,
                optimization_applied=True
            )
            
            return optimized
            
        except Exception as e:
            self.logger.error(
                "Response optimization failed",
                error=str(e),
                session_id=context.session_id
            )
            return message

    async def generate_message(
        self, 
        context: Context, 
        guidelines: Guideline
    ) -> MessageGenerationResult:
        """Generate a message with DSPy enhancements.
        
        This method orchestrates the enhanced message generation process:
        1. Classifies the context
        2. Enhances the guidelines
        3. Generates the base message
        4. Optimizes the response
        
        Args:
            context: Current conversation context
            guidelines: Guidelines to follow
            
        Returns:
            Generated message with metrics
        """
        metrics: Dict[str, Any] = {
            "dspy_enhancement_status": "not_started",
            "guideline_optimization_applied": False
        }
        
        try:
            # Pre-generation enhancements
            await self.classify_context(context)
            enhanced_guidelines = await self.enhance_guidelines(context, guidelines)
            
            # Generate base message
            base_result = await super().generate_message(context, enhanced_guidelines)
            
            # Post-generation optimization
            optimized_message = await self.optimize_response(
                message=base_result.message,
                context=context
            )
            
            metrics.update({
                "dspy_enhancement_status": "success",
                "guideline_optimization_applied": True
            })
            
            return MessageGenerationResult(
                message=optimized_message,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(
                "DSPy enhanced generation failed",
                error=str(e),
                session_id=context.session_id
            )
            # Fallback to base implementation
            metrics["dspy_enhancement_status"] = "failed"
            base_result = await super().generate_message(context, guidelines)
            return MessageGenerationResult(
                message=base_result.message,
                metrics=metrics
            ) 
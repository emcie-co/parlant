"""
Integration between Discord bot and COPRO optimization functionality.

This module provides the bridge between the Discord bot and the COPRO optimization
functionality in the Parlant repository.
"""
import logging
from typing import Dict, Any, List, Optional

from parlant.core.engines.types import Context
from parlant.core.guidelines import Guideline
from parlant.dspy_integration.guideline_optimizer import GuidelineOptimizer
from parlant.dspy_integration.metrics import DSPyMetrics

# Setup logging
logger = logging.getLogger(__name__)

class COPROBridge:
    """Bridge between Discord bot and COPRO optimization functionality."""
    
    def __init__(self, optimizer: GuidelineOptimizer, metrics: DSPyMetrics):
        """Initialize the COPRO bridge.
        
        Args:
            optimizer: The guideline optimizer
            metrics: The DSPy metrics
        """
        self.optimizer = optimizer
        self.metrics = metrics
        self.optimization_history = []
    
    async def process_conversation(self, user_id: str, message: str, history: List[Dict[str, Any]]) -> str:
        """Process a conversation message using COPRO.
        
        Args:
            user_id: The user ID
            message: The user message
            history: The conversation history
            
        Returns:
            The assistant's response
        """
        try:
            # Create context from conversation
            context = Context(
                user_id=user_id,
                message=message,
                conversation_history=history
            )
            
            # Create a simple guideline for the conversation
            guideline = Guideline(
                name="conversation",
                description="Respond to the user's message in a helpful and coherent way.",
                input_fields=["user_message", "conversation_history"],
                output_fields=["assistant_message"]
            )
            
            # Use the optimizer to generate a response
            result = await self.optimizer.optimize_response(context, guideline)
            
            return result.get("assistant_message", "I'm not sure how to respond to that.")
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    async def optimize_prompts(self, user_id: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize prompts based on conversation history.
        
        Args:
            user_id: The user ID
            history: The conversation history
            
        Returns:
            Optimization results
        """
        try:
            # Create context from conversation history
            context = Context(
                user_id=user_id,
                conversation_history=history
            )
            
            # Create a guideline for optimization
            guideline = Guideline(
                name="conversation_optimization",
                description="Optimize prompts for better conversation quality.",
                input_fields=["conversation_history"],
                output_fields=["optimized_prompts"]
            )
            
            # Use the optimizer to optimize prompts
            before_metrics = await self.metrics.evaluate_guideline(guideline, context)
            optimized_guideline = await self.optimizer.optimize_guideline(guideline, context)
            after_metrics = await self.metrics.evaluate_guideline(optimized_guideline, context)
            
            # Calculate improvement
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            
            # Record optimization in history
            optimization_record = {
                "user_id": user_id,
                "history_length": len(history),
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "improvement": improvement
            }
            
            self.optimization_history.append(optimization_record)
            
            return {
                "improvement": improvement,
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "optimized_guideline": optimized_guideline.dict()
            }
        except Exception as e:
            logger.error(f"Error optimizing prompts: {e}")
            raise
    
    def _calculate_improvement(self, before_metrics: Dict[str, float], after_metrics: Dict[str, float]) -> float:
        """Calculate the improvement percentage between before and after metrics.
        
        Args:
            before_metrics: Metrics before optimization
            after_metrics: Metrics after optimization
            
        Returns:
            The improvement percentage
        """
        if not before_metrics or not after_metrics:
            return 0.0
        
        # Calculate average improvement across all metrics
        total_improvement = 0.0
        metric_count = 0
        
        for metric_name in before_metrics:
            if metric_name in after_metrics:
                before_value = before_metrics[metric_name]
                after_value = after_metrics[metric_name]
                
                if before_value > 0:
                    improvement = ((after_value - before_value) / before_value) * 100
                    total_improvement += improvement
                    metric_count += 1
        
        if metric_count == 0:
            return 0.0
        
        return total_improvement / metric_count
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization performance.
        
        Returns:
            Statistics about optimization performance
        """
        if not self.optimization_history:
            return {
                "total_optimizations": 0,
                "average_improvement": 0.0,
                "success_rate": 0.0,
                "metrics": {}
            }
        
        # Calculate statistics
        total_optimizations = len(self.optimization_history)
        
        # Calculate average improvement
        total_improvement = sum(record["improvement"] for record in self.optimization_history)
        average_improvement = total_improvement / total_optimizations
        
        # Calculate success rate (improvement > 0)
        successful_optimizations = sum(1 for record in self.optimization_history if record["improvement"] > 0)
        success_rate = (successful_optimizations / total_optimizations) * 100
        
        # Calculate metric-specific improvements
        metric_improvements = {}
        
        for record in self.optimization_history:
            before_metrics = record["before_metrics"]
            after_metrics = record["after_metrics"]
            
            for metric_name in before_metrics:
                if metric_name in after_metrics:
                    before_value = before_metrics[metric_name]
                    after_value = after_metrics[metric_name]
                    
                    if before_value > 0:
                        improvement = ((after_value - before_value) / before_value) * 100
                        
                        if metric_name not in metric_improvements:
                            metric_improvements[metric_name] = []
                        
                        metric_improvements[metric_name].append(improvement)
        
        # Calculate average improvement for each metric
        metrics = {}
        
        for metric_name, improvements in metric_improvements.items():
            if improvements:
                metrics[metric_name] = sum(improvements) / len(improvements)
            else:
                metrics[metric_name] = 0.0
        
        return {
            "total_optimizations": total_optimizations,
            "average_improvement": average_improvement,
            "success_rate": success_rate,
            "metrics": metrics
        }
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the optimization history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            The optimization history
        """
        # Return the most recent entries first
        return sorted(
            self.optimization_history,
            key=lambda record: record.get("timestamp", 0),
            reverse=True
        )[:limit]

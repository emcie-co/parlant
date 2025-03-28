"""Type definitions for DSPy integration.

This module defines the types used throughout the DSPy integration components.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from parlant.core.guidelines import Guideline


@dataclass(frozen=True)
class EnhancedGuidelines(Guideline):
    """Guidelines enhanced with DSPy optimizations.
    
    This extends the base Guidelines type with additional DSPy-specific
    optimizations and metadata.
    
    Args:
        optimization_score: Confidence score of the optimization
        feature_importance: Importance scores for different features
        context_relevance: How relevant the guidelines are to current context
        metadata: Additional DSPy-specific metadata
        insights: List of insights derived from the guidelines
    """
    
    optimization_score: float
    feature_importance: Dict[str, float]
    context_relevance: float
    metadata: Dict[str, Any]
    insights: Optional[List[str]] = None


@dataclass
class ClassificationResult:
    """Result of DSPy context classification.
    
    Args:
        success: Whether classification was successful
        features: Extracted features from the context
        scores: Classification scores for different categories
        error: Error message if classification failed
        metadata: Additional classification metadata
    """
    
    success: bool
    features: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class OptimizationMetrics:
    """Metrics collected during DSPy optimization.
    
    Args:
        latency_ms: Time taken for optimization in milliseconds
        success_rate: Rate of successful optimizations
        enhancement_scores: Scores for different enhancement aspects
        error_counts: Count of different types of errors
    """
    
    latency_ms: float
    success_rate: float
    enhancement_scores: Dict[str, float]
    error_counts: Dict[str, int]
"""MM-RegCheck: Training-Free Causal-Aware Multimodal Reasoning for Regulatory Evasion Detection."""

from __future__ import annotations

from mmregcheck.evaluator import MedRegEvaluator, MetricTriple
from mmregcheck.schemas import (
    MEDREG_CANARY_GUID,
    BusinessMetadata,
    ComplianceVerdict,
    EvaluationSubset,
    MedRegSampleInput,
    RegulatoryDomain,
    SubmissionPrediction,
    TechniqueVerdictItem,
    TFBearBranchTrace,
    aggregate_strict_hierarchy,
)

__version__ = "0.1.0"

__all__ = [
    "MEDREG_CANARY_GUID",
    "BusinessMetadata",
    "ComplianceVerdict",
    "EvaluationSubset",
    "MedRegEvaluator",
    "MedRegSampleInput",
    "MetricTriple",
    "RegulatoryDomain",
    "SubmissionPrediction",
    "TFBearBranchTrace",
    "TechniqueVerdictItem",
    "__version__",
    "aggregate_strict_hierarchy",
]

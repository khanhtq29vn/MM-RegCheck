"""Pillar 2: Evidence Adjudicator via TF-BEAR (Section 4.2).

`TFBear` is the public entry point of the pipeline; it is configured with `TFBearConfig` and
runs over `TFBearState`, returning a `TFBearResult`. `EndpointConfig` describes an
OpenAI-compatible endpoint for one or more stages and `TFBearAdjudicator` is the math-only
dual-threshold adjudicator. Custom stages subclass the matching `Base*Strategy` and are passed
to `TFBearConfig`.

Importing this package does not import the optional SDK: `openai` and `json_repair` stay lazy
inside `mmregcheck.pillar2.llm`.
"""

from __future__ import annotations

from mmregcheck.pillar2.adjudication import TFBearAdjudicator
from mmregcheck.pillar2.base import (
    BaseAdjudicationStrategy,
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.pipeline_state import Node, TFBearState
from mmregcheck.pillar2.results import TFBearResult
from mmregcheck.pillar2.tf_bear import TFBear

__all__ = [
    "BaseAdjudicationStrategy",
    "BaseExpansionStrategy",
    "BaseExplanationStrategy",
    "BasePriorStrategy",
    "BaseScoringStrategy",
    "EndpointConfig",
    "Node",
    "TFBear",
    "TFBearAdjudicator",
    "TFBearConfig",
    "TFBearResult",
    "TFBearState",
]

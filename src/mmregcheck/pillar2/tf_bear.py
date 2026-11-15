"""TF-BEAR pipeline entry point (Section 4.2).

The five stages of the pipeline and the module that implements each one:

    Stage 1  Prior        `mmregcheck.pillar2.prior`
    Stage 2  Expansion    `mmregcheck.pillar2.expansion`
    Stage 3  Scoring      `mmregcheck.pillar2.scoring`
    Stage 4  Adjudication `mmregcheck.pillar2.adjudication`
    Stage 5  Explanation  `mmregcheck.pillar2.explainer`

The orchestration lives in `mmregcheck.pillar2.engine`, the strategy wiring in
`mmregcheck.pillar2.config`, and the shared state in `mmregcheck.pillar2.pipeline_state`.
"""

from __future__ import annotations

from mmregcheck.pillar2.adjudication import TFBearAdjudicator
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.engine import run_tfbear
from mmregcheck.pillar2.results import TFBearAdjudicationResult, TFBearResult

__all__ = [
    "TFBearAdjudicationResult",
    "TFBearAdjudicator",
    "TFBearConfig",
    "TFBearResult",
    "run_tfbear",
]

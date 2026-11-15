"""Result types of the TF-BEAR pipeline.

These dataclasses are the return values of the pipeline and of its adjudication stage. They sit
at the bottom of the import graph together with `base.py`: `pipeline_state` and `base` import
them, never the other way round, so the state can hold an adjudication result without importing
the orchestration module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace

if TYPE_CHECKING:
    from mmregcheck.pillar2.pipeline_state import TFBearState


@dataclass(frozen=True, slots=True)
class TFBearAdjudicationResult:
    """Output of TF-BEAR branch pruning and majority voting."""

    final_verdict: ComplianceVerdict
    surviving_branches: list[TFBearBranchTrace]
    pruned_branches: list[TFBearBranchTrace]
    used_fallback: bool = False


@dataclass(frozen=True, slots=True)
class StageUsage:
    """Latency and token usage of one stage."""

    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class TFBearResult:
    """Outcome of one TF-BEAR run.

    Attributes:
        state: Final state of the run.
        verdict: Final adjudicated verdict.
        explanation: Stage 5 output.
        usage: Latency and token usage per stage.
        diagnostics: Non-fatal conditions recorded during the run, in occurrence order. The list
            does not depend on `verbose`, so a benchmark artefact can be audited after the run.
    """

    state: TFBearState
    verdict: ComplianceVerdict
    explanation: str
    usage: dict[str, StageUsage]
    diagnostics: list[str] = field(default_factory=list)

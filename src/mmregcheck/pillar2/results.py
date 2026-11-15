"""Result types of the TF-BEAR pipeline (Section 4.2).

These dataclasses are the return values of the pipeline and of its adjudication stage. They sit
at the bottom of the import graph together with `base.py`: `pipeline_state` and `base` import
them, never the other way round, so the state can hold an adjudication result without importing
the orchestration module.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Outcome of one TF-BEAR run."""

    state: TFBearState
    verdict: ComplianceVerdict
    explanation: str
    usage: dict[str, StageUsage]

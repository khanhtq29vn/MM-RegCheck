"""Result types of the TF-BEAR pipeline.

These dataclasses are the return values of the pipeline and of its adjudication stage. They sit
at the bottom of the import graph together with `base.py`: `pipeline_state` and `base` import
them, never the other way round, so the state can hold an adjudication result without importing
the orchestration module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

    def to_trace_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable audit record of the run.

        The record exposes what the leaderboard prediction deliberately hides: the prior
        distribution, every scored branch with its Delta and whether it survived the dual gate,
        the fallback flag, the per-stage usage, the raw explanation and the diagnostics. Branch
        membership of the gate is resolved by `branch_id`, which is unique within one run.

        This is a pure function: it reads the result and allocates a new dictionary, so it can be
        called repeatedly and from several call sites without changing the run.
        """
        adjudication = self.state.adjudication
        surviving_branch_ids = (
            {branch.branch_id for branch in adjudication.surviving_branches}
            if adjudication is not None
            else set()
        )
        branches: list[dict[str, Any]] = [
            {
                "branch_id": branch.branch_id,
                "strategy": branch.strategy,
                "predicted_verdict": branch.predicted_verdict.value,
                "prior_prob": branch.prior_prob,
                "posterior_prob": branch.posterior_prob,
                "log_odds_delta": branch.log_odds_delta,
                "gate": "survived" if branch.branch_id in surviving_branch_ids else "pruned",
            }
            for branch in self.state.branches
        ]
        return {
            "verdict": self.verdict.value,
            "used_fallback": adjudication.used_fallback if adjudication is not None else False,
            "p_prior": {
                verdict.value: probability for verdict, probability in self.state.p_prior.items()
            },
            "branches": branches,
            "n_branches": len(branches),
            "n_surviving": len(adjudication.surviving_branches) if adjudication is not None else 0,
            "usage": {
                stage: {
                    "latency_seconds": usage.latency_seconds,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                for stage, usage in self.usage.items()
            },
            "explanation": self.explanation,
            "diagnostics": list(self.diagnostics),
        }

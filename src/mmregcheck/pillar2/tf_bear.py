"""Module 3: Training-Free Bayesian Evidence Aggregation (TF-BEAR — Section 4.2).

Implements the exact mathematical log-odds filtering and majority-vote adjudication
from Section 4.2 and Section 5.4 (Case Study). Hoang-An Vo can plug the VLM logprobs
sampler directly into `TFBearAdjudicator.adjudicate_branches()`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace


@dataclass(frozen=True, slots=True)
class TFBearAdjudicationResult:
    """Output of TF-BEAR branch pruning and majority voting."""

    final_verdict: ComplianceVerdict
    surviving_branches: list[TFBearBranchTrace]
    pruned_branches: list[TFBearBranchTrace]


class TFBearAdjudicator:
    """Executes TF-BEAR dual-threshold log-odds filtering and majority voting."""

    def __init__(self, tau_posterior: float = 0.50, tau_delta: float = 0.50) -> None:
        self.tau_posterior = tau_posterior
        self.tau_delta = tau_delta

    def adjudicate_branches(
        self,
        branches: list[TFBearBranchTrace],
    ) -> TFBearAdjudicationResult:
        """Prune hallucinated branches failing the dual-threshold gate and vote."""
        if not branches:
            raise ValueError("TFBearAdjudicator requires at least one reasoning branch.")

        surviving: list[TFBearBranchTrace] = []
        pruned: list[TFBearBranchTrace] = []

        for branch in branches:
            if branch.passes_dual_threshold_gate(
                tau_posterior=self.tau_posterior,
                tau_delta=self.tau_delta,
            ):
                surviving.append(branch)
            else:
                pruned.append(branch)

        active_pool = surviving if surviving else branches
        vote_counts = Counter(b.predicted_verdict for b in active_pool)

        # Strict Liability tie-breaking: prioritize Non-Compliant on ties
        priority = {
            ComplianceVerdict.NON_COMPLIANT: 3,
            ComplianceVerdict.PARTLY_COMPLIANT: 2,
            ComplianceVerdict.COMPLIANT: 1,
        }
        winner = max(
            vote_counts.keys(),
            key=lambda v: (vote_counts[v], priority[v]),
        )

        return TFBearAdjudicationResult(
            final_verdict=winner,
            surviving_branches=surviving,
            pruned_branches=pruned,
        )

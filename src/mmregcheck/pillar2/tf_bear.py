"""Module 3: Training-Free Bayesian Evidence Aggregation (TF-BEAR — Section 4.2).

Implements the log-odds filtering and majority-vote adjudication
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
    used_fallback: bool = False


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
        posterior_tally = self._posterior_tally(branches)

        if surviving:
            # Majority vote; ties resolved by the summed posterior over *all* branches.
            winner = max(vote_counts, key=lambda v: (vote_counts[v], posterior_tally[v]))
        else:
            # No branch passed the causal filter: pick the label with the highest
            # summed posterior across all generated branches (Appendix H fallback).
            winner = max(ComplianceVerdict, key=lambda v: posterior_tally[v])

        return TFBearAdjudicationResult(
            final_verdict=winner,
            surviving_branches=surviving,
            pruned_branches=pruned,
            used_fallback=not surviving,
        )

    @staticmethod
    def _posterior_tally(branches: list[TFBearBranchTrace]) -> dict[ComplianceVerdict, float]:
        """Sum each label's posterior over every branch.

        A branch without a full distribution contributes only the posterior of its own verdict.
        """
        tally = dict.fromkeys(ComplianceVerdict, 0.0)
        for branch in branches:
            if branch.posterior_distribution is None:
                tally[branch.predicted_verdict] += branch.posterior_prob
            else:
                for verdict, prob in branch.posterior_distribution.items():
                    tally[verdict] += prob
        return tally

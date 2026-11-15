"""Adjudication of the TF-BEAR pipeline (Section 4.2, Stage 4).

`TFBearAdjudicator` is the authoritative math: the dual-threshold gate, the majority vote, the
summed-posterior tie-break and the Appendix H fallback. `summed_posterior_tally` is the one
posterior aggregation every strategy shares.

The strategies wire that math into the pipeline. `TFBearAdjudicationStrategy` is the default and
calls the adjudicator directly; the two extra strategies port the draft's `selection.py`
behaviour onto `TFBearBranchTrace` and are opt-in because they report more than the paper's
adjudicator:

    MajorityVoteAdjudicationStrategy  same gate/vote/fallback, plus the normalised summed
                                      posterior in `state.p_posterior`
    JointAdjudicationStrategy         one joint logprob call over every qualifying path
"""

from __future__ import annotations

from collections import Counter

from mmregcheck.pillar2.base import BaseAdjudicationStrategy
from mmregcheck.pillar2.llm import ChatClient, label_distribution
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.results import TFBearAdjudicationResult
from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace


def summed_posterior_tally(
    branches: list[TFBearBranchTrace],
) -> dict[ComplianceVerdict, float]:
    """Sum each label's posterior over every branch.

    A branch without a full distribution contributes only the posterior of its own verdict. This
    is the shared aggregation of the adjudicator's tie-break and fallback and of the
    majority-vote strategy's reported distribution.
    """
    tally = dict.fromkeys(ComplianceVerdict, 0.0)
    for branch in branches:
        if branch.posterior_distribution is None:
            tally[branch.predicted_verdict] += branch.posterior_prob
        else:
            for verdict, probability in branch.posterior_distribution.items():
                tally[verdict] += probability
    return tally


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
        posterior_tally = summed_posterior_tally(branches)

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


def _validate_thresholds(tau_posterior: float, tau_delta: float) -> None:
    if not 0.0 <= tau_posterior <= 1.0:
        raise ValueError(f"tau_posterior must be in [0, 1], got {tau_posterior}.")
    if not tau_delta >= 0.0:
        raise ValueError(f"tau_delta must be non-negative, got {tau_delta}.")


class TFBearAdjudicationStrategy(BaseAdjudicationStrategy):
    """Default adjudication: the paper's dual-threshold gate and vote.

    This is a thin wrapper around `TFBearAdjudicator` so the thresholds live in
    `TFBearConfig`; it uses `state.branches` produced by the scoring stage.
    """

    def __init__(
        self,
        tau_posterior: float = 0.50,
        tau_delta: float = 0.50,
        prompt_template: str = "",
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        _validate_thresholds(tau_posterior, tau_delta)
        self.tau_posterior = tau_posterior
        self.tau_delta = tau_delta

    def adjudicate(
        self,
        state: TFBearState,
        client: ChatClient,
        model: str,
    ) -> TFBearAdjudicationResult:
        """Apply the dual-threshold gate and majority vote over `state.branches`."""
        if not state.branches:
            raise ValueError(f"{type(self).__name__}: state.branches is empty.")
        return TFBearAdjudicator(
            tau_posterior=self.tau_posterior,
            tau_delta=self.tau_delta,
        ).adjudicate_branches(state.branches)


class MajorityVoteAdjudicationStrategy(BaseAdjudicationStrategy):
    """Draft majority vote: `TFBearAdjudicator` plus the reported aggregate posterior.

    The gate, vote, tie-break and fallback are exactly `TFBearAdjudicator`'s, so this strategy
    delegates to it instead of duplicating the math. Its only extra work is writing the
    normalised `summed_posterior_tally` to `state.p_posterior`, which the draft's evaluation
    scripts read. Use it when reproducing those numbers.
    """

    def __init__(
        self,
        tau_posterior: float = 0.50,
        tau_delta: float = 0.50,
        prompt_template: str = "",
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        _validate_thresholds(tau_posterior, tau_delta)
        self.tau_posterior = tau_posterior
        self.tau_delta = tau_delta

    def adjudicate(
        self,
        state: TFBearState,
        client: ChatClient,
        model: str,
    ) -> TFBearAdjudicationResult:
        """Vote over gate-passing branches; fall back to the summed posterior."""
        if not state.branches:
            raise ValueError(f"{type(self).__name__}: state.branches is empty.")

        posterior = summed_posterior_tally(state.branches)
        total = sum(posterior.values())
        state.p_posterior = (
            {verdict: probability / total for verdict, probability in posterior.items()}
            if total > 0
            else posterior
        )

        return TFBearAdjudicator(
            tau_posterior=self.tau_posterior,
            tau_delta=self.tau_delta,
        ).adjudicate_branches(state.branches)


class JointAdjudicationStrategy(BaseAdjudicationStrategy):
    """Draft joint selection: classify every qualifying path in one logprob call.

    Qualifying paths (posterior at the branch's own verdict at least `tau_posterior`) are
    serialised into `{paths}` and classified together. If none qualifies, the single
    highest-posterior path is used and the result is marked as a fallback. The joint
    distribution is written to `state.p_posterior`, as in the draft.
    """

    def __init__(
        self,
        prompt_template: str,
        tau_posterior: float = 0.50,
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        if not 0.0 <= tau_posterior <= 1.0:
            raise ValueError(f"tau_posterior must be in [0, 1], got {tau_posterior}.")
        if not prompt_template.strip():
            raise ValueError(
                f"{type(self).__name__} requires a prompt_template that references a "
                "`{paths}` placeholder and asks for a single digit."
            )
        self.tau_posterior = tau_posterior

    def adjudicate(
        self,
        state: TFBearState,
        client: ChatClient,
        model: str,
    ) -> TFBearAdjudicationResult:
        """Classify the joint set of qualifying paths with one logprob call."""
        if not state.branches:
            raise ValueError(f"{type(self).__name__}: state.branches is empty.")

        qualified = [
            branch for branch in state.branches if branch.posterior_prob >= self.tau_posterior
        ]
        used_fallback = not qualified
        joint_branches = (
            qualified
            if qualified
            else [max(state.branches, key=lambda branch: branch.posterior_prob)]
        )
        paths = "\n\n".join(
            f"Path {position} [predicted: {branch.predicted_verdict.value}, "
            f"delta={branch.log_odds_delta:+.4f}]:\n{branch.rationale}"
            for position, branch in enumerate(joint_branches, start=1)
        )
        prompt = state.render(self.prompt_template, paths=paths)

        distribution = label_distribution(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=self.max_retry,
        )
        winner = max(distribution, key=lambda verdict: distribution[verdict])
        state.p_posterior = distribution

        return TFBearAdjudicationResult(
            final_verdict=winner,
            surviving_branches=joint_branches,
            pruned_branches=[branch for branch in state.branches if branch not in joint_branches],
            used_fallback=used_fallback,
        )

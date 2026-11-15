"""Stage 3: per-branch posterior scoring and causal log-odds attribution (Section 4.2).

Every scoreable node is classified once with the shared label-token logprob helper. The draft
computed Delta inline with a second copy of the log-odds formula; here the trace's
`TFBearBranchTrace.log_odds_delta` property owns the epsilon-stabilised transform, so scoring
and adjudication can never disagree about Delta.
"""

from __future__ import annotations

from mmregcheck.pillar2.base import BaseScoringStrategy
from mmregcheck.pillar2.clients import ChatClient
from mmregcheck.pillar2.logprobs import label_distribution
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.schemas import TFBearBranchTrace


class CausalDeltaScoringStrategy(BaseScoringStrategy):
    """Score every reasoning path with P_posterior and a causal log-odds Delta.

    For each node the strategy renders the scoring prompt with the node's reasoning text as
    `{state}`, elicits the verdict distribution conditioned on claim, context and path, and
    builds one `TFBearBranchTrace` per scoreable node:

    * `prior_prob` is `state.p_prior` at the branch's predicted verdict,
    * `posterior_prob` is the posterior at the same verdict,
    * `posterior_distribution` is the full softmax (needed by the adjudicator's fallback),
    * `strategy` is `node.meta["strategy"]` (expansion strategies label their paths there),
    * `rationale` is the node's reasoning text,
    * Delta is read from `TFBearBranchTrace.log_odds_delta`.

    Hollow paths follow the draft: a node that is not terminal, or whose state and answer are
    both blank, is skipped and produces no trace. The result can therefore be shorter than
    `state.nodes`, and empty when nothing is scoreable. The strategy never writes
    `state.p_posterior`; per-branch distributions live on the traces, and inventing an
    aggregate would misreport any single branch.

    The prior must already cover the predicted verdict; a missing entry is an error rather
    than the draft's silent 1/3 default, which would fabricate a Delta.
    """

    def score(
        self,
        state: TFBearState,
        client: ChatClient,
        model: str,
    ) -> list[TFBearBranchTrace]:
        """Return one trace per terminal, non-empty node of `state.nodes`."""
        traces: list[TFBearBranchTrace] = []

        for position, node in enumerate(state.nodes, start=1):
            if not node.is_terminal:
                if self.verbose:
                    self.logger.log_info(f"Skipping non-terminal node {position}.", style="dim")
                continue

            rationale = node.state or node.answer or ""
            if not rationale.strip():
                if self.verbose:
                    self.logger.log_warning(f"Skipping node {position} with empty reasoning.")
                continue

            prompt = state.render(self.prompt_template, state=rationale)
            distribution = label_distribution(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_retry=self.max_retry,
            )
            # Distribution order follows the digit order, so an exact tie resolves to "0"
            # (Non-Compliant) as in the draft's `max` over the same insertion order.
            predicted_verdict = max(distribution, key=lambda verdict: distribution[verdict])

            if predicted_verdict not in state.p_prior:
                available = [verdict.value for verdict in state.p_prior]
                raise ValueError(
                    f"{type(self).__name__}: node {position} predicts "
                    f"{predicted_verdict.value!r}, but the prior has no probability for it "
                    f"(available: {available!r}). Run the prior stage before scoring."
                )

            trace = TFBearBranchTrace(
                branch_id=f"P{position}",
                strategy=str(node.meta.get("strategy") or "unknown"),
                prior_prob=state.p_prior[predicted_verdict],
                posterior_prob=distribution[predicted_verdict],
                predicted_verdict=predicted_verdict,
                rationale=rationale,
                posterior_distribution=distribution,
            )

            if self.verbose:
                self.logger.log_branch(
                    trace.branch_id,
                    trace.predicted_verdict.value,
                    trace.log_odds_delta,
                    {verdict.value: prob for verdict, prob in distribution.items()},
                )
            traces.append(trace)

        return traces

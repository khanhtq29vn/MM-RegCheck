"""Stage 3: per-branch posterior scoring and causal log-odds attribution.

Every scoreable node is classified once with the shared label-token logprob helper, so the
posterior is elicited exactly like the prior and the two are directly comparable. The trace's
`TFBearBranchTrace.log_odds_delta` property owns the epsilon-stabilised log-odds transform, so
scoring and adjudication can never disagree about Delta.

The elicitation sampling is caller-controlled: `temperature` (default 0.0), `top_p` (default
0.95) and `top_logprobs` (default 5) are constructor arguments.
"""

from __future__ import annotations

from typing import Any

from mmregcheck.pillar2.base import BaseScoringStrategy
from mmregcheck.pillar2.diagnostics import record_diagnostic
from mmregcheck.pillar2.llm import ChatClient, label_distribution
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.tie_break import select_verdict
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

    An exact tie in the posterior resolves through the shared tie rule of
    `mmregcheck.pillar2.tie_break` (Non-Compliant first), so the predicted verdict does not
    depend on dictionary order.

    Hollow paths are skipped: a node that is not terminal, or whose state and answer are both
    blank, produces no trace. The result can therefore be shorter than `state.nodes`, and empty
    when nothing is scoreable. Every skip is reported as a `TFBearWarning` and recorded on the
    run's diagnostics. The strategy never writes `state.p_posterior`; per-branch distributions
    live on the traces, and inventing an aggregate would misreport any single branch.

    The prior must already cover the predicted verdict; a missing entry is an error rather than
    a uniform fallback, because a default probability would fabricate a Delta.

    Args:
        prompt_template: Scoring prompt; may reference claim, context and `{state}`.
        temperature: Sampling temperature of the elicitation call; defaults to 0.0.
        top_p: Nucleus threshold of the elicitation call; defaults to 0.95.
        top_logprobs: Number of candidate tokens requested; defaults to 5.
        verbose: Enables verbose stage logging.
        max_retry: Retry budget of the LLM call.
        reasoning: Provider-specific reasoning payload.
    """

    def __init__(
        self,
        prompt_template: str = "",
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_logprobs: int = 5,
        verbose: bool = False,
        max_retry: int = 3,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            prompt_template=prompt_template,
            verbose=verbose,
            max_retry=max_retry,
            reasoning=reasoning,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.top_logprobs = top_logprobs

    def score(
        self,
        state: TFBearState,
        client: ChatClient,
        model: str,
    ) -> list[TFBearBranchTrace]:
        """Return one trace per terminal, non-empty node of `state.nodes`."""
        traces: list[TFBearBranchTrace] = []
        skipped_non_terminal = 0
        skipped_empty = 0

        for position, node in enumerate(state.nodes, start=1):
            if not node.is_terminal:
                skipped_non_terminal += 1
                if self.verbose:
                    self.logger.log_info(f"Skipping non-terminal node {position}.", style="dim")
                continue

            rationale = node.state or node.answer or ""
            if not rationale.strip():
                skipped_empty += 1
                if self.verbose:
                    self.logger.log_warning(f"Skipping node {position} with empty reasoning.")
                continue

            prompt = state.render(self.prompt_template, state=rationale)
            distribution = label_distribution(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_retry=self.max_retry,
                temperature=self.temperature,
                top_p=self.top_p,
                top_logprobs=self.top_logprobs,
                diagnostics=state.diagnostics,
                stage=type(self).__name__,
                extra_body=self.extra_body,
            )
            predicted_verdict = select_verdict(distribution)

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

        skipped = skipped_non_terminal + skipped_empty
        if skipped:
            reasons: list[str] = []
            if skipped_non_terminal:
                reasons.append(f"{skipped_non_terminal} non-terminal")
            if skipped_empty:
                reasons.append(f"{skipped_empty} with empty reasoning")
            record_diagnostic(
                state.diagnostics,
                f"{type(self).__name__}: skipped {skipped} of {len(state.nodes)} node(s) "
                f"({', '.join(reasons)}); {len(traces)} path(s) reached scoring.",
                stacklevel=2,
            )

        return traces

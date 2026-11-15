"""Stage 1: the evidence-free prior distribution P_prior(y | text_claim) (Section 4.2).

The prior is elicited with the same single label-token logprob call as the posterior, so the
two distributions differ only in the evidence they see. The prompt must instruct the model to
answer with exactly one digit; the draft's per-domain packs (`examples/beauty/*.py`,
`examples/healthcare/*.py`) are the reference prompts, supplied here as the
`prompt_template` of the strategy. Only `text_claim` is available at this stage, so a template
must not reference the regulatory context.
"""

from __future__ import annotations

from mmregcheck.pillar2.base import BasePriorStrategy
from mmregcheck.pillar2.llm import ChatClient, label_distribution
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.schemas import ComplianceVerdict


class LogprobPriorStrategy(BasePriorStrategy):
    """Compute P_prior from one label-token logprob call over `state.text_claim`.

    The returned mapping is keyed by `ComplianceVerdict` and covers all three members. A label
    token absent from the top-K candidates gets probability 0.0, matching the draft; if none
    of the three label tokens is returned, elicitation fails rather than fabricating a uniform
    prior (the draft's silent 1/3 fallback lived further downstream, in scoring).
    """

    def compute_prior(
        self,
        state: TFBearState,
        client: ChatClient,
        model: str,
    ) -> dict[ComplianceVerdict, float]:
        """Return the verdict distribution using only `state.text_claim`."""
        prompt = state.render(self.prompt_template)
        distribution = label_distribution(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=self.max_retry,
        )
        if self.verbose:
            summary = ", ".join(
                f"{verdict.value}={prob:.4f}" for verdict, prob in distribution.items()
            )
            self.logger.log_info(f"P_prior: {summary}", style="dim")
        return distribution

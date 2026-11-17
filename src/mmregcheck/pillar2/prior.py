"""Stage 1: the evidence-free prior distribution P_prior(y | text_claim).

The prior is elicited with the same single label-token logprob call as the posterior, so the two
distributions differ only in the evidence they see. The prompt must instruct the model to answer
with exactly one digit; the per-domain packs in `mmregcheck.pillar2.prompts` provide such prompts
and are supplied as the `prompt_template` of the strategy. Only `text_claim` is available at
this stage, so a template must not reference the regulatory context.

The elicitation sampling is caller-controlled: `temperature` (default 0.0), `top_p` (default
0.95) and `top_logprobs` (default 5) are constructor arguments.
"""

from __future__ import annotations

from typing import Any

from mmregcheck.pillar2.base import BasePriorStrategy
from mmregcheck.pillar2.llm import ChatClient, label_distribution
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.schemas import ComplianceVerdict


class LogprobPriorStrategy(BasePriorStrategy):
    """Compute P_prior from one label-token logprob call over `state.text_claim`.

    The returned mapping is keyed by `ComplianceVerdict` and covers all three members. A label
    token absent from the top-K candidates gets probability 0.0, because a token the endpoint
    did not return carries no evidence of being the answer; the present labels are renormalised
    into a proper distribution and the absence is recorded on the run's diagnostics. If none of
    the three label tokens is returned, elicitation fails rather than fabricating a uniform
    prior, so the downstream Delta never rests on an invented number.

    Args:
        prompt_template: Prior prompt; may reference `{text_claim}` only.
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
            temperature=self.temperature,
            top_p=self.top_p,
            top_logprobs=self.top_logprobs,
            diagnostics=state.diagnostics,
            stage=type(self).__name__,
            extra_body=self.extra_body,
        )
        if self.verbose:
            summary = ", ".join(
                f"{verdict.value}={prob:.4f}" for verdict, prob in distribution.items()
            )
            self.logger.log_info(f"P_prior: {summary}", style="dim")
        return distribution

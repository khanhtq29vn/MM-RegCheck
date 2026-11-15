"""Typed container for one per-domain prompt pack and its strategy wiring.

A `PromptPack` holds the prompt templates the draft assembled in each
`examples/{beauty,healthcare}/*.py` module (prior, expansion, scoring, explanation and, for
tree-of-thoughts, the extra state-evaluation prompt) plus the hyperparameters the draft passed
to the expansion strategy. Packs are immutable and validate every template at construction, so
a placeholder the pipeline cannot fill fails at import instead of midway through a run.

`PromptPack.to_config` follows `mmregcheck.pillar2.config`: it builds the matching built-in
strategies and lets `TFBearConfig` own endpoint selection, thresholds and the propagation of
`verbose`/`max_retry`/`reasoning`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mmregcheck.pillar2.base import (
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.clients import EndpointConfig
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.expansion import (
    CoTSCExpansionStrategy,
    StandardExpansionStrategy,
    ToTExpansionStrategy,
)
from mmregcheck.pillar2.explainer import GenerativeExplanationStrategy
from mmregcheck.pillar2.pipeline_state import validate_template
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy
from mmregcheck.schemas import RegulatoryDomain

#: Supported expansion variants, matching the draft's six example files.
ExpansionName = Literal["standard", "cot-sc", "tot"]

#: Ordered tuple of the supported expansion variants (for CLI validation and tests).
EXPANSIONS: tuple[ExpansionName, ...] = ("standard", "cot-sc", "tot")


@dataclass(frozen=True, slots=True)
class PromptPack:
    """The prompts and expansion hyperparameters of one domain/expansion pair.

    Attributes:
        domain: Target regulatory domain of the pack.
        expansion: Expansion variant the pack belongs to.
        prior: Stage 1 template. May reference `{text_claim}` only.
        expansion_prompt: Stage 2 template. Standard/CoT-SC reference `{text_claim}` and
            `{retrieved_regulatory_context}`; ToT also references `{state}`.
        scoring: Stage 3 template. References claim, context and `{state}`.
        explanation: Stage 5 template. References claim, context, `{verdict}` and `{paths}`.
        tot_eval: State-evaluation template of the ToT engine; only set for `"tot"`.
        n_paths: Completions requested by the standard and CoT-SC strategies.
        expansion_max_tokens: Token budget of one expansion completion.
        expansion_temperature: Sampling temperature of the expansion completions.
        tot_max_tokens_thought_gen: Token budget of each ToT thought-generation call.
        tot_n_steps: Maximum BFS depth of the ToT search.
        tot_k: Candidate thoughts requested per expanded ToT node.
        tot_n_evals: Independent state evaluations per ToT branch.
        tot_breadth_limit: Top-`b` nodes retained after each ToT level.
        tot_min_steps: BFS levels expanded before honouring `terminate=true`.
    """

    domain: RegulatoryDomain
    expansion: ExpansionName
    prior: str
    expansion_prompt: str
    scoring: str
    explanation: str
    tot_eval: str | None = None

    n_paths: int = 3
    expansion_max_tokens: int = 512
    expansion_temperature: float = 0.7

    tot_max_tokens_thought_gen: int = 1024
    tot_n_steps: int = 3
    tot_k: int = 3
    tot_n_evals: int = 2
    tot_breadth_limit: int = 3
    tot_min_steps: int = 2

    def __post_init__(self) -> None:
        if self.expansion not in EXPANSIONS:
            raise ValueError(
                f"PromptPack: unknown expansion {self.expansion!r}; "
                f"expected one of {list(EXPANSIONS)!r}."
            )
        label = f"{self.domain.value}/{self.expansion}"
        templates = (
            ("prior", self.prior),
            ("expansion", self.expansion_prompt),
            ("scoring", self.scoring),
            ("explanation", self.explanation),
        )
        for name, template in templates:
            validate_template(template, f"PromptPack[{label}].{name}")
        if self.tot_eval is not None:
            validate_template(self.tot_eval, f"PromptPack[{label}].tot_eval")

        if self.expansion == "tot" and self.tot_eval is None:
            raise ValueError("PromptPack: the 'tot' expansion requires tot_eval.")
        if self.expansion != "tot" and self.tot_eval is not None:
            raise ValueError("PromptPack: tot_eval is only valid for the 'tot' expansion.")
        if self.n_paths < 1:
            raise ValueError(f"PromptPack: n_paths must be >= 1, got {self.n_paths}.")
        if self.expansion_max_tokens < 1:
            raise ValueError(
                f"PromptPack: expansion_max_tokens must be >= 1, got {self.expansion_max_tokens}."
            )
        for name, value in (
            ("tot_max_tokens_thought_gen", self.tot_max_tokens_thought_gen),
            ("tot_n_steps", self.tot_n_steps),
            ("tot_k", self.tot_k),
            ("tot_n_evals", self.tot_n_evals),
            ("tot_breadth_limit", self.tot_breadth_limit),
            ("tot_min_steps", self.tot_min_steps),
        ):
            if value < 1:
                raise ValueError(f"PromptPack: {name} must be >= 1, got {value}.")

    def expansion_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments for the expansion strategy of this pack.

        Standard and CoT-SC share the `(n_paths, max_tokens, temperature)` shape; the ToT
        strategy additionally takes the evaluation template and the search parameters. The
        draft used the same values for the three domains.
        """
        if self.expansion == "tot":
            return {
                "prompt_template": self.expansion_prompt,
                "eval_prompt_template": self.tot_eval,
                "n_steps": self.tot_n_steps,
                "k": self.tot_k,
                "n_evals": self.tot_n_evals,
                "breadth_limit": self.tot_breadth_limit,
                "max_tokens_thought_gen": self.tot_max_tokens_thought_gen,
                "min_steps": self.tot_min_steps,
            }
        return {
            "prompt_template": self.expansion_prompt,
            "n_paths": self.n_paths,
            "max_tokens": self.expansion_max_tokens,
            "temperature": self.expansion_temperature,
        }

    def build_strategies(
        self,
        *,
        explanation_max_tokens: int = 256,
    ) -> tuple[
        BasePriorStrategy,
        BaseExpansionStrategy,
        BaseScoringStrategy,
        BaseExplanationStrategy,
    ]:
        """Build fresh stage strategies wired to this pack's prompts.

        The draft left the explanation strategy disabled; the pack builds it so a caller gets a
        complete pipeline. Pass `explanation_max_tokens` to match a different budget (the draft
        comment used 256).
        """
        prior = LogprobPriorStrategy(prompt_template=self.prior)
        expansion_type = {
            "standard": StandardExpansionStrategy,
            "cot-sc": CoTSCExpansionStrategy,
            "tot": ToTExpansionStrategy,
        }[self.expansion]
        expansion = expansion_type(**self.expansion_kwargs())
        scoring = CausalDeltaScoringStrategy(prompt_template=self.scoring)
        explanation = GenerativeExplanationStrategy(
            prompt_template=self.explanation,
            max_tokens=explanation_max_tokens,
        )
        return prior, expansion, scoring, explanation

    def to_config(
        self,
        *,
        endpoint: EndpointConfig | None = None,
        stage_endpoints: dict[str, EndpointConfig] | None = None,
        tau_posterior: float = 0.50,
        tau_delta: float = 0.50,
        verbose: bool = False,
        max_retry: int = 3,
        reasoning: dict[str, Any] | None = None,
        explanation_max_tokens: int = 256,
    ) -> TFBearConfig:
        """Build a `TFBearConfig` that reproduces the draft's pipeline for this pack."""
        prior, expansion, scoring, explanation = self.build_strategies(
            explanation_max_tokens=explanation_max_tokens
        )
        return TFBearConfig(
            endpoint=endpoint,
            stage_endpoints=dict(stage_endpoints or {}),
            prior_strategy=prior,
            expansion_strategy=expansion,
            scoring_strategy=scoring,
            explanation_strategy=explanation,
            tau_posterior=tau_posterior,
            tau_delta=tau_delta,
            verbose=verbose,
            max_retry=max_retry,
            reasoning=reasoning if reasoning is not None else {"enabled": False},
        )

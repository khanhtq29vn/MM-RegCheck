"""Typed container for one per-domain prompt pack and its strategy wiring.

A `PromptPack` holds the prompt templates needed by one domain/expansion pair (prior, expansion,
scoring, explanation and, for tree-of-thoughts, the extra state-evaluation prompt) plus the
hyperparameters of its expansion strategy. Packs are immutable and validate every template at
construction, so a placeholder the pipeline cannot fill fails at import instead of midway through
a run.

`PromptPack.to_config` follows `mmregcheck.pillar2.config`: it builds the matching built-in
strategies and lets `TFBearConfig` own endpoint selection, thresholds and the propagation of
`verbose`/`max_retry`/`reasoning`. Expansion-specific parameters can be overridden at run time
through `expansion_params`; a key the chosen strategy does not accept raises `ValueError` instead
of being ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mmregcheck.pillar2.adjudication import TFBearAdjudicationStrategy
from mmregcheck.pillar2.base import (
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.expansion import (
    CoTSCExpansionStrategy,
    StandardExpansionStrategy,
    ToTExpansionStrategy,
)
from mmregcheck.pillar2.explainer import GenerativeExplanationStrategy
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.pipeline_state import validate_template
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy
from mmregcheck.schemas import RegulatoryDomain

#: Supported expansion variants; every domain ships a pack for each of them.
ExpansionName = Literal["standard", "cot-sc", "tot"]

#: Ordered tuple of the supported expansion variants (for CLI validation and tests).
EXPANSIONS: tuple[ExpansionName, ...] = ("standard", "cot-sc", "tot")

#: Run-time expansion parameters accepted by each built-in expansion strategy.
_EXPANSION_PARAM_KEYS: dict[type[BaseExpansionStrategy], frozenset[str]] = {
    StandardExpansionStrategy: frozenset({"n_paths", "max_tokens", "temperature", "top_p"}),
    CoTSCExpansionStrategy: frozenset({"n_paths", "max_tokens", "temperature", "top_p"}),
    ToTExpansionStrategy: frozenset(
        {
            "n_steps",
            "k",
            "n_evals",
            "breadth_limit",
            "max_tokens_thought_gen",
            "min_steps",
            "thought_temperature",
            "eval_temperature",
            "thought_top_p",
            "eval_top_p",
        }
    ),
}


def apply_expansion_params(
    strategy: BaseExpansionStrategy,
    params: Mapping[str, Any] | None,
) -> None:
    """Apply run-time expansion parameters to an already constructed strategy.

    The parameters are the strategy's own tuning knobs, not stage protocol: for example
    `{"breadth_limit": 2}` narrows a tree search. They are applied after `TFBearConfig` has
    propagated its shared sampling settings, so an explicit expansion parameter always wins.

    Args:
        strategy: Expansion strategy built by a pack factory.
        params: Parameter names and values; `None` or an empty mapping changes nothing.

    Raises:
        ValueError: If the strategy is not a built-in with documented parameters, or if `params`
            contains a key the strategy does not accept.
    """
    if not params:
        return
    accepted = _EXPANSION_PARAM_KEYS.get(type(strategy))
    if accepted is None:
        raise ValueError(
            f"{type(strategy).__name__} does not accept expansion_params; "
            "pass the parameters to the strategy constructor instead."
        )
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ValueError(
            f"expansion_params has unknown key(s) {unknown!r} for "
            f"{type(strategy).__name__}; accepted keys: {sorted(accepted)!r}."
        )
    for key, value in params.items():
        setattr(strategy, key, value)


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
        expansion_max_tokens: Token budget of one standard/CoT-SC expansion completion.
        expansion_temperature: Sampling temperature of the standard/CoT-SC expansions.
        tot_max_tokens_thought_gen: Token budget of each ToT thought-generation call.
        tot_temperature: Sampling temperature of ToT thought generation.
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
    expansion_max_tokens: int = 1024
    expansion_temperature: float = 0.2

    tot_max_tokens_thought_gen: int = 1024
    tot_temperature: float = 0.2
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
        if not self.expansion_temperature >= 0.0:
            raise ValueError(
                "PromptPack: expansion_temperature must be non-negative, "
                f"got {self.expansion_temperature}."
            )
        if not self.tot_temperature >= 0.0:
            raise ValueError(
                f"PromptPack: tot_temperature must be non-negative, got {self.tot_temperature}."
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
        values belong to the pack, so each domain can tune its own expansion.
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
                "thought_temperature": self.tot_temperature,
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

        The explanation strategy is built as well, so a pack always yields a config whose final
        stage is wired like the others. Pass `explanation_max_tokens` to change the budget of
        the explanation call.
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
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        elicitation_temperature: float = 0.0,
        elicitation_top_p: float = 0.95,
        top_logprobs: int = 5,
        stage_temperature: Mapping[str, float] | None = None,
        stage_top_p: Mapping[str, float] | None = None,
        stage_max_tokens: Mapping[str, int] | None = None,
        stage_top_logprobs: Mapping[str, int] | None = None,
        expansion_params: Mapping[str, Any] | None = None,
    ) -> TFBearConfig:
        """Build a `TFBearConfig` wired to this pack's prompts and hyperparameters.

        The pack's explanation budget is preserved as the explanation-stage override, so it is
        not replaced by the shared generation budget. Expansion sampling comes from the shared
        and per-stage settings; `expansion_params` can then refine the chosen strategy.

        Args:
            endpoint: Default endpoint of every stage.
            stage_endpoints: Optional per-stage endpoint overrides.
            tau_posterior: Posterior threshold of the dual gate.
            tau_delta: Delta threshold of the dual gate.
            verbose: Enables verbose stage logging.
            max_retry: Retry budget of the LLM calls.
            reasoning: Provider-specific reasoning payload.
            explanation_max_tokens: Token budget of the explanation call.
            temperature: Shared generation temperature; defaults to 0.2.
            top_p: Shared generation nucleus threshold; defaults to 0.95.
            max_tokens: Shared generation token budget; defaults to 1024.
            elicitation_temperature: Shared elicitation temperature; defaults to 0.0.
            elicitation_top_p: Shared elicitation nucleus threshold; defaults to 0.95.
            top_logprobs: Candidate cut of the elicitation calls; defaults to 5.
            stage_temperature: Per-stage temperature overrides.
            stage_top_p: Per-stage top_p overrides.
            stage_max_tokens: Per-stage generation budget overrides.
            stage_top_logprobs: Per-stage elicitation candidate-cut overrides.
            expansion_params: Run-time parameters of the chosen expansion strategy; unknown
                keys raise `ValueError`.
        """
        prior, expansion, scoring, explanation = self.build_strategies(
            explanation_max_tokens=explanation_max_tokens
        )

        pack_stage_max_tokens: dict[str, int] = {"explanation": explanation_max_tokens}

        config = TFBearConfig(
            prior_strategy=prior,
            expansion_strategy=expansion,
            scoring_strategy=scoring,
            adjudication_strategy=TFBearAdjudicationStrategy(
                tau_posterior=tau_posterior,
                tau_delta=tau_delta,
            ),
            explanation_strategy=explanation,
            endpoint=endpoint,
            stage_endpoints=dict(stage_endpoints or {}),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            elicitation_temperature=elicitation_temperature,
            elicitation_top_p=elicitation_top_p,
            top_logprobs=top_logprobs,
            stage_temperature=dict(stage_temperature or {}),
            stage_top_p=dict(stage_top_p or {}),
            stage_max_tokens={**pack_stage_max_tokens, **(stage_max_tokens or {})},
            stage_top_logprobs=dict(stage_top_logprobs or {}),
            tau_posterior=tau_posterior,
            tau_delta=tau_delta,
            verbose=verbose,
            max_retry=max_retry,
            reasoning=reasoning if reasoning is not None else {"enabled": False},
        )
        apply_expansion_params(config.expansion_strategy, expansion_params)
        return config

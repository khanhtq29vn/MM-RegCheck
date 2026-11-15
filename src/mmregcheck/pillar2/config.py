"""Configuration of the TF-BEAR pipeline.

`TFBearConfig` wires one injectable strategy per stage and holds the endpoint, thresholds and
global retry/verbosity settings. It is also the single place that knows how to reach the LLM
endpoint(s).

Every stage defaults to the built-in strategy, so a caller that only wants the stock pipeline
can write `TFBearConfig(endpoint=EndpointConfig(...))`. Custom strategies must subclass the
matching `Base*Strategy`; a wrong object is rejected at construction instead of failing halfway
through a run. The explanation strategy is imported lazily because it pulls the optional LLM
path and is developed independently of the other stages.

Prompts are caller data. The defaults below are domain-neutral English prompts built on the
strict hierarchy (one Non-Compliant technique makes the advertisement Non-Compliant, otherwise
one Partly-Compliant technique makes it Partly-Compliant, otherwise it is Compliant) and the
`0/1/2` answer convention (0 = Non-Compliant, 1 = Partly-Compliant, 2 = Compliant). The
per-domain packs in `mmregcheck.pillar2.prompts` are the reference for benchmark runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmregcheck.pillar2.adjudication import TFBearAdjudicationStrategy
from mmregcheck.pillar2.base import (
    BaseAdjudicationStrategy,
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.expansion import StandardExpansionStrategy
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.logger import TFBearLogger
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy

#: Stages that may have their own endpoint override.
STAGES: tuple[str, ...] = ("prior", "expansion", "scoring", "adjudication", "explanation")

DEFAULT_PRIOR_PROMPT = """\
You are a regulatory compliance classifier for medical and aesthetic advertising.

[Techniques]: {text_claim}

Apply the strict hierarchy: one Non-Compliant technique makes the advertisement
Non-Compliant; otherwise one Partly-Compliant technique makes it Partly-Compliant; otherwise
it is Compliant.

Answer with exactly one digit and nothing else:
0 = Non-Compliant, 1 = Partly-Compliant, 2 = Compliant.
Answer:"""

DEFAULT_EXPANSION_PROMPT = """\
You are a regulatory compliance auditor. Reason step by step about whether the advertised
techniques are permitted by the facility's registered scope.

[Registered scope]: {retrieved_regulatory_context}
[Techniques]: {text_claim}

Map each technique to the relevant rule, check the facility's registered category, and end
with a conclusion that names the verdict suggested by the strict hierarchy.
"""

DEFAULT_SCORING_PROMPT = """\
You are a regulatory compliance auditor. Judge the advertisement using only the claim, the
registered scope and the reasoning path below.

[Techniques]: {text_claim}
[Registered scope]: {retrieved_regulatory_context}
[Reasoning path]: {state}

Apply the strict hierarchy: one Non-Compliant technique => 0; otherwise one Partly-Compliant
technique => 1; otherwise Compliant => 2.

Answer with exactly one digit and nothing else (0, 1 or 2):"""


def _default_explanation_strategy() -> BaseExplanationStrategy:
    """Build the generative explanation strategy, importing it only when needed."""
    from importlib import import_module

    module = import_module("mmregcheck.pillar2.explainer")
    strategy_type = getattr(module, "GenerativeExplanationStrategy", None)
    if not isinstance(strategy_type, type):
        raise ImportError(
            "The default explanation strategy is unavailable: "
            "mmregcheck.pillar2.explainer.GenerativeExplanationStrategy is missing. "
            "Pass `explanation_strategy` explicitly."
        )
    strategy = strategy_type()
    if not isinstance(strategy, BaseExplanationStrategy):
        raise TypeError(
            "mmregcheck.pillar2.explainer.GenerativeExplanationStrategy must subclass "
            "BaseExplanationStrategy."
        )
    return strategy


@dataclass
class TFBearConfig:
    """Strategy wiring, endpoint selection and global settings of one TF-BEAR run.

    Attributes:
        endpoint: Default OpenAI-compatible endpoint used by every stage.
        stage_endpoints: Optional per-stage overrides, keyed by a name from `STAGES`.
        prior_strategy: Stage 1; defaults to `LogprobPriorStrategy` with a generic prompt.
        expansion_strategy: Stage 2; defaults to `StandardExpansionStrategy`.
        scoring_strategy: Stage 3; defaults to `CausalDeltaScoringStrategy`.
        adjudication_strategy: Stage 4; defaults to `TFBearAdjudicationStrategy` using the
            thresholds below.
        explanation_strategy: Stage 5; defaults to `GenerativeExplanationStrategy` from
            `mmregcheck.pillar2.explainer` (imported lazily).
        tau_posterior: Minimum posterior at the branch verdict for the dual-threshold gate.
        tau_delta: Minimum log-odds Delta for the dual-threshold gate.
        verbose: Enables stage and strategy logging. Propagated to every strategy.
        max_retry: Retry budget of the LLM calls. Propagated to every strategy.
        reasoning: Provider-specific reasoning payload. Propagated to every strategy.
    """

    endpoint: EndpointConfig | None = None
    stage_endpoints: dict[str, EndpointConfig] = field(default_factory=dict)

    prior_strategy: BasePriorStrategy | None = None
    expansion_strategy: BaseExpansionStrategy | None = None
    scoring_strategy: BaseScoringStrategy | None = None
    adjudication_strategy: BaseAdjudicationStrategy | None = None
    explanation_strategy: BaseExplanationStrategy | None = None

    tau_posterior: float = 0.50
    tau_delta: float = 0.50
    verbose: bool = False
    max_retry: int = 3
    reasoning: dict[str, Any] = field(default_factory=lambda: {"enabled": False})

    logger: TFBearLogger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_posterior <= 1.0:
            raise ValueError(f"tau_posterior must be in [0, 1], got {self.tau_posterior}.")
        if not self.tau_delta >= 0.0:
            raise ValueError(f"tau_delta must be non-negative, got {self.tau_delta}.")
        if self.max_retry < 1:
            raise ValueError(f"max_retry must be at least 1, got {self.max_retry}.")
        unknown_stages = sorted(set(self.stage_endpoints) - set(STAGES))
        if unknown_stages:
            raise ValueError(
                f"stage_endpoints has unknown stage(s) {unknown_stages!r}; "
                f"expected a subset of {list(STAGES)!r}."
            )

        self.reasoning = dict(self.reasoning)
        self._resolve_defaults()
        self._validate_strategy_types()

        self.logger = TFBearLogger(verbose=self.verbose)
        for strategy in self.strategies():
            strategy.verbose = self.verbose
            strategy.logger = self.logger
            strategy.max_retry = self.max_retry
            strategy.reasoning = self.reasoning

    def _resolve_defaults(self) -> None:
        if self.prior_strategy is None:
            self.prior_strategy = LogprobPriorStrategy(prompt_template=DEFAULT_PRIOR_PROMPT)
        if self.expansion_strategy is None:
            self.expansion_strategy = StandardExpansionStrategy(
                prompt_template=DEFAULT_EXPANSION_PROMPT
            )
        if self.scoring_strategy is None:
            self.scoring_strategy = CausalDeltaScoringStrategy(
                prompt_template=DEFAULT_SCORING_PROMPT
            )
        if self.adjudication_strategy is None:
            self.adjudication_strategy = TFBearAdjudicationStrategy(
                tau_posterior=self.tau_posterior,
                tau_delta=self.tau_delta,
            )
        if self.explanation_strategy is None:
            self.explanation_strategy = _default_explanation_strategy()

    def _validate_strategy_types(self) -> None:
        checks: tuple[tuple[str, type[Any]], ...] = (
            ("prior_strategy", BasePriorStrategy),
            ("expansion_strategy", BaseExpansionStrategy),
            ("scoring_strategy", BaseScoringStrategy),
            ("adjudication_strategy", BaseAdjudicationStrategy),
            ("explanation_strategy", BaseExplanationStrategy),
        )
        for field_name, base in checks:
            strategy = getattr(self, field_name)
            if not isinstance(strategy, base):
                raise TypeError(
                    f"TFBearConfig.{field_name} must be a {base.__name__}, "
                    f"got {type(strategy).__name__}."
                )

    def strategies(
        self,
    ) -> tuple[
        BasePriorStrategy,
        BaseExpansionStrategy,
        BaseScoringStrategy,
        BaseAdjudicationStrategy,
        BaseExplanationStrategy,
    ]:
        """Return the five resolved strategies, or fail if the config was left unresolved."""
        prior = self.prior_strategy
        expansion = self.expansion_strategy
        scoring = self.scoring_strategy
        adjudication = self.adjudication_strategy
        explanation = self.explanation_strategy
        if (
            prior is None
            or expansion is None
            or scoring is None
            or adjudication is None
            or explanation is None
        ):
            raise RuntimeError("TFBearConfig strategies were not resolved.")
        return prior, expansion, scoring, adjudication, explanation

    def endpoint_for(self, stage: str) -> EndpointConfig:
        """Return the stage override if present, otherwise the default endpoint.

        Raises:
            ValueError: If `stage` is unknown or no endpoint is configured for it.
        """
        if stage not in STAGES:
            raise ValueError(f"Unknown TF-BEAR stage {stage!r}; expected one of {list(STAGES)!r}.")
        override = self.stage_endpoints.get(stage)
        if override is not None:
            return override
        if self.endpoint is None:
            raise ValueError(
                f"TFBearConfig has no endpoint for stage {stage!r}; set `endpoint` or "
                f"`stage_endpoints[{stage!r}]`."
            )
        return self.endpoint

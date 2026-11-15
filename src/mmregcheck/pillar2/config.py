"""Configuration of the TF-BEAR pipeline.

`TFBearConfig` wires one injectable strategy per stage, holds the endpoint, the shared sampling
settings, the gate thresholds and the global retry/verbosity settings. It is also the single
place that knows how to reach the LLM endpoint(s).

The five stage strategies are required arguments: there is no implicit pipeline, because a
different default would silently run a different experiment. `TFBear.from_domain` is the main
entry point for benchmark runs; it builds the matching per-domain prompt pack and then hands the
strategies to this class.

Custom strategies must subclass the matching `Base*Strategy`; a wrong object is rejected at
construction instead of failing halfway through a run.

Sampling settings are shared per call family and can be overridden per stage:

    generation     expansion and explanation   temperature, top_p, max_tokens
    elicitation    prior, scoring, adjudication temperature, top_p, top_logprobs

The shared values apply to the strategies that expose the matching attribute; a custom strategy
that does not expose one keeps its own behaviour. The per-stage dictionaries take precedence over
the shared values. `top_logprobs` is the size of the candidate cut used to build a label
distribution; labels outside that cut receive probability 0.0, which the run records as a
diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmregcheck.pillar2.base import (
    BaseAdjudicationStrategy,
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.expansion import ToTExpansionStrategy
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.logger import TFBearLogger

#: Stages that may have their own endpoint override.
STAGES: tuple[str, ...] = ("prior", "expansion", "scoring", "adjudication", "explanation")

#: Stages that generate free text and therefore accept temperature, top_p and max_tokens.
GENERATION_STAGES: tuple[str, ...] = ("expansion", "explanation")

#: Stages that elicit a label distribution and therefore accept temperature, top_p and
#: top_logprobs. The default adjudication strategy makes no call, but the joint strategy does.
ELICITATION_STAGES: tuple[str, ...] = ("prior", "scoring", "adjudication")


@dataclass
class TFBearConfig:
    """Strategy wiring, endpoint selection, sampling settings and thresholds of one run.

    Attributes:
        prior_strategy: Stage 1; computes the evidence-free prior distribution.
        expansion_strategy: Stage 2; generates reasoning paths.
        scoring_strategy: Stage 3; computes the per-path posterior and Delta.
        adjudication_strategy: Stage 4; applies the dual-threshold gate and vote.
        explanation_strategy: Stage 5; writes the rationale of the final verdict.
        endpoint: Default OpenAI-compatible endpoint used by every stage.
        stage_endpoints: Optional per-stage endpoint overrides, keyed by a name from `STAGES`.
        temperature: Shared sampling temperature of the generation stages; defaults to 0.2.
        top_p: Shared nucleus threshold of the generation stages; defaults to 0.95.
        max_tokens: Shared completion budget of the generation stages; defaults to 1024.
        elicitation_temperature: Shared temperature of the label-token elicitation stages;
            defaults to 0.0 (greedy).
        elicitation_top_p: Shared nucleus threshold of the elicitation stages; defaults to 0.95.
        top_logprobs: Number of candidate tokens requested by the elicitation stages; defaults
            to 5.
        stage_temperature: Per-stage temperature overrides for any stage in `STAGES`.
        stage_top_p: Per-stage top_p overrides for any stage in `STAGES`.
        stage_max_tokens: Per-stage completion budget for `GENERATION_STAGES`.
        stage_top_logprobs: Per-stage candidate cut for `ELICITATION_STAGES`.
        tau_posterior: Minimum posterior at the branch verdict for the dual-threshold gate.
        tau_delta: Minimum log-odds Delta for the dual-threshold gate.
        verbose: Enables stage and strategy logging. Propagated to every strategy.
        max_retry: Retry budget of the LLM calls. Propagated to every strategy.
        reasoning: Provider-specific reasoning payload. Propagated to every strategy.
    """

    prior_strategy: BasePriorStrategy
    expansion_strategy: BaseExpansionStrategy
    scoring_strategy: BaseScoringStrategy
    adjudication_strategy: BaseAdjudicationStrategy
    explanation_strategy: BaseExplanationStrategy

    endpoint: EndpointConfig | None = None
    stage_endpoints: dict[str, EndpointConfig] = field(default_factory=dict)

    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 1024

    elicitation_temperature: float = 0.0
    elicitation_top_p: float = 0.95
    top_logprobs: int = 5

    stage_temperature: dict[str, float] = field(default_factory=dict)
    stage_top_p: dict[str, float] = field(default_factory=dict)
    stage_max_tokens: dict[str, int] = field(default_factory=dict)
    stage_top_logprobs: dict[str, int] = field(default_factory=dict)

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
        self._validate_sampling()
        unknown_stages = sorted(set(self.stage_endpoints) - set(STAGES))
        if unknown_stages:
            raise ValueError(
                f"stage_endpoints has unknown stage(s) {unknown_stages!r}; "
                f"expected a subset of {list(STAGES)!r}."
            )

        self.reasoning = dict(self.reasoning)
        self._validate_strategy_types()
        self._apply_thresholds()
        self._apply_sampling_settings()

        self.logger = TFBearLogger(verbose=self.verbose)
        for strategy in self.strategies():
            strategy.verbose = self.verbose
            strategy.logger = self.logger
            strategy.max_retry = self.max_retry
            strategy.reasoning = self.reasoning

    def _validate_sampling(self) -> None:
        for name, value in (
            ("temperature", self.temperature),
            ("elicitation_temperature", self.elicitation_temperature),
        ):
            if not value >= 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        for name, value in (
            ("top_p", self.top_p),
            ("elicitation_top_p", self.elicitation_top_p),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("top_logprobs", self.top_logprobs),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}.")

        self._validate_stage_keys("stage_temperature", self.stage_temperature, STAGES)
        self._validate_stage_keys("stage_top_p", self.stage_top_p, STAGES)
        self._validate_stage_keys("stage_max_tokens", self.stage_max_tokens, GENERATION_STAGES)
        self._validate_stage_keys(
            "stage_top_logprobs", self.stage_top_logprobs, ELICITATION_STAGES
        )
        for name, values in (
            ("stage_temperature", self.stage_temperature),
            ("stage_top_p", self.stage_top_p),
            ("stage_max_tokens", self.stage_max_tokens),
            ("stage_top_logprobs", self.stage_top_logprobs),
        ):
            for stage, value in values.items():
                if name in ("stage_max_tokens", "stage_top_logprobs"):
                    if value < 1:
                        raise ValueError(f"{name}[{stage!r}] must be at least 1, got {value}.")
                elif name == "stage_top_p":
                    if not 0.0 < value <= 1.0:
                        raise ValueError(f"{name}[{stage!r}] must be in (0, 1], got {value}.")
                elif not value >= 0.0:
                    raise ValueError(f"{name}[{stage!r}] must be non-negative, got {value}.")

    @staticmethod
    def _validate_stage_keys(name: str, values: dict[str, Any], allowed: tuple[str, ...]) -> None:
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise ValueError(
                f"{name} has unknown stage(s) {unknown!r}; expected a subset of {list(allowed)!r}."
            )

    def _apply_thresholds(self) -> None:
        """Copy the configured gate thresholds onto adjudication strategies that expose them."""
        if hasattr(self.adjudication_strategy, "tau_posterior"):
            self.adjudication_strategy.tau_posterior = self.tau_posterior
        if hasattr(self.adjudication_strategy, "tau_delta"):
            self.adjudication_strategy.tau_delta = self.tau_delta

    def _apply_sampling_settings(self) -> None:
        """Copy the resolved sampling settings onto the strategies that expose them.

        The per-stage dictionaries win over the shared values. Expansion strategies with separate
        thought-generation and state-evaluation calls (tree search) receive the resolved
        expansion temperature and top_p for both calls; `expansion_params` can still refine them
        afterwards through the pack factory.
        """
        for stage, strategy in zip(STAGES, self.strategies(), strict=True):
            generation = stage in GENERATION_STAGES
            temperature = self.stage_temperature.get(
                stage, self.temperature if generation else self.elicitation_temperature
            )
            top_p = self.stage_top_p.get(
                stage, self.top_p if generation else self.elicitation_top_p
            )

            if isinstance(strategy, ToTExpansionStrategy):
                strategy.thought_temperature = temperature
                strategy.eval_temperature = temperature
                strategy.thought_top_p = top_p
                strategy.eval_top_p = top_p
            else:
                if hasattr(strategy, "temperature"):
                    strategy.temperature = temperature
                if hasattr(strategy, "top_p"):
                    strategy.top_p = top_p

            if generation:
                max_tokens = self.stage_max_tokens.get(stage, self.max_tokens)
                if hasattr(strategy, "max_tokens_thought_gen"):
                    strategy.max_tokens_thought_gen = max_tokens
                elif hasattr(strategy, "max_tokens"):
                    strategy.max_tokens = max_tokens

            if stage in ELICITATION_STAGES:
                top_logprobs = self.stage_top_logprobs.get(stage, self.top_logprobs)
                if hasattr(strategy, "top_logprobs"):
                    strategy.top_logprobs = top_logprobs

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
        """Return the five resolved strategies in pipeline order."""
        return (
            self.prior_strategy,
            self.expansion_strategy,
            self.scoring_strategy,
            self.adjudication_strategy,
            self.explanation_strategy,
        )

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

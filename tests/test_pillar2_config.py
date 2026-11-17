"""Tests for TF-BEAR configuration, sampling propagation and endpoint selection."""

from __future__ import annotations

from typing import Any

import pytest

from mmregcheck.pillar2.adjudication import TFBearAdjudicationStrategy
from mmregcheck.pillar2.base import (
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.expansion import (
    StandardExpansionStrategy,
    ToTExpansionStrategy,
)
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy


class _Prior(BasePriorStrategy):
    def compute_prior(self, state: TFBearState, client: Any, model: str) -> dict[Any, float]:
        return {}


class _Expansion(BaseExpansionStrategy):
    def expand(self, state: TFBearState, client: Any, model: str) -> list[Any]:
        return []


class _Scoring(BaseScoringStrategy):
    def score(self, state: TFBearState, client: Any, model: str) -> list[Any]:
        return []


class _Explanation(BaseExplanationStrategy):
    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        return "test explanation"


def _endpoint(model: str = "test-model") -> EndpointConfig:
    return EndpointConfig(
        base_url="https://endpoint.invalid/v1",
        env_var="MMREG_TEST_API_KEY",
        model=model,
    )


def _strategies() -> dict[str, Any]:
    return {
        "prior_strategy": _Prior(),
        "expansion_strategy": _Expansion(),
        "scoring_strategy": _Scoring(),
        "adjudication_strategy": TFBearAdjudicationStrategy(),
        "explanation_strategy": _Explanation(),
    }


def _config(**overrides: Any) -> TFBearConfig:
    values = _strategies()
    values.update(overrides)
    return TFBearConfig(**values)


def test_the_five_stage_strategies_are_required() -> None:
    with pytest.raises(TypeError):
        TFBearConfig()  # type: ignore[call-arg]

    values = _strategies()
    values.pop("explanation_strategy")
    with pytest.raises(TypeError):
        TFBearConfig(**values)  # type: ignore[arg-type]


def test_default_sampling_values_are_the_documented_ones() -> None:
    config = _config(endpoint=_endpoint())

    assert config.temperature == 0.2
    assert config.top_p == 0.95
    assert config.max_tokens == 1024
    assert config.elicitation_temperature == 0.0
    assert config.elicitation_top_p == 0.95
    assert config.top_logprobs == 5


def test_shared_sampling_settings_reach_the_built_in_strategies() -> None:
    prior = LogprobPriorStrategy()
    expansion = StandardExpansionStrategy()
    scoring = CausalDeltaScoringStrategy()
    config = _config(
        prior_strategy=prior,
        expansion_strategy=expansion,
        scoring_strategy=scoring,
        endpoint=_endpoint(),
    )

    assert (prior.temperature, prior.top_p, prior.top_logprobs) == (0.0, 0.95, 5)
    assert (scoring.temperature, scoring.top_p, scoring.top_logprobs) == (0.0, 0.95, 5)
    assert (expansion.temperature, expansion.top_p, expansion.max_tokens) == (0.2, 0.95, 1024)
    explanation = config.explanation_strategy
    assert isinstance(explanation, _Explanation)


def test_shared_generation_temperature_reaches_the_explanation_stage() -> None:
    from mmregcheck.pillar2.explainer import GenerativeExplanationStrategy

    explanation = GenerativeExplanationStrategy()
    _config(explanation_strategy=explanation)

    assert explanation.temperature == 0.2
    assert explanation.top_p == 0.95
    assert explanation.max_tokens == 1024


def test_toT_strategy_receives_the_expansion_sampling_settings() -> None:
    expansion = ToTExpansionStrategy()
    _config(expansion_strategy=expansion, temperature=0.6, top_p=0.8, max_tokens=512)

    assert expansion.thought_temperature == 0.6
    assert expansion.eval_temperature == 0.6
    assert expansion.thought_top_p == 0.8
    assert expansion.eval_top_p == 0.8
    assert expansion.max_tokens_thought_gen == 512


def test_per_stage_overrides_win_over_the_shared_values() -> None:
    prior = LogprobPriorStrategy()
    expansion = StandardExpansionStrategy()
    explanation = _Explanation()
    config = _config(
        prior_strategy=prior,
        expansion_strategy=expansion,
        explanation_strategy=explanation,
        temperature=0.2,
        top_p=0.95,
        max_tokens=1024,
        elicitation_temperature=0.0,
        elicitation_top_p=0.95,
        top_logprobs=5,
        stage_temperature={"prior": 0.4, "expansion": 0.8},
        stage_top_p={"prior": 0.5, "expansion": 0.6},
        stage_max_tokens={"expansion": 2048},
        stage_top_logprobs={"prior": 9},
    )

    assert config.temperature == 0.2
    assert (prior.temperature, prior.top_p, prior.top_logprobs) == (0.4, 0.5, 9)
    assert (expansion.temperature, expansion.top_p, expansion.max_tokens) == (0.8, 0.6, 2048)


def test_adjudication_thresholds_are_stored_on_the_strategy() -> None:
    config = _config(tau_posterior=0.8, tau_delta=0.9)

    adjudication = config.adjudication_strategy
    assert isinstance(adjudication, TFBearAdjudicationStrategy)
    assert adjudication.tau_posterior == 0.8
    assert adjudication.tau_delta == 0.9


def test_global_settings_are_propagated_to_every_strategy() -> None:
    explanation = _Explanation()
    config = _config(
        explanation_strategy=explanation,
        verbose=True,
        max_retry=7,
        reasoning={"enabled": True},
    )

    assert config.explanation_strategy is explanation
    assert explanation.verbose is True
    assert explanation.max_retry == 7
    assert explanation.reasoning == {"enabled": True}
    assert explanation.logger is config.logger
    assert config.logger.verbose is True


def test_injected_strategy_of_the_wrong_family_is_rejected() -> None:
    with pytest.raises(TypeError, match="prior_strategy"):
        _config(prior_strategy=_Explanation())


@pytest.mark.parametrize(
    ("field", "value"),
    [("tau_posterior", -0.1), ("tau_posterior", 1.1), ("tau_delta", -1.0)],
)
def test_posterior_threshold_is_a_probability_and_delta_is_non_negative(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_delta_threshold_may_exceed_one() -> None:
    config = _config(tau_delta=2.5)

    adjudication = config.adjudication_strategy
    assert isinstance(adjudication, TFBearAdjudicationStrategy)
    assert adjudication.tau_delta == 2.5


def test_max_retry_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_retry"):
        _config(max_retry=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("max_tokens", 0),
        ("elicitation_temperature", -0.1),
        ("elicitation_top_p", 0.0),
        ("top_logprobs", 0),
    ],
)
def test_sampling_values_are_validated(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("stage_temperature", {"selection": 0.2}),
        ("stage_top_p", {"selection": 0.5}),
        ("stage_max_tokens", {"prior": 10}),
        ("stage_top_logprobs", {"expansion": 10}),
        ("stage_temperature", {"expansion": -0.1}),
        ("stage_top_p", {"expansion": 0.0}),
        ("stage_max_tokens", {"expansion": 0}),
        ("stage_top_logprobs", {"prior": 0}),
    ],
)
def test_stage_sampling_overrides_are_validated(field: str, overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: overrides})


def test_unknown_stage_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        _config(stage_endpoints={"seleciton": _endpoint()})


def test_endpoint_for_prefers_the_stage_override() -> None:
    base = _endpoint("base-model")
    scoring = _endpoint("scoring-model")
    config = _config(endpoint=base, stage_endpoints={"scoring": scoring})

    assert config.endpoint_for("prior") is base
    assert config.endpoint_for("scoring") is scoring


def test_endpoint_for_requires_a_configured_endpoint() -> None:
    config = _config()

    with pytest.raises(ValueError, match="no endpoint"):
        config.endpoint_for("prior")
    with pytest.raises(ValueError, match="Unknown TF-BEAR stage"):
        config.endpoint_for("selection")

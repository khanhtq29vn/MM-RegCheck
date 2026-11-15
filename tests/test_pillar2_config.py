"""Tests for TF-BEAR configuration defaults, validation and endpoint selection."""

from __future__ import annotations

from typing import Any

import pytest

from mmregcheck.pillar2.adjudication import TFBearAdjudicationStrategy
from mmregcheck.pillar2.base import BaseExplanationStrategy
from mmregcheck.pillar2.clients import EndpointConfig
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.expansion import StandardExpansionStrategy
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy


class _Explanation(BaseExplanationStrategy):
    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        return "test explanation"


def _endpoint(model: str = "test-model") -> EndpointConfig:
    return EndpointConfig(
        base_url="https://endpoint.invalid/v1",
        api_key_env="MMREG_TEST_API_KEY",
        model=model,
    )


def _config(**overrides: Any) -> TFBearConfig:
    values: dict[str, Any] = {"explanation_strategy": _Explanation()}
    values.update(overrides)
    return TFBearConfig(**values)


def test_defaults_are_the_built_in_strategies() -> None:
    config = _config(endpoint=_endpoint())

    assert isinstance(config.prior_strategy, LogprobPriorStrategy)
    assert isinstance(config.expansion_strategy, StandardExpansionStrategy)
    assert isinstance(config.scoring_strategy, CausalDeltaScoringStrategy)
    assert isinstance(config.adjudication_strategy, TFBearAdjudicationStrategy)
    assert isinstance(config.explanation_strategy, _Explanation)
    assert config.adjudication_strategy.tau_posterior == 0.50
    assert config.adjudication_strategy.tau_delta == 0.50
    assert len(config.strategies()) == 5


def test_default_adjudication_uses_configured_thresholds() -> None:
    config = _config(endpoint=_endpoint(), tau_posterior=0.8, tau_delta=0.9)

    adjudication = config.adjudication_strategy
    assert isinstance(adjudication, TFBearAdjudicationStrategy)
    assert adjudication.tau_posterior == 0.8
    assert adjudication.tau_delta == 0.9


def test_global_settings_are_propagated_to_custom_strategies() -> None:
    explanation = _Explanation()
    config = _config(
        endpoint=_endpoint(),
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
        _config(endpoint=_endpoint(), prior_strategy=_Explanation())


@pytest.mark.parametrize(
    ("field", "value"),
    [("tau_posterior", -0.1), ("tau_posterior", 1.1), ("tau_delta", -1.0)],
)
def test_posterior_threshold_is_a_probability_and_delta_is_non_negative(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        _config(endpoint=_endpoint(), **{field: value})


def test_delta_threshold_may_exceed_one() -> None:
    config = _config(endpoint=_endpoint(), tau_delta=2.5)

    adjudication = config.adjudication_strategy
    assert isinstance(adjudication, TFBearAdjudicationStrategy)
    assert adjudication.tau_delta == 2.5


def test_max_retry_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_retry"):
        _config(endpoint=_endpoint(), max_retry=0)


def test_unknown_stage_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        _config(endpoint=_endpoint(), stage_endpoints={"seleciton": _endpoint()})


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

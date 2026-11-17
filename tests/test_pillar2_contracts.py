"""Tests for the TF-BEAR pipeline state, template validation and strategy base classes."""

from __future__ import annotations

from typing import Any

import pytest

from mmregcheck.pillar2.base import BaseExplanationStrategy, BasePriorStrategy
from mmregcheck.pillar2.pipeline_state import TFBearState, validate_template
from mmregcheck.pillar2.tf_bear import TFBearAdjudicator
from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace


def test_validate_template_accepts_known_placeholders() -> None:
    validate_template("{text_claim} / {retrieved_regulatory_context} / {paths}", "Demo")
    validate_template("{p_prior[x]} {state.attr}", "Demo")


def test_validate_template_rejects_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        validate_template("{text_claim} {typo_field}", "Demo")


def test_render_fills_state_fields_and_extras() -> None:
    state = TFBearState(text_claim="claim", retrieved_regulatory_context="law")
    out = state.render("{text_claim}|{retrieved_regulatory_context}|{paths}", paths="p1")
    assert out == "claim|law|p1"


def test_render_reports_missing_variables() -> None:
    with pytest.raises(ValueError, match="paths"):
        TFBearState().render("{paths}")


def test_verdict_is_empty_until_adjudication_has_run() -> None:
    state = TFBearState()
    assert state.verdict == ""
    branch = TFBearBranchTrace(
        branch_id="b",
        strategy="standard",
        prior_prob=0.2,
        posterior_prob=0.9,
        predicted_verdict=ComplianceVerdict.NON_COMPLIANT,
        rationale="r",
    )
    state.adjudication = TFBearAdjudicator().adjudicate_branches([branch])
    assert state.verdict == ComplianceVerdict.NON_COMPLIANT.value


def test_custom_strategy_is_validated_at_construction() -> None:
    class MyPrior(BasePriorStrategy):
        def compute_prior(
            self, state: TFBearState, client: Any, model: str
        ) -> dict[ComplianceVerdict, float]:
            return dict.fromkeys(ComplianceVerdict, 1 / 3)

    assert MyPrior(prompt_template="{text_claim}").max_retry == 3
    with pytest.raises(ValueError, match="MyPrior"):
        MyPrior(prompt_template="{nope}")


def test_base_strategies_cannot_be_instantiated_without_implementing_the_stage() -> None:
    with pytest.raises(TypeError):
        BaseExplanationStrategy()  # type: ignore[abstract]

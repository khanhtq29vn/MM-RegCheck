"""Tests for the default and alternative adjudication strategies."""

from __future__ import annotations

import math
from typing import Any

import pytest

from mmregcheck.pillar2.adjudication import (
    JointAdjudicationStrategy,
    MajorityVoteAdjudicationStrategy,
    TFBearAdjudicationStrategy,
)
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.tf_bear import TFBearAdjudicator
from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace

NON = ComplianceVerdict.NON_COMPLIANT
PARTLY = ComplianceVerdict.PARTLY_COMPLIANT
COMPLIANT = ComplianceVerdict.COMPLIANT


def _branch(
    branch_id: str,
    verdict: ComplianceVerdict,
    prior: float,
    posterior: float,
    distribution: dict[ComplianceVerdict, float] | None = None,
) -> TFBearBranchTrace:
    return TFBearBranchTrace(
        branch_id=branch_id,
        strategy="R1: Strict",
        prior_prob=prior,
        posterior_prob=posterior,
        predicted_verdict=verdict,
        rationale=f"reasoning for {branch_id}",
        posterior_distribution=distribution,
    )


class _TopLogprob:
    def __init__(self, token: str, logprob: float) -> None:
        self.token = token
        self.logprob = logprob


class _TokenLogprob:
    def __init__(self, token: str, logprob: float, candidates: list[_TopLogprob]) -> None:
        self.token = token
        self.logprob = logprob
        self.top_logprobs = candidates


class _ChoiceLogprobs:
    def __init__(self, content: list[_TokenLogprob]) -> None:
        self.content = content


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: list[_TokenLogprob]) -> None:
        self.message = _Message("0")
        self.logprobs = _ChoiceLogprobs(content)


class _Response:
    def __init__(self, content: list[_TokenLogprob]) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, distribution: dict[str, float]) -> None:
        self._distribution = distribution
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        top = [_TopLogprob(token, math.log(prob)) for token, prob in self._distribution.items()]
        return _Response([_TokenLogprob(top[0].token, top[0].logprob, top)])


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)


def test_default_strategy_wraps_the_tf_bear_adjudicator() -> None:
    branches = [
        _branch("P1", NON, 0.227, 0.982),
        _branch("P2", COMPLIANT, 0.617, 0.420),
    ]
    state = TFBearState(branches=branches)

    result = TFBearAdjudicationStrategy(tau_posterior=0.5, tau_delta=0.5).adjudicate(
        state, _Client(_Completions({})), "m"
    )
    expected = TFBearAdjudicator(tau_posterior=0.5, tau_delta=0.5).adjudicate_branches(branches)

    assert result == expected
    assert result.final_verdict == NON
    assert len(result.surviving_branches) == 1


def test_default_strategy_rejects_an_empty_branch_list() -> None:
    with pytest.raises(ValueError, match=r"state\.branches is empty"):
        TFBearAdjudicationStrategy().adjudicate(TFBearState(), _Client(_Completions({})), "m")


def test_majority_vote_counts_gate_passing_branches() -> None:
    branches = [
        _branch("P1", NON, 0.227, 0.982),
        _branch("P2", NON, 0.227, 0.704),
        _branch("P3", COMPLIANT, 0.500, 0.900),
    ]
    state = TFBearState(branches=branches)

    result = MajorityVoteAdjudicationStrategy().adjudicate(state, _Client(_Completions({})), "m")

    assert result.final_verdict == NON
    assert not result.used_fallback
    assert len(result.surviving_branches) == 3
    assert sum(state.p_posterior.values()) == pytest.approx(1.0)
    assert set(state.p_posterior) == set(ComplianceVerdict)


def test_majority_vote_fallback_matches_the_default_and_reports_the_tally() -> None:
    """Both fallbacks pick the summed-posterior argmax; the majority-vote strategy reports it."""
    branches = [
        _branch(
            "P1",
            COMPLIANT,
            0.6,
            0.4,
            {COMPLIANT: 0.4, PARTLY: 0.1, NON: 0.5},
        ),
        _branch(
            "P2",
            PARTLY,
            0.6,
            0.4,
            {COMPLIANT: 0.1, PARTLY: 0.4, NON: 0.5},
        ),
    ]
    state = TFBearState(branches=branches)

    majority_result = MajorityVoteAdjudicationStrategy().adjudicate(
        state, _Client(_Completions({})), "m"
    )
    default_result = TFBearAdjudicator().adjudicate_branches(branches)

    assert majority_result.used_fallback is True
    assert majority_result.surviving_branches == []
    assert majority_result.final_verdict == NON
    assert default_result.final_verdict == NON
    assert state.p_posterior[NON] == pytest.approx(0.5)


def test_joint_strategy_classifies_the_qualified_paths_in_one_call() -> None:
    branches = [
        _branch("P1", NON, 0.227, 0.900),
        _branch("P2", COMPLIANT, 0.400, 0.300),
    ]
    state = TFBearState(text_claim="claim", branches=branches)
    completions = _Completions({"0": 0.8, "1": 0.1, "2": 0.1})
    client = _Client(completions)

    strategy = JointAdjudicationStrategy(prompt_template="Claim: {text_claim}\n{paths}")
    result = strategy.adjudicate(state, client, "test-model")

    assert result.final_verdict == NON
    assert result.used_fallback is False
    assert result.surviving_branches == [branches[0]]
    assert result.pruned_branches == [branches[1]]
    assert state.p_posterior == pytest.approx({NON: 0.8, PARTLY: 0.1, COMPLIANT: 0.1})
    prompt = completions.calls[0]["messages"][0]["content"]
    assert "reasoning for P1" in prompt
    assert "reasoning for P2" not in prompt
    assert "Claim: claim" in prompt


def test_joint_strategy_falls_back_to_the_highest_posterior_path() -> None:
    branches = [_branch("P1", COMPLIANT, 0.5, 0.400)]
    state = TFBearState(text_claim="claim", branches=branches)

    strategy = JointAdjudicationStrategy(prompt_template="{paths}", tau_posterior=0.6)
    result = strategy.adjudicate(state, _Client(_Completions({"0": 0.1, "1": 0.1, "2": 0.8})), "m")

    assert result.used_fallback is True
    assert result.final_verdict == COMPLIANT
    assert result.surviving_branches == branches


def test_joint_strategy_requires_a_prompt_template() -> None:
    with pytest.raises(ValueError, match="prompt_template"):
        JointAdjudicationStrategy(prompt_template="  ")


@pytest.mark.parametrize("tau", [-0.1, 1.1])
def test_adjudication_thresholds_are_validated(tau: float) -> None:
    with pytest.raises(ValueError, match="tau_posterior"):
        TFBearAdjudicationStrategy(tau_posterior=tau)
    with pytest.raises(ValueError, match="tau_posterior"):
        MajorityVoteAdjudicationStrategy(tau_posterior=tau)
    with pytest.raises(ValueError, match="tau_posterior"):
        JointAdjudicationStrategy(prompt_template="{paths}", tau_posterior=tau)


def test_delta_threshold_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="tau_delta"):
        TFBearAdjudicationStrategy(tau_delta=-0.1)
    with pytest.raises(ValueError, match="tau_delta"):
        MajorityVoteAdjudicationStrategy(tau_delta=-0.1)


@pytest.mark.parametrize("reverse", [False, True])
def test_vote_tie_resolves_to_non_compliant_regardless_of_branch_order(reverse: bool) -> None:
    branches = [
        _branch(
            "P1",
            NON,
            0.3,
            0.9,
            {NON: 0.9, PARTLY: 0.05, COMPLIANT: 0.05},
        ),
        _branch(
            "P2",
            COMPLIANT,
            0.3,
            0.9,
            {NON: 0.05, PARTLY: 0.05, COMPLIANT: 0.9},
        ),
    ]
    if reverse:
        branches.reverse()

    result = TFBearAdjudicator().adjudicate_branches(branches)

    assert result.used_fallback is False
    assert result.final_verdict is NON


@pytest.mark.parametrize("reverse", [False, True])
def test_fallback_with_all_equal_summed_posteriors_is_non_compliant(reverse: bool) -> None:
    distribution = {NON: 1 / 3, PARTLY: 1 / 3, COMPLIANT: 1 / 3}
    branches = [
        _branch("P1", NON, 0.9, 1 / 3, distribution),
        _branch("P2", COMPLIANT, 0.9, 1 / 3, distribution),
    ]
    if reverse:
        branches.reverse()

    result = TFBearAdjudicator().adjudicate_branches(branches)

    assert result.used_fallback is True
    assert result.final_verdict is NON


@pytest.mark.parametrize("reverse", [False, True])
def test_all_equal_summed_posteriors_without_survivors_is_non_compliant(reverse: bool) -> None:
    branches = [
        _branch("P1", NON, 0.9, 0.4, {NON: 0.4, PARTLY: 0.3, COMPLIANT: 0.3}),
        _branch("P2", PARTLY, 0.9, 0.4, {NON: 0.3, PARTLY: 0.4, COMPLIANT: 0.3}),
        _branch("P3", COMPLIANT, 0.9, 0.4, {NON: 0.3, PARTLY: 0.3, COMPLIANT: 0.4}),
    ]
    if reverse:
        branches.reverse()

    result = TFBearAdjudicator().adjudicate_branches(branches)

    assert result.used_fallback is True
    assert result.final_verdict is NON


def test_joint_all_equal_distribution_resolves_to_non_compliant() -> None:
    branch = _branch("P1", NON, 0.3, 0.9)
    state = TFBearState(text_claim="claim", branches=[branch])
    strategy = JointAdjudicationStrategy(prompt_template="{paths}")

    result = strategy.adjudicate(
        state,
        _Client(_Completions({"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})),
        "m",
    )

    assert result.final_verdict is NON

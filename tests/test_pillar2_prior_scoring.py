"""Tests for the prior and causal-delta scoring stages with a fake client (no network)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest

from mmregcheck.pillar2.diagnostics import TFBearWarning
from mmregcheck.pillar2.llm import TOKEN_TO_VERDICT
from mmregcheck.pillar2.pipeline_state import Node, TFBearState
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy
from mmregcheck.schemas import ComplianceVerdict

MODEL = "test-model"


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
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, logprobs_payload: _ChoiceLogprobs) -> None:
        self.message = _Message("0")
        self.logprobs = logprobs_payload


class _Response:
    def __init__(self, logprobs_payload: _ChoiceLogprobs) -> None:
        self.choices = [_Choice(logprobs_payload)]


class _FakeCompletions:
    """Replays one verdict distribution per call, in order."""

    def __init__(self, distributions: list[dict[str, float]]) -> None:
        self._distributions = distributions
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        distribution = self._distributions.pop(0)
        candidates = [_TopLogprob(token, math.log(p)) for token, p in distribution.items()]
        content = [_TokenLogprob(candidates[0].token, candidates[0].logprob, candidates)]
        return _Response(_ChoiceLogprobs(content))


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def test_token_mapping_covers_the_enum() -> None:
    assert set(TOKEN_TO_VERDICT.values()) == set(ComplianceVerdict)
    assert TOKEN_TO_VERDICT["0"] is ComplianceVerdict.NON_COMPLIANT


def test_prior_then_scoring_reproduces_section_5_4() -> None:
    completions = _FakeCompletions(
        [
            {"0": 0.227, "1": 0.400, "2": 0.373},
            {"0": 0.982, "1": 0.012, "2": 0.006},
        ]
    )
    client = _FakeClient(completions)
    state = TFBearState(
        text_claim="A spa advertises a chemical peel.",
        retrieved_regulatory_context="SPA establishments may not use chemical peels.",
    )

    prior_strategy = LogprobPriorStrategy(prompt_template="Classify: {text_claim}")
    prior = prior_strategy.compute_prior(state, client, MODEL)

    assert set(prior) == set(ComplianceVerdict)
    assert prior[ComplianceVerdict.NON_COMPLIANT] == pytest.approx(0.227)
    assert completions.calls[0]["messages"] == [
        {"role": "user", "content": "Classify: A spa advertises a chemical peel."}
    ]

    state.p_prior = prior
    state.nodes = [
        Node(
            state="Chemical peels are invasive and prohibited for a SPA.",
            is_terminal=True,
            meta={"strategy": "R1: Strict"},
        )
    ]
    scoring_strategy = CausalDeltaScoringStrategy(
        prompt_template=(
            "Claim: {text_claim}\n"
            "Context: {retrieved_regulatory_context}\n"
            "Reasoning: {state}\n"
            "Answer with one digit."
        )
    )
    traces = scoring_strategy.score(state, client, MODEL)

    assert len(traces) == 1
    trace = traces[0]
    assert trace.branch_id == "P1"
    assert trace.strategy == "R1: Strict"
    assert trace.predicted_verdict == ComplianceVerdict.NON_COMPLIANT
    assert trace.prior_prob == pytest.approx(0.227)
    assert trace.posterior_prob == pytest.approx(0.982)
    assert trace.posterior_distribution == pytest.approx(
        {
            ComplianceVerdict.NON_COMPLIANT: 0.982,
            ComplianceVerdict.PARTLY_COMPLIANT: 0.012,
            ComplianceVerdict.COMPLIANT: 0.006,
        }
    )
    assert trace.rationale == "Chemical peels are invasive and prohibited for a SPA."
    # Delta is the schema property: L(0.982) - L(0.227) = 5.2245.
    assert trace.log_odds_delta == pytest.approx(5.224, abs=0.02)

    scoring_prompt = completions.calls[1]["messages"][0]["content"]
    assert "Chemical peels are invasive and prohibited for a SPA." in scoring_prompt
    assert "SPA establishments may not use chemical peels." in scoring_prompt


def test_hollow_nodes_are_skipped_and_do_not_consume_calls() -> None:
    completions = _FakeCompletions([{"0": 0.9, "1": 0.05, "2": 0.05}])
    state = TFBearState(
        text_claim="claim",
        p_prior=dict.fromkeys(ComplianceVerdict, 1 / 3),
        nodes=[
            Node(state="unfinished", is_terminal=False),
            Node(state="   ", answer=None, is_terminal=True),
            Node(state="scoreable path", is_terminal=True, meta={"strategy": "R2: Permissive"}),
        ],
    )

    with pytest.warns(TFBearWarning, match=r"skipped 2 of 3"):
        traces = CausalDeltaScoringStrategy(prompt_template="{text_claim}\n{state}").score(
            state, _FakeClient(completions), MODEL
        )

    assert len(traces) == 1
    assert traces[0].branch_id == "P3"
    assert traces[0].strategy == "R2: Permissive"
    assert len(completions.calls) == 1
    assert len(state.diagnostics) == 1
    assert "1 non-terminal" in state.diagnostics[0]
    assert "1 with empty reasoning" in state.diagnostics[0]


def test_scoring_does_not_warn_when_every_node_is_scoreable() -> None:
    completions = _FakeCompletions([{"0": 0.9, "1": 0.05, "2": 0.05}])
    state = TFBearState(
        text_claim="claim",
        p_prior=dict.fromkeys(ComplianceVerdict, 1 / 3),
        nodes=[Node(state="path", is_terminal=True)],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", TFBearWarning)
        traces = CausalDeltaScoringStrategy(prompt_template="{text_claim}\n{state}").score(
            state, _FakeClient(completions), MODEL
        )

    assert len(traces) == 1
    assert state.diagnostics == []


def test_all_equal_posterior_resolves_to_non_compliant() -> None:
    completions = _FakeCompletions([{"0": 1 / 3, "1": 1 / 3, "2": 1 / 3}])
    state = TFBearState(
        text_claim="claim",
        p_prior=dict.fromkeys(ComplianceVerdict, 1 / 3),
        nodes=[Node(state="balanced path", is_terminal=True)],
    )

    trace = CausalDeltaScoringStrategy(prompt_template="{text_claim}\n{state}").score(
        state, _FakeClient(completions), MODEL
    )[0]

    assert trace.predicted_verdict is ComplianceVerdict.NON_COMPLIANT
    assert trace.posterior_prob == pytest.approx(1 / 3)
    assert trace.log_odds_delta == pytest.approx(0.0)


def test_scoring_without_scoreable_nodes_returns_an_empty_list() -> None:
    completions = _FakeCompletions([])
    state = TFBearState(nodes=[Node(state="unfinished", is_terminal=False)])

    traces = CausalDeltaScoringStrategy().score(state, _FakeClient(completions), MODEL)

    assert traces == []
    assert completions.calls == []


def test_node_answer_is_used_when_the_state_is_empty() -> None:
    completions = _FakeCompletions([{"0": 0.9, "1": 0.05, "2": 0.05}])
    state = TFBearState(
        text_claim="claim",
        p_prior=dict.fromkeys(ComplianceVerdict, 1 / 3),
        nodes=[Node(state="", answer="final answer", is_terminal=True)],
    )

    traces = CausalDeltaScoringStrategy().score(state, _FakeClient(completions), MODEL)

    assert len(traces) == 1
    assert traces[0].rationale == "final answer"
    assert traces[0].strategy == "unknown"


def test_missing_prior_probability_is_an_error() -> None:
    completions = _FakeCompletions([{"0": 0.9, "1": 0.05, "2": 0.05}])
    state = TFBearState(
        text_claim="claim",
        p_prior={
            ComplianceVerdict.COMPLIANT: 0.5,
            ComplianceVerdict.PARTLY_COMPLIANT: 0.5,
        },
        nodes=[Node(state="path", is_terminal=True)],
    )

    with pytest.raises(ValueError, match="prior has no probability"):
        CausalDeltaScoringStrategy().score(state, _FakeClient(completions), MODEL)


def test_saturated_probabilities_produce_a_finite_delta() -> None:
    completions = _FakeCompletions([{"0": 1.0}])
    state = TFBearState(
        text_claim="claim",
        p_prior={
            ComplianceVerdict.NON_COMPLIANT: 0.0,
            ComplianceVerdict.PARTLY_COMPLIANT: 0.5,
            ComplianceVerdict.COMPLIANT: 0.5,
        },
        nodes=[Node(state="path", is_terminal=True, meta={"strategy": "R1: Strict"})],
    )

    trace = CausalDeltaScoringStrategy().score(state, _FakeClient(completions), MODEL)[0]

    assert trace.posterior_prob == 1.0
    assert trace.prior_prob == 0.0
    assert trace.posterior_distribution == {
        ComplianceVerdict.NON_COMPLIANT: 1.0,
        ComplianceVerdict.PARTLY_COMPLIANT: 0.0,
        ComplianceVerdict.COMPLIANT: 0.0,
    }
    assert trace.log_odds_delta == pytest.approx(23.0259, abs=1e-3)

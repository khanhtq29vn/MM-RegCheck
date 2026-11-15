"""Tests for the shared label-token logprob elicitation (no network)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from mmregcheck.pillar2.llm import logprobs
from mmregcheck.pillar2.llm.logprobs import (
    TOKEN_TO_VERDICT,
    LabelTokenMissingError,
    LogprobElicitationError,
    MissingLogprobsError,
    label_distribution,
)
from mmregcheck.schemas import ComplianceVerdict

PROMPT = [{"role": "user", "content": "Answer with one digit."}]


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
    def __init__(self, logprobs_payload: _ChoiceLogprobs | None) -> None:
        self.message = _Message("0")
        self.logprobs = logprobs_payload


class _Response:
    def __init__(self, logprobs_payload: _ChoiceLogprobs | None) -> None:
        self.choices = [_Choice(logprobs_payload)]


class _FakeCompletions:
    def __init__(self, responses: list[_Response], errors_before_success: int = 0) -> None:
        self._responses = responses
        self._errors_before_success = errors_before_success
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if self._errors_before_success:
            self._errors_before_success -= 1
            raise RuntimeError("transient endpoint error")
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def _response(candidates: dict[str, float]) -> _Response:
    top = [_TopLogprob(token, logprob) for token, logprob in candidates.items()]
    content = [_TokenLogprob(top[0].token, top[0].logprob, top)]
    return _Response(_ChoiceLogprobs(content))


def _client(response: _Response) -> _FakeClient:
    return _FakeClient(_FakeCompletions([response]))


def test_token_mapping_matches_the_compliance_enum() -> None:
    assert TOKEN_TO_VERDICT == {
        "0": ComplianceVerdict.NON_COMPLIANT,
        "1": ComplianceVerdict.PARTLY_COMPLIANT,
        "2": ComplianceVerdict.COMPLIANT,
    }
    assert set(TOKEN_TO_VERDICT.values()) == set(ComplianceVerdict)


def test_restricted_softmax_ignores_non_label_tokens_and_normalises() -> None:
    # Raw log-probabilities are not normalised over the labels (1 + 3 + 1 = 5).
    client = _client(
        _response({"0": 0.0, "1": math.log(3.0), "2": 0.0, "X": math.log(10.0)})
    )
    distribution = label_distribution(client, model="m", messages=PROMPT)

    assert distribution[ComplianceVerdict.NON_COMPLIANT] == pytest.approx(0.2)
    assert distribution[ComplianceVerdict.PARTLY_COMPLIANT] == pytest.approx(0.6)
    assert distribution[ComplianceVerdict.COMPLIANT] == pytest.approx(0.2)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_tokens_with_a_leading_space_are_accepted() -> None:
    client = _client(
        _response({" 0": math.log(0.7), " 1": math.log(0.2), " 2": math.log(0.1)})
    )
    distribution = label_distribution(client, model="m", messages=PROMPT)

    assert distribution[ComplianceVerdict.NON_COMPLIANT] == pytest.approx(0.7)
    assert distribution[ComplianceVerdict.PARTLY_COMPLIANT] == pytest.approx(0.2)
    assert distribution[ComplianceVerdict.COMPLIANT] == pytest.approx(0.1)


def test_label_outside_the_candidates_gets_zero_probability() -> None:
    client = _client(_response({"1": math.log(0.4), "X": math.log(0.6)}))
    distribution = label_distribution(client, model="m", messages=PROMPT)

    assert distribution[ComplianceVerdict.PARTLY_COMPLIANT] == pytest.approx(1.0)
    assert distribution[ComplianceVerdict.NON_COMPLIANT] == 0.0
    assert distribution[ComplianceVerdict.COMPLIANT] == 0.0


def test_single_surviving_label_saturates_to_one() -> None:
    client = _client(_response({" 2": 0.0}))
    distribution = label_distribution(client, model="m", messages=PROMPT)

    assert distribution[ComplianceVerdict.COMPLIANT] == 1.0
    assert distribution[ComplianceVerdict.NON_COMPLIANT] == 0.0
    assert distribution[ComplianceVerdict.PARTLY_COMPLIANT] == 0.0


def test_missing_label_token_raises_instead_of_fabricating_a_distribution() -> None:
    client = _client(_response({"A": math.log(0.5), "B": math.log(0.5)}))

    with pytest.raises(LabelTokenMissingError, match="label tokens"):
        label_distribution(client, model="m", messages=PROMPT)


def test_missing_logprobs_payload_raises() -> None:
    client = _client(_Response(None))

    with pytest.raises(MissingLogprobsError, match="no logprobs"):
        label_distribution(client, model="m", messages=PROMPT)


def test_empty_logprob_content_raises() -> None:
    client = _client(_Response(_ChoiceLogprobs([])))

    with pytest.raises(MissingLogprobsError, match="no logprob content"):
        label_distribution(client, model="m", messages=PROMPT)


def test_request_is_greedy_single_token_and_asks_for_top_logprobs() -> None:
    completions = _FakeCompletions([_response({"0": 0.0, "1": 0.0, "2": 0.0})])
    label_distribution(_FakeClient(completions), model="test-model", messages=PROMPT)

    call = completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == PROMPT
    assert call["max_tokens"] == 1
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 5
    assert call["temperature"] == 0.0
    assert call["top_p"] == 1.0


def test_transient_error_is_retried_within_max_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(logprobs.time, "sleep", sleeps.append)
    completions = _FakeCompletions([_response({"0": 0.0, "1": 0.0, "2": 0.0})], errors_before_success=1)

    distribution = label_distribution(_FakeClient(completions), model="m", messages=PROMPT, max_retry=2)

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert len(completions.calls) == 2
    assert sleeps == [2]


def test_retry_budget_is_bounded_and_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logprobs.time, "sleep", lambda _seconds: None)
    completions = _FakeCompletions([], errors_before_success=5)

    with pytest.raises(LogprobElicitationError, match="failed after 2 attempt"):
        label_distribution(_FakeClient(completions), model="m", messages=PROMPT, max_retry=2)
    assert len(completions.calls) == 2


def test_max_retry_must_be_positive() -> None:
    client = _client(_response({"0": 0.0, "1": 0.0, "2": 0.0}))

    with pytest.raises(ValueError, match="max_retry"):
        label_distribution(client, model="m", messages=PROMPT, max_retry=0)

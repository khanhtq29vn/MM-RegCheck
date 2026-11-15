"""Tests for the generative and structured explanation strategies (Stage 5).

Clients are fakes that mimic the shared OpenAI-compatible shape
(`client.chat.completions.create(...)` -> `.choices[i].message.content`), so no test uses the
network. The deterministic `synthesize_overall_explanation` helper is covered too because it is
public API that the port must not break.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from mmregcheck.pillar2.explainer import (
    DEFAULT_SCHEMA_NAME,
    ExplanationOutput,
    GenerativeExplanationStrategy,
    StructuredGenerativeExplanationStrategy,
    synthesize_overall_explanation,
)
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.tf_bear import TFBearAdjudicator
from mmregcheck.schemas import ComplianceVerdict, TechniqueVerdictItem, TFBearBranchTrace


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _FakeCompletions:
    def __init__(self, handler: Callable[[str], list[str]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._handler = handler

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        return _Response(choices=[_Choice(_Message(text)) for text in self._handler(prompt)])


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeClient:
    """Minimal stand-in for an OpenAI-compatible chat-completion client."""

    def __init__(self, handler: Callable[[str], list[str]]) -> None:
        self.chat = _FakeChat(_FakeCompletions(handler))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


SUPPORTING_RATIONALE = "The clinic advertises laser treatment without a licence."
PRUNED_RATIONALE = "No regulatory issue was found in this branch."


def _state() -> TFBearState:
    supporting = TFBearBranchTrace(
        branch_id="P1",
        strategy="standard",
        prior_prob=0.2,
        posterior_prob=0.9,
        predicted_verdict=ComplianceVerdict.NON_COMPLIANT,
        rationale=SUPPORTING_RATIONALE,
    )
    pruned = TFBearBranchTrace(
        branch_id="P2",
        strategy="standard",
        prior_prob=0.2,
        posterior_prob=0.2,
        predicted_verdict=ComplianceVerdict.COMPLIANT,
        rationale=PRUNED_RATIONALE,
    )
    state = TFBearState(
        text_claim="Laser hair removal advertised without a licence",
        retrieved_regulatory_context="Laser treatment requires a licensed practitioner.",
    )
    state.branches = [supporting, pruned]
    state.adjudication = TFBearAdjudicator().adjudicate_branches(state.branches)
    return state


def _prompt_of(client: FakeClient, index: int = 0) -> str:
    return client.calls[index]["messages"][-1]["content"]


def test_generative_happy_path_renders_state_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(lambda _prompt: ["The claim breaches the licensed-practitioner scope."])
    strategy = GenerativeExplanationStrategy()
    logged: list[str] = []
    monkeypatch.setattr(strategy.logger, "log_explanation", logged.append)

    result = strategy.explain(_state(), client, "fake-model")

    assert result == "The claim breaches the licensed-practitioner scope."
    assert logged == [result]

    prompt = _prompt_of(client)
    assert "Laser hair removal advertised without a licence" in prompt
    assert "Laser treatment requires a licensed practitioner." in prompt
    assert "Verdict: Non-Compliant" in prompt
    assert f"Path 1:\n{SUPPORTING_RATIONALE}" in prompt
    # Branches pruned by the dual-threshold gate are not offered as supporting evidence.
    assert PRUNED_RATIONALE not in prompt

    call = client.calls[0]
    assert call["model"] == "fake-model"
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 1024
    assert call["reasoning"] == {"enabled": False}


def test_generative_uses_all_branches_before_adjudication() -> None:
    state = _state()
    state.adjudication = None
    client = FakeClient(lambda _prompt: ["ok"])
    strategy = GenerativeExplanationStrategy(prompt_template="{paths}")

    strategy.explain(state, client, "fake-model")

    prompt = _prompt_of(client)
    assert SUPPORTING_RATIONALE in prompt
    assert PRUNED_RATIONALE in prompt


def test_generative_empty_completion_is_returned_unchanged() -> None:
    client = FakeClient(lambda _prompt: [""])
    strategy = GenerativeExplanationStrategy()

    assert strategy.explain(_state(), client, "fake-model") == ""


def test_generative_retry_budget_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def _always_fails(_prompt: str) -> list[str]:
        raise RuntimeError("api down")

    client = FakeClient(_always_fails)
    strategy = GenerativeExplanationStrategy(max_retry=2)
    monkeypatch.setattr("mmregcheck.pillar2.expansion.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        strategy.explain(_state(), client, "fake-model")

    assert len(client.calls) == 2


def test_structured_happy_path_returns_validated_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {"verdict": "Non-Compliant", "explanation": "The licence scope is exceeded."}
    )
    client = FakeClient(lambda _prompt: [payload])
    strategy = StructuredGenerativeExplanationStrategy()
    logged: list[str] = []
    monkeypatch.setattr(strategy.logger, "log_explanation", logged.append)

    result = strategy.explain(_state(), client, "fake-model")

    assert json.loads(result) == {
        "verdict": "Non-Compliant",
        "explanation": "The licence scope is exceeded.",
    }
    assert logged == [result]

    prompt = _prompt_of(client)
    assert "Verdict: Non-Compliant" in prompt
    assert f"Path 1:\n{SUPPORTING_RATIONALE}" in prompt

    call = client.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 1024
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": DEFAULT_SCHEMA_NAME,
            "schema": ExplanationOutput.model_json_schema(),
        },
    }


def test_structured_custom_schema_is_sent_and_validated() -> None:
    class CustomExplanation(BaseModel):
        rationale: str

    client = FakeClient(lambda _prompt: [json.dumps({"rationale": "custom rationale"})])
    strategy = StructuredGenerativeExplanationStrategy(
        output_schema=CustomExplanation, schema_name="custom_rationale"
    )

    result = strategy.explain(_state(), client, "fake-model")

    assert json.loads(result) == {"rationale": "custom rationale"}
    assert client.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "custom_rationale",
            "schema": CustomExplanation.model_json_schema(),
        },
    }


def test_structured_retries_malformed_output_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = ["this is not json", json.dumps({"verdict": "Compliant", "explanation": "ok"})]
    client = FakeClient(lambda _prompt: [payloads.pop(0)])
    strategy = StructuredGenerativeExplanationStrategy(max_retry=2)
    monkeypatch.setattr("mmregcheck.pillar2.explainer.time.sleep", lambda _seconds: None)

    result = strategy.explain(_state(), client, "fake-model")

    assert json.loads(result) == {"verdict": "Compliant", "explanation": "ok"}
    assert len(client.calls) == 2


def test_structured_exhausted_retries_raise_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(lambda _prompt: ["not json"])
    strategy = StructuredGenerativeExplanationStrategy(max_retry=2)
    monkeypatch.setattr("mmregcheck.pillar2.explainer.time.sleep", lambda _seconds: None)

    with pytest.raises(ValueError, match="failed after 2 attempts"):
        strategy.explain(_state(), client, "fake-model")

    assert len(client.calls) == 2


def test_structured_schema_violation_raises_a_clear_error() -> None:
    client = FakeClient(lambda _prompt: [json.dumps({"verdict": "Compliant"})])
    strategy = StructuredGenerativeExplanationStrategy(max_retry=1)

    with pytest.raises(ValueError, match="failed after 1 attempts"):
        strategy.explain(_state(), client, "fake-model")


def test_structured_rejects_a_zero_retry_budget() -> None:
    client = FakeClient(lambda _prompt: ["unused"])
    strategy = StructuredGenerativeExplanationStrategy(max_retry=0)

    with pytest.raises(ValueError, match="max_retry"):
        strategy.explain(_state(), client, "fake-model")

    assert client.calls == []


@pytest.mark.parametrize(
    "strategy_cls",
    [GenerativeExplanationStrategy, StructuredGenerativeExplanationStrategy],
)
def test_unknown_placeholder_is_rejected_at_construction(
    strategy_cls: type[GenerativeExplanationStrategy]
    | type[StructuredGenerativeExplanationStrategy],
) -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        strategy_cls(prompt_template="{text_claim} {imaginary_field}")


def test_default_constructors_need_no_arguments() -> None:
    assert GenerativeExplanationStrategy().prompt_template
    assert StructuredGenerativeExplanationStrategy().output_schema is ExplanationOutput


def test_synthesize_overall_explanation_still_works() -> None:
    items = [
        TechniqueVerdictItem(
            technique_name="Laser",
            verdict=ComplianceVerdict.COMPLIANT,
            explanation="Covered by the facility licence.",
        ),
        TechniqueVerdictItem(
            technique_name="Peel",
            verdict=ComplianceVerdict.NON_COMPLIANT,
            explanation="Outside the licensed scope.",
        ),
    ]

    verdict, details = synthesize_overall_explanation(items)

    assert verdict == ComplianceVerdict.NON_COMPLIANT.value
    assert "[Laser: Compliant] Covered by the facility licence." in details
    assert "[Peel: Non-Compliant] Outside the licensed scope." in details


def test_synthesize_overall_explanation_handles_no_items() -> None:
    assert synthesize_overall_explanation([]) == ("Compliant", "No techniques extracted.")

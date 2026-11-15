"""Tests for the Standard and CoT-SC reasoning-path expansion strategies.

Every client is a fake that mimics the shared OpenAI-compatible shape
(`client.chat.completions.create(...)` -> `.choices[i].message.content`), so the suite never
touches the network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from mmregcheck.pillar2.expansion import (
    COT_SC_STRATEGY,
    STANDARD_STRATEGY,
    CoTSCExpansionStrategy,
    StandardExpansionStrategy,
)
from mmregcheck.pillar2.pipeline_state import TFBearState


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
        messages = kwargs["messages"]
        prompt = messages[-1]["content"]
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


def _state() -> TFBearState:
    return TFBearState(
        text_claim="Laser hair removal advertised without a licence",
        retrieved_regulatory_context="Laser treatment requires a licensed practitioner.",
    )


def test_standard_returns_one_direct_path() -> None:
    client = FakeClient(lambda _prompt: ["Step 1: the clinic holds no laser licence."])
    strategy = StandardExpansionStrategy(
        prompt_template="{text_claim} | {retrieved_regulatory_context}"
    )

    nodes = strategy.expand(_state(), client, "fake-model")

    assert len(nodes) == 1
    node = nodes[0]
    assert node.state == "Step 1: the clinic holds no laser licence."
    assert node.thought == node.state
    assert node.is_terminal is True
    assert node.meta == {"strategy": STANDARD_STRATEGY, "choice_index": 0}

    call = client.calls[0]
    assert call["model"] == "fake-model"
    assert call["messages"] == [
        {
            "role": "user",
            "content": (
                "Laser hair removal advertised without a licence | "
                "Laser treatment requires a licensed practitioner."
            ),
        }
    ]
    assert call["n"] == 1
    assert call["temperature"] == 0.0
    assert call["reasoning"] == {"enabled": False}


def test_cot_sc_returns_n_paths_with_distinct_meta_indices() -> None:
    texts = ["path a", "path b", "path c"]
    client = FakeClient(lambda _prompt: texts)
    strategy = CoTSCExpansionStrategy(
        n_paths=3, prompt_template="{text_claim} :: take {n_paths} independent paths"
    )

    nodes = strategy.expand(_state(), client, "fake-model")

    assert [node.state for node in nodes] == texts
    assert [node.meta["cot_index"] for node in nodes] == [0, 1, 2]
    assert len({node.meta["cot_index"] for node in nodes}) == 3
    assert all(node.meta["strategy"] == COT_SC_STRATEGY for node in nodes)
    assert all(node.is_terminal for node in nodes)

    call = client.calls[0]
    assert call["n"] == 3
    assert call["temperature"] == 0.7
    assert call["messages"][0]["content"].endswith(":: take 3 independent paths")


def test_cot_sc_temperature_override_is_forwarded() -> None:
    client = FakeClient(lambda _prompt: ["only path"])
    strategy = CoTSCExpansionStrategy(
        n_paths=1, temperature=0.2, prompt_template="{text_claim}"
    )

    strategy.expand(_state(), client, "fake-model")

    assert client.calls[0]["temperature"] == 0.2


def test_blank_completions_are_not_turned_into_nodes() -> None:
    client = FakeClient(lambda _prompt: ["   ", ""])
    strategy = StandardExpansionStrategy(prompt_template="{text_claim}")

    with pytest.raises(ValueError, match="no usable completion"):
        strategy.expand(_state(), client, "fake-model")


def test_empty_content_choices_are_skipped() -> None:
    client = FakeClient(lambda _prompt: ["", "usable path"])
    strategy = StandardExpansionStrategy(
        n_paths=2, prompt_template="{text_claim}"
    )

    nodes = strategy.expand(_state(), client, "fake-model")

    assert len(nodes) == 1
    assert nodes[0].state == "usable path"
    assert nodes[0].meta["choice_index"] == 1


def test_retry_budget_is_bounded_and_raises_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _always_fails(_prompt: str) -> list[str]:
        raise RuntimeError("api down")

    client = FakeClient(_always_fails)
    strategy = StandardExpansionStrategy(prompt_template="{text_claim}", max_retry=2)
    monkeypatch.setattr("mmregcheck.pillar2.expansion.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        strategy.expand(_state(), client, "fake-model")

    assert len(client.calls) == 2


def test_retry_recovers_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def _flaky(_prompt: str) -> list[str]:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return ["recovered path"]

    client = FakeClient(_flaky)
    strategy = CoTSCExpansionStrategy(
        n_paths=1, prompt_template="{text_claim}", max_retry=3
    )
    monkeypatch.setattr("mmregcheck.pillar2.expansion.time.sleep", lambda _seconds: None)

    nodes = strategy.expand(_state(), client, "fake-model")

    assert len(attempts) == 3
    assert [node.state for node in nodes] == ["recovered path"]


def test_zero_retry_budget_is_rejected() -> None:
    client = FakeClient(lambda _prompt: ["unused"])
    strategy = StandardExpansionStrategy(prompt_template="{text_claim}", max_retry=0)

    with pytest.raises(ValueError, match="max_retry"):
        strategy.expand(_state(), client, "fake-model")

    assert client.calls == []


@pytest.mark.parametrize("strategy_cls", [StandardExpansionStrategy, CoTSCExpansionStrategy])
def test_unknown_placeholder_is_rejected_at_construction(
    strategy_cls: type[StandardExpansionStrategy] | type[CoTSCExpansionStrategy],
) -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        strategy_cls(prompt_template="{text_claim} {imaginary_field}")


@pytest.mark.parametrize("strategy_cls", [StandardExpansionStrategy, CoTSCExpansionStrategy])
def test_default_strategy_name_is_exposed(
    strategy_cls: type[StandardExpansionStrategy] | type[CoTSCExpansionStrategy],
) -> None:
    strategy = strategy_cls(prompt_template="{text_claim}")
    client = FakeClient(lambda _prompt: ["path"])

    nodes = strategy.expand(_state(), client, "fake-model")

    expected = STANDARD_STRATEGY if strategy_cls is StandardExpansionStrategy else COT_SC_STRATEGY
    assert nodes[0].meta["strategy"] == expected

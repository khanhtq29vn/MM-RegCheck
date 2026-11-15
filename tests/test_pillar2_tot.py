"""Tests for the Tree-of-Thoughts expansion strategy and its BFS engine.

The fake client answers a scripted prompt, so the search tree, its pruning and its error paths are
exercised without any network access.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from mmregcheck.pillar2.expansion.tot import (
    TOT_STRATEGY,
    ExpansionParseError,
    ToTExpansionStrategy,
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


_EVALUATION = json.dumps({"evaluation": "sure", "reasoning": "progress"})


def _thought(text: str, *, terminate: bool = False, answer: str | None = None) -> str:
    return json.dumps({"thought": text, "terminate": terminate, "answer": answer})


def _scripted_client(
    on_thought: Callable[[str], list[str]],
    evaluation: str = _EVALUATION,
) -> FakeClient:
    """Route thought-generation prompts to `on_thought` and every evaluation to one answer."""

    def handler(prompt: str) -> list[str]:
        if prompt.startswith("EVALUATE["):
            return [evaluation]
        assert prompt.startswith("THINK["), prompt
        return on_thought(prompt[len("THINK[") : -1])

    return FakeClient(handler)


def _state() -> TFBearState:
    return TFBearState(
        text_claim="Laser hair removal advertised without a licence",
        retrieved_regulatory_context="Laser treatment requires a licensed practitioner.",
    )


def _tot(**overrides: Any) -> ToTExpansionStrategy:
    parameters: dict[str, Any] = {
        "n_steps": 2,
        "k": 2,
        "n_evals": 1,
        "breadth_limit": 2,
        "prompt_template": "THINK[{state}]",
        "eval_prompt_template": "EVALUATE[{state}]",
    }
    parameters.update(overrides)
    return ToTExpansionStrategy(**parameters)


def _calls_for(client: FakeClient, prefix: str) -> list[dict[str, Any]]:
    return [call for call in client.calls if call["messages"][-1]["content"].startswith(prefix)]


def test_bfs_respects_configured_breadth_and_depth() -> None:
    def _grow_two(current: str) -> list[str]:
        depth = current.count("\n") + (1 if current else 0)
        return [_thought(f"L{depth}A"), _thought(f"L{depth}B")]

    client = _scripted_client(_grow_two)

    nodes = _tot(n_steps=3, breadth_limit=1).expand(_state(), client, "fake-model")

    # One node survives each of the first two levels, so the third level ends with two leaves.
    assert [node.state for node in nodes] == ["L0A\nL1A\nL2A", "L0A\nL1A\nL2B"]
    assert all(node.meta["depth"] == 3 for node in nodes)
    assert all(node.meta["strategy"] == TOT_STRATEGY for node in nodes)
    # No terminal node was found, so the fallback leaves are promoted for the scorer.
    assert all(node.is_terminal for node in nodes)

    thought_calls = _calls_for(client, "THINK[")
    eval_calls = _calls_for(client, "EVALUATE[")
    assert len(thought_calls) == 3
    # Final-level nodes are never expanded, so they are not evaluated either.
    assert len(eval_calls) == 4
    assert all(call["n"] == 2 for call in thought_calls)
    assert all(call["response_format"] == {"type": "json_object"} for call in thought_calls)
    assert all(call["n"] == 1 for call in eval_calls)


def test_bfs_collects_terminals_and_keeps_expanding_siblings() -> None:
    def _grow_or_terminate(current: str) -> list[str]:
        if current == "":
            return [_thought("root-left"), _thought("root-right")]
        if current == "root-left":
            return [_thought("done", terminate=True, answer="Compliant")]
        return [_thought("right-a"), _thought("right-b")]

    client = _scripted_client(_grow_or_terminate)

    nodes = _tot(n_steps=2, breadth_limit=2).expand(_state(), client, "fake-model")

    assert len(nodes) == 3
    terminal_nodes = [node for node in nodes if node.is_terminal]
    assert len(terminal_nodes) == 1
    terminal = terminal_nodes[0]
    assert terminal.state == "root-left\nAnswer: Compliant"
    assert terminal.thought == "Answer: Compliant"
    assert terminal.answer == "Compliant"
    assert terminal.meta["depth"] == 2

    leaves = [node for node in nodes if not node.is_terminal]
    assert sorted(node.state for node in leaves) == [
        "root-right\nright-a",
        "root-right\nright-b",
    ]
    # The terminal at level 2 did not stop the sibling branch from being expanded.
    assert len(_calls_for(client, "THINK[")) == 3


def test_malformed_thought_output_raises_a_clear_error() -> None:
    client = _scripted_client(lambda _current: ["this is not json"])

    with pytest.raises(ExpansionParseError, match="JSON"):
        _tot(n_steps=1, k=1).expand(_state(), client, "fake-model")


def test_json_object_without_thought_or_terminate_is_rejected() -> None:
    client = _scripted_client(lambda _current: ["{}"])

    with pytest.raises(ExpansionParseError, match="neither a non-empty 'thought'"):
        _tot(n_steps=1, k=1).expand(_state(), client, "fake-model")


def test_fenced_json_output_is_tolerated() -> None:
    fenced = '```json\n{"thought": "fenced thought", "terminate": false}\n```'
    client = _scripted_client(lambda _current: [fenced])

    nodes = _tot(n_steps=1, k=1).expand(_state(), client, "fake-model")

    assert [node.state for node in nodes] == ["fenced thought"]


def test_malformed_evaluation_output_raises_a_clear_error() -> None:
    client = _scripted_client(lambda _current: [_thought("branch")], evaluation="not json")

    with pytest.raises(ExpansionParseError, match="state evaluation"):
        _tot(n_steps=2, k=1, breadth_limit=1).expand(_state(), client, "fake-model")


def test_retry_budget_is_bounded_and_raises_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _always_fails(_prompt: str) -> list[str]:
        raise RuntimeError("api down")

    client = FakeClient(_always_fails)
    strategy = _tot(n_steps=1, k=1, max_retry=2)
    monkeypatch.setattr("mmregcheck.pillar2.llm.completion.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        strategy.expand(_state(), client, "fake-model")

    assert len(client.calls) == 2


def test_terminate_below_min_steps_is_suppressed_without_fabricating_a_node() -> None:
    def _terminate_immediately(_current: str) -> list[str]:
        return [json.dumps({"terminate": True, "answer": "early"})]

    client = _scripted_client(_terminate_immediately)

    nodes = _tot(n_steps=2, k=1, breadth_limit=1, min_steps=2).expand(
        _state(), client, "fake-model"
    )

    assert nodes == []
    assert len(_calls_for(client, "THINK[")) == 1


@pytest.mark.parametrize(
    "parameters",
    [{"n_steps": 0}, {"k": 0}, {"n_evals": 0}, {"breadth_limit": 0}, {"max_retry": 0}],
)
def test_non_positive_search_budget_is_rejected(parameters: dict[str, int]) -> None:
    client = FakeClient(lambda _prompt: ["unused"])

    with pytest.raises(ValueError, match="must be >= 1"):
        _tot(**parameters).expand(_state(), client, "fake-model")

    assert client.calls == []


def test_unknown_placeholders_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        _tot(prompt_template="{claim}")
    with pytest.raises(ValueError, match="eval_prompt_template"):
        _tot(eval_prompt_template="{retrieved_context}")

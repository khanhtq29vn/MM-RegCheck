"""Orchestration of the five TF-BEAR stages.

`run_tfbear` runs prior, expansion, scoring, adjudication and explanation over one
`TFBearState`, logs each stage, and records latency and token usage per stage. It returns typed
objects rather than a plain dict, and it resolves one OpenAI-compatible client per stage so
multiple endpoints can be mixed in a single run.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.llm import ChatClient, ChatCompletions, ChatResource, build_client
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.results import StageUsage, TFBearResult


@dataclass(frozen=True, slots=True)
class StageClient:
    """The client and model one stage should use."""

    client: ChatClient
    model: str


@dataclass(frozen=True, slots=True)
class StageClients:
    """One `StageClient` per pipeline stage."""

    prior: StageClient
    expansion: StageClient
    scoring: StageClient
    adjudication: StageClient
    explanation: StageClient

    @classmethod
    def single(cls, client: ChatClient, model: str) -> StageClients:
        """Use one client and model for every stage (tests, single-endpoint runs)."""
        stage = StageClient(client=client, model=model)
        return cls(
            prior=stage,
            expansion=stage,
            scoring=stage,
            adjudication=stage,
            explanation=stage,
        )


def _usage_payload(response: Any) -> Mapping[str, Any]:
    """Return the `usage` mapping of a completion response, tolerating missing payloads."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return usage
    for method_name in ("model_dump", "to_dict"):
        method = getattr(usage, method_name, None)
        if callable(method):
            payload = method()
            if isinstance(payload, Mapping):
                return payload
    try:
        attributes = vars(usage)
    except TypeError:
        return {}
    return attributes if isinstance(attributes, Mapping) else {}


class _TrackedCompletions:
    """Delegates `create` and accumulates token counts on the response."""

    def __init__(self, completions: ChatCompletions, tracker: _TokenTracker) -> None:
        self._completions = completions
        self._tracker = tracker

    def create(self, **kwargs: Any) -> Any:
        response = self._completions.create(**kwargs)
        payload = _usage_payload(response)
        self._tracker.prompt_tokens += int(payload.get("prompt_tokens") or 0)
        self._tracker.completion_tokens += int(payload.get("completion_tokens") or 0)
        self._tracker.total_tokens += int(payload.get("total_tokens") or 0)
        return response


class _TrackedChat:
    def __init__(self, chat: ChatResource, tracker: _TokenTracker) -> None:
        self._chat = chat
        self._tracker = tracker

    @property
    def completions(self) -> _TrackedCompletions:
        return _TrackedCompletions(self._chat.completions, self._tracker)


class _TokenTracker:
    """Transparent client wrapper that counts tokens on every chat-completion call."""

    def __init__(self, client: ChatClient) -> None:
        self._chat = client.chat
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    @property
    def chat(self) -> _TrackedChat:
        return _TrackedChat(self._chat, self)


def _build_stage_client(config: TFBearConfig, stage: str) -> StageClient:
    endpoint = config.endpoint_for(stage)
    return StageClient(client=build_client(endpoint), model=endpoint.model)


def build_stage_clients(config: TFBearConfig) -> StageClients:
    """Build one SDK client per stage, honouring per-stage endpoint overrides."""
    return StageClients(
        prior=_build_stage_client(config, "prior"),
        expansion=_build_stage_client(config, "expansion"),
        scoring=_build_stage_client(config, "scoring"),
        adjudication=_build_stage_client(config, "adjudication"),
        explanation=_build_stage_client(config, "explanation"),
    )


def _snapshot(tracker: _TokenTracker, started: float) -> StageUsage:
    return StageUsage(
        latency_seconds=time.perf_counter() - started,
        prompt_tokens=tracker.prompt_tokens,
        completion_tokens=tracker.completion_tokens,
        total_tokens=tracker.total_tokens,
    )


def run_tfbear(
    state: TFBearState,
    config: TFBearConfig,
    *,
    clients: StageClients | None = None,
) -> TFBearResult:
    """Run the TF-BEAR pipeline over `state` and return the final state and summary.

    Args:
        state: Inputs for the run. `text_claim` is required; the other fields are filled here.
        config: Strategy wiring, endpoints, thresholds and global settings.
        clients: Optional per-stage clients. When omitted, one client is built per stage from
            the configured endpoints, which requires the endpoint's API-key variable to be set.

    Raises:
        ValueError: If the claim is empty, or a stage produces no nodes/branches to continue
            with. Failing here keeps the error next to the stage that caused it instead of
            letting an empty run reach the adjudicator.
    """
    if not state.text_claim.strip():
        raise ValueError("run_tfbear: state.text_claim must not be empty.")

    prior, expansion, scoring, adjudication, explanation_strategy = config.strategies()
    stage_clients = clients if clients is not None else build_stage_clients(config)
    logger = config.logger
    usage: dict[str, StageUsage] = {}

    if config.verbose:
        logger.log_stage(1, "Computing P_prior", color="green")
    tracker = _TokenTracker(stage_clients.prior.client)
    started = time.perf_counter()
    state.p_prior = prior.compute_prior(state, tracker, stage_clients.prior.model)
    usage["prior"] = _snapshot(tracker, started)

    if config.verbose:
        logger.log_stage(2, "Expanding reasoning paths", color="blue")
    tracker = _TokenTracker(stage_clients.expansion.client)
    started = time.perf_counter()
    state.nodes = expansion.expand(state, tracker, stage_clients.expansion.model)
    usage["expansion"] = _snapshot(tracker, started)
    if not state.nodes:
        raise ValueError("run_tfbear: expansion produced zero nodes.")

    if config.verbose:
        logger.log_stage(3, "Causal scoring (P_posterior + delta)", color="purple")
    tracker = _TokenTracker(stage_clients.scoring.client)
    started = time.perf_counter()
    state.branches = scoring.score(state, tracker, stage_clients.scoring.model)
    usage["scoring"] = _snapshot(tracker, started)
    if not state.branches:
        raise ValueError("run_tfbear: scoring produced zero branches.")

    if config.verbose:
        logger.log_stage(4, "Adjudicating branches", color="yellow")
    tracker = _TokenTracker(stage_clients.adjudication.client)
    started = time.perf_counter()
    state.adjudication = adjudication.adjudicate(state, tracker, stage_clients.adjudication.model)
    usage["adjudication"] = _snapshot(tracker, started)
    if config.verbose:
        logger.log_adjudication(
            state.adjudication.final_verdict.value,
            fallback=state.adjudication.used_fallback,
        )

    if config.verbose:
        logger.log_stage(5, "Generating explanation", color="cyan")
    tracker = _TokenTracker(stage_clients.explanation.client)
    started = time.perf_counter()
    explanation = explanation_strategy.explain(state, tracker, stage_clients.explanation.model)
    usage["explanation"] = _snapshot(tracker, started)
    if config.verbose:
        logger.log_explanation(explanation)

    return TFBearResult(
        state=state,
        verdict=state.adjudication.final_verdict,
        explanation=explanation,
        usage=usage,
        diagnostics=list(state.diagnostics),
    )

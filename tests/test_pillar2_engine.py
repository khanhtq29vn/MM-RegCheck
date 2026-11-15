"""Tests for the TF-BEAR engine orchestration with fake clients and strategies."""

from __future__ import annotations

from types import SimpleNamespace
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
from mmregcheck.pillar2.engine import (
    StageClient,
    StageClients,
    build_stage_clients,
    run_tfbear,
)
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.pipeline_state import Node, TFBearState
from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace

NON = ComplianceVerdict.NON_COMPLIANT
PARTLY = ComplianceVerdict.PARTLY_COMPLIANT
COMPLIANT = ComplianceVerdict.COMPLIANT


class _NullCompletions:
    def create(self, **kwargs: Any) -> Any:
        raise AssertionError("fake strategies must not call the model")


class _NullClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_NullCompletions())


class _FixedPrior(BasePriorStrategy):
    def compute_prior(
        self, state: TFBearState, client: Any, model: str
    ) -> dict[ComplianceVerdict, float]:
        return {NON: 0.227, PARTLY: 0.400, COMPLIANT: 0.373}


class _FixedExpansion(BaseExpansionStrategy):
    def __init__(self, nodes: list[Node] | None = None) -> None:
        super().__init__()
        self._nodes = (
            nodes
            if nodes is not None
            else [Node(state="strict reasoning", is_terminal=True, meta={"strategy": "R1: Strict"})]
        )
        self.contexts: list[str] = []

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        self.contexts.append(state.retrieved_regulatory_context)
        return list(self._nodes)


class _FixedScoring(BaseScoringStrategy):
    def __init__(self, prior: float = 0.227, posterior: float = 0.982) -> None:
        super().__init__()
        self._prior = prior
        self._posterior = posterior

    def score(self, state: TFBearState, client: Any, model: str) -> list[TFBearBranchTrace]:
        distribution = {NON: self._posterior, PARTLY: (1 - self._posterior) / 2}
        distribution[COMPLIANT] = 1 - sum(distribution.values())
        return [
            TFBearBranchTrace(
                branch_id="P1",
                strategy="R1: Strict",
                prior_prob=self._prior,
                posterior_prob=self._posterior,
                predicted_verdict=NON,
                rationale="strict reasoning",
                posterior_distribution=distribution,
            )
        ]


class _FixedExplanation(BaseExplanationStrategy):
    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        return "The strict branch drives the verdict."


def _config(
    *,
    prior: BasePriorStrategy | None = None,
    expansion: BaseExpansionStrategy | None = None,
    scoring: BaseScoringStrategy | None = None,
    adjudication: TFBearAdjudicationStrategy | None = None,
    explanation: BaseExplanationStrategy | None = None,
) -> TFBearConfig:
    return TFBearConfig(
        prior_strategy=prior or _FixedPrior(),
        expansion_strategy=expansion or _FixedExpansion(),
        scoring_strategy=scoring or _FixedScoring(),
        adjudication_strategy=adjudication or TFBearAdjudicationStrategy(),
        explanation_strategy=explanation or _FixedExplanation(),
    )


def _run(config: TFBearConfig, state: TFBearState | None = None) -> Any:
    return run_tfbear(
        state or TFBearState(text_claim="A spa advertises a chemical peel."),
        config,
        clients=StageClients.single(_NullClient(), "test-model"),
    )


def test_pipeline_runs_every_stage_and_returns_the_final_state() -> None:
    expansion = _FixedExpansion()
    result = _run(_config(expansion=expansion))

    assert result.verdict == NON
    assert result.state.adjudication is not None
    assert result.state.adjudication.final_verdict == NON
    assert len(result.state.nodes) == 1
    assert len(result.state.branches) == 1
    assert result.state.p_prior[NON] == pytest.approx(0.227)
    assert result.explanation == "The strict branch drives the verdict."
    assert set(result.usage) == {
        "prior",
        "expansion",
        "scoring",
        "adjudication",
        "explanation",
    }
    for usage in result.usage.values():
        assert usage.latency_seconds >= 0.0
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


def test_injected_strategies_are_the_ones_actually_used() -> None:
    prior = _FixedPrior()
    expansion = _FixedExpansion()
    scoring = _FixedScoring(prior=0.3, posterior=0.9)
    explanation = _FixedExplanation()

    state = TFBearState(
        text_claim="claim",
        retrieved_regulatory_context="rule one\nrule two",
    )
    result = _run(
        _config(prior=prior, expansion=expansion, scoring=scoring, explanation=explanation),
        state,
    )

    assert result.state.p_prior == {NON: 0.227, PARTLY: 0.400, COMPLIANT: 0.373}
    assert expansion.contexts == ["rule one\nrule two"]
    assert result.state.branches[0].prior_prob == 0.3
    assert result.state.branches[0].posterior_prob == 0.9


def test_no_surviving_branch_uses_the_adjudicator_fallback() -> None:
    # Negative Delta: the branch fails the gate and the fallback vote selects its verdict.
    result = _run(_config(scoring=_FixedScoring(prior=0.6, posterior=0.4)))

    assert result.state.adjudication is not None
    assert result.state.adjudication.used_fallback is True
    assert result.verdict == NON


def test_empty_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match="text_claim"):
        _run(_config(), TFBearState(text_claim="   "))


def test_zero_expansion_nodes_is_reported() -> None:
    expansion = _FixedExpansion(nodes=[])
    with pytest.raises(ValueError, match="expansion produced zero nodes"):
        _run(_config(expansion=expansion))


def test_zero_scored_branches_is_reported() -> None:
    class _EmptyScoring(BaseScoringStrategy):
        def score(self, state: TFBearState, client: Any, model: str) -> list[TFBearBranchTrace]:
            return []

    with pytest.raises(ValueError, match="scoring produced zero branches"):
        _run(_config(scoring=_EmptyScoring()))


def test_diagnostics_are_copied_from_the_state_to_the_result() -> None:
    state = TFBearState(text_claim="claim", diagnostics=["recorded before the run"])

    result = _run(_config(), state)

    assert result.diagnostics == ["recorded before the run"]
    assert result.diagnostics is not state.diagnostics


def test_token_usage_is_tracked_per_stage() -> None:
    class _UsageResponse:
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)

    class _UsageCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            return _UsageResponse()

    class _UsagePrior(BasePriorStrategy):
        def compute_prior(
            self, state: TFBearState, client: Any, model: str
        ) -> dict[ComplianceVerdict, float]:
            client.chat.completions.create(model=model, messages=[])
            return {NON: 0.5, PARTLY: 0.25, COMPLIANT: 0.25}

    completions = _UsageCompletions()
    client = _NullClient()
    client.chat = SimpleNamespace(completions=completions)

    result = run_tfbear(
        TFBearState(text_claim="claim"),
        _config(prior=_UsagePrior()),
        clients=StageClients.single(client, "test-model"),
    )

    assert completions.calls == 1
    assert result.usage["prior"].prompt_tokens == 3
    assert result.usage["prior"].completion_tokens == 1
    assert result.usage["prior"].total_tokens == 4
    assert result.usage["scoring"].total_tokens == 0


def test_build_stage_clients_honours_stage_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    def fake_build_client(endpoint: EndpointConfig) -> Any:
        built.append(endpoint.base_url)
        return _NullClient()

    monkeypatch.setattr("mmregcheck.pillar2.engine.build_client", fake_build_client)
    base = EndpointConfig(
        base_url="https://base.invalid/v1",
        api_key_env="MMREG_TEST_API_KEY",
        model="base-model",
    )
    scoring = EndpointConfig(
        base_url="https://scoring.invalid/v1",
        api_key_env="MMREG_TEST_API_KEY",
        model="scoring-model",
    )
    config = TFBearConfig(
        prior_strategy=_FixedPrior(),
        expansion_strategy=_FixedExpansion(),
        scoring_strategy=_FixedScoring(),
        adjudication_strategy=TFBearAdjudicationStrategy(),
        explanation_strategy=_FixedExplanation(),
        endpoint=base,
        stage_endpoints={"scoring": scoring},
    )

    clients = build_stage_clients(config)

    assert len(built) == 5
    assert clients.prior == StageClient(client=clients.prior.client, model="base-model")
    assert clients.scoring.model == "scoring-model"
    assert built.count("https://scoring.invalid/v1") == 1


def test_stage_clients_single_reuses_one_pair() -> None:
    client = _NullClient()
    clients = StageClients.single(client, "test-model")

    assert clients.prior == clients.explanation
    assert clients.prior.client is client
    assert clients.prior.model == "test-model"

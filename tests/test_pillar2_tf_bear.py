"""Tests for the public TF-BEAR facade: runs, prompt packs, samples and the exported API.

Clients are fakes that mimic the OpenAI-compatible shape, so no test uses the network.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mmregcheck
import mmregcheck.pillar2 as pillar2
from mmregcheck.pillar2 import (
    TFBear,
    TFBearAdjudicator,
    TFBearConfig,
    TFBearResult,
)
from mmregcheck.pillar2.adjudication import TFBearAdjudicationStrategy
from mmregcheck.pillar2.base import (
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.engine import StageClients, run_tfbear
from mmregcheck.pillar2.expansion import (
    CoTSCExpansionStrategy,
    StandardExpansionStrategy,
    ToTExpansionStrategy,
)
from mmregcheck.pillar2.explainer import GenerativeExplanationStrategy
from mmregcheck.pillar2.pipeline_state import Node, TFBearState
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.prompts import EXPANSIONS, ExpansionName, load_prompt_pack
from mmregcheck.pillar2.results import TFBearAdjudicationResult
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy
from mmregcheck.schemas import (
    BusinessMetadata,
    ComplianceVerdict,
    MedRegSampleInput,
    RegulatoryDomain,
    TFBearBranchTrace,
)

NON = ComplianceVerdict.NON_COMPLIANT
PARTLY = ComplianceVerdict.PARTLY_COMPLIANT
COMPLIANT = ComplianceVerdict.COMPLIANT

DOMAINS = ("Beauty", "Healthcare")

EXPECTED_PUBLIC_API = {
    "TFBear",
    "TFBearConfig",
    "TFBearState",
    "TFBearResult",
    "EndpointConfig",
    "TFBearAdjudicator",
    "Node",
    "BasePriorStrategy",
    "BaseExpansionStrategy",
    "BaseScoringStrategy",
    "BaseAdjudicationStrategy",
    "BaseExplanationStrategy",
}

# The scoring request is the only logprob prompt that renders a reasoning path.
_SCORING_MARKER = "[Reasoning path]"
_PRIOR_DISTRIBUTION = {"0": 0.20, "1": 0.30, "2": 0.50}
_SCORING_DISTRIBUTION = {"0": 0.90, "1": 0.06, "2": 0.04}
_REASONING = "The advertised peel exceeds the facility's registered scope."
_EXPLANATION = "The peel exceeds the registered scope, so the claim is Non-Compliant."


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
    def __init__(self, content: str, logprobs: _ChoiceLogprobs | None = None) -> None:
        self.message = _Message(content)
        self.logprobs = logprobs


class _Response:
    def __init__(self, choices: list[_Choice]) -> None:
        self.choices = choices


def _logprob_response(distribution: dict[str, float]) -> _Response:
    top = [_TopLogprob(token, math.log(probability)) for token, probability in distribution.items()]
    position = _TokenLogprob(top[0].token, top[0].logprob, top)
    return _Response([_Choice("0", _ChoiceLogprobs([position]))])


class _PipelineCompletions:
    """Answers logprob requests with a distribution and text requests with fixed text."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if kwargs.get("logprobs"):
            distribution = (
                _SCORING_DISTRIBUTION if _SCORING_MARKER in prompt else _PRIOR_DISTRIBUTION
            )
            return _logprob_response(distribution)
        text = _EXPLANATION if "Verdict:" in prompt else _REASONING
        return _Response([_Choice(text)])


class _PipelineClient:
    def __init__(self) -> None:
        self.completions = _PipelineCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


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


class _RecordingExpansion(BaseExpansionStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[str] = []

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        self.contexts.append(state.retrieved_regulatory_context)
        return [Node(state="strict reasoning", is_terminal=True, meta={"strategy": "R1: Strict"})]


class _FixedScoring(BaseScoringStrategy):
    def score(self, state: TFBearState, client: Any, model: str) -> list[TFBearBranchTrace]:
        return [
            TFBearBranchTrace(
                branch_id="P1",
                strategy="R1: Strict",
                prior_prob=0.227,
                posterior_prob=0.982,
                predicted_verdict=NON,
                rationale="strict reasoning",
                posterior_distribution={NON: 0.982, PARTLY: 0.010, COMPLIANT: 0.008},
            )
        ]


class _FixedExplanation(BaseExplanationStrategy):
    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        return "The strict branch drives the verdict."


def _fake_config(
    *,
    prior: BasePriorStrategy | None = None,
    expansion: BaseExpansionStrategy | None = None,
) -> TFBearConfig:
    return TFBearConfig(
        prior_strategy=prior or _FixedPrior(),
        expansion_strategy=expansion or _RecordingExpansion(),
        scoring_strategy=_FixedScoring(),
        adjudication_strategy=TFBearAdjudicationStrategy(),
        explanation_strategy=_FixedExplanation(),
    )


def _fake_bear(
    *,
    prior: BasePriorStrategy | None = None,
    expansion: BaseExpansionStrategy | None = None,
) -> TFBear:
    return TFBear(
        _fake_config(prior=prior, expansion=expansion),
        clients=StageClients.single(_NullClient(), "test-model"),
    )


def _sample() -> MedRegSampleInput:
    return MedRegSampleInput(
        sample_id="MEDREG-TEST-0001",
        domain=RegulatoryDomain.BEAUTY,
        text_claim="A spa advertises a TCA chemical peel.",
        image_rel_path="images/ignored.jpg",
        business_info=BusinessMetadata(facility_name="Test Spa"),
        retrieved_regulatory_context=["Rule one.", "Rule two."],
    )


def test_run_executes_the_five_stages_with_fake_clients() -> None:
    client = _PipelineClient()
    config = TFBearConfig(
        prior_strategy=LogprobPriorStrategy(prompt_template="Classify: {text_claim}"),
        expansion_strategy=StandardExpansionStrategy(
            n_paths=1,
            prompt_template="{text_claim}|{retrieved_regulatory_context}",
        ),
        scoring_strategy=CausalDeltaScoringStrategy(
            prompt_template=f"{_SCORING_MARKER} {{state}}\n{{text_claim}}"
        ),
        adjudication_strategy=TFBearAdjudicationStrategy(),
        explanation_strategy=GenerativeExplanationStrategy(
            prompt_template="Verdict: {verdict}\n{paths}"
        ),
    )
    bear = TFBear(config, clients=StageClients.single(client, "test-model"))

    result = bear.run(
        "A spa advertises a TCA chemical peel.",
        "Basic Skincare Spa: non-invasive cleansing only.",
    )

    assert isinstance(result, TFBearResult)
    assert result.verdict == NON
    assert result.state.adjudication is not None
    assert result.state.adjudication.used_fallback is False
    assert result.state.branches[0].rationale == _REASONING
    assert result.explanation == _EXPLANATION
    assert set(result.usage) == {
        "prior",
        "expansion",
        "scoring",
        "adjudication",
        "explanation",
    }
    # Prior, expansion, scoring and explanation each make one call; adjudication makes none.
    assert len(client.completions.calls) == 4
    expansion_prompt = next(
        call["messages"][-1]["content"]
        for call in client.completions.calls
        if not call.get("logprobs") and "Verdict:" not in call["messages"][-1]["content"]
    )
    assert "Basic Skincare Spa: non-invasive cleansing only." in expansion_prompt


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_from_domain_builds_every_prompt_pack(domain: str, expansion: ExpansionName) -> None:
    endpoint = pillar2.EndpointConfig(
        base_url="https://endpoint.invalid/v1",
        api_key_env="MMREG_TEST_API_KEY",
        model="test-model",
    )

    bear = TFBear.from_domain(
        domain,
        expansion,
        endpoint=endpoint,
        tau_posterior=0.6,
        tau_delta=0.3,
    )

    pack = load_prompt_pack(RegulatoryDomain(domain), expansion)
    expected_expansion = {
        "standard": StandardExpansionStrategy,
        "cot-sc": CoTSCExpansionStrategy,
        "tot": ToTExpansionStrategy,
    }[expansion]
    assert bear.config.endpoint == endpoint
    assert bear.config.tau_posterior == 0.6
    assert bear.config.tau_delta == 0.3
    assert bear.config.prior_strategy is not None
    assert bear.config.prior_strategy.prompt_template == pack.prior
    assert isinstance(bear.config.expansion_strategy, expected_expansion)


def test_run_sample_keeps_the_retrieved_context_when_open_book() -> None:
    expansion = _RecordingExpansion()
    bear = _fake_bear(expansion=expansion)

    prediction = bear.run_sample(_sample())

    assert prediction.sample_id == "MEDREG-TEST-0001"
    assert prediction.overall_verdict == NON
    assert len(prediction.explanation) >= 5
    assert expansion.contexts == ["Rule one.\nRule two."]


def test_run_sample_drops_the_retrieved_context_when_close_book() -> None:
    expansion = _RecordingExpansion()
    bear = _fake_bear(expansion=expansion)

    prediction = bear.run_sample(_sample(), open_book=False)

    assert prediction.sample_id == "MEDREG-TEST-0001"
    assert prediction.overall_verdict == NON
    assert expansion.contexts == [""]


def test_custom_strategy_injected_through_the_config_is_used() -> None:
    result = _fake_bear(prior=_FixedPrior()).run("A spa advertises a chemical peel.")

    assert result.state.p_prior == {NON: 0.227, PARTLY: 0.400, COMPLIANT: 0.373}
    assert result.verdict == NON


def test_public_api_names_are_exported() -> None:
    assert set(pillar2.__all__) == EXPECTED_PUBLIC_API
    for name in pillar2.__all__:
        assert getattr(pillar2, name) is not None


def test_tf_bear_reexports_the_legacy_names() -> None:
    import mmregcheck.pillar2.tf_bear as tf_bear_module

    assert tf_bear_module.TFBearAdjudicator is TFBearAdjudicator
    assert tf_bear_module.TFBearAdjudicationResult is TFBearAdjudicationResult
    assert tf_bear_module.TFBearConfig is TFBearConfig
    assert tf_bear_module.TFBearResult is TFBearResult
    assert tf_bear_module.run_tfbear is run_tfbear


def test_from_domain_applies_expansion_params() -> None:
    endpoint = pillar2.EndpointConfig(
        base_url="https://endpoint.invalid/v1",
        api_key_env="MMREG_TEST_API_KEY",
        model="test-model",
    )

    bear = TFBear.from_domain(
        "Beauty",
        "tot",
        endpoint=endpoint,
        expansion_params={"breadth_limit": 2},
    )

    expansion = bear.config.expansion_strategy
    assert isinstance(expansion, ToTExpansionStrategy)
    assert expansion.breadth_limit == 2


def test_from_domain_rejects_an_unknown_expansion_param() -> None:
    endpoint = pillar2.EndpointConfig(
        base_url="https://endpoint.invalid/v1",
        api_key_env="MMREG_TEST_API_KEY",
        model="test-model",
    )

    with pytest.raises(ValueError, match="accepted keys"):
        TFBear.from_domain(
            "Beauty",
            "standard",
            endpoint=endpoint,
            expansion_params={"breadth_limit": 2},
        )


def test_importing_pillar2_does_not_import_the_optional_llm_dependencies() -> None:
    src_root = Path(mmregcheck.__file__).resolve().parents[1]
    code = "\n".join(
        [
            "import sys",
            "from importlib.abc import MetaPathFinder",
            "from pathlib import Path",
            "import mmregcheck",
            "",
            "class _BlockOptional(MetaPathFinder):",
            "    def find_spec(self, fullname, path, target=None):",
            "        blocked = fullname == 'openai' or fullname.startswith('openai.')",
            "        if blocked or fullname == 'json_repair':",
            "            raise ImportError(fullname + ' is not installed')",
            "        return None",
            "",
            "sys.meta_path.insert(0, _BlockOptional())",
            "import mmregcheck.pillar2",
            f"assert Path(mmregcheck.__file__).resolve().parents[1] == Path({str(src_root)!r})",
            "for name in ('openai', 'json_repair'):",
            "    assert name not in sys.modules, name + ' was imported eagerly'",
        ]
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root)

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

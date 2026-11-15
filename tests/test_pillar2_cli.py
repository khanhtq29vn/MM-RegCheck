"""Tests for the `run-tfbear` CLI command using fake strategies and clients."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import mmregcheck.cli as cli
from mmregcheck.evaluator import MedRegEvaluator
from mmregcheck.pillar2.adjudication import TFBearAdjudicationStrategy
from mmregcheck.pillar2.base import (
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.engine import StageClients, TFBearResult
from mmregcheck.pillar2.pipeline_state import Node, TFBearState
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

runner = CliRunner()


class _NullClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))


class _Prior(BasePriorStrategy):
    def compute_prior(
        self, state: TFBearState, client: Any, model: str
    ) -> dict[ComplianceVerdict, float]:
        return {NON: 0.500, PARTLY: 0.300, COMPLIANT: 0.200}


class _Expansion(BaseExpansionStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[str] = []

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        self.contexts.append(state.retrieved_regulatory_context)
        return [Node(state="strict reasoning", is_terminal=True, meta={"strategy": "R1: Strict"})]


class _Scoring(BaseScoringStrategy):
    def score(self, state: TFBearState, client: Any, model: str) -> list[TFBearBranchTrace]:
        return [
            TFBearBranchTrace(
                branch_id="P1",
                strategy="R1: Strict",
                prior_prob=0.500,
                posterior_prob=0.950,
                predicted_verdict=NON,
                rationale="strict reasoning",
                posterior_distribution={NON: 0.95, PARTLY: 0.03, COMPLIANT: 0.02},
            )
        ]


class _Explanation(BaseExplanationStrategy):
    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        return "The strict branch drives the non-compliant verdict."


def _config() -> TFBearConfig:
    return TFBearConfig(
        prior_strategy=_Prior(),
        expansion_strategy=_Expansion(),
        scoring_strategy=_Scoring(),
        adjudication_strategy=TFBearAdjudicationStrategy(),
        explanation_strategy=_Explanation(),
    )


def _write_samples(path: Path, count: int = 2) -> None:
    lines = []
    for index in range(count):
        sample = MedRegSampleInput(
            sample_id=f"MEDREG-TEST-{index + 1:04d}",
            domain=RegulatoryDomain.BEAUTY,
            text_claim="A spa advertises a TCA chemical peel.",
            image_rel_path="images/ignored.jpg",
            business_info=BusinessMetadata(facility_name="Test Spa"),
            retrieved_regulatory_context=["Rule one.", "Rule two."],
        )
        lines.append(sample.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_fakes(monkeypatch: pytest.MonkeyPatch, config: TFBearConfig) -> None:
    def fake_build_config(**kwargs: Any) -> TFBearConfig:
        return config

    def fake_build_clients(_config: TFBearConfig) -> StageClients:
        return StageClients.single(_NullClient(), "test-model")

    monkeypatch.setattr(cli, "_build_config", fake_build_config)
    monkeypatch.setattr(cli, "build_stage_clients", fake_build_clients)


def test_run_tfbear_writes_submission_compatible_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples)
    config = _config()
    _install_fakes(monkeypatch, config)

    result = runner.invoke(
        cli.app,
        [
            "run-tfbear",
            str(samples),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    predictions = MedRegEvaluator.validate_submission_file(output)
    assert [prediction.sample_id for prediction in predictions] == [
        "MEDREG-TEST-0001",
        "MEDREG-TEST-0002",
    ]
    assert all(prediction.overall_verdict == NON for prediction in predictions)
    assert all(len(prediction.explanation) >= 5 for prediction in predictions)
    expansion = config.expansion_strategy
    assert isinstance(expansion, _Expansion)
    assert expansion.contexts == ["Rule one.\nRule two.", "Rule one.\nRule two."]


def test_run_tfbear_close_book_drops_the_retrieved_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    config = _config()
    _install_fakes(monkeypatch, config)

    result = runner.invoke(
        cli.app,
        [
            "run-tfbear",
            str(samples),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
            "--out",
            str(output),
            "--close-book",
        ],
    )

    assert result.exit_code == 0, result.stdout
    expansion = config.expansion_strategy
    assert isinstance(expansion, _Expansion)
    assert expansion.contexts == [""]


def test_run_tfbear_requires_an_endpoint_and_model(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    _write_samples(samples, count=1)

    result = runner.invoke(cli.app, ["run-tfbear", str(samples)])

    assert result.exit_code != 0


def _write_samples_for(path: Path, domains: list[RegulatoryDomain]) -> None:
    lines = []
    for index, domain in enumerate(domains, start=1):
        sample = MedRegSampleInput(
            sample_id=f"MEDREG-TEST-{index:04d}",
            domain=domain,
            text_claim="A facility advertises a treatment.",
            image_rel_path="images/ignored.jpg",
            business_info=BusinessMetadata(facility_name="Test Facility"),
            retrieved_regulatory_context=["Rule one."],
        )
        lines.append(sample.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_pack_fakes(
    monkeypatch: pytest.MonkeyPatch,
    loaded: list[tuple[RegulatoryDomain, str]],
) -> None:
    real_load_prompt_pack = cli.load_prompt_pack

    def fake_load_prompt_pack(domain: RegulatoryDomain, expansion: str) -> Any:
        loaded.append((domain, expansion))
        return real_load_prompt_pack(domain, expansion)  # type: ignore[arg-type]

    def fake_build_clients(_config: TFBearConfig) -> StageClients:
        return StageClients.single(_NullClient(), "test-model")

    def fake_run(
        state: TFBearState, config: TFBearConfig, *, clients: StageClients
    ) -> TFBearResult:
        return TFBearResult(
            state=state,
            verdict=NON,
            explanation="The strict branch drives the verdict.",
            usage={},
        )

    monkeypatch.setattr(cli, "load_prompt_pack", fake_load_prompt_pack)
    monkeypatch.setattr(cli, "build_stage_clients", fake_build_clients)
    monkeypatch.setattr(cli, "run_tfbear", fake_run)


def test_run_tfbear_uses_an_explicit_domain_and_expansion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    loaded: list[tuple[RegulatoryDomain, str]] = []
    _install_pack_fakes(monkeypatch, loaded)

    result = runner.invoke(
        cli.app,
        [
            "run-tfbear",
            str(samples),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
            "--out",
            str(output),
            "--domain",
            "Healthcare",
            "--expansion",
            "tot",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert loaded == [(RegulatoryDomain.HEALTHCARE, "tot")]
    predictions = MedRegEvaluator.validate_submission_file(output)
    assert len(predictions) == 1


def test_run_tfbear_prefers_each_sample_domain_when_domain_is_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples_for(samples, [RegulatoryDomain.BEAUTY, RegulatoryDomain.HEALTHCARE])
    loaded: list[tuple[RegulatoryDomain, str]] = []
    _install_pack_fakes(monkeypatch, loaded)

    result = runner.invoke(
        cli.app,
        [
            "run-tfbear",
            str(samples),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
            "--out",
            str(output),
            "--expansion",
            "cot-sc",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert loaded == [
        (RegulatoryDomain.BEAUTY, "cot-sc"),
        (RegulatoryDomain.HEALTHCARE, "cot-sc"),
    ]


def test_run_tfbear_rejects_an_unknown_expansion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    _write_samples(samples, count=1)
    loaded: list[tuple[RegulatoryDomain, str]] = []
    _install_pack_fakes(monkeypatch, loaded)

    result = runner.invoke(
        cli.app,
        [
            "run-tfbear",
            str(samples),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
            "--expansion",
            "cot",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown expansion" in result.output
    assert loaded == []

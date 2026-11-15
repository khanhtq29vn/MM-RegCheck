"""Tests for the `run-tfbear` CLI command using fake strategies and clients."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import mmregcheck.cli as cli
from mmregcheck.evaluator import MedRegEvaluator
from mmregcheck.pillar2 import run_support
from mmregcheck.pillar2.adjudication import TFBearAdjudicationStrategy
from mmregcheck.pillar2.base import (
    BaseExpansionStrategy,
    BaseExplanationStrategy,
    BasePriorStrategy,
    BaseScoringStrategy,
)
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.diagnostics import record_diagnostic
from mmregcheck.pillar2.engine import StageClients
from mmregcheck.pillar2.pipeline_state import Node, TFBearState
from mmregcheck.pillar2.prompts import PromptPack
from mmregcheck.pillar2.results import TFBearResult
from mmregcheck.pillar2.sample_adapter import write_predictions
from mmregcheck.schemas import (
    BusinessMetadata,
    ComplianceVerdict,
    MedRegSampleInput,
    RegulatoryDomain,
    SubmissionPrediction,
    TFBearBranchTrace,
)

NON = ComplianceVerdict.NON_COMPLIANT
PARTLY = ComplianceVerdict.PARTLY_COMPLIANT
COMPLIANT = ComplianceVerdict.COMPLIANT

runner = CliRunner()

BASE_ARGS = ["--base-url", "https://endpoint.invalid/v1", "--model", "test-model"]


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


class _WarnPrior(BasePriorStrategy):
    def compute_prior(
        self, state: TFBearState, client: Any, model: str
    ) -> dict[ComplianceVerdict, float]:
        record_diagnostic(state.diagnostics, "synthetic warning one")
        record_diagnostic(state.diagnostics, "synthetic warning two")
        return {NON: 0.500, PARTLY: 0.300, COMPLIANT: 0.200}


def _config(**overrides: Any) -> TFBearConfig:
    strategies: dict[str, Any] = {
        "prior_strategy": _Prior(),
        "expansion_strategy": _Expansion(),
        "scoring_strategy": _Scoring(),
        "adjudication_strategy": TFBearAdjudicationStrategy(),
        "explanation_strategy": _Explanation(),
    }
    strategies.update(overrides)
    return TFBearConfig(**strategies)


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


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    config: TFBearConfig,
    *,
    captured: list[dict[str, Any]] | None = None,
) -> None:
    def fake_build_config(_pack: Any, **kwargs: Any) -> TFBearConfig:
        if captured is not None:
            captured.append(kwargs)
        return config

    def fake_build_clients(_config: TFBearConfig) -> StageClients:
        return StageClients.single(_NullClient(), "test-model")

    monkeypatch.setattr(run_support, "build_config", fake_build_config)
    monkeypatch.setattr(run_support, "build_stage_clients", fake_build_clients)


def _invoke(samples: Path, output: Path, *extra: str) -> Any:
    return runner.invoke(
        cli.app,
        ["run-tfbear", str(samples), *BASE_ARGS, "--out", str(output), *extra],
    )


def _trace_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _result_for(sample: MedRegSampleInput) -> TFBearResult:
    return TFBearResult(
        state=TFBearState(text_claim=sample.text_claim),
        verdict=NON,
        explanation="The strict branch drives the verdict.",
        usage={},
    )


def test_run_tfbear_writes_submission_compatible_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples)
    config = _config()
    _install_fakes(monkeypatch, config)

    result = _invoke(samples, output)

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

    result = _invoke(samples, output, "--close-book")

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


def _install_to_config_capture(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[dict[str, Any]],
) -> None:
    """Capture the keyword arguments the run passes to `PromptPack.to_config`."""
    real_to_config = PromptPack.to_config

    def fake_to_config(self: PromptPack, **kwargs: Any) -> TFBearConfig:
        captured.append(kwargs)
        return real_to_config(self, **kwargs)

    def fake_build_clients(_config: TFBearConfig) -> StageClients:
        return StageClients.single(_NullClient(), "test-model")

    def fake_run_sample(
        _config: TFBearConfig,
        _clients: StageClients,
        sample: MedRegSampleInput,
        *,
        close_book: bool,
    ) -> TFBearResult:
        return _result_for(sample)

    monkeypatch.setattr(PromptPack, "to_config", fake_to_config)
    monkeypatch.setattr(run_support, "build_stage_clients", fake_build_clients)
    monkeypatch.setattr(run_support, "run_one_sample", fake_run_sample)


def _install_pack_fakes(
    monkeypatch: pytest.MonkeyPatch,
    loaded: list[tuple[RegulatoryDomain, str]],
) -> None:
    real_load_prompt_pack = run_support.load_prompt_pack

    def fake_load_prompt_pack(domain: RegulatoryDomain, expansion: str) -> Any:
        loaded.append((domain, expansion))
        return real_load_prompt_pack(domain, expansion)  # type: ignore[arg-type]

    def fake_build_clients(_config: TFBearConfig) -> StageClients:
        return StageClients.single(_NullClient(), "test-model")

    def fake_run_sample(
        _config: TFBearConfig,
        _clients: StageClients,
        sample: MedRegSampleInput,
        *,
        close_book: bool,
    ) -> TFBearResult:
        return _result_for(sample)

    monkeypatch.setattr(run_support, "load_prompt_pack", fake_load_prompt_pack)
    monkeypatch.setattr(run_support, "build_stage_clients", fake_build_clients)
    monkeypatch.setattr(run_support, "run_one_sample", fake_run_sample)


def test_run_tfbear_uses_an_explicit_domain_and_expansion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    loaded: list[tuple[RegulatoryDomain, str]] = []
    _install_pack_fakes(monkeypatch, loaded)

    result = _invoke(samples, output, "--domain", "Healthcare", "--expansion", "tot")

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

    result = _invoke(samples, output, "--expansion", "cot-sc")

    assert result.exit_code == 0, result.stdout
    assert loaded == [
        (RegulatoryDomain.BEAUTY, "cot-sc"),
        (RegulatoryDomain.HEALTHCARE, "cot-sc"),
    ]


def test_run_tfbear_defaults_to_standard_and_the_sample_domain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples_for(samples, [RegulatoryDomain.BEAUTY, RegulatoryDomain.HEALTHCARE])
    loaded: list[tuple[RegulatoryDomain, str]] = []
    _install_pack_fakes(monkeypatch, loaded)

    result = _invoke(samples, output)

    assert result.exit_code == 0, result.stdout
    assert loaded == [
        (RegulatoryDomain.BEAUTY, "standard"),
        (RegulatoryDomain.HEALTHCARE, "standard"),
    ]


def test_run_tfbear_rejects_an_unknown_expansion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    _write_samples(samples, count=1)
    loaded: list[tuple[RegulatoryDomain, str]] = []
    _install_pack_fakes(monkeypatch, loaded)

    result = _invoke(samples, tmp_path / "predictions.jsonl", "--expansion", "cot")

    assert result.exit_code != 0
    assert "Unknown expansion" in (result.output + result.stderr)
    assert loaded == []


def test_omitted_sampling_flags_pass_no_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    captured: list[dict[str, Any]] = []
    _install_to_config_capture(monkeypatch, captured)

    result = _invoke(samples, output)

    assert result.exit_code == 0, result.stdout
    assert len(captured) == 1
    assert set(captured[0]) == {"endpoint", "tau_posterior", "tau_delta"}
    assert captured[0]["tau_posterior"] == 0.50
    assert captured[0]["tau_delta"] == 0.50


def test_sampling_and_expansion_flags_are_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    captured: list[dict[str, Any]] = []
    _install_to_config_capture(monkeypatch, captured)

    result = _invoke(
        samples,
        output,
        "--temperature",
        "0.7",
        "--top-p",
        "0.8",
        "--max-tokens",
        "256",
        "--elicitation-temperature",
        "0.1",
        "--elicitation-top-p",
        "0.9",
        "--top-logprobs",
        "7",
        "--expansion-param",
        "n_paths=2",
        "--expansion-param",
        "temperature=0.3",
        "--verbose",
    )

    assert result.exit_code == 0, result.stdout
    run_calls = [kwargs for kwargs in captured if "endpoint" in kwargs]
    assert len(run_calls) == 1
    kwargs = run_calls[0]
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_tokens"] == 256
    assert kwargs["elicitation_temperature"] == 0.1
    assert kwargs["elicitation_top_p"] == 0.9
    assert kwargs["top_logprobs"] == 7
    assert kwargs["verbose"] is True
    assert kwargs["expansion_params"] == {"n_paths": 2, "temperature": 0.3}


def test_unknown_expansion_param_key_fails_before_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    captured: list[dict[str, Any]] = []
    _install_fakes(monkeypatch, _config(), captured=captured)

    result = _invoke(samples, output, "--expansion-param", "bogus=1")

    assert result.exit_code != 0
    message = result.output + result.stderr
    assert "bogus" in message
    assert "n_paths" in message
    assert captured == []
    assert not output.exists()


def test_bad_expansion_param_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    _install_fakes(monkeypatch, _config())

    result = _invoke(samples, output, "--expansion-param", "n_paths=two")

    assert result.exit_code != 0
    message = result.output + result.stderr
    assert "expected an int" in message
    assert "float" in message
    assert not output.exists()


def test_warnings_are_visible_on_stderr_without_verbose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    _install_fakes(monkeypatch, _config(prior_strategy=_WarnPrior()))

    result = _invoke(samples, output)

    assert result.exit_code == 0, result.stdout
    assert result.stderr.count("synthetic warning one") == 1
    assert result.stderr.count("synthetic warning two") == 1
    assert "samples_done=1" in result.stderr
    assert "samples_failed=0" in result.stderr
    assert "samples_skipped=0" in result.stderr
    assert "diagnostics=2" in result.stderr


def test_trace_file_records_the_header_and_every_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    _install_fakes(monkeypatch, _config())

    result = _invoke(samples, output, "--domain", "Beauty")

    assert result.exit_code == 0, result.stdout
    trace_path = output.with_name(output.name + ".trace.jsonl")
    records = _trace_records(trace_path)
    assert [record["record"] for record in records] == ["header", "sample"]

    header = records[0]
    assert header["schema"] == "tfbear-trace-v1"
    assert header["domain"] == "Beauty"
    assert header["expansion"] == "standard"
    assert header["model"] == "test-model"
    assert header["base_url"] == "https://endpoint.invalid/v1"
    assert header["tau_posterior"] == 0.50
    assert header["tau_delta"] == 0.50
    assert header["sampling"]["temperature"] == 0.2
    assert header["sampling"]["top_logprobs"] == 5
    assert header["expansion_params"] is None
    assert header["packs"]["Beauty"]["expansion_strategy"]["strategy"] == "_Expansion"

    sample = records[1]
    assert sample["sample_id"] == "MEDREG-TEST-0001"
    assert sample["status"] == "ok"
    assert sample["verdict"] == NON.value
    assert sample["used_fallback"] is False
    assert sample["p_prior"] == {
        NON.value: 0.500,
        PARTLY.value: 0.300,
        COMPLIANT.value: 0.200,
    }
    assert len(sample["branches"]) == 1
    branch = sample["branches"][0]
    assert branch["branch_id"] == "P1"
    assert branch["strategy"] == "R1: Strict"
    assert branch["predicted_verdict"] == NON.value
    assert branch["prior_prob"] == 0.500
    assert branch["posterior_prob"] == 0.950
    assert (
        branch["log_odds_delta"]
        == TFBearBranchTrace(
            branch_id="P1",
            strategy="R1: Strict",
            prior_prob=0.500,
            posterior_prob=0.950,
            predicted_verdict=NON,
            rationale="strict reasoning",
        ).log_odds_delta
    )
    assert branch["gate"] == "survived"
    assert sample["n_surviving"] == 1
    assert set(sample["usage"]) == {"prior", "expansion", "scoring", "adjudication", "explanation"}
    assert sample["explanation"] == "The strict branch drives the non-compliant verdict."
    assert sample["diagnostics"] == []


def test_trace_file_never_contains_the_api_key_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=1)
    _install_fakes(monkeypatch, _config())
    monkeypatch.setenv("MMREG_TRACE_KEY", "sk-super-secret-value")

    result = _invoke(samples, output, "--api-key-env", "MMREG_TRACE_KEY")

    assert result.exit_code == 0, result.stdout
    trace_text = output.with_name(output.name + ".trace.jsonl").read_text(encoding="utf-8")
    assert "MMREG_TRACE_KEY" in trace_text
    assert "sk-super-secret-value" not in trace_text


def test_explicit_trace_path_is_honoured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    trace = tmp_path / "custom" / "trace.jsonl"
    _write_samples(samples, count=1)
    _install_fakes(monkeypatch, _config())

    result = _invoke(samples, output, "--trace-path", str(trace))

    assert result.exit_code == 0, result.stdout
    assert [record["record"] for record in _trace_records(trace)] == ["header", "sample"]
    assert not output.with_name(output.name + ".trace.jsonl").exists()


def test_resume_skips_samples_already_in_the_predictions_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=2)
    _install_fakes(monkeypatch, _config())
    write_predictions(
        output,
        [
            SubmissionPrediction(
                sample_id="MEDREG-TEST-0001",
                overall_verdict=NON,
                explanation="Kept from the interrupted run.",
            )
        ],
    )
    processed: list[str] = []

    def fake_run_sample(
        _config: TFBearConfig,
        _clients: StageClients,
        sample: MedRegSampleInput,
        *,
        close_book: bool,
    ) -> TFBearResult:
        processed.append(sample.sample_id)
        return _result_for(sample)

    monkeypatch.setattr(run_support, "run_one_sample", fake_run_sample)

    result = _invoke(samples, output, "--resume")

    assert result.exit_code == 0, result.stdout
    assert processed == ["MEDREG-TEST-0002"]
    predictions = MedRegEvaluator.validate_submission_file(output)
    assert [prediction.sample_id for prediction in predictions] == [
        "MEDREG-TEST-0001",
        "MEDREG-TEST-0002",
    ]
    assert predictions[0].explanation == "Kept from the interrupted run."
    assert "samples_done=1" in result.stderr
    assert "samples_skipped=1" in result.stderr
    records = _trace_records(output.with_name(output.name + ".trace.jsonl"))
    assert [record["record"] for record in records] == ["header", "sample"]
    assert records[1]["sample_id"] == "MEDREG-TEST-0002"


def test_without_resume_a_previous_predictions_file_is_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=2)
    _install_fakes(monkeypatch, _config())
    write_predictions(
        output,
        [
            SubmissionPrediction(
                sample_id="STALE-0001",
                overall_verdict=NON,
                explanation="Stale prediction from an earlier run.",
            )
        ],
    )
    processed: list[str] = []

    def fake_run_sample(
        _config: TFBearConfig,
        _clients: StageClients,
        sample: MedRegSampleInput,
        *,
        close_book: bool,
    ) -> TFBearResult:
        processed.append(sample.sample_id)
        return _result_for(sample)

    monkeypatch.setattr(run_support, "run_one_sample", fake_run_sample)

    result = _invoke(samples, output)

    assert result.exit_code == 0, result.stdout
    assert processed == ["MEDREG-TEST-0001", "MEDREG-TEST-0002"]
    predictions = MedRegEvaluator.validate_submission_file(output)
    assert [prediction.sample_id for prediction in predictions] == [
        "MEDREG-TEST-0001",
        "MEDREG-TEST-0002",
    ]


def test_a_failing_sample_is_recorded_and_the_run_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples, count=3)
    _install_fakes(monkeypatch, _config())
    processed: list[str] = []

    def fake_run_sample(
        _config: TFBearConfig,
        _clients: StageClients,
        sample: MedRegSampleInput,
        *,
        close_book: bool,
    ) -> TFBearResult:
        processed.append(sample.sample_id)
        if sample.sample_id == "MEDREG-TEST-0002":
            raise RuntimeError("endpoint rejected sample MEDREG-TEST-0002")
        return _result_for(sample)

    monkeypatch.setattr(run_support, "run_one_sample", fake_run_sample)

    result = _invoke(samples, output)

    assert result.exit_code != 0
    assert processed == ["MEDREG-TEST-0001", "MEDREG-TEST-0002", "MEDREG-TEST-0003"]
    predictions = MedRegEvaluator.validate_submission_file(output)
    assert [prediction.sample_id for prediction in predictions] == [
        "MEDREG-TEST-0001",
        "MEDREG-TEST-0003",
    ]
    trace_text = output.with_name(output.name + ".trace.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in trace_text.splitlines() if line]
    failed = [record for record in records if record.get("status") == "failed"]
    assert failed == [
        {
            "record": "sample",
            "status": "failed",
            "sample_id": "MEDREG-TEST-0002",
            "error_type": "RuntimeError",
            "error": "endpoint rejected sample MEDREG-TEST-0002",
        }
    ]
    assert "Traceback" not in trace_text
    assert "samples_failed=1" in result.stderr
    assert "samples_done=2" in result.stderr

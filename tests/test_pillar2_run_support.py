"""Tests for the shared TF-BEAR run loop, the trace record and the benchmark script front end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import scripts.run_tfbear_benchmark as benchmark

from mmregcheck.pillar2.expansion import StandardExpansionStrategy
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.prompts import load_prompt_pack
from mmregcheck.pillar2.results import (
    StageUsage,
    TFBearAdjudicationResult,
    TFBearResult,
)
from mmregcheck.pillar2.run_support import (
    RunSummary,
    SamplingSettings,
    build_config,
    default_trace_path,
    parse_expansion_param,
    parse_expansion_params,
    validate_expansion_params,
)
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

ENDPOINT = EndpointConfig(
    base_url="https://endpoint.invalid/v1",
    env_var="MMREG_TEST_API_KEY",
    model="test-model",
)


def _pack() -> Any:
    return load_prompt_pack(RegulatoryDomain.BEAUTY, "standard")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("n_paths=3", ("n_paths", 3)),
        ("temperature=0.7", ("temperature", 0.7)),
        ("min_steps=-0", ("min_steps", 0)),
        ("thought_temperature=1", ("thought_temperature", 1)),
        ("eval_top_p=true", ("eval_top_p", True)),
        ("breadth_limit=FALSE", ("breadth_limit", False)),
        ("  n_evals = 2 ", ("n_evals", 2)),
    ],
)
def test_parse_expansion_param_reads_int_float_and_bool(
    raw: str, expected: tuple[str, Any]
) -> None:
    assert parse_expansion_param(raw) == expected


@pytest.mark.parametrize("raw", ["n_paths", "=1", "n_paths=", "n_paths=two", "n_paths=1 2"])
def test_parse_expansion_param_rejects_malformed_input(raw: str) -> None:
    with pytest.raises(ValueError, match=r"--expansion-param|expected an int"):
        parse_expansion_param(raw)


def test_parse_expansion_params_collects_repeated_keys() -> None:
    assert parse_expansion_params(["n_paths=2", "temperature=0.3"]) == {
        "n_paths": 2,
        "temperature": 0.3,
    }
    assert parse_expansion_params([]) == {}


def test_build_config_keeps_the_defaults_when_nothing_is_requested() -> None:
    config = build_config(
        _pack(),
        endpoint=ENDPOINT,
        tau_posterior=0.40,
        tau_delta=0.60,
        sampling=SamplingSettings(),
    )

    assert config.temperature == 0.2
    assert config.top_p == 0.95
    assert config.max_tokens == 1024
    assert config.elicitation_temperature == 0.0
    assert config.elicitation_top_p == 0.95
    assert config.top_logprobs == 5
    assert config.verbose is False
    assert config.tau_posterior == 0.40
    assert config.tau_delta == 0.60
    assert isinstance(config.expansion_strategy, StandardExpansionStrategy)
    assert config.expansion_strategy.n_paths == 3
    assert config.expansion_strategy.temperature == 0.2


def test_build_config_applies_explicit_overrides_last() -> None:
    config = build_config(
        _pack(),
        endpoint=ENDPOINT,
        tau_posterior=0.50,
        tau_delta=0.50,
        sampling=SamplingSettings(
            temperature=0.6,
            top_p=0.7,
            max_tokens=333,
            elicitation_temperature=0.2,
            elicitation_top_p=0.8,
            top_logprobs=9,
            verbose=True,
        ),
        expansion_params={"n_paths": 2, "max_tokens": 111},
    )

    assert config.temperature == 0.6
    assert config.top_p == 0.7
    assert config.max_tokens == 333
    assert config.elicitation_temperature == 0.2
    assert config.elicitation_top_p == 0.8
    assert config.top_logprobs == 9
    assert config.verbose is True
    assert isinstance(config.expansion_strategy, StandardExpansionStrategy)
    assert config.expansion_strategy.n_paths == 2
    assert config.expansion_strategy.max_tokens == 111


def test_build_config_rejects_an_unknown_expansion_param() -> None:
    with pytest.raises(ValueError, match="unknown key") as excinfo:
        build_config(
            _pack(),
            endpoint=ENDPOINT,
            tau_posterior=0.50,
            tau_delta=0.50,
            sampling=SamplingSettings(),
            expansion_params={"bogus": 1},
        )

    assert "bogus" in str(excinfo.value)
    assert "n_paths" in str(excinfo.value)


def test_validate_expansion_params_checks_the_chosen_variant() -> None:
    validate_expansion_params("standard", {})
    validate_expansion_params("standard", {"n_paths": 2})
    validate_expansion_params("tot", {"breadth_limit": 2})

    with pytest.raises(ValueError, match="breadth_limit") as excinfo:
        validate_expansion_params("standard", {"breadth_limit": 2})
    assert "n_paths" in str(excinfo.value)


def test_default_trace_path_appends_the_suffix() -> None:
    assert default_trace_path(Path("out/predictions.jsonl")) == Path(
        "out/predictions.jsonl.trace.jsonl"
    )


def _branch(
    branch_id: str,
    *,
    prior: float,
    posterior: float,
    verdict: ComplianceVerdict = NON,
) -> TFBearBranchTrace:
    return TFBearBranchTrace(
        branch_id=branch_id,
        strategy=f"R{branch_id}: test",
        prior_prob=prior,
        posterior_prob=posterior,
        predicted_verdict=verdict,
        rationale=f"rationale {branch_id}",
    )


def _result(*, used_fallback: bool = False) -> TFBearResult:
    surviving = _branch("P1", prior=0.50, posterior=0.95)
    pruned = _branch("P2", prior=0.60, posterior=0.40, verdict=COMPLIANT)
    state = TFBearState(
        text_claim="A spa advertises a TCA chemical peel.",
        p_prior={NON: 0.500, PARTLY: 0.300, COMPLIANT: 0.200},
        branches=[surviving, pruned],
        adjudication=TFBearAdjudicationResult(
            final_verdict=NON,
            surviving_branches=[surviving],
            pruned_branches=[pruned],
            used_fallback=used_fallback,
        ),
    )
    return TFBearResult(
        state=state,
        verdict=NON,
        explanation="The strict branch drives the verdict.",
        usage={
            "prior": StageUsage(
                latency_seconds=0.5,
                prompt_tokens=10,
                completion_tokens=1,
                total_tokens=11,
            )
        },
        diagnostics=["a recorded diagnostic"],
    )


def test_to_trace_dict_exposes_the_branches_and_the_gate() -> None:
    result = _result()
    trace = result.to_trace_dict()

    assert trace["verdict"] == NON.value
    assert trace["used_fallback"] is False
    assert trace["p_prior"] == {
        NON.value: 0.500,
        PARTLY.value: 0.300,
        COMPLIANT.value: 0.200,
    }
    assert trace["n_branches"] == 2
    assert trace["n_surviving"] == 1
    assert [branch["gate"] for branch in trace["branches"]] == ["survived", "pruned"]
    assert trace["branches"][0]["log_odds_delta"] == result.state.branches[0].log_odds_delta
    assert trace["branches"][1]["predicted_verdict"] == COMPLIANT.value
    assert trace["usage"]["prior"] == {
        "latency_seconds": 0.5,
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "total_tokens": 11,
    }
    assert trace["explanation"] == "The strict branch drives the verdict."
    assert trace["diagnostics"] == ["a recorded diagnostic"]
    assert json.loads(json.dumps(trace)) == trace


def test_to_trace_dict_is_pure_and_repeatable() -> None:
    result = _result(used_fallback=True)
    branch_ids = [branch.branch_id for branch in result.state.branches]

    first = result.to_trace_dict()
    second = result.to_trace_dict()

    assert first == second
    assert first is not second
    assert first["used_fallback"] is True
    assert [branch.branch_id for branch in result.state.branches] == branch_ids
    first["diagnostics"].append("mutated copy")
    assert result.diagnostics == ["a recorded diagnostic"]


def test_to_trace_dict_handles_a_result_without_adjudication() -> None:
    result = TFBearResult(
        state=TFBearState(text_claim="claim"),
        verdict=NON,
        explanation="explanation",
        usage={},
    )

    trace = result.to_trace_dict()

    assert trace["used_fallback"] is False
    assert trace["n_surviving"] == 0
    assert trace["branches"] == []
    assert trace["usage"] == {}


def _write_samples(path: Path, count: int = 1) -> None:
    lines = []
    for index in range(count):
        sample = MedRegSampleInput(
            sample_id=f"MEDREG-TEST-{index + 1:04d}",
            domain=RegulatoryDomain.BEAUTY,
            text_claim="A spa advertises a TCA chemical peel.",
            image_rel_path="images/ignored.jpg",
            business_info=BusinessMetadata(facility_name="Test Spa"),
            retrieved_regulatory_context=["Rule one."],
        )
        lines.append(sample.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_script_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed: int = 0,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run_benchmark(samples: Any, **kwargs: Any) -> RunSummary:
        captured["samples"] = samples
        captured.update(kwargs)
        return RunSummary(
            done=len(samples),
            failed=failed,
            skipped=0,
            diagnostics=0,
            predictions_path=kwargs["predictions_path"],
            trace_path=kwargs["trace_path"],
        )

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)
    return captured


def test_benchmark_script_forwards_the_new_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples)
    captured = _install_script_capture(monkeypatch)

    code = benchmark.main(
        [
            "--input",
            str(samples),
            "--out",
            str(output),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
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
            "--resume",
        ]
    )

    assert code == 0
    sampling = captured["sampling"]
    assert isinstance(sampling, SamplingSettings)
    assert sampling.temperature == 0.7
    assert sampling.top_p == 0.8
    assert sampling.max_tokens == 256
    assert sampling.elicitation_temperature == 0.1
    assert sampling.elicitation_top_p == 0.9
    assert sampling.top_logprobs == 7
    assert sampling.verbose is True
    assert captured["expansion_params"] == {"n_paths": 2, "temperature": 0.3}
    assert captured["resume"] is True
    assert captured["domain"] is None
    assert captured["expansion"] == "standard"
    assert captured["trace_path"] == default_trace_path(output)


def test_benchmark_script_passes_nothing_when_the_flags_are_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples)
    captured = _install_script_capture(monkeypatch)

    code = benchmark.main(
        [
            "--input",
            str(samples),
            "--out",
            str(output),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
            "--trace-path",
            str(tmp_path / "explicit-trace.jsonl"),
        ]
    )

    assert code == 0
    assert captured["sampling"] == SamplingSettings()
    assert captured["expansion_params"] == {}
    assert captured["resume"] is False
    assert captured["trace_path"] == tmp_path / "explicit-trace.jsonl"


def test_benchmark_script_returns_non_zero_when_a_sample_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples)
    _install_script_capture(monkeypatch, failed=1)

    code = benchmark.main(
        [
            "--input",
            str(samples),
            "--out",
            str(output),
            "--base-url",
            "https://endpoint.invalid/v1",
            "--model",
            "test-model",
        ]
    )

    assert code == 1


def test_benchmark_script_rejects_a_bad_expansion_param(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    _write_samples(samples)
    common = [
        "--input",
        str(samples),
        "--out",
        str(tmp_path / "predictions.jsonl"),
        "--base-url",
        "https://endpoint.invalid/v1",
        "--model",
        "test-model",
    ]

    with pytest.raises(SystemExit, match="bogus"):
        benchmark.main([*common, "--expansion-param", "bogus=1"])
    with pytest.raises(SystemExit, match="expected an int"):
        benchmark.main([*common, "--expansion-param", "n_paths=two"])


def test_benchmark_script_forwards_extra_body_and_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_samples(samples)
    captured = _install_script_capture(monkeypatch)
    common = [
        "--input",
        str(samples),
        "--out",
        str(output),
        "--base-url",
        "https://endpoint.invalid/v1",
        "--model",
        "test-model",
    ]

    code = benchmark.main(
        [
            *common,
            "--extra-body",
            '{"chat_template_kwargs": {"enable_thinking": false}}',
        ]
    )
    assert code == 0
    assert captured["sampling"].extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    with pytest.raises(SystemExit, match="invalid --extra-body JSON"):
        benchmark.main([*common, "--extra-body", "{bad-json"])
    with pytest.raises(SystemExit, match="expected a JSON object"):
        benchmark.main([*common, "--extra-body", "[1, 2]"])

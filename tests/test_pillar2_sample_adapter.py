"""Tests for the MedReg-MM sample and submission adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmregcheck.evaluator import MedRegEvaluator
from mmregcheck.pillar2.engine import TFBearResult
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.sample_adapter import (
    FALLBACK_EXPLANATION,
    load_samples,
    state_from_sample,
    to_submission_prediction,
    write_predictions,
)
from mmregcheck.schemas import (
    BusinessMetadata,
    ComplianceVerdict,
    MedRegSampleInput,
    RegulatoryDomain,
)


def _sample(sample_id: str = "MEDREG-TEST-0001") -> MedRegSampleInput:
    return MedRegSampleInput(
        sample_id=sample_id,
        domain=RegulatoryDomain.BEAUTY,
        text_claim="A spa advertises a TCA chemical peel.",
        image_rel_path="images/ignored.jpg",
        business_info=BusinessMetadata(facility_name="Test Spa"),
        retrieved_regulatory_context=[
            "Basic Skincare Spa Scope: non-invasive cleansing only.",
            "   ",
            "Chemical peeling (TCA) is prohibited.",
        ],
    )


def _result(explanation: str = "A valid rationale.") -> TFBearResult:
    return TFBearResult(
        state=TFBearState(text_claim="claim"),
        verdict=ComplianceVerdict.NON_COMPLIANT,
        explanation=explanation,
        usage={},
    )


def test_state_mapping_joins_context_and_ignores_the_image() -> None:
    state = state_from_sample(_sample())

    assert state.text_claim == "A spa advertises a TCA chemical peel."
    assert state.retrieved_regulatory_context == (
        "Basic Skincare Spa Scope: non-invasive cleansing only.\n"
        "Chemical peeling (TCA) is prohibited."
    )
    assert state.nodes == []
    assert state.branches == []


def test_empty_context_stays_empty() -> None:
    sample = _sample().model_copy(update={"retrieved_regulatory_context": []})
    assert state_from_sample(sample).retrieved_regulatory_context == ""


def test_submission_mapping_keeps_a_valid_explanation() -> None:
    prediction = to_submission_prediction(_result("A valid rationale."), "MEDREG-TEST-0001")

    assert prediction.sample_id == "MEDREG-TEST-0001"
    assert prediction.overall_verdict == ComplianceVerdict.NON_COMPLIANT
    assert prediction.explanation == "A valid rationale."
    assert prediction.extracted_techniques == []


@pytest.mark.parametrize("explanation", ["", "   ", "tiny"])
def test_submission_mapping_fills_a_schema_valid_fallback(explanation: str) -> None:
    prediction = to_submission_prediction(_result(explanation), "MEDREG-TEST-0001")

    assert len(prediction.explanation) >= 5
    assert prediction.explanation == FALLBACK_EXPLANATION.format(
        verdict=ComplianceVerdict.NON_COMPLIANT.value
    )


def test_load_samples_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        _sample("MEDREG-TEST-0001").model_dump_json()
        + "\n\n"
        + _sample("MEDREG-TEST-0002").model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    samples = load_samples(path)

    assert [sample.sample_id for sample in samples] == ["MEDREG-TEST-0001", "MEDREG-TEST-0002"]


def test_load_samples_reports_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text('{"sample_id": "only"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=":1:"):
        load_samples(path)


def test_load_samples_rejects_missing_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_samples(tmp_path / "missing.jsonl")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_samples(empty)


def test_predictions_round_trip_through_the_submission_validator(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "predictions.jsonl"
    predictions = [
        to_submission_prediction(_result(), "MEDREG-TEST-0001"),
        to_submission_prediction(_result(""), "MEDREG-TEST-0002"),
    ]

    write_predictions(out, predictions)

    validated = MedRegEvaluator.validate_submission_file(out)
    assert [prediction.sample_id for prediction in validated] == [
        "MEDREG-TEST-0001",
        "MEDREG-TEST-0002",
    ]
    assert out.read_text(encoding="utf-8").endswith("\n")

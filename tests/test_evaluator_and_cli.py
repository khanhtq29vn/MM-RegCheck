"""Unit tests for `MedRegEvaluator`, CLI commands, and encrypted vault."""

from __future__ import annotations

from pathlib import Path

from scripts.crypto_vault import decrypt_bytes, encrypt_bytes
from typer.testing import CliRunner

from mmregcheck.cli import app
from mmregcheck.evaluator import MedRegEvaluator
from mmregcheck.schemas import ComplianceVerdict

runner = CliRunner()


def test_macro_f1_calculation_and_submission_validator() -> None:
    y_true = [
        ComplianceVerdict.COMPLIANT,
        ComplianceVerdict.PARTLY_COMPLIANT,
        ComplianceVerdict.NON_COMPLIANT,
    ]
    y_pred = [
        ComplianceVerdict.COMPLIANT,
        ComplianceVerdict.PARTLY_COMPLIANT,
        ComplianceVerdict.NON_COMPLIANT,
    ]
    metrics = MedRegEvaluator.compute_metrics(y_true, y_pred)
    assert metrics.macro_f1 == 1.0
    assert metrics.sample_count == 3

    baseline_path = Path("submissions/baselines/qwen3_5_9b_sample.jsonl")
    res = runner.invoke(app, ["validate-submission", str(baseline_path)])
    assert res.exit_code == 0

    demo_res = runner.invoke(app, ["demo-tfbear"])
    assert demo_res.exit_code == 0
    assert "Non-Compliant" in demo_res.stdout


def test_crypto_vault_roundtrip() -> None:
    secret = "medreg-2026-aes-secret"
    payload = b'{"sample_id": "MEDREG-TEST-0001", "overall_verdict": "Non-Compliant"}\n'
    encrypted = encrypt_bytes(payload, secret)
    assert b"Non-Compliant" not in encrypted
    decrypted = decrypt_bytes(encrypted, secret)
    assert decrypted == payload

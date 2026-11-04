"""Unit tests for `TF-BEAR` log-odds attribution math and Strict Liability hierarchy."""

from __future__ import annotations

import pytest

from mmregcheck.pillar2.tf_bear import TFBearAdjudicator
from mmregcheck.schemas import (
    ComplianceVerdict,
    TFBearBranchTrace,
    aggregate_strict_hierarchy,
)


def test_tfbear_log_odds_delta_matches_paper_section_5_4() -> None:
    """Verify Section 5.4 Case Study (Figure 3) log-odds delta and gate pruning."""
    r1_strict = TFBearBranchTrace(
        branch_id="P1",
        strategy="R1: Strict",
        prior_prob=0.227,
        posterior_prob=0.982,
        predicted_verdict=ComplianceVerdict.NON_COMPLIANT,
        rationale="Chemical peels are prohibited for basic skincare.",
    )
    r2_permissive = TFBearBranchTrace(
        branch_id="P2",
        strategy="R2: Permissive",
        prior_prob=0.617,
        posterior_prob=0.420,
        predicted_verdict=ComplianceVerdict.COMPLIANT,
        rationale="Hallucinated permissive path.",
    )
    r3_partial = TFBearBranchTrace(
        branch_id="P3",
        strategy="R3: Partial",
        prior_prob=0.227,
        posterior_prob=0.704,
        predicted_verdict=ComplianceVerdict.NON_COMPLIANT,
        rationale="Exceeds basic skincare scope.",
    )

    assert r1_strict.log_odds_delta == pytest.approx(5.224, abs=0.02)
    assert r3_partial.log_odds_delta == pytest.approx(2.091, abs=0.02)
    assert r1_strict.passes_dual_threshold_gate()
    assert not r2_permissive.passes_dual_threshold_gate()
    assert r3_partial.passes_dual_threshold_gate()

    res = TFBearAdjudicator().adjudicate_branches([r1_strict, r2_permissive, r3_partial])
    assert res.final_verdict == ComplianceVerdict.NON_COMPLIANT
    assert len(res.surviving_branches) == 2
    assert len(res.pruned_branches) == 1


def test_strict_liability_hierarchy_aggregation() -> None:
    """Verify `Non-Compliant` overrides `Compliant` and `Partly-Compliant`."""
    assert (
        aggregate_strict_hierarchy([ComplianceVerdict.COMPLIANT, ComplianceVerdict.NON_COMPLIANT])
        == ComplianceVerdict.NON_COMPLIANT
    )
    assert (
        aggregate_strict_hierarchy([ComplianceVerdict.COMPLIANT, ComplianceVerdict.PARTLY_COMPLIANT])
        == ComplianceVerdict.PARTLY_COMPLIANT
    )
    assert (
        aggregate_strict_hierarchy([ComplianceVerdict.COMPLIANT, ComplianceVerdict.COMPLIANT])
        == ComplianceVerdict.COMPLIANT
    )

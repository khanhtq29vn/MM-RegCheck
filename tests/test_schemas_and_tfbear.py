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


def _branch(
    branch_id: str,
    prior: float,
    posterior: float,
    verdict: ComplianceVerdict,
    distribution: dict[ComplianceVerdict, float] | None = None,
) -> TFBearBranchTrace:
    return TFBearBranchTrace(
        branch_id=branch_id,
        strategy="test",
        prior_prob=prior,
        posterior_prob=posterior,
        predicted_verdict=verdict,
        rationale="test branch",
        posterior_distribution=distribution,
    )


def test_log_odds_epsilon_keeps_saturated_probabilities_finite() -> None:
    """Appendix H: epsilon = 1e-5 stabilises p = 0 and p = 1, which softmax can produce."""
    assert TFBearBranchTrace.log_odds(0.0) == pytest.approx(-11.5129, abs=1e-3)
    assert TFBearBranchTrace.log_odds(1.0) == pytest.approx(11.5129, abs=1e-3)
    saturated = _branch("S", 0.0, 1.0, ComplianceVerdict.NON_COMPLIANT)
    assert saturated.log_odds_delta == pytest.approx(23.0259, abs=1e-3)


def test_section_5_4_deltas_match_paper_to_three_decimals() -> None:
    cases = [(0.227, 0.982, 5.224), (0.617, 0.420, -0.7996), (0.227, 0.704, 2.0917)]
    for prior, posterior, expected in cases:
        branch = _branch("P", prior, posterior, ComplianceVerdict.NON_COMPLIANT)
        assert branch.log_odds_delta == pytest.approx(expected, abs=1e-3)


def test_tie_is_broken_by_summed_posterior_not_by_verdict_priority() -> None:
    """One vote each: the label with the larger summed posterior wins, even if Compliant."""
    non_compliant = _branch("A", 0.2, 0.80, ComplianceVerdict.NON_COMPLIANT)
    compliant = _branch("B", 0.2, 0.95, ComplianceVerdict.COMPLIANT)
    result = TFBearAdjudicator().adjudicate_branches([non_compliant, compliant])
    assert result.final_verdict == ComplianceVerdict.COMPLIANT
    assert not result.used_fallback


def test_fallback_uses_full_summed_posterior_when_no_branch_passes_gate() -> None:
    """Appendix H: with no surviving branch, pick the label with the highest summed posterior."""
    first = _branch(
        "A",
        0.6,
        0.4,
        ComplianceVerdict.COMPLIANT,
        {
            ComplianceVerdict.COMPLIANT: 0.4,
            ComplianceVerdict.NON_COMPLIANT: 0.35,
            ComplianceVerdict.PARTLY_COMPLIANT: 0.25,
        },
    )
    second = _branch(
        "B",
        0.6,
        0.4,
        ComplianceVerdict.PARTLY_COMPLIANT,
        {
            ComplianceVerdict.COMPLIANT: 0.1,
            ComplianceVerdict.NON_COMPLIANT: 0.5,
            ComplianceVerdict.PARTLY_COMPLIANT: 0.4,
        },
    )
    result = TFBearAdjudicator().adjudicate_branches([first, second])
    assert result.used_fallback
    assert result.surviving_branches == []
    assert result.final_verdict == ComplianceVerdict.NON_COMPLIANT


def test_fallback_without_distribution_counts_own_verdict_posterior() -> None:
    low = _branch("A", 0.6, 0.30, ComplianceVerdict.NON_COMPLIANT)
    higher = _branch("B", 0.6, 0.45, ComplianceVerdict.COMPLIANT)
    result = TFBearAdjudicator().adjudicate_branches([low, higher])
    assert result.used_fallback
    assert result.final_verdict == ComplianceVerdict.COMPLIANT


def test_separate_thresholds_are_applied_independently() -> None:
    branch = _branch("A", 0.2, 0.9, ComplianceVerdict.NON_COMPLIANT)
    assert branch.passes_dual_threshold_gate(tau_posterior=0.5, tau_delta=0.5)
    assert not branch.passes_dual_threshold_gate(tau_posterior=0.95, tau_delta=0.5)
    assert not branch.passes_dual_threshold_gate(tau_posterior=0.5, tau_delta=5.0)

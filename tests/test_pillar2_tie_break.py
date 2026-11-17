"""Tests for the single deterministic label tie rule."""

from __future__ import annotations

import pytest

from mmregcheck.pillar2.tie_break import (
    TIE_BREAK_ORDER,
    select_verdict,
    select_verdict_keyed,
)
from mmregcheck.schemas import ComplianceVerdict

NON = ComplianceVerdict.NON_COMPLIANT
PARTLY = ComplianceVerdict.PARTLY_COMPLIANT
COMPLIANT = ComplianceVerdict.COMPLIANT


def test_tie_break_order_is_strictest_first() -> None:
    assert TIE_BREAK_ORDER == (NON, PARTLY, COMPLIANT)


def test_all_equal_scores_resolve_to_non_compliant() -> None:
    assert select_verdict({NON: 1 / 3, PARTLY: 1 / 3, COMPLIANT: 1 / 3}) is NON


@pytest.mark.parametrize(
    "order",
    [
        (NON, PARTLY, COMPLIANT),
        (COMPLIANT, PARTLY, NON),
        (PARTLY, COMPLIANT, NON),
    ],
)
def test_all_equal_scores_ignore_dict_order(
    order: tuple[ComplianceVerdict, ...],
) -> None:
    scores = dict.fromkeys(order, 1 / 3)

    assert select_verdict(scores) is NON


def test_partly_and_compliant_tie_resolves_to_partly() -> None:
    assert select_verdict({COMPLIANT: 0.5, PARTLY: 0.5, NON: 0.0}) is PARTLY


def test_a_strictly_higher_score_beats_the_tie_order() -> None:
    assert select_verdict({NON: 0.2, PARTLY: 0.2, COMPLIANT: 0.6}) is COMPLIANT


def test_missing_labels_count_as_zero() -> None:
    assert select_verdict({NON: 0.0, PARTLY: 0.0, COMPLIANT: 0.1}) is COMPLIANT


def test_keyed_all_equal_candidates_resolve_to_non_compliant() -> None:
    candidates = {NON: 2, PARTLY: 2, COMPLIANT: 2}

    winner = select_verdict_keyed(candidates, key=lambda verdict: (candidates[verdict], 1.0))

    assert winner is NON


def test_keyed_tie_ignores_candidate_iteration_order() -> None:
    candidates = [COMPLIANT, NON, PARTLY]

    winner = select_verdict_keyed(candidates, key=lambda _verdict: (1, 1.0))

    assert winner is NON


def test_keyed_primary_key_wins_before_the_tie_rule() -> None:
    candidates = [NON, COMPLIANT]

    winner = select_verdict_keyed(candidates, key=lambda verdict: 1 if verdict is COMPLIANT else 0)

    assert winner is COMPLIANT


def test_keyed_selection_requires_a_candidate() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        select_verdict_keyed([], key=lambda _verdict: 0.0)

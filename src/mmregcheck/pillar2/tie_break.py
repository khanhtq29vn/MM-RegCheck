"""Deterministic resolution of exact ties between verdict labels.

Every label-valued maximum in the pipeline goes through this module, so an exact tie does not
depend on branch order, on the declaration order of `ComplianceVerdict`, or on dictionary order.

The single tie rule, preferred first:

    Non-Compliant, Partly-Compliant, Compliant

`Non-Compliant` is the strictest verdict, so an unbroken tie resolves toward the label that keeps
the stricter reading of the advertisement. The rule is documented once here and reused by the
per-branch argmax of scoring, the adjudicator's vote tie-break, its no-survivor fallback and the
joint adjudication strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

from mmregcheck.schemas import ComplianceVerdict

#: Tie-break preference, strictest first. The first label of an exact tie wins.
TIE_BREAK_ORDER: tuple[ComplianceVerdict, ...] = (
    ComplianceVerdict.NON_COMPLIANT,
    ComplianceVerdict.PARTLY_COMPLIANT,
    ComplianceVerdict.COMPLIANT,
)

_LabelKey = TypeVar("_LabelKey")


def select_verdict(scores: Mapping[ComplianceVerdict, float]) -> ComplianceVerdict:
    """Return the label with the highest score, resolving an exact tie by `TIE_BREAK_ORDER`.

    Labels missing from `scores` count as `0.0`, so a distribution that covers only part of the
    enum still resolves deterministically.

    Args:
        scores: Label -> score mapping, for example a posterior distribution or a summed tally.

    Returns:
        The highest-scoring label; on an exact tie, the earliest label of `TIE_BREAK_ORDER`.
    """
    winner = TIE_BREAK_ORDER[0]
    best = scores.get(winner, 0.0)
    for verdict in TIE_BREAK_ORDER[1:]:
        score = scores.get(verdict, 0.0)
        if score > best:
            winner, best = verdict, score
    return winner


def select_verdict_keyed(
    candidates: Iterable[ComplianceVerdict],
    key: Callable[[ComplianceVerdict], Any],
) -> ComplianceVerdict:
    """Return the candidate with the greatest `key`, resolving an exact tie by `TIE_BREAK_ORDER`.

    Args:
        candidates: Labels that are eligible to win, for example the keys of a vote counter.
        key: Total-order key of one label, for example `(vote_count, summed_posterior)`.

    Returns:
        The winning candidate; on an exact tie, the earliest candidate of `TIE_BREAK_ORDER`.

    Raises:
        ValueError: If `candidates` is empty.
    """
    eligible = set(candidates)
    winner: ComplianceVerdict | None = None
    best_key: Any = None
    for verdict in TIE_BREAK_ORDER:
        if verdict not in eligible:
            continue
        current = key(verdict)
        if winner is None or current > best_key:
            winner, best_key = verdict, current
    if winner is None:
        raise ValueError("select_verdict_keyed requires at least one candidate.")
    return winner

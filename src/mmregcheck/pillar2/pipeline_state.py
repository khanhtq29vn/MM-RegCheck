"""Shared state of the TF-BEAR pipeline.

A single `TFBearState` accumulates the inputs and outputs of every stage. Strategies receive
the whole object and must only read the fields their stage contract allows:

    Stage 1 (Prior):        text_claim
    Stage 2 (Expansion):    text_claim, retrieved_regulatory_context
    Stage 3 (Scoring):      ... + p_prior, nodes
    Stage 4 (Adjudication): ... + branches
    Stage 5 (Explanation):  ... + adjudication

Later fields stay empty until the stage that owns them writes them.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any

from mmregcheck.pillar2.results import TFBearAdjudicationResult
from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace

# Placeholders a prompt template may reference. `paths`, `state` and `n_paths` are injected by the
# strategies at render time, the others come from the state fields.
STATE_FIELDS: frozenset[str] = frozenset(
    {
        "text_claim",
        "retrieved_regulatory_context",
        "p_prior",
        "p_posterior",
        "nodes",
        "branches",
        "verdict",
        "paths",
        "state",
        "n_paths",
    }
)


def _root_name(field_name: str) -> str:
    return field_name.split(".")[0].split("[")[0]


def validate_template(template: str, strategy_name: str) -> None:
    """Raise `ValueError` if `template` references a placeholder the pipeline cannot fill."""
    unknown = [
        _root_name(name)
        for _, name, _, _ in string.Formatter().parse(template)
        if name is not None and _root_name(name) and _root_name(name) not in STATE_FIELDS
    ]
    if unknown:
        raise ValueError(
            f"{strategy_name}: prompt_template references unknown placeholder(s) {unknown!r}. "
            f"Available placeholders: {sorted(STATE_FIELDS)!r}."
        )


@dataclass
class Node:
    """A reasoning path produced by an expansion strategy (before scoring).

    Attributes:
        state: Full reasoning text from the root to this node.
        thought: The last reasoning step that produced this node.
        is_terminal: Whether the expansion strategy flagged the path as complete.
        answer: Raw answer emitted by the model when `is_terminal`.
        value: Heuristic score assigned during tree search, if any.
        meta: Strategy-specific metadata (e.g. tree depth, CoT sample index).
    """

    state: str = ""
    thought: str = ""
    is_terminal: bool = False
    answer: str | None = None
    value: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TFBearState:
    """Inputs and per-stage outputs of one TF-BEAR run."""

    text_claim: str = ""
    retrieved_regulatory_context: str = ""
    p_prior: dict[ComplianceVerdict, float] = field(default_factory=dict)
    p_posterior: dict[ComplianceVerdict, float] = field(default_factory=dict)
    nodes: list[Node] = field(default_factory=list)
    branches: list[TFBearBranchTrace] = field(default_factory=list)
    adjudication: TFBearAdjudicationResult | None = None

    @property
    def verdict(self) -> str:
        """Winning verdict value, empty until the adjudication stage has run."""
        return self.adjudication.final_verdict.value if self.adjudication else ""

    def render(self, template: str, **extra: Any) -> str:
        """Render `template` with the state fields; `extra` overrides fields of the same name."""
        context: dict[str, Any] = {
            "text_claim": self.text_claim,
            "retrieved_regulatory_context": self.retrieved_regulatory_context,
            "p_prior": self.p_prior,
            "p_posterior": self.p_posterior,
            "nodes": self.nodes,
            "branches": self.branches,
            "verdict": self.verdict,
        }
        context.update(extra)
        missing = {
            _root_name(name)
            for _, name, _, _ in string.Formatter().parse(template)
            if name is not None and _root_name(name) not in context
        }
        if missing:
            raise ValueError(f"Missing variables in template: {sorted(missing)!r}.")
        return template.format_map(context)
